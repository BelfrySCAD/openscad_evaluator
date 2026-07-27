"""Command-line entry point: evaluate an OpenSCAD script and export a mesh.

    openscad-evaluator model.scad -o model.stl
    openscad-evaluator model.scad -o model.3mf --debug
    openscad-evaluator model.scad -o model.stl --profile profile.txt
    openscad-evaluator model.scad -o model.stl --profile profile.csv --profile-format csv \
        --profile-sort cumulative --profile-min-calls 10

echo()/warning output goes to stdout. With --debug, drops into a gdb-style
interactive debugger (breakpoints, step/next/finish, print, backtrace) before
and during evaluation -- see `DebugRepl` in `_debug_repl.py`. With --profile
FILENAME, writes a per-call-site self/cumulative timing report to FILENAME
(see Evaluator's profile=True instrumentation and ProfileResult) --
--profile-format selects text (default) or csv, --profile-sort picks the
sort key (self/cumulative/calls/name), and --profile-min-self/
--profile-min-calls filter out call sites below a threshold.
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import signal
import sys

from openscad_lalr_parser import FunctionDeclaration, ModuleDeclaration, getASTfromFile

from openscad_evaluator._debug_repl import DebugRepl, DeclInfo
from openscad_evaluator.evaluator import (
    DEBUGGING_STOPPED_MESSAGE, EvalError, Evaluator, resolve_use_scopes, to_renderable_bodies,
)
from openscad_evaluator.export import export_bodies, format_for_path


def _collect_declarations(nodes: list, cls: type, main_path: str) -> list[DeclInfo]:
    """"info functions"/"info modules"/"list <name>": one DeclInfo per
    top-level FunctionDeclaration/ModuleDeclaration node in the fully
    use-resolved node list, so a `use <file>`-injected declaration shows up
    too, matching what's actually callable -- sorted by name for
    deterministic "info" output (DebugRepl.set_declared_names() itself
    resolves each origin, so the raw node.position.origin here is enough).
    Ported identically to the C++ port's own collectDeclarations in
    cli_lib.cpp."""
    decls = [
        DeclInfo(
            name=n.name.name,
            params=", ".join(str(p) for p in n.parameters),
            origin=n.position.origin or main_path,
            line=n.position.line,
        )
        for n in nodes if isinstance(n, cls)
    ]
    decls.sort(key=lambda d: d.name)
    return decls


def _select_and_sort_call_sites(profile, sort: str, min_self: float, min_calls: int) -> list:
    """Filters call_sites to self_time >= min_self and call_count >=
    min_calls, then sorts by `sort` ("self" | "cumulative" | "calls" |
    "name"). Every non-"name" order ties-broken by (call_origin,
    call_line, name) -- call_sites' own storage order isn't sorted by any
    of these, so this needs an explicit, deterministic tie-break. Ported
    identically to the C++ port's own selectAndSortCallSites in
    cli_lib.cpp."""
    sites = [s for s in profile.call_sites if s.self_time >= min_self and s.call_count >= min_calls]
    if sort == "cumulative":
        sites.sort(key=lambda s: (-s.cumulative_time, s.call_origin, s.call_line, s.name))
    elif sort == "calls":
        sites.sort(key=lambda s: (-s.call_count, s.call_origin, s.call_line, s.name))
    elif sort == "name":
        sites.sort(key=lambda s: (s.name, s.call_origin, s.call_line))
    else:  # "self" (default)
        sites.sort(key=lambda s: (-s.self_time, s.call_origin, s.call_line, s.name))
    return sites


def _render_profile_report_text(source_path: str, profile, sites: list) -> str:
    lines = [
        f"Profile report for {source_path}",
        "",
        f"Total time:      {profile.total_time:.6f}s",
        f"  resolve:       {profile.resolve_time:.6f}s",
        f"  generate:      {profile.generate_time:.6f}s",
        f"  unattributed:  {profile.unattributed_time:.6f}s",
        "",
        f"{'kind':<8} {'name':<24} {'caller':<24} {'location':<28} {'calls':>6} {'self(s)':>12} {'cumulative(s)':>14}",
    ]
    for s in sites:
        origin = s.call_origin or source_path
        location = f"{os.path.basename(origin)}:{s.call_line}"
        lines.append(
            f"{s.kind:<8} {s.name:<24} {s.caller_name:<24} {location:<28} "
            f"{s.call_count:>6} {s.self_time:>12.6f} {s.cumulative_time:>14.6f}"
        )
    return "\n".join(lines) + "\n"


def _render_profile_report_csv(source_path: str, profile, sites: list) -> str:
    # The summary lives in "#"-prefixed comment lines ahead of the real
    # CSV header/rows -- readers that want just the tabular data can skip
    # them (e.g. pandas.read_csv(..., comment="#")) or a plain `grep -v
    # '^#'`; these aren't real CSV fields so aren't comma-escaped.
    buf = io.StringIO()
    buf.write(f"# source,{source_path}\n")
    buf.write(f"# total_time,{profile.total_time:.6f}\n")
    buf.write(f"# resolve_time,{profile.resolve_time:.6f}\n")
    buf.write(f"# generate_time,{profile.generate_time:.6f}\n")
    buf.write(f"# unattributed_time,{profile.unattributed_time:.6f}\n")
    writer = csv.writer(buf)
    writer.writerow(["kind", "name", "caller", "call_origin", "call_line", "call_count", "self_time", "cumulative_time"])
    for s in sites:
        origin = s.call_origin or source_path
        writer.writerow([
            s.kind, s.name, s.caller_name, origin, s.call_line, s.call_count,
            f"{s.self_time:.6f}", f"{s.cumulative_time:.6f}",
        ])
    return buf.getvalue()


def _format_profile_report(
    source_path: str, profile, fmt: str = "text", sort: str = "self", min_self: float = 0.0, min_calls: int = 0,
) -> str:
    """--profile report: a summary of resolve/generate/total time, then one
    row per call site (optionally filtered/sorted), as plain text (default)
    or CSV. Ported identically (same column layout, same tie-break rule,
    same CSV columns) to the C++ port's own formatProfileReport in
    cli_lib.cpp, so a report generated by either CLI on the same script
    looks the same."""
    sites = _select_and_sort_call_sites(profile, sort, min_self, min_calls)
    if fmt == "csv":
        return _render_profile_report_csv(source_path, profile, sites)
    return _render_profile_report_text(source_path, profile, sites)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="openscad-evaluator",
        description="Evaluate an OpenSCAD script and export a mesh.",
    )
    parser.add_argument("input", help="Path to the .scad file to evaluate")
    parser.add_argument("-o", "--output", required=True, help="Output mesh file (.stl, .obj, .off, or .3mf)")
    parser.add_argument(
        "--format", choices=["stl", "obj", "off", "3mf"],
        help="Explicit output format (default: inferred from --output's extension)",
    )
    parser.add_argument("--debug", action="store_true", help="Run under an interactive, gdb-style debugger")
    parser.add_argument("--profile", metavar="FILENAME", help="Write a per-call-site profiling report to FILENAME")
    parser.add_argument(
        "--profile-format", choices=["text", "csv"], default="text",
        help="Format for --profile's report (default: text)",
    )
    parser.add_argument(
        "--profile-sort", choices=["self", "cumulative", "calls", "name"], default="self",
        help="Sort key for --profile's call-site rows (default: self)",
    )
    parser.add_argument(
        "--profile-min-self", type=float, default=0.0, metavar="SECONDS",
        help="Omit call sites with self time below SECONDS (default: 0)",
    )
    parser.add_argument(
        "--profile-min-calls", type=int, default=0, metavar="N",
        help="Omit call sites with fewer than N calls (default: 0)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    try:
        fmt = args.format or format_for_path(args.output)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    nodes = getASTfromFile(args.input, include_comments=False)
    if nodes is None:
        return 1  # the parser already printed a formatted syntax error

    try:
        nodes, _own_nodes, root_scope = resolve_use_scopes(nodes, args.input, print)
    except RecursionError:
        print("error: AST too deeply nested (recursion limit exceeded while resolving 'use')", file=sys.stderr)
        return 1

    repl = None
    if args.debug:
        repl = DebugRepl(args.input)
        # Ctrl+C during a running evaluate() pauses like a breakpoint
        # (see DebugRepl.request_pause()'s own doc comment) instead of
        # raising an unhandled KeyboardInterrupt mid-AST-walk, which is
        # Python's own default SIGINT behavior otherwise.
        signal.signal(signal.SIGINT, lambda signum, frame: repl.request_pause())
        repl.set_declared_names(
            _collect_declarations(nodes, FunctionDeclaration, args.input),
            _collect_declarations(nodes, ModuleDeclaration, args.input),
        )

    # Runs at least once; loops again only when a paused --debug session
    # issues "stop" (back to the pre-run prompt) or "restart" (skip the
    # prompt, go straight back into a fresh run) -- both unwind out of
    # evaluate() via the same shared DEBUGGING_STOPPED_MESSAGE EvalError
    # "quit" already used, disambiguated below via
    # DebugRepl.take_post_run_action(). Without --debug this loop always
    # runs exactly once (need_prompt/repl-related branches are all no-ops
    # when not args.debug), so this is the same single-pass behavior as
    # before, not a new code path.
    need_prompt = True
    while True:
        if args.debug:
            if need_prompt and not repl.run_prompt():
                return 0  # user quit before ever running
            repl.prepare_for_run()
        need_prompt = True

        if args.debug:
            evaluator = Evaluator(
                debug_hook=repl.debug_hook, error_break_fn=repl.error_break, return_hook=repl.return_hook,
                profile=bool(args.profile),
            )
            repl._evaluator = evaluator  # lets "child" read Evaluator._last_children_positions
        else:
            evaluator = Evaluator(profile=bool(args.profile))

        try:
            bodies, _id_to_node = evaluator.evaluate(nodes, root_scope)
        except RecursionError:
            print("error: AST too deeply nested (recursion limit exceeded during evaluation)", file=sys.stderr)
            return 1
        except EvalError as e:
            if args.debug and str(e) == DEBUGGING_STOPPED_MESSAGE:
                action = repl.take_post_run_action()
                if action == "stopped":
                    print("Evaluation stopped.")
                    continue  # need_prompt stays True -> back to the pre-run prompt
                if action == "restart":
                    need_prompt = False  # skip the prompt, run again immediately
                    continue
                # action in ("quit", None): fall through to the generic
                # error path below -- str(e) is already
                # DEBUGGING_STOPPED_MESSAGE, matching this exact behavior
                # from before restart/stop existed.
            print(str(e), file=sys.stderr)
            return 1

        if args.profile:
            try:
                with open(args.profile, "w", encoding="utf-8") as f:
                    f.write(_format_profile_report(
                        args.input, evaluator.profile_result,
                        fmt=args.profile_format, sort=args.profile_sort,
                        min_self=args.profile_min_self, min_calls=args.profile_min_calls,
                    ))
            except OSError as e:
                print(f"error: {e}", file=sys.stderr)
                return 1

        bodies = to_renderable_bodies(bodies)
        try:
            export_bodies(args.output, bodies, fmt=fmt)
        except (ValueError, ImportError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1

        print(f"Exported to {args.output}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
