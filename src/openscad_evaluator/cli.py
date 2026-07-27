"""Command-line entry point: evaluate an OpenSCAD script and export a mesh.

    openscad-evaluator model.scad -o model.stl
    openscad-evaluator model.scad -o model.3mf --debug
    openscad-evaluator model.scad -o model.stl --profile profile.txt

echo()/warning output goes to stdout. With --debug, drops into a gdb-style
interactive debugger (breakpoints, step/next/finish, print, backtrace) before
and during evaluation -- see `DebugRepl` in `_debug_repl.py`. With --profile
FILENAME, writes a per-call-site self/cumulative timing report to FILENAME
(see Evaluator's profile=True instrumentation and ProfileResult).
"""
from __future__ import annotations

import argparse
import os
import signal
import sys

from openscad_lalr_parser import getASTfromFile

from openscad_evaluator._debug_repl import DebugRepl
from openscad_evaluator.evaluator import EvalError, Evaluator, resolve_use_scopes, to_renderable_bodies
from openscad_evaluator.export import export_bodies, format_for_path


def _format_profile_report(source_path: str, profile) -> str:
    """Plain-text --profile report: a summary of resolve/generate/total
    time, then one row per call site sorted by self time descending (ties
    broken by call location/name, so the ordering is deterministic run to
    run -- ProfileResult.call_sites isn't self-time ordered). Ported
    identically (same column layout, same tie-break rule) to the C++
    port's own format_profile_report in cli_lib.cpp, so a report generated
    by either CLI on the same script looks the same."""
    sites = sorted(profile.call_sites, key=lambda s: (-s.self_time, s.call_origin, s.call_line, s.name))
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

    if args.debug:
        repl = DebugRepl(args.input)
        # Ctrl+C during a running evaluate() pauses like a breakpoint
        # (see DebugRepl.request_pause()'s own doc comment) instead of
        # raising an unhandled KeyboardInterrupt mid-AST-walk, which is
        # Python's own default SIGINT behavior otherwise.
        signal.signal(signal.SIGINT, lambda signum, frame: repl.request_pause())
        if not repl.run_prompt():
            return 0
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
        print(str(e), file=sys.stderr)
        return 1

    if args.profile:
        try:
            with open(args.profile, "w", encoding="utf-8") as f:
                f.write(_format_profile_report(args.input, evaluator.profile_result))
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
