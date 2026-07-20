"""Command-line entry point: evaluate an OpenSCAD script and export a mesh.

    openscad-evaluator model.scad -o model.stl
    openscad-evaluator model.scad -o model.3mf --debug

echo()/warning output goes to stdout. With --debug, drops into a gdb-style
interactive debugger (breakpoints, step/next/finish, print, backtrace) before
and during evaluation -- see `DebugRepl` in `_debug_repl.py`.
"""
from __future__ import annotations

import argparse
import sys

from openscad_lalr_parser import getASTfromFile

from openscad_evaluator._debug_repl import DebugRepl
from openscad_evaluator.evaluator import EvalError, Evaluator, resolve_use_scopes, to_renderable_bodies
from openscad_evaluator.export import export_bodies, format_for_path


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
        if not repl.run_prompt():
            return 0
        evaluator = Evaluator(
            debug_hook=repl.debug_hook, error_break_fn=repl.error_break, return_hook=repl.return_hook,
        )
    else:
        evaluator = Evaluator()

    try:
        bodies, _id_to_node = evaluator.evaluate(nodes, root_scope)
    except RecursionError:
        print("error: AST too deeply nested (recursion limit exceeded during evaluation)", file=sys.stderr)
        return 1
    except EvalError as e:
        print(str(e), file=sys.stderr)
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
