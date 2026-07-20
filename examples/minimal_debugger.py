"""Minimal example: hook a debugger into `Evaluator` via `debug_hook`.

`debug_hook` is called at every statement (and, for some constructs, at
every sub-expression when `expr_level=True`) before it runs. It receives
enough context to log/inspect current state, optionally override a
variable's value for that scope via the returned `mods` dict, and decide
whether to keep going (any `cmd` other than `"stop"`) or abort the whole
`evaluate()` call (`cmd="stop"`, which raises `EvalError`).

Real debuggers (breakpoints, step-over/step-into, blocking on user input)
are built entirely on top of this one hook -- the evaluator itself has no
stepping state of its own; it just calls `debug_hook` and does whatever it
says.

Run: python examples/minimal_debugger.py
"""
from openscad_lalr_parser import getASTfromString, build_scopes
from openscad_evaluator import Evaluator, EvalError

SCRIPT = """\
width = 10;
cube([width, width, width]);
translate([20, 0, 0]) sphere(r=5);
"""


def trace_every_statement():
    """Print (line, call depth, in-scope variables) at every statement."""
    def debug_hook(line, depth, forced, expr_level, expr_depth, origin, get_frames):
        (locals_, _all_frames), _call_stack = get_frames()
        print(f"  line {line:>2}  depth={depth}  locals={locals_}")
        return "continue", {}

    nodes = getASTfromString(SCRIPT)
    root_scope = build_scopes(nodes)
    ev = Evaluator(debug_hook=debug_hook)
    bodies, _id_to_node = ev.evaluate(nodes, root_scope)
    print(f"  -> {len(bodies)} bodies\n")
    assert len(bodies) == 2


def stop_at_breakpoint():
    """Abort evaluation as soon as a chosen line is reached."""
    break_line = 3  # the `translate(...) sphere(...)` statement

    def debug_hook(line, depth, forced, expr_level, expr_depth, origin, get_frames):
        if line == break_line:
            print(f"  breakpoint hit at line {line}")
            return "stop", {}
        return "continue", {}

    nodes = getASTfromString(SCRIPT)
    root_scope = build_scopes(nodes)
    ev = Evaluator(debug_hook=debug_hook)
    try:
        ev.evaluate(nodes, root_scope)
        raise AssertionError("expected evaluation to stop at the breakpoint")
    except EvalError:
        print("  evaluation aborted as expected\n")


def override_variable_via_mods():
    """The `mods` dict lets a debugger inject a "set variable" command
    mid-run -- e.g. a "change this value and keep going" watch expression."""
    def debug_hook(line, depth, forced, expr_level, expr_depth, origin, get_frames):
        if line == 2:  # about to run `cube([width, width, width])`
            return "continue", {"width": 2}  # override just for this scope
        return "continue", {}

    nodes = getASTfromString(SCRIPT)
    root_scope = build_scopes(nodes)
    ev = Evaluator(debug_hook=debug_hook)
    bodies, _id_to_node = ev.evaluate(nodes, root_scope)
    cube_body = bodies[0]
    lo_x, _lo_y, _lo_z, hi_x, _hi_y, _hi_z = cube_body.body.bounding_box()
    print(f"  cube size after override: {hi_x - lo_x} (script said width=10)")
    assert hi_x - lo_x == 2


def main():
    print("1. trace every statement:")
    trace_every_statement()
    print("2. stop at a breakpoint:")
    stop_at_breakpoint()
    print("3. override a variable via mods:")
    override_variable_via_mods()


if __name__ == "__main__":
    main()
