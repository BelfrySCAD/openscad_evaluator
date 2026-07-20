"""A minimal, gdb-style interactive debugger for the CLI's `--debug` flag.

Wires into `Evaluator` via the same `debug_hook`/`error_break_fn`/`return_hook`
callback contract any caller uses (see `examples/minimal_debugger.py`). Unlike
a GUI debugger, which needs a worker thread so pausing doesn't block the event
loop, this blocks synchronously on `input()` from inside the hook itself --
`evaluate()` is on the same thread as the prompt, so there's nothing else that
needs to keep running while paused.

Breakpoint/step-into/step-over/step-out semantics mirror BelfrySCAD's
`DebugSession._make_hook` (src/belfryscad/window/debugger.py), which already
went through several rounds of fixes for pausing correctly at geometry
statements -- ported rather than re-derived to avoid reintroducing those bugs.
"""
from __future__ import annotations

import os
from pathlib import Path


def _fmt(v) -> str:
    if v is None:
        return "undef"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, list):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    if isinstance(v, str):
        return f'"{v}"'
    from openscad_evaluator.evaluator import OscObject
    if isinstance(v, OscObject):
        inner = ", ".join(f"{k} = {_fmt(val)}" for k, val in v.items())
        return f"object({inner})"
    return str(v)


def _parse_value(s: str):
    if s == "undef":
        return None
    if s == "true":
        return True
    if s == "false":
        return False
    try:
        return float(s)
    except ValueError:
        pass
    if s.startswith('"') and s.endswith('"'):
        return s[1:-1]
    return None


_PRE_RUN_HELP = """\
Commands (before "run"):
  run, r                 Start evaluating the script
  break [file:]line, b   Set a breakpoint
  delete [file:]line, d  Delete a breakpoint (no args: delete all)
  info breakpoints       List breakpoints
  list [line], l         Show source around a line (default: start of file)
  quit, q                Exit without running
  help, h                Show this text"""

_PAUSED_HELP = """\
Commands (while paused):
  continue, c             Resume until the next breakpoint
  step, s                 Step into the next statement/call
  next, n                 Step over the next statement (don't descend into calls)
  finish, fin             Run until the current call returns
  print <name>, p         Print a variable's value
  backtrace, bt, where    Show the call stack (innermost first)
  list [line], l          Show source around a line (default: current line)
  break [file:]line, b    Set a breakpoint
  delete [file:]line, d   Delete a breakpoint (no args: delete all)
  set <name>=<value>      Override a variable's value on resume
  quit, q                 Abort evaluation
  help, h                 Show this text"""


class DebugRepl:
    """One instance per `--debug` run. Construct, wire its three methods into
    `Evaluator(debug_hook=repl.debug_hook, error_break_fn=repl.error_break,
    return_hook=repl.return_hook)`, call `run_prompt()` first, and only call
    `evaluate()` if it returns True."""

    def __init__(self, source_path: str):
        self._source_path = os.path.realpath(source_path)
        try:
            self._source_lines = Path(source_path).read_text(encoding="utf-8").splitlines()
        except OSError:
            self._source_lines = []
        self._breakpoints: dict[str, set[int]] = {}
        self._break_on_first = True
        self._step_cmd: str | None = None   # "into" / "over" / "out"
        self._step_line = 0
        self._step_depth = 0
        self._step_origin = ""
        self._pending_mods: dict = {}
        self._print_count = 0
        self._quit = False

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _resolve(self, origin: str | None) -> str:
        return os.path.realpath(origin) if origin else self._source_path

    def _parse_location(self, arg: str):
        arg = arg.strip()
        if ":" in arg:
            file_part, _, line_part = arg.rpartition(":")
            origin = os.path.realpath(file_part)
        else:
            origin, line_part = self._source_path, arg
        try:
            return origin, int(line_part)
        except ValueError:
            return origin, None

    def _add_breakpoint(self, arg: str):
        origin, line = self._parse_location(arg)
        if line is None:
            print("Usage: break [file:]line")
            return
        self._breakpoints.setdefault(origin, set()).add(line)
        print(f"Breakpoint set at {os.path.basename(origin)}:{line}")

    def _delete_breakpoint(self, arg: str):
        if not arg.strip():
            self._breakpoints.clear()
            print("All breakpoints deleted")
            return
        origin, line = self._parse_location(arg)
        if line is None:
            print("Usage: delete [file:]line")
            return
        self._breakpoints.get(origin, set()).discard(line)

    def _print_breakpoints(self):
        rows = [(o, l) for o, lines in self._breakpoints.items() for l in sorted(lines)]
        if not rows:
            print("No breakpoints set.")
            return
        for origin, line in rows:
            print(f"breakpoint at {os.path.basename(origin)}:{line}")

    def _list_source(self, arg: str, current_line: int | None = None):
        target = current_line if current_line is not None else 1
        if arg.strip():
            try:
                target = int(arg.strip())
            except ValueError:
                pass
        if not self._source_lines:
            print("No source available.")
            return
        lo = max(1, target - 5)
        hi = min(len(self._source_lines), target + 4)
        for n in range(lo, hi + 1):
            marker = "->" if n == current_line else "  "
            print(f"{marker}{n:4d}\t{self._source_lines[n - 1]}")

    def _set_var(self, arg: str):
        if "=" not in arg:
            print("Usage: set <name>=<value>")
            return
        name, _, val = arg.partition("=")
        name = name.strip()
        parsed = _parse_value(val.strip())
        self._pending_mods[name] = parsed
        print(f"{name} will be set to {_fmt(parsed)} on resume")

    def _print_var(self, arg: str, visible_vars: dict):
        name = arg.strip()
        if not name:
            print("Usage: print <name>")
            return
        if name not in visible_vars:
            print(f'No symbol "{name}" in current context.')
            return
        self._print_count += 1
        print(f"${self._print_count} = {_fmt(visible_vars[name])}")

    def _print_backtrace(self, call_stack: list, origin: str | None, line: int):
        frames = list(call_stack)  # outermost..innermost
        n = len(frames)
        cur_origin, cur_line = origin, line
        for k in range(n + 1):
            name = frames[n - 1 - k][1] if k < n else None
            label = f"{name}()" if name else "<toplevel>"
            print(f"#{k}  {label} at {os.path.basename(cur_origin) if cur_origin else '?'}:{cur_line}")
            if k < n:
                call_pos = frames[n - 1 - k][2]
                cur_origin = getattr(call_pos, "origin", None) or self._source_path
                cur_line = getattr(call_pos, "line", "?")

    @staticmethod
    def _visible_vars(frame: dict) -> dict:
        return {**frame.get("outer_scope", {}), **frame.get("local_scope", {})}

    # ------------------------------------------------------------------
    # Pre-run prompt
    # ------------------------------------------------------------------

    def run_prompt(self) -> bool:
        """Interactive prompt shown before evaluation starts. Returns False
        if the user quit without running."""
        print(f"Reading symbols from {self._source_path}...")
        while True:
            try:
                raw = input("(scad-dbg) ")
            except EOFError:
                print()
                return False
            cmd, _, arg = raw.strip().partition(" ")
            arg = arg.strip()
            if not cmd:
                continue
            if cmd in ("run", "r"):
                return True
            elif cmd in ("break", "b"):
                self._add_breakpoint(arg)
            elif cmd in ("delete", "d"):
                self._delete_breakpoint(arg)
            elif cmd == "info" and arg.startswith("break"):
                self._print_breakpoints()
            elif cmd in ("list", "l"):
                self._list_source(arg)
            elif cmd in ("quit", "q"):
                return False
            elif cmd in ("help", "h"):
                print(_PRE_RUN_HELP)
            else:
                print(f'Undefined command: "{cmd}". Try "help".')

    # ------------------------------------------------------------------
    # Evaluator callbacks
    # ------------------------------------------------------------------

    def debug_hook(self, line, depth, forced=False, expr_level=False, expr_depth=0, origin=None, get_frames=None):
        if self._quit:
            return "stop", {}

        resolved = self._resolve(origin)
        step = self._step_cmd
        step_hit = False
        if step == "over":
            step_hit = (
                depth <= self._step_depth and resolved == self._step_origin
                and line != self._step_line and not expr_level
            )
        elif step == "into":
            step_hit = (line != self._step_line or resolved != self._step_origin) and not expr_level
        elif step == "out":
            step_hit = depth < self._step_depth and not expr_level

        should_pause = (
            forced
            or (self._break_on_first and not expr_level and resolved == self._source_path)
            or (line in self._breakpoints.get(resolved, set()) and not expr_level)
            or step_hit
        )
        if not should_pause:
            return "continue", {}

        self._break_on_first = False
        self._step_cmd = None

        (_narrow_locals, all_frames), call_stack = get_frames()
        print(f"\nBreakpoint hit at {os.path.basename(resolved)}:{line}")
        self._list_source("", current_line=line)
        visible_vars = self._visible_vars(all_frames[0]) if all_frames else {}
        return self._interact(line, depth, resolved, visible_vars, call_stack)

    def error_break(self, line, msg, all_frame_locals, call_stack, origin=None):
        if self._quit:
            return
        resolved = self._resolve(origin)
        print(f"\n{msg}")
        self._list_source("", current_line=line)
        visible_vars = self._visible_vars(all_frame_locals[0]) if all_frame_locals else {}
        print("(evaluation will abort once you resume; inspect state, then continue/quit)")
        self._interact(line, len(call_stack), resolved, visible_vars, call_stack)

    def return_hook(self, name, value, depth):
        if self._step_cmd == "out" and depth == self._step_depth:
            self._print_count += 1
            print(f"Value returned is ${self._print_count} = {_fmt(value)}")

    # ------------------------------------------------------------------
    # Paused prompt
    # ------------------------------------------------------------------

    def _interact(self, line: int, depth: int, origin: str, visible_vars: dict, call_stack: list):
        while True:
            try:
                raw = input("(scad-dbg) ")
            except EOFError:
                print()
                self._quit = True
                return "stop", {}
            cmd, _, arg = raw.strip().partition(" ")
            arg = arg.strip()
            if not cmd:
                continue

            if cmd in ("continue", "c"):
                return self._resume(None)
            elif cmd in ("step", "s"):
                return self._resume("into", line, depth, origin)
            elif cmd in ("next", "n"):
                return self._resume("over", line, depth, origin)
            elif cmd in ("finish", "fin"):
                return self._resume("out", line, depth, origin)
            elif cmd in ("print", "p"):
                self._print_var(arg, visible_vars)
            elif cmd in ("backtrace", "bt", "where"):
                self._print_backtrace(call_stack, origin, line)
            elif cmd in ("list", "l"):
                self._list_source(arg, current_line=line)
            elif cmd in ("break", "b"):
                self._add_breakpoint(arg)
            elif cmd in ("delete", "d"):
                self._delete_breakpoint(arg)
            elif cmd == "set":
                self._set_var(arg)
            elif cmd in ("quit", "q"):
                self._quit = True
                return "stop", {}
            elif cmd in ("help", "h"):
                print(_PAUSED_HELP)
            else:
                print(f'Undefined command: "{cmd}". Try "help".')

    def _resume(self, step_cmd: str | None, line: int = 0, depth: int = 0, origin: str = ""):
        if step_cmd is not None:
            self._step_cmd, self._step_line, self._step_depth, self._step_origin = step_cmd, line, depth, origin
        mods, self._pending_mods = self._pending_mods, {}
        return "continue", mods
