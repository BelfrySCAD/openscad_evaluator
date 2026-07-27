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
from dataclasses import dataclass
from pathlib import Path

try:
    # Importing this is the entire fix: it hooks stdlib input() into
    # GNU readline (or libedit, which is what Python's own readline module
    # actually links against on macOS) for the rest of the process, giving
    # every input() call arrow-key history/line-editing for free -- no
    # other code here needs to change. Not available on Windows (no
    # built-in equivalent), so this degrades to plain input() there.
    import readline  # noqa: F401
except ImportError:
    pass


@dataclass
class DeclInfo:
    """One user-defined function/module declaration -- backs both
    "info functions"/"info modules" (display) and "list <name>" (lookup by
    name, jumping to wherever it's actually declared, which may be a
    different file than wherever the debugger happens to be paused right
    now -- e.g. a `use <file>`-injected declaration). Constructed by the
    caller (cli.py, which has direct AST access) and handed to
    set_declared_names(); `origin` need not be pre-resolved --
    set_declared_names() canonicalizes it the same way every other origin
    this class stores/compares already is."""
    name: str
    params: str  # "a, b=2" ("" if no parameters)
    origin: str
    line: int


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
  run, r, restart        Start evaluating the script
  break [file:]line, b   Set a breakpoint
  delete [file:]line, d  Delete a breakpoint (no args: delete all)
  info breakpoints       List breakpoints
  info modules           List user-defined modules
  info functions         List user-defined functions
  list [line|name], l    Show source around a line, or a function/module
                          ("foo" if unambiguous, else "function:foo"/"module:foo")
  quit, q, exit          Exit without running
  help, h                Show this text
(Enter on a blank line repeats the last restart/list)"""

_PAUSED_HELP = """\
Commands (while paused):
  continue, c             Resume until the next breakpoint
  step, s                 Step into the next statement/call
  next, n                 Step over the next statement (don't descend into calls)
  finish, fin             Run until the current call returns
  child, sc               Step to child: run until children()/children(N) forwards
                          control to one of this call's own { ... } children
                          (or it returns, if it never calls children() at all)
  stop                    Abort the current evaluation, return to the pre-run prompt
  restart, r              Abort the current evaluation and run again from the start
  print <name>, p         Print a variable's value
  backtrace, bt, where    Show the call stack (innermost first)
  up                      Select the caller frame (view its variables)
  down                    Select the callee frame
  frame [n], f            Select frame #n (no arg: show the current frame)
  info breakpoints        List breakpoints
  info variables          List the selected frame's variables
  info modules            List user-defined modules
  info functions          List user-defined functions
  list [line|name], l     Show source around a line, or a function/module
                          ("foo" if unambiguous, else "function:foo"/"module:foo")
  break [file:]line, b    Set a breakpoint
  delete [file:]line, d   Delete a breakpoint (no args: delete all)
  set <name>=<value>      Override a variable's value on resume
  quit, q, exit           Abort evaluation
  help, h                 Show this text
(Enter on a blank line repeats the last step/next/child/restart/
 continue/finish/list)"""


class DebugRepl:
    """One instance per `--debug` run. Construct, wire its three methods into
    `Evaluator(debug_hook=repl.debug_hook, error_break_fn=repl.error_break,
    return_hook=repl.return_hook)`, call `run_prompt()` first, and only call
    `evaluate()` if it returns True."""

    def __init__(self, source_path: str):
        self._source_path = os.path.realpath(source_path)
        # Lazily populated per-origin (not just the main script) -- a
        # breakpoint/step can land inside a `use <file>`-injected function
        # or module, which lives in a different file than source_path;
        # `list`/pause-time source display must read *that* file's lines,
        # not always the main script's. See _lines_for().
        self._source_lines_by_origin: dict[str, list[str]] = {self._source_path: self._read_lines(source_path)}
        self._breakpoints: dict[str, set[int]] = {}
        self._break_on_first = True
        self._step_cmd: str | None = None   # "into" / "over" / "out" / "to_child"
        self._step_line = 0
        self._step_depth = 0
        self._step_origin = ""
        self._step_to_child_targets: set[tuple[str | None, int]] = set()
        self._pending_mods: dict = {}
        self._print_count = 0
        self._quit = False
        # Set via request_pause() -- either by a real SIGINT handler
        # (cli.py installs one around the evaluate() call) or, in tests,
        # directly (no subprocess means no real OS signal can be
        # delivered). Read-and-cleared in debug_hook(), folded into the
        # exact same should_pause check breakpoints/steps use -- mirrors
        # BelfrySCAD's own DebugSession._pause_requested/pause() exactly,
        # just signal-triggered here instead of a GUI button click.
        self._pause_requested = False
        # Set by the caller (cli.py) once the Evaluator exists -- "child"
        # reads Evaluator._last_children_positions (see that field's own
        # doc comment), computed fresh on every _check_debug call, to know
        # which of the currently-paused call's own { ... } children a
        # children()/children(N) forward might land on. Left unwired
        # (None), "child" behaves like "finish" (the depth-drop fallback
        # below still applies), matching what happens if the paused call
        # never invokes children() at all.
        self._evaluator = None
        # Fed by set_declared_names() (cli.py, which has direct AST
        # access) for "info functions"/"info modules" (display) and
        # "list <name>" (lookup), so this class stays fully decoupled
        # from parser/AST types beyond the one DeclInfo dataclass.
        self._declared_functions: list[DeclInfo] = []
        self._declared_modules: list[DeclInfo] = []
        # What the paused session's own "stop"/"restart"/"quit" command
        # decided, once it raises the shared DEBUGGING_STOPPED_MESSAGE
        # EvalError -- read by cli.py (via take_post_run_action()) after
        # catching that specific exception, to decide whether to return to
        # the pre-run prompt ("stopped"), immediately re-run ("restart"),
        # or actually exit the CLI ("quit"). None is the initial/reset
        # state -- also what a genuine (non-debugger-triggered) EvalError
        # leaves it at.
        self._post_run_action: str | None = None
        # Hitting Enter on a blank line re-issues this exact command+arg,
        # mirroring gdb's own "repeat last command" convention -- only
        # step/next/child/restart/continue/finish/list ever set this
        # (None means "nothing to repeat yet", so a blank line before any
        # of those is a no-op like it always was). Shared across
        # run_prompt() and _interact() rather than reset between them,
        # matching gdb's own single persistent "last command" register.
        self._last_repeatable_cmd: str | None = None
        self._last_repeatable_arg = ""

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _resolve(self, origin: str | None) -> str:
        return os.path.realpath(origin) if origin else self._source_path

    @staticmethod
    def _read_lines(path: str) -> list[str]:
        try:
            return Path(path).read_text(encoding="utf-8").splitlines()
        except OSError:
            return []

    def _lines_for(self, origin: str) -> list[str]:
        lines = self._source_lines_by_origin.get(origin)
        if lines is None:
            lines = self._read_lines(origin)
            self._source_lines_by_origin[origin] = lines
        return lines

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

    def _list_source(self, arg: str, current_line: int | None = None, origin: str | None = None):
        """`arg` also accepts a name instead of a line number -- "fib"
        (looked up unqualified, erroring if ambiguous between the
        function/module namespaces) or "function:fib"/"module:fib"
        (explicit), jumping to that declaration's own file:line
        regardless of the current pause location. See DeclInfo's own
        doc comment for why this needs _declared_functions/_declared_modules,
        not just a line number."""
        list_origin = origin or self._source_path
        target = current_line if current_line is not None else 1
        t = arg.strip()

        if t:
            try:
                target = int(t)
            except ValueError:
                # "function:name" / "module:name" qualifies which namespace
                # to search -- reuses this REPL's existing "prefix:rest"
                # colon convention (break/delete's own [file:]line parsing).
                # Unqualified: search both, erroring if the name exists in
                # both (a function and a module CAN share a name in OpenSCAD).
                qualifier, sep, name = t.partition(":")
                if not sep:
                    qualifier, name = "", t
                fn = self._find_decl(self._declared_functions, name) if qualifier in ("", "function") else None
                mod = self._find_decl(self._declared_modules, name) if qualifier in ("", "module") else None
                if fn and mod:
                    print(f'Both a function and a module are named "{name}" -- '
                          f'use "list function:{name}" or "list module:{name}".')
                    return
                decl = fn or mod
                if not decl:
                    print(f'No symbol "{name}" in current context.')
                    return
                list_origin = decl.origin
                target = decl.line
                current_line = None  # jumping to a declaration, not the paused line -- no "->" marker

        lines = self._lines_for(list_origin)
        if not lines:
            print("No source available.")
            return
        lo = max(1, target - 5)
        hi = min(len(lines), target + 4)
        for n in range(lo, hi + 1):
            marker = "->" if n == current_line else "  "
            print(f"{marker}{n:4d}\t{lines[n - 1]}")

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

    def _frame_location(self, k: int, call_stack: list, origin: str | None, line: int):
        """(origin, line) shown for backtrace/frame level `k` -- walks call
        positions outward. Shared by backtrace and up/down/frame."""
        n = len(call_stack)
        cur_origin, cur_line = origin, line
        for i in range(min(k, n)):
            call_pos = call_stack[n - 1 - i][2]
            cur_origin = getattr(call_pos, "origin", None) or self._source_path
            cur_line = getattr(call_pos, "line", "?")
        return cur_origin, cur_line

    def _print_backtrace(self, call_stack: list, origin: str | None, line: int):
        n = len(call_stack)  # call_stack is outermost..innermost
        for k in range(n + 1):
            name = call_stack[n - 1 - k][1] if k < n else None
            label = f"{name}()" if name else "<toplevel>"
            o, l = self._frame_location(k, call_stack, origin, line)
            print(f"#{k}  {label} at {os.path.basename(o) if o else '?'}:{l}")

    @staticmethod
    def _visible_vars(frame: dict) -> dict:
        return {**frame.get("outer_scope", {}), **frame.get("local_scope", {})}

    def _print_variables(self, visible_vars: dict):
        if not visible_vars:
            print("No variables in current context.")
            return
        for name in sorted(visible_vars):
            print(f"{name} = {_fmt(visible_vars[name])}")

    @staticmethod
    def _print_decls(decls: list[DeclInfo], kind_label: str):
        if not decls:
            print(f"No user-defined {kind_label}.")
            return
        print(f"User-defined {kind_label}:")
        for d in decls:
            print(f"  {d.name}({d.params}) at {os.path.basename(d.origin)}:{d.line}")

    def _print_declared_functions(self):
        self._print_decls(self._declared_functions, "functions")

    def _print_declared_modules(self):
        self._print_decls(self._declared_modules, "modules")

    @staticmethod
    def _find_decl(decls: list[DeclInfo], name: str) -> DeclInfo | None:
        for d in decls:
            if d.name == name:
                return d
        return None

    def set_declared_names(self, functions: list[DeclInfo], modules: list[DeclInfo]):
        """Feeds "info functions"/"info modules" (display) and
        "list <name>" (lookup) their declaration data -- computed once by
        the caller (cli.py, which has direct AST access) from the fully
        use-resolved top-level node list. Call before run_prompt(); an
        empty list just means "no user-defined functions/modules" (a
        real, reportable state, not an error)."""
        for d in functions:
            d.origin = self._resolve(d.origin)
        for d in modules:
            d.origin = self._resolve(d.origin)
        self._declared_functions = functions
        self._declared_modules = modules

    def prepare_for_run(self):
        """Resets everything that must start fresh for a (re)run: the
        break-on-first-statement flag, any in-flight step command, and
        pending `set` overrides -- called once per run, including the
        very first one, so "restart" is indistinguishable from a genuine
        fresh "run" except that breakpoints/history/print-counter carry
        over (matching gdb's own `run`-after-`kill` behavior). Does NOT
        touch _breakpoints or _print_count."""
        self._quit = False
        self._post_run_action = None
        self._break_on_first = True
        self._step_cmd = None
        self._step_to_child_targets = set()
        self._pending_mods = {}

    def take_post_run_action(self) -> str | None:
        """Reads and resets the outcome of the paused session's own
        "stop"/"restart"/"quit" command (None if evaluation completed
        normally, or hasn't been asked to abort at all)."""
        action, self._post_run_action = self._post_run_action, None
        return action

    def request_pause(self):
        """Requests a pause at the next debug_hook() call, exactly like
        hitting a breakpoint. Called by a real SIGINT handler (see cli.py)
        so Ctrl+C during a running evaluate() drops into the paused
        prompt; also callable directly (bypassing signal.signal entirely)
        so tests -- which drive this in-process via monkeypatched
        input(), no subprocess, no real OS signal deliverable -- can
        simulate "the user just pressed Ctrl+C"."""
        self._pause_requested = True

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
                # Hitting Enter on a blank line repeats the last "restart"/
                # "list" (the only two of the repeatable commands valid at
                # this prompt) -- mirrors gdb's own repeat-last-command
                # convention. No prior repeatable command yet: unchanged
                # no-op behavior.
                if self._last_repeatable_cmd is None:
                    continue
                cmd, arg = self._last_repeatable_cmd, self._last_repeatable_arg
            # "restart" is also accepted here (not just "run"/"r") so a
            # user who just typed "stop" can reflexively type "restart"
            # again -- with nothing currently running, the two commands
            # mean the same thing at this prompt.
            if cmd in ("run", "r", "restart"):
                self._last_repeatable_cmd, self._last_repeatable_arg = "restart", ""
                return True
            elif cmd in ("break", "b"):
                self._add_breakpoint(arg)
            elif cmd in ("delete", "d"):
                self._delete_breakpoint(arg)
            elif cmd == "info":
                sub = arg.strip()
                if sub.startswith("break"):
                    self._print_breakpoints()
                elif sub == "modules":
                    self._print_declared_modules()
                elif sub == "functions":
                    self._print_declared_functions()
                elif sub == "variables":
                    print('No variables to show before "run".')
                else:
                    print(f'Undefined info command: "{sub}". Try "help".')
            elif cmd in ("list", "l"):
                self._last_repeatable_cmd, self._last_repeatable_arg = "list", arg
                self._list_source(arg)
            elif cmd in ("quit", "q", "exit"):
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
        elif step == "to_child":
            # Pause the first time control reaches one of the paused call's
            # own children (wherever children()/children(N) forwards to
            # them) -- or, if the call never invokes children() at all,
            # fall back to the same "call returned" safety net step-out
            # uses, so this can never hang. Mirrors BelfrySCAD's
            # DebugSession._make_hook exactly.
            step_hit = not expr_level and (
                (resolved, line) in self._step_to_child_targets or depth < self._step_depth
            )

        # Read-and-clear, same as BelfrySCAD's own DebugSession.pause()/
        # pause_now -- a stray SIGINT that arrives while already blocked
        # in input() at a prompt (not mid-evaluate()) just gets consumed
        # here on the next statement check once the user resumes, causing
        # an immediate re-pause; a harmless quirk, not a hang or crash.
        pause_requested = self._pause_requested
        self._pause_requested = False

        should_pause = (
            forced
            or pause_requested
            or (self._break_on_first and not expr_level and resolved == self._source_path)
            or (line in self._breakpoints.get(resolved, set()) and not expr_level)
            or step_hit
        )
        if not should_pause:
            return "continue", {}

        self._break_on_first = False
        self._step_cmd = None

        (_narrow_locals, all_frames), call_stack = get_frames()
        if pause_requested:
            print(f"\nInterrupted at {os.path.basename(resolved)}:{line}")
        else:
            print(f"\nBreakpoint hit at {os.path.basename(resolved)}:{line}")
        self._list_source("", current_line=line, origin=resolved)
        return self._interact(line, depth, resolved, all_frames, call_stack)

    def error_break(self, line, msg, all_frame_locals, call_stack, origin=None):
        if self._quit:
            return
        resolved = self._resolve(origin)
        print(f"\n{msg}")
        self._list_source("", current_line=line, origin=resolved)
        print("(evaluation will abort once you resume; inspect state, then continue/quit)")
        self._interact(line, len(call_stack), resolved, all_frame_locals, call_stack)

    def return_hook(self, name, value, depth):
        if self._step_cmd == "out" and depth == self._step_depth:
            self._print_count += 1
            print(f"Value returned is ${self._print_count} = {_fmt(value)}")

    # ------------------------------------------------------------------
    # Paused prompt
    # ------------------------------------------------------------------

    def _interact(self, line: int, depth: int, origin: str, all_frames: list, call_stack: list):
        # Which frame `print`/`info variables` inspect. 0 = innermost (the
        # paused statement); `up`/`down`/`frame n` move it. Aligns with
        # backtrace #k. The closures read cur_frame live (late binding).
        cur_frame = 0

        def visible():
            return self._visible_vars(all_frames[cur_frame]) if cur_frame < len(all_frames) else {}

        def print_frame_header():
            n = len(call_stack)
            name = call_stack[n - 1 - cur_frame][1] if cur_frame < n else None
            label = f"{name}()" if name else "<toplevel>"
            o, l = self._frame_location(cur_frame, call_stack, origin, line)
            print(f"#{cur_frame}  {label} at {os.path.basename(o) if o else '?'}:{l}")

        while True:
            try:
                raw = input("(scad-dbg) ")
            except EOFError:
                print()
                self._quit = True
                self._post_run_action = "quit"
                return "stop", {}
            cmd, _, arg = raw.strip().partition(" ")
            arg = arg.strip()
            if not cmd:
                # Hitting Enter on a blank line repeats the last step/next/
                # child/restart/continue/finish/list -- mirrors gdb's own
                # repeat-last-command convention. No prior repeatable
                # command yet: unchanged no-op behavior.
                if self._last_repeatable_cmd is None:
                    continue
                cmd, arg = self._last_repeatable_cmd, self._last_repeatable_arg

            if cmd in ("continue", "c"):
                self._last_repeatable_cmd, self._last_repeatable_arg = "continue", ""
                return self._resume(None)
            elif cmd in ("step", "s"):
                self._last_repeatable_cmd, self._last_repeatable_arg = "step", ""
                return self._resume("into", line, depth, origin)
            elif cmd in ("next", "n"):
                self._last_repeatable_cmd, self._last_repeatable_arg = "next", ""
                return self._resume("over", line, depth, origin)
            elif cmd in ("finish", "fin"):
                self._last_repeatable_cmd, self._last_repeatable_arg = "finish", ""
                return self._resume("out", line, depth, origin)
            elif cmd in ("child", "sc"):
                self._last_repeatable_cmd, self._last_repeatable_arg = "child", ""
                return self._resume("to_child", line, depth, origin)
            elif cmd == "stop":
                self._quit = True
                self._post_run_action = "stopped"
                return "stop", {}
            elif cmd in ("restart", "r"):
                self._last_repeatable_cmd, self._last_repeatable_arg = "restart", ""
                self._quit = True
                self._post_run_action = "restart"
                return "stop", {}
            elif cmd in ("print", "p"):
                self._print_var(arg, visible())
            elif cmd == "up":
                if cur_frame + 1 < len(all_frames):
                    cur_frame += 1
                    print_frame_header()
                else:
                    print("Already at the outermost frame.")
            elif cmd == "down":
                if cur_frame > 0:
                    cur_frame -= 1
                    print_frame_header()
                else:
                    print("Already at the innermost frame.")
            elif cmd in ("frame", "f"):
                a = arg.strip()
                if not a:
                    print_frame_header()
                else:
                    try:
                        idx = int(a)
                    except ValueError:
                        print("Usage: frame <n>")
                    else:
                        if 0 <= idx < len(all_frames):
                            cur_frame = idx
                            print_frame_header()
                        else:
                            print(f"No frame #{a} (have 0..{len(all_frames) - 1}).")
            elif cmd in ("backtrace", "bt", "where"):
                self._print_backtrace(call_stack, origin, line)
            elif cmd == "info":
                sub = arg.strip()
                if sub.startswith("break"):
                    self._print_breakpoints()
                elif sub == "variables":
                    self._print_variables(visible())
                elif sub == "modules":
                    self._print_declared_modules()
                elif sub == "functions":
                    self._print_declared_functions()
                else:
                    print(f'Undefined info command: "{sub}". Try "help".')
            elif cmd in ("list", "l"):
                self._last_repeatable_cmd, self._last_repeatable_arg = "list", arg
                self._list_source(arg, current_line=line, origin=origin)
            elif cmd in ("break", "b"):
                self._add_breakpoint(arg)
            elif cmd in ("delete", "d"):
                self._delete_breakpoint(arg)
            elif cmd == "set":
                self._set_var(arg)
            elif cmd in ("quit", "q", "exit"):
                self._quit = True
                self._post_run_action = "quit"
                return "stop", {}
            elif cmd in ("help", "h"):
                print(_PAUSED_HELP)
            else:
                print(f'Undefined command: "{cmd}". Try "help".')

    def _resume(self, step_cmd: str | None, line: int = 0, depth: int = 0, origin: str = ""):
        if step_cmd is not None:
            self._step_cmd, self._step_line, self._step_depth, self._step_origin = step_cmd, line, depth, origin
        if step_cmd == "to_child":
            targets = getattr(self._evaluator, "_last_children_positions", None) if self._evaluator else None
            # Normalize each target's origin through the same _resolve()
            # used on the hook's own `resolved` before comparing -- these
            # positions are captured raw (whatever the AST's own position
            # carries), which can disagree with the realpath'd form (e.g.
            # macOS's /var -> /private/var) even for the objectively same
            # file.
            self._step_to_child_targets = {(self._resolve(o), ln) for o, ln in (targets or [])}
        mods, self._pending_mods = self._pending_mods, {}
        return "continue", mods
