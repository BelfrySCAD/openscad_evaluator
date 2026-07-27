"""Tests for the `openscad-evaluator` CLI (src/openscad_evaluator/cli.py) and
its --debug REPL (src/openscad_evaluator/_debug_repl.py)."""
import builtins
import sys

import pytest

from openscad_evaluator import cli
from openscad_evaluator._debug_repl import DebugRepl

CUBE_SCRIPT = "cube([10, 10, 10]);\n"

MODULE_SCRIPT = (
    "width = 10;\n"
    "cube([width, width, width]);\n"
    "echo(\"hi\");\n"
)

CHILDREN_SCRIPT = (
    "module wrapper() {\n"
    "    children();\n"
    "}\n"
    "wrapper() {\n"
    "    cube(1);\n"
    "    sphere(1);\n"
    "}\n"
)


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return p


def _feed_input(monkeypatch, responses):
    """Make every `input()` call pop the next canned response, matching
    real stdin one-line-per-call behavior. Raises EOFError once exhausted,
    same as real `input()` on a closed pipe."""
    it = iter(responses)

    def fake_input(prompt=""):
        sys.stdout.write(prompt)
        try:
            return next(it)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr(builtins, "input", fake_input)


class TestExportFormats:
    def test_stl_export(self, tmp_path):
        src = _write(tmp_path, "cube.scad", CUBE_SCRIPT)
        out = tmp_path / "cube.stl"
        assert cli.main([str(src), "-o", str(out)]) == 0
        assert out.stat().st_size > 0

    def test_obj_export(self, tmp_path):
        src = _write(tmp_path, "cube.scad", CUBE_SCRIPT)
        out = tmp_path / "cube.obj"
        assert cli.main([str(src), "-o", str(out)]) == 0
        assert out.read_text().startswith("v ")

    def test_off_export(self, tmp_path):
        src = _write(tmp_path, "cube.scad", CUBE_SCRIPT)
        out = tmp_path / "cube.off"
        assert cli.main([str(src), "-o", str(out)]) == 0
        assert out.read_text().startswith("OFF\n")

    def test_3mf_export(self, tmp_path):
        src = _write(tmp_path, "cube.scad", CUBE_SCRIPT)
        out = tmp_path / "cube.3mf"
        assert cli.main([str(src), "-o", str(out)]) == 0
        assert out.stat().st_size > 0

    def test_unrecognized_extension_errors(self, tmp_path, capsys):
        src = _write(tmp_path, "cube.scad", CUBE_SCRIPT)
        out = tmp_path / "cube.xyz"
        assert cli.main([str(src), "-o", str(out)]) == 1
        assert not out.exists()
        assert "Unrecognized export extension" in capsys.readouterr().err

    def test_explicit_format_overrides_extension(self, tmp_path):
        src = _write(tmp_path, "cube.scad", CUBE_SCRIPT)
        out = tmp_path / "cube.mesh"
        assert cli.main([str(src), "-o", str(out), "--format", "stl"]) == 0
        assert out.stat().st_size > 0


class TestProfile:
    def test_writes_report_with_call_site_and_summary(self, tmp_path):
        # A recursive user function gives real call sites to report on --
        # fib(0)/fib(1) never recurse further, so the tree stays small.
        src = _write(
            tmp_path, "profile.scad",
            "function fib(n) = n < 2 ? n : fib(n-1) + fib(n-2);\ncube([fib(4)+1, 1, 1]);\n",
        )
        out = tmp_path / "profile_out.stl"
        report = tmp_path / "profile_out.txt"
        assert cli.main([str(src), "-o", str(out), "--profile", str(report)]) == 0
        assert out.stat().st_size > 0
        text = report.read_text()
        assert "Profile report for" in text
        assert "Total time:" in text
        assert "fib" in text

    def test_unwritable_profile_path_returns_1(self, tmp_path, capsys):
        src = _write(tmp_path, "profile2.scad", CUBE_SCRIPT)
        out = tmp_path / "profile2_out.stl"
        # A path inside a nonexistent directory can never be opened for writing.
        bad_path = tmp_path / "no_such_dir_xyz" / "p.txt"
        assert cli.main([str(src), "-o", str(out), "--profile", str(bad_path)]) == 1
        assert "error:" in capsys.readouterr().err

    def test_sort_calls_orders_higher_call_count_first(self, tmp_path):
        # fib(6): the recursive call site is "fib(n-1)"/"fib(n-2)" on line 1
        # (many calls); the toplevel call site is "fib(6)" on line 2 (one
        # call). --profile-sort calls must put the former's row first
        # regardless of self time. Keyed on the unique "file:line" location
        # column rather than exact column spacing, an implementation detail
        # this test shouldn't depend on.
        src = _write(
            tmp_path, "profile3.scad",
            "function fib(n) = n < 2 ? n : fib(n-1) + fib(n-2);\ncube([fib(6)+1, 1, 1]);\n",
        )
        out = tmp_path / "profile3_out.stl"
        report = tmp_path / "profile3_out.txt"
        assert cli.main([str(src), "-o", str(out), "--profile", str(report), "--profile-sort", "calls"]) == 0
        text = report.read_text()
        recursive_row = text.find("profile3.scad:1")
        toplevel_row = text.find("profile3.scad:2")
        assert recursive_row != -1
        assert toplevel_row != -1
        assert recursive_row < toplevel_row

    def test_min_calls_filters_out_low_volume_call_sites(self, tmp_path):
        # Same script/call sites as above -- the line-2 (toplevel, 1 call)
        # site is filtered out by a threshold the line-1 (recursive, dozens
        # of calls) site clears.
        src = _write(
            tmp_path, "profile4.scad",
            "function fib(n) = n < 2 ? n : fib(n-1) + fib(n-2);\ncube([fib(6)+1, 1, 1]);\n",
        )
        out = tmp_path / "profile4_out.stl"
        report = tmp_path / "profile4_out.txt"
        assert cli.main([str(src), "-o", str(out), "--profile", str(report), "--profile-min-calls", "10"]) == 0
        text = report.read_text()
        assert "profile4.scad:1" in text
        assert "profile4.scad:2" not in text

    def test_csv_format_writes_header_and_comma_separated_rows(self, tmp_path):
        src = _write(
            tmp_path, "profile5.scad",
            "function fib(n) = n < 2 ? n : fib(n-1) + fib(n-2);\ncube([fib(4)+1, 1, 1]);\n",
        )
        out = tmp_path / "profile5_out.stl"
        report = tmp_path / "profile5_out.csv"
        assert cli.main([str(src), "-o", str(out), "--profile", str(report), "--profile-format", "csv"]) == 0
        text = report.read_text()
        assert "# total_time," in text
        assert "kind,name,caller,call_origin,call_line,call_count,self_time,cumulative_time\n" in text
        assert "function,fib," in text

    def test_invalid_profile_sort_returns_2(self, tmp_path, capsys):
        # argparse's own `choices=` validation rejects this before main()
        # ever runs -- exit code 2 and a SystemExit, same as an invalid
        # --format value already does (an existing, not new, precedent).
        src = _write(tmp_path, "profile6.scad", CUBE_SCRIPT)
        out = tmp_path / "profile6_out.stl"
        report = tmp_path / "profile6_out.txt"
        with pytest.raises(SystemExit) as exc_info:
            cli.main([str(src), "-o", str(out), "--profile", str(report), "--profile-sort", "bogus"])
        assert exc_info.value.code == 2
        assert not report.exists()

    def test_invalid_profile_format_returns_2(self, tmp_path, capsys):
        src = _write(tmp_path, "profile7.scad", CUBE_SCRIPT)
        out = tmp_path / "profile7_out.stl"
        report = tmp_path / "profile7_out.txt"
        with pytest.raises(SystemExit) as exc_info:
            cli.main([str(src), "-o", str(out), "--profile", str(report), "--profile-format", "bogus"])
        assert exc_info.value.code == 2
        assert not report.exists()


class TestErrorHandling:
    def test_syntax_error_returns_1(self, tmp_path):
        src = _write(tmp_path, "bad.scad", "cube([10,10,10]\n")
        out = tmp_path / "bad.stl"
        assert cli.main([str(src), "-o", str(out)]) == 1
        assert not out.exists()

    def test_eval_error_returns_1_and_prints_to_stderr(self, tmp_path, capsys):
        src = _write(tmp_path, "err.scad", 'assert(false, "boom");\n')
        out = tmp_path / "err.stl"
        assert cli.main([str(src), "-o", str(out)]) == 1
        assert "boom" in capsys.readouterr().err

    def test_echo_goes_to_stdout(self, tmp_path, capsys):
        src = _write(tmp_path, "echo.scad", MODULE_SCRIPT)
        out = tmp_path / "echo.stl"
        assert cli.main([str(src), "-o", str(out)]) == 0
        assert 'ECHO: "hi"' in capsys.readouterr().out


class TestDebugRepl:
    def test_breakpoint_then_continue_exports(self, tmp_path, monkeypatch):
        src = _write(tmp_path, "m.scad", MODULE_SCRIPT)
        out = tmp_path / "m.stl"
        # "run" itself pauses at line 1 first (break-on-first, gdb "start"
        # style); the first "continue" resumes to the line-2 breakpoint, the
        # second runs it to completion.
        _feed_input(monkeypatch, ["break 2", "run", "continue", "continue"])
        assert cli.main([str(src), "-o", str(out), "--debug"]) == 0
        assert out.stat().st_size > 0

    def test_print_shows_variable_after_assignment(self, tmp_path, monkeypatch, capsys):
        src = _write(tmp_path, "m.scad", MODULE_SCRIPT)
        out = tmp_path / "m.stl"
        # break_on_first pauses at line 1 (before `width` is assigned), "next"
        # steps to line 2, where `width` is now visible.
        _feed_input(monkeypatch, ["run", "next", "print width", "continue"])
        assert cli.main([str(src), "-o", str(out), "--debug"]) == 0
        assert "$1 = 10" in capsys.readouterr().out

    def test_quit_mid_debug_aborts_without_exporting(self, tmp_path, monkeypatch):
        src = _write(tmp_path, "m.scad", MODULE_SCRIPT)
        out = tmp_path / "m.stl"
        _feed_input(monkeypatch, ["run", "quit"])
        assert cli.main([str(src), "-o", str(out), "--debug"]) == 1
        assert not out.exists()

    def test_quit_before_run_exits_cleanly_without_exporting(self, tmp_path, monkeypatch):
        src = _write(tmp_path, "m.scad", MODULE_SCRIPT)
        out = tmp_path / "m.stl"
        _feed_input(monkeypatch, ["quit"])
        assert cli.main([str(src), "-o", str(out), "--debug"]) == 0
        assert not out.exists()

    def test_set_overrides_variable_on_resume(self, tmp_path, monkeypatch):
        src = _write(tmp_path, "m.scad", MODULE_SCRIPT)
        out = tmp_path / "m.off"
        _feed_input(monkeypatch, ["break 2", "run", "continue", "set width=2", "continue"])
        assert cli.main([str(src), "-o", str(out), "--debug"]) == 0
        verts = [
            line for line in out.read_text().splitlines()[2:]
            if line and not line.startswith("3 ")
        ]
        max_coord = max(abs(float(v)) for line in verts for v in line.split())
        assert max_coord == 2.0  # would be 10.0 without the override

    def test_error_break_lets_user_inspect_then_aborts(self, tmp_path, monkeypatch, capsys):
        src = _write(tmp_path, "err.scad", 'assert(false, "boom");\n')
        out = tmp_path / "err.stl"
        _feed_input(monkeypatch, ["run", "continue"])
        assert cli.main([str(src), "-o", str(out), "--debug"]) == 1
        assert "boom" in capsys.readouterr().err

    def test_child_steps_to_children_call_forwarded_statement(self, tmp_path, monkeypatch, capsys):
        # Paused at the `wrapper() { cube(1); sphere(1); } ` call itself
        # (line 4), "child" should run until wrapper's own `children();`
        # (line 2) forwards control to one of that call's own children --
        # here, `cube(1)` at line 5, its first child statement -- not stop
        # at line 2 itself (children() is not one of the snapshotted
        # targets, only the block's own top-level statements are).
        src = _write(tmp_path, "m.scad", CHILDREN_SCRIPT)
        out = tmp_path / "m.stl"
        # "run" itself already pauses directly at line 4 (break-on-first and
        # the explicit breakpoint coincide there, since line 1 is a
        # declaration and never checked) -- no preceding "continue" needed.
        _feed_input(monkeypatch, ["break 4", "run", "child", "continue"])
        assert cli.main([str(src), "-o", str(out), "--debug"]) == 0
        stdout = capsys.readouterr().out
        assert "Breakpoint hit at m.scad:5" in stdout
        assert "cube(1);" in stdout

    def test_child_falls_back_to_call_return_when_children_never_invoked(self, tmp_path, monkeypatch, capsys):
        # A module that never calls children() at all -- the target
        # position is never reached, so "child" relies purely on the
        # depth-drop fallback (or, issued at top level with nothing
        # shallower to drop to, simply lets the rest of the script run
        # normally) -- either way evaluation must still complete, not hang.
        src = _write(tmp_path, "m.scad", "module noop() {\n}\nnoop() {\n    cube(1);\n}\nsphere(1);\n")
        out = tmp_path / "m.stl"
        # "run" pauses directly at line 3 (break-on-first + the explicit
        # breakpoint coincide, same reasoning as the test above).
        _feed_input(monkeypatch, ["break 3", "run", "child", "continue"])
        assert cli.main([str(src), "-o", str(out), "--debug"]) == 0
        assert out.stat().st_size > 0

    def test_request_pause_causes_next_debug_hook_call_to_pause_like_a_breakpoint(self, tmp_path, monkeypatch, capsys):
        # request_pause() sets the exact same flag a real SIGINT handler
        # would (see its own docstring -- this in-process test harness,
        # monkeypatched input() with no subprocess, can't deliver a real
        # OS signal). Constructs a DebugRepl directly (not through
        # cli.main()) and drives debug_hook() by hand so this is isolated
        # from _break_on_first/_breakpoints/step_hit -- a different origin
        # than the constructed source path means break-on-first can't be
        # what's causing the pause, so this specifically proves
        # request_pause()'s own contribution to the should_pause check.
        src = _write(tmp_path, "m.scad", MODULE_SCRIPT)
        _feed_input(monkeypatch, ["continue"])
        repl = DebugRepl(str(src))
        repl.request_pause()
        cmd, mods = repl.debug_hook(
            5, 0, forced=False, origin="/some/other/file.scad",
            get_frames=lambda: (({}, [{}]), []),
        )
        assert cmd == "continue"
        assert "Interrupted at" in capsys.readouterr().out

    def test_list_shows_source_from_use_injected_file_when_paused_there(self, tmp_path, monkeypatch, capsys):
        # A breakpoint hit inside a `use <file>`-injected module's own body
        # lives in a *different* file than the main script -- `list` (both
        # the automatic display on the breakpoint hit and an explicit `list`
        # command) must show that file's lines, not the main script's.
        lib = _write(tmp_path, "lib.scad", "module lib_cube(s) {\n    cube(s);\n}\n")
        src = _write(tmp_path, "main.scad", f'use <{lib}>\nlib_cube(5);\n')
        out = tmp_path / "main.stl"
        # "run" pauses at main.scad:1 first (break-on-first); "continue"
        # resumes to the lib.scad:2 breakpoint (whose hit auto-lists);
        # "list" then re-lists explicitly, from the same paused location.
        _feed_input(monkeypatch, [f"break {lib}:2", "run", "continue", "list", "continue"])
        assert cli.main([str(src), "-o", str(out), "--debug"]) == 0
        # "cube(s);" only exists in lib.scad -- before the fix, _list_source
        # always read main.scad's own lines regardless of which file the
        # debugger was actually paused in, so this string could never appear
        # (main.scad's own line 2 is "lib_cube(5);", not "cube(s);").
        assert "cube(s);" in capsys.readouterr().out

    def test_stop_returns_to_pre_run_prompt_then_run_exports(self, tmp_path, monkeypatch, capsys):
        # "stop" aborts the current evaluation but -- unlike "quit" --
        # returns to the pre-run prompt instead of exiting the CLI; a
        # plain "run" from there starts a fresh evaluation that completes
        # normally.
        src = _write(tmp_path, "m.scad", MODULE_SCRIPT)
        out = tmp_path / "m.stl"
        _feed_input(monkeypatch, ["run", "stop", "run", "continue", "continue"])
        assert cli.main([str(src), "-o", str(out), "--debug"]) == 0
        assert "Evaluation stopped." in capsys.readouterr().out
        assert out.stat().st_size > 0

    def test_stop_then_quit_aborts_without_exporting(self, tmp_path, monkeypatch):
        # If the user never restarts after "stop", quitting from the
        # pre-run prompt behaves exactly like never having run at all --
        # no export, but a clean exit (0), matching
        # test_quit_before_run_exits_cleanly_without_exporting.
        src = _write(tmp_path, "m.scad", MODULE_SCRIPT)
        out = tmp_path / "m.stl"
        _feed_input(monkeypatch, ["run", "stop", "quit"])
        assert cli.main([str(src), "-o", str(out), "--debug"]) == 0
        assert not out.exists()

    def test_restart_while_paused_aborts_and_runs_again_from_start(self, tmp_path, monkeypatch, capsys):
        # "restart" while paused aborts the current run and immediately
        # re-runs (no intervening pre-run prompt) -- breakpoints carry
        # over, so the fresh run pauses at the same breakpoint again
        # before finally being allowed to complete. Each full run pauses
        # twice (break-on-first at line 1, then the explicit breakpoint at
        # line 2); running twice means "Breakpoint hit" must appear
        # exactly 4 times -- proving the second run genuinely started over
        # rather than "restart" being a no-op or immediately finishing.
        src = _write(tmp_path, "m.scad", MODULE_SCRIPT)
        out = tmp_path / "m.stl"
        _feed_input(monkeypatch, ["break 2", "run", "continue", "restart", "continue", "continue"])
        assert cli.main([str(src), "-o", str(out), "--debug"]) == 0
        assert capsys.readouterr().out.count("Breakpoint hit") == 4
        assert out.stat().st_size > 0

    def test_restart_accepted_at_pre_run_prompt_after_stop(self, tmp_path, monkeypatch, capsys):
        # A user who just typed "stop" naturally reaches for "restart"
        # again out of habit -- with nothing currently running, it must
        # behave exactly like "run" at the pre-run prompt, not
        # "Undefined command".
        src = _write(tmp_path, "m.scad", MODULE_SCRIPT)
        out = tmp_path / "m.stl"
        _feed_input(monkeypatch, ["run", "stop", "restart", "continue", "continue"])
        assert cli.main([str(src), "-o", str(out), "--debug"]) == 0
        assert "Undefined command" not in capsys.readouterr().out
        assert out.exists()

    def test_exit_alias_works_like_quit_before_and_during_run(self, tmp_path, monkeypatch):
        src = _write(tmp_path, "m.scad", MODULE_SCRIPT)
        out1 = tmp_path / "m1.stl"
        _feed_input(monkeypatch, ["exit"])
        assert cli.main([str(src), "-o", str(out1), "--debug"]) == 0
        assert not out1.exists()

        out2 = tmp_path / "m2.stl"
        _feed_input(monkeypatch, ["run", "exit"])
        assert cli.main([str(src), "-o", str(out2), "--debug"]) == 1
        assert not out2.exists()

    def test_info_functions_and_modules_list_user_declarations_before_and_during_run(self, tmp_path, monkeypatch, capsys):
        src = _write(
            tmp_path, "m.scad",
            "function fib(n) = n < 2 ? n : fib(n-1) + fib(n-2);\n"
            "module wrapper(s) {\n"
            "    children();\n"
            "}\n"
            "wrapper(1) {\n"
            "    cube(1);\n"
            "}\n",
        )
        out = tmp_path / "m.stl"
        _feed_input(monkeypatch, ["info functions", "info modules", "run", "info functions", "info modules", "continue"])
        assert cli.main([str(src), "-o", str(out), "--debug"]) == 0
        text = capsys.readouterr().out
        assert "User-defined functions:" in text
        assert "fib(n) at m.scad:1" in text
        assert "User-defined modules:" in text
        assert "wrapper(s) at m.scad:2" in text

    def test_info_functions_and_modules_report_none_when_script_declares_neither(self, tmp_path, monkeypatch, capsys):
        src = _write(tmp_path, "m.scad", CUBE_SCRIPT)
        out = tmp_path / "m.stl"
        _feed_input(monkeypatch, ["info functions", "info modules", "quit"])
        assert cli.main([str(src), "-o", str(out), "--debug"]) == 0
        text = capsys.readouterr().out
        assert "No user-defined functions." in text
        assert "No user-defined modules." in text

    def test_info_variables_shows_currently_visible_variables_only_while_paused(self, tmp_path, monkeypatch, capsys):
        src = _write(tmp_path, "m.scad", MODULE_SCRIPT)
        out = tmp_path / "m.stl"
        # Pre-run: no variables to show yet (nothing has executed). Paused
        # after "next" (past the `width` assignment): `width` is visible.
        _feed_input(monkeypatch, ["info variables", "run", "next", "info variables", "continue"])
        assert cli.main([str(src), "-o", str(out), "--debug"]) == 0
        text = capsys.readouterr().out
        assert 'No variables to show before "run".' in text
        assert "width = 10" in text

    def test_blank_line_repeats_last_step_command(self, tmp_path, monkeypatch):
        # "next" once, then two blank lines -- mirrors gdb's own repeat-
        # last-command convention. Four statement lines means four total
        # pauses (break-on-first at line 1, then one "next" advance per
        # subsequent command) if the repeat genuinely re-issued "next"
        # each time.
        src = _write(tmp_path, "m.scad", "a = 1;\nb = 2;\nc = 3;\ncube(1);\n")
        out = tmp_path / "m.stl"
        _feed_input(monkeypatch, ["run", "next", "", "", "continue"])
        assert cli.main([str(src), "-o", str(out), "--debug"]) == 0
        assert out.exists()

    def test_info_unrecognized_sub_command_reports_undefined(self, tmp_path, monkeypatch, capsys):
        src = _write(tmp_path, "m.scad", MODULE_SCRIPT)
        out = tmp_path / "m.stl"
        _feed_input(monkeypatch, ["info bogus_sub_command", "run", "info bogus_sub_command", "continue"])
        assert cli.main([str(src), "-o", str(out), "--debug"]) == 0
        assert capsys.readouterr().out.count('Undefined info command: "bogus_sub_command"') == 2

    def test_blank_line_before_any_repeatable_command_is_a_noop(self, tmp_path, monkeypatch, capsys):
        # A blank line before any of step/next/child/restart/continue/
        # finish/list has ever been issued has nothing to repeat -- must
        # fall back to the pre-feature behavior (silently re-prompt), not
        # crash or treat it as some other command.
        src = _write(tmp_path, "m.scad", MODULE_SCRIPT)
        out = tmp_path / "m.stl"
        _feed_input(monkeypatch, ["run", "", "print width", "continue"])
        assert cli.main([str(src), "-o", str(out), "--debug"]) == 0
        assert "Undefined command" not in capsys.readouterr().out
