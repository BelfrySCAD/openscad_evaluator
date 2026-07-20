"""Tests for the `openscad-evaluator` CLI (src/openscad_evaluator/cli.py) and
its --debug REPL (src/openscad_evaluator/_debug_repl.py)."""
import builtins
import sys

import pytest

from openscad_evaluator import cli

CUBE_SCRIPT = "cube([10, 10, 10]);\n"

MODULE_SCRIPT = (
    "width = 10;\n"
    "cube([width, width, width]);\n"
    "echo(\"hi\");\n"
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
        pytest.importorskip("lib3mf")
        src = _write(tmp_path, "cube.scad", CUBE_SCRIPT)
        out = tmp_path / "cube.3mf"
        assert cli.main([str(src), "-o", str(out)]) == 0
        assert out.stat().st_size > 0

    def test_missing_lib3mf_gives_clear_error(self, tmp_path, monkeypatch, capsys):
        real_import = builtins.__import__

        def blocked_import(name, *args, **kwargs):
            if name == "lib3mf":
                raise ImportError("simulated missing lib3mf")
            return real_import(name, *args, **kwargs)

        monkeypatch.delitem(sys.modules, "lib3mf", raising=False)
        monkeypatch.setattr(builtins, "__import__", blocked_import)
        src = _write(tmp_path, "cube.scad", CUBE_SCRIPT)
        out = tmp_path / "cube.3mf"
        assert cli.main([str(src), "-o", str(out)]) == 1
        assert "lib3mf" in capsys.readouterr().err

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
