"""Unit tests for src/openscad_evaluator/export.py -- format-level detail
that the CLI's black-box tests (tests/test_cli.py) don't exercise directly,
especially the pure-Python 3MF writer (no lib3mf dependency; see CLAUDE.md)."""
import zipfile
import xml.etree.ElementTree as ET

import pytest
from openscad_lalr_parser import build_scopes, getASTfromString

from openscad_evaluator.evaluator import Evaluator
from openscad_evaluator.export import (
    export_bodies, format_for_path, write_3mf, write_obj, write_off, write_stl,
)

_CORE_NS = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
_MATERIAL_NS = "http://schemas.microsoft.com/3dmanufacturing/material/2015/02"


def _evaluate(script):
    nodes = getASTfromString(script)
    scope = build_scopes(nodes)
    bodies, _ = Evaluator().evaluate(nodes, scope)
    return bodies


class TestFormatForPath:
    @pytest.mark.parametrize("path,expected", [
        ("out.stl", "stl"), ("OUT.STL", "stl"), ("out.obj", "obj"),
        ("out.off", "off"), ("out.3mf", "3mf"),
    ])
    def test_recognized_extensions(self, path, expected):
        assert format_for_path(path) == expected

    def test_unrecognized_extension_raises(self):
        with pytest.raises(ValueError, match="Unrecognized export extension"):
            format_for_path("out.xyz")


class TestNoGeometry:
    @pytest.mark.parametrize("writer", [write_stl, write_obj, write_off, write_3mf])
    def test_empty_body_list_raises(self, tmp_path, writer):
        with pytest.raises(ValueError, match="No geometry to export"):
            writer(str(tmp_path / "out"), [])


class TestWrite3mf:
    def _read_model(self, path):
        with zipfile.ZipFile(path) as z:
            assert z.testzip() is None
            names = z.namelist()
            assert "[Content_Types].xml" in names
            assert "_rels/.rels" in names
            assert "3D/3dmodel.model" in names
            # Every part must be well-formed XML on its own.
            for name in names:
                ET.fromstring(z.read(name))
            return ET.fromstring(z.read("3D/3dmodel.model"))

    def test_single_body_structure_and_round_trip(self, tmp_path):
        bodies = _evaluate("cube([10, 10, 10]);")
        out = tmp_path / "cube.3mf"
        write_3mf(str(out), bodies)

        root = self._read_model(str(out))
        objects = root.findall(f".//{{{_CORE_NS}}}object")
        assert len(objects) == 1
        vertices = objects[0].findall(f".//{{{_CORE_NS}}}vertex")
        triangles = objects[0].findall(f".//{{{_CORE_NS}}}triangle")
        assert len(vertices) == 8    # cube corners
        assert len(triangles) == 12  # 2 triangles per face * 6 faces

        verts, tris = Evaluator()._load_3mf(str(out))
        assert len(verts) == 8
        assert len(tris) == 12

    def test_per_body_color_written_as_hex_rgba(self, tmp_path):
        bodies = _evaluate('color("red") cube(1); color([0, 1, 1, 0.5]) translate([5, 0, 0]) sphere(r=1);')
        out = tmp_path / "colors.3mf"
        write_3mf(str(out), bodies)

        root = self._read_model(str(out))
        colors = [el.get("color") for el in root.findall(f".//{{{_MATERIAL_NS}}}color")]
        assert "#FF0000FF" in colors        # opaque red
        assert "#00FFFF80" in colors        # 50%-alpha cyan (0.5 * 255 rounds to 0x80)

    def test_default_color_used_when_body_has_none(self, tmp_path):
        bodies = _evaluate("cube(1);")
        assert bodies[0].color is None
        out = tmp_path / "uncolored.3mf"
        write_3mf(str(out), bodies)
        root = self._read_model(str(out))
        colors = root.findall(f".//{{{_MATERIAL_NS}}}color")
        assert len(colors) == 1
        assert colors[0].get("color") == "#CCCCCCFF"

    def test_multiple_bodies_get_distinct_resource_ids(self, tmp_path):
        bodies = _evaluate("cube(1); translate([5, 0, 0]) sphere(r=1);")
        out = tmp_path / "multi.3mf"
        write_3mf(str(out), bodies)
        root = self._read_model(str(out))
        object_ids = [el.get("id") for el in root.findall(f".//{{{_CORE_NS}}}object")]
        assert len(object_ids) == len(set(object_ids)) == 2
        build_refs = [el.get("objectid") for el in root.findall(f".//{{{_CORE_NS}}}item")]
        assert sorted(build_refs) == sorted(object_ids)

    def test_no_lib3mf_import(self, tmp_path, monkeypatch):
        """The whole point of the pure-Python writer: it must not import
        lib3mf, which has limited platform availability (aarch64/ARM64)."""
        import builtins
        real_import = builtins.__import__

        def blow_up_on_lib3mf(name, *args, **kwargs):
            if name == "lib3mf":
                raise AssertionError("write_3mf must not import lib3mf")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blow_up_on_lib3mf)
        bodies = _evaluate("cube(1);")
        write_3mf(str(tmp_path / "out.3mf"), bodies)


class TestExportBodiesDispatch:
    def test_infers_format_from_extension(self, tmp_path):
        bodies = _evaluate("cube(1);")
        out = tmp_path / "out.off"
        export_bodies(str(out), bodies)
        assert out.read_text().startswith("OFF\n")

    def test_explicit_format_overrides_extension(self, tmp_path):
        bodies = _evaluate("cube(1);")
        out = tmp_path / "out.mesh"
        export_bodies(str(out), bodies, fmt="obj")
        assert out.read_text().startswith("v ")
