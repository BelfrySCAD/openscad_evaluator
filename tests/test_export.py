"""Unit tests for src/openscad_evaluator/export.py -- format-level detail
that the CLI's black-box tests (tests/test_cli.py) don't exercise directly,
especially the pure-Python 3MF writer (no lib3mf dependency; see CLAUDE.md)."""
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from openscad_lalr_parser import build_scopes, getASTfromString

from openscad_evaluator.evaluator import Evaluator
from openscad_evaluator.export import (
    export_bodies, export_model, format_for_path, write_3mf, write_obj, write_off, write_stl,
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

    def test_same_colour_bodies_weld_into_one_object(self, tmp_path):
        bodies = _evaluate("cube(1); translate([5, 0, 0]) sphere(r=1);")
        out = tmp_path / "one.3mf"
        write_3mf(str(out), bodies)
        assert len(self._read_model(str(out)).findall(f".//{{{_CORE_NS}}}object")) == 1

    def test_multiple_bodies_get_distinct_resource_ids(self, tmp_path):
        bodies = _evaluate('color("red") cube(1); translate([5, 0, 0]) sphere(r=1);')
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
        assert out.read_text().startswith("mtllib out.mtl\n")


_OPEN_TETRA = "polyhedron([[0,0,0],[10,0,0],[0,10,0],[0,0,10]], [[0,1,2],[0,3,1],[0,2,3]]);"


class TestOpenMesh:
    """An open polyhedron is exported as the surface it is, as OpenSCAD does."""

    def test_off_has_the_open_surface(self, tmp_path):
        out = tmp_path / "open.off"
        write_off(str(out), _evaluate(_OPEN_TETRA))
        assert out.read_text().splitlines()[1] == "4 3 0"

    def test_open_surface_appended_after_solids(self, tmp_path):
        out = tmp_path / "both.off"
        write_off(str(out), _evaluate("cube(1); translate([5,0,0]) " + _OPEN_TETRA))
        assert out.read_text().splitlines()[1] == "12 15 0"  # cube 8/12 + surface 4/3

    @pytest.mark.parametrize("writer", [write_stl, write_obj, write_3mf])
    def test_every_format_writes_it(self, tmp_path, writer):
        out = tmp_path / "open.out"
        writer(str(out), _evaluate(_OPEN_TETRA))
        assert out.stat().st_size > 0


class TestExportModel:
    """export_model's object split and checks. Every structure below matches
    openscad_cpp_evaluator's export_model on the same script: object count,
    triangles, volume and colours per object, and the warnings."""

    SCRIPT = ('color("red") cube(10); color("blue") translate([5,5,5]) cube(10);'
              'union() { color("green") translate([30,0,0]) cube(8); color("yellow") translate([34,0,0]) cube(8); }'
              'translate([60,0,0]) cube(4); translate([70,0,0]) cube(4); %translate([0,40,0]) cube(5);')

    def _objects(self, tmp_path, script=SCRIPT, **kw):
        out = tmp_path / "m.3mf"
        warnings = export_model(str(out), _evaluate(script), **kw)
        root = ET.fromstring(zipfile.ZipFile(out).read("3D/3dmodel.model"))
        groups = {g.get("id"): [c.get("color") for c in g] for g in root.iter(f"{{{_MATERIAL_NS}}}colorgroup")}
        objs = []
        for o in root.iter(f"{{{_CORE_NS}}}object"):
            v = np.array([[float(x.get(a)) for a in "xyz"] for x in o.iter(f"{{{_CORE_NS}}}vertex")])
            t = np.array([[int(x.get(a)) for a in ("v1", "v2", "v3")] for x in o.iter(f"{{{_CORE_NS}}}triangle")])
            vol = np.einsum("ij,ij->i", v[t[:, 0]], np.cross(v[t[:, 1]], v[t[:, 2]])).sum() / 6
            objs.append((round(float(vol), 2), groups[o.get("pid")]))
        return sorted(objs), warnings

    def test_one_object_per_colour_later_colour_wins(self, tmp_path):
        objs, warnings = self._objects(tmp_path)
        assert objs == [(128.0, ["#CCCCCCFF"]), (768.0, ["#008000FF", "#FFFF00FF"]),
                        (875.0, ["#FF0000FF"]), (1000.0, ["#0000FFFF"])]
        assert warnings == []

    def test_split_components_and_single_material(self, tmp_path):
        objs, _ = self._objects(tmp_path, split_components=True)
        assert [o[0] for o in objs] == [64.0, 64.0, 768.0, 875.0, 1000.0]
        objs, _ = self._objects(tmp_path, split_colors=False)
        assert objs == [(2771.0, ["#FF0000FF"])]

    def test_open_shell_is_written_and_reported(self, tmp_path):
        warnings = export_model(str(tmp_path / "m.stl"), _evaluate("cube(1); translate([5,0,0]) " + _OPEN_TETRA))
        assert warnings == ["part 2 is not a closed solid; its surface is written as-is, and most slicers "
                            "will reject it.", "exported mesh 3 boundary edges"]

    def test_touching_bodies_are_unioned_not_concatenated(self, tmp_path):
        out = tmp_path / "m.off"
        assert export_model(str(out), _evaluate("cube(1); translate([1,0,0]) cube(1);")) == []
        assert out.read_text().splitlines()[1] == "12 20 0"  # one welded 2x1x1 box

    def test_obj_writes_its_materials(self, tmp_path):
        export_model(str(tmp_path / "m.obj"), _evaluate("color([1,0,0,0.5]) cube(1);"))
        assert (tmp_path / "m.mtl").read_text() == "newmtl color_1\nKd 1 0 0\nd 0.5\n\n"
