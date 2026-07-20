"""Headless mesh export for `ColoredBody` lists: STL, OBJ, OFF, and 3MF.

No GUI/toolkit dependency -- a caller needing a file dialog, progress bar,
etc. wraps these directly. 3MF needs the optional `lib3mf` package (not
available on every platform; see this project's `pyproject.toml`).
"""
from __future__ import annotations

import struct
from pathlib import Path

import manifold3d as m3d
import numpy as np

from openscad_evaluator.evaluator import ColoredBody

_EXTENSION_FORMATS = {".stl": "stl", ".obj": "obj", ".off": "off", ".3mf": "3mf"}


def format_for_path(path: str) -> str:
    """Infer an export format ("stl"/"obj"/"off"/"3mf") from `path`'s extension."""
    ext = Path(path).suffix.lower()
    if ext not in _EXTENSION_FORMATS:
        raise ValueError(f"Unrecognized export extension '{ext}' (expected one of {sorted(_EXTENSION_FORMATS)})")
    return _EXTENSION_FORMATS[ext]


def _compose_mesh(bodies: list[ColoredBody]):
    manifolds = [b.body for b in bodies if b.body is not None and not b.body.is_empty()]
    if not manifolds:
        return None
    return m3d.Manifold.compose(manifolds).to_mesh()


def write_stl(path: str, bodies: list[ColoredBody]) -> None:
    """Write a binary STL, composing every body's mesh into one solid."""
    mesh = _compose_mesh(bodies)
    if mesh is None:
        raise ValueError("No geometry to export")
    verts = np.asarray(mesh.vert_properties[:, :3], dtype=np.float32)
    tris = np.asarray(mesh.tri_verts, dtype=np.int32)

    v0, v1, v2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0).astype(np.float32)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.where(lengths > 0, lengths, 1.0)

    dtype = np.dtype([
        ("normal", np.float32, (3,)),
        ("v0", np.float32, (3,)),
        ("v1", np.float32, (3,)),
        ("v2", np.float32, (3,)),
        ("attr", np.uint16),
    ])
    data = np.zeros(len(tris), dtype=dtype)
    data["normal"], data["v0"], data["v1"], data["v2"] = normals, v0, v1, v2

    with open(path, "wb") as f:
        f.write(b"\0" * 80)
        f.write(struct.pack("<I", len(tris)))
        f.write(data.tobytes())


def write_obj(path: str, bodies: list[ColoredBody]) -> None:
    """Write a Wavefront OBJ, composing every body's mesh into one solid."""
    mesh = _compose_mesh(bodies)
    if mesh is None:
        raise ValueError("No geometry to export")
    verts = np.asarray(mesh.vert_properties[:, :3], dtype=np.float32)
    tris = np.asarray(mesh.tri_verts, dtype=np.int32)

    with open(path, "w", encoding="utf-8") as f:
        for v in verts:
            f.write(f"v {v[0]:.6g} {v[1]:.6g} {v[2]:.6g}\n")
        f.write("\n")
        for tri in tris:
            f.write(f"f {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")


def write_off(path: str, bodies: list[ColoredBody]) -> None:
    """Write an OFF (Object File Format) file, composing every body's mesh
    into one solid."""
    mesh = _compose_mesh(bodies)
    if mesh is None:
        raise ValueError("No geometry to export")
    verts = np.asarray(mesh.vert_properties[:, :3], dtype=np.float32)
    tris = np.asarray(mesh.tri_verts, dtype=np.int32)

    with open(path, "w", encoding="utf-8") as f:
        f.write("OFF\n")
        f.write(f"{len(verts)} {len(tris)} 0\n")
        for v in verts:
            f.write(f"{v[0]:.6g} {v[1]:.6g} {v[2]:.6g}\n")
        for tri in tris:
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")


def write_3mf(path: str, bodies: list[ColoredBody]) -> None:
    """Write a 3MF file, one mesh object + color group per body. Needs the
    optional `lib3mf` package (`pip install openscad-evaluator[3mf]`)."""
    try:
        import lib3mf
    except ImportError as e:
        raise ImportError(
            "3MF export needs the optional 'lib3mf' package: pip install openscad-evaluator[3mf]"
        ) from e

    fa3 = type(lib3mf.Position().Coordinates)
    ui3 = type(lib3mf.Triangle().Indices)

    def _identity_transform():
        t = lib3mf.Transform()
        col = type(t.Fields[0])
        t.Fields[0] = col(1, 0, 0)
        t.Fields[1] = col(0, 1, 0)
        t.Fields[2] = col(0, 0, 1)
        t.Fields[3] = col(0, 0, 0)
        return t

    wrapper = lib3mf.Wrapper()
    model = wrapper.CreateModel()
    wrote_any = False

    for colored_body in bodies:
        if colored_body.body is None or colored_body.body.is_empty():
            continue
        mesh = colored_body.body.to_mesh()
        verts = np.asarray(mesh.vert_properties[:, :3], dtype=np.float32)
        tris = np.asarray(mesh.tri_verts, dtype=np.int32)
        if len(tris) == 0:
            continue
        wrote_any = True

        mesh_obj = model.AddMeshObject()
        positions = []
        for v in verts:
            p = lib3mf.Position()
            p.Coordinates = fa3(float(v[0]), float(v[1]), float(v[2]))
            positions.append(p)
        triangles = []
        for t in tris:
            tri = lib3mf.Triangle()
            tri.Indices = ui3(int(t[0]), int(t[1]), int(t[2]))
            triangles.append(tri)
        mesh_obj.SetGeometry(positions, triangles)

        rgba = colored_body.color or (0.8, 0.8, 0.8, 1.0)
        cg = model.AddColorGroup()
        c = lib3mf.Color()
        c.Red = max(0, min(255, int(rgba[0] * 255)))
        c.Green = max(0, min(255, int(rgba[1] * 255)))
        c.Blue = max(0, min(255, int(rgba[2] * 255)))
        c.Alpha = max(0, min(255, int(rgba[3] * 255)))
        color_id = cg.AddColor(c)
        cg_uid = cg.GetUniqueResourceID()

        props = []
        for _ in range(len(tris)):
            tp = lib3mf.TriangleProperties()
            tp.ResourceID = cg_uid
            tp.PropertyIDs = ui3(color_id, color_id, color_id)
            props.append(tp)
        mesh_obj.SetAllTriangleProperties(props)
        model.AddBuildItem(mesh_obj, _identity_transform())

    if not wrote_any:
        raise ValueError("No geometry to export")

    writer = model.QueryWriter("3mf")
    writer.WriteToFile(str(path))


_WRITERS = {"stl": write_stl, "obj": write_obj, "off": write_off, "3mf": write_3mf}


def export_bodies(path: str, bodies: list[ColoredBody], fmt: str | None = None) -> None:
    """Write `bodies` to `path`, inferring the format ("stl"/"obj"/"off"/"3mf")
    from its extension unless `fmt` is given explicitly."""
    fmt = fmt or format_for_path(path)
    _WRITERS[fmt](path, bodies)
