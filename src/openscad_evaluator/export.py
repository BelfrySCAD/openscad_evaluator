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


_3MF_CORE_NS = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
_3MF_MATERIAL_NS = "http://schemas.microsoft.com/3dmanufacturing/material/2015/02"
_3MF_XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"

_3MF_CONTENT_TYPES = (
    b'<?xml version="1.0" encoding="UTF-8"?>\n'
    b'<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    b'<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    b'<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>'
    b'</Types>'
)
_3MF_RELS = (
    b'<?xml version="1.0" encoding="UTF-8"?>\n'
    b'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    b'<Relationship Target="/3D/3dmodel.model" Id="rel0" '
    b'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>'
    b'</Relationships>'
)


def _3mf_hex_color(rgba) -> str:
    r, g, b, a = (max(0, min(255, round(c * 255))) for c in rgba)
    return f"#{r:02X}{g:02X}{b:02X}{a:02X}"


def write_3mf(path: str, bodies: list[ColoredBody]) -> None:
    """Write a 3MF file, one mesh object + base color per body.

    Pure Python -- a 3MF file is just a ZIP package holding an XML mesh
    description (plus a couple of small fixed manifest files), so this needs
    only the standard library (`zipfile` + `xml.etree.ElementTree`), unlike
    `lib3mf`, which isn't available on every platform (e.g. aarch64/ARM64).
    """
    import zipfile
    import xml.etree.ElementTree as ET

    ET.register_namespace("", _3MF_CORE_NS)
    ET.register_namespace("m", _3MF_MATERIAL_NS)

    model_el = ET.Element(f"{{{_3MF_CORE_NS}}}model", {"unit": "millimeter", _3MF_XML_LANG: "en-US"})
    resources_el = ET.SubElement(model_el, f"{{{_3MF_CORE_NS}}}resources")
    build_el = ET.SubElement(model_el, f"{{{_3MF_CORE_NS}}}build")

    next_id = 1
    object_ids = []

    for colored_body in bodies:
        if colored_body.body is None or colored_body.body.is_empty():
            continue
        mesh = colored_body.body.to_mesh()
        verts = np.asarray(mesh.vert_properties[:, :3], dtype=np.float64)
        tris = np.asarray(mesh.tri_verts, dtype=np.int64)
        if len(tris) == 0:
            continue

        color_group_id, next_id = next_id, next_id + 1
        colorgroup_el = ET.SubElement(resources_el, f"{{{_3MF_MATERIAL_NS}}}colorgroup", {"id": str(color_group_id)})
        rgba = colored_body.color or (0.8, 0.8, 0.8, 1.0)
        ET.SubElement(colorgroup_el, f"{{{_3MF_MATERIAL_NS}}}color", {"color": _3mf_hex_color(rgba)})

        object_id, next_id = next_id, next_id + 1
        object_el = ET.SubElement(resources_el, f"{{{_3MF_CORE_NS}}}object", {
            "id": str(object_id), "type": "model", "pid": str(color_group_id), "pindex": "0",
        })
        mesh_el = ET.SubElement(object_el, f"{{{_3MF_CORE_NS}}}mesh")
        vertices_el = ET.SubElement(mesh_el, f"{{{_3MF_CORE_NS}}}vertices")
        for v in verts:
            ET.SubElement(vertices_el, f"{{{_3MF_CORE_NS}}}vertex",
                          {"x": f"{v[0]:.6g}", "y": f"{v[1]:.6g}", "z": f"{v[2]:.6g}"})
        triangles_el = ET.SubElement(mesh_el, f"{{{_3MF_CORE_NS}}}triangles")
        for t in tris:
            ET.SubElement(triangles_el, f"{{{_3MF_CORE_NS}}}triangle",
                          {"v1": str(int(t[0])), "v2": str(int(t[1])), "v3": str(int(t[2]))})

        object_ids.append(object_id)

    if not object_ids:
        raise ValueError("No geometry to export")

    for object_id in object_ids:
        ET.SubElement(build_el, f"{{{_3MF_CORE_NS}}}item", {"objectid": str(object_id)})

    model_xml = b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n' + ET.tostring(model_el, encoding="utf-8")

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", _3MF_CONTENT_TYPES)
        z.writestr("_rels/.rels", _3MF_RELS)
        z.writestr("3D/3dmodel.model", model_xml)


_WRITERS = {"stl": write_stl, "obj": write_obj, "off": write_off, "3mf": write_3mf}


def export_bodies(path: str, bodies: list[ColoredBody], fmt: str | None = None) -> None:
    """Write `bodies` to `path`, inferring the format ("stl"/"obj"/"off"/"3mf")
    from its extension unless `fmt` is given explicitly."""
    fmt = fmt or format_for_path(path)
    _WRITERS[fmt](path, bodies)
