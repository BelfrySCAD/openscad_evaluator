"""Headless export for `ColoredBody` lists. A port of openscad_cpp_evaluator's
export.cpp: the same object split, colour rules, checks and file layouts.

`export_model` is the one entry point. Single-mesh formats (STL, OFF) write
the implicit top-level union as one mesh; multi-object formats (3MF, OBJ)
write one object per colour -- the per-colour volume claim, later `color()`
winning an overlap -- optionally one per connected component, carrying
per-triangle colour where a CSG merge produced it. Pure Python: a 3MF is a
ZIP of XML, so no `lib3mf`.
"""
from __future__ import annotations

import struct
import zipfile
from pathlib import Path

import manifold3d as m3d
import numpy as np

from openscad_evaluator.evaluator import ColoredBody, to_renderable_bodies
from openscad_evaluator.mesh_check import check_mesh, strip_slivers

DEFAULT_EXPORT_COLOR = (0.8, 0.8, 0.8, 1.0)

# The single source of the format list: the CLI, format_for_path and any
# front end read it from here rather than restating it (cpp #118).
_EXTENSIONS = (".3mf", ".stl", ".obj", ".off")
_MULTI_OBJECT = {".3mf", ".obj"}


def export_extensions() -> list[str]:
    """The dot-prefixed extensions export_model writes."""
    return list(_EXTENSIONS)


def format_for_path(path: str) -> str:
    """The export format ("stl", "3mf", ...) for `path`'s extension."""
    ext = Path(path).suffix.lower()
    if ext not in _EXTENSIONS:
        raise ValueError(f"Unrecognized export extension '{ext}' (expected one of {sorted(_EXTENSIONS)})")
    return ext[1:]


def _g6(v) -> str:
    return f"{float(v):.6g}"


def _mesh_arrays(body: m3d.Manifold):
    """(verts float32 (N, 3), tris int (M, 3)) as Manifold hands them out --
    float, as the C++ port's GetMeshGL() is, so formatted output agrees."""
    mesh = body.to_mesh()
    return np.asarray(mesh.vert_properties, dtype=np.float32)[:, :3], np.asarray(mesh.tri_verts, dtype=np.int64)


def _is_display_only(b: ColoredBody) -> bool:
    return b.raw_mesh is not None and (b.body is None or b.body.is_empty())


class ExportObject:
    """One object in a multi-object file: its mesh, base colour, and
    per-triangle colours (empty when the object is one flat colour)."""
    __slots__ = ("verts", "tris", "color", "tri_colors")

    def __init__(self, verts, tris, color, tri_colors=None):
        self.verts, self.tris, self.color = verts, tris, tuple(float(c) for c in color)
        self.tri_colors = [] if tri_colors is None else [tuple(float(c) for c in t) for t in tri_colors]


def _add_all(parts: list) -> m3d.Manifold:
    return parts[0] if len(parts) == 1 else m3d.Manifold.batch_boolean(parts, m3d.OpType.Add)


def _boxes_overlap(a, b) -> bool:
    return not (a[3] < b[0] or b[3] < a[0] or a[4] < b[1] or b[4] < a[1] or a[5] < b[2] or b[5] < a[2])


def split_bodies_for_export(bodies: list[ColoredBody], open_parts: list | None = None,
                            split_components: bool = False, split_colors: bool = True) -> list[ExportObject]:
    """The objects a multi-object file holds. Background (`%`) bodies are
    scenery and never exported; a 2D preview slab is dropped beside real
    solids, as the reference does, but exported when it is all there is.
    Open shells can join no boolean, so they are written as-is and their
    1-based index appended to `open_parts`.

    With several colours, each body keeps only the volume no LATER body
    claims (later `color()` wins an overlap), cut only by the later bodies
    whose boxes reach it -- a running union was quadratic (#177), and
    `A - disjoint B` would reorder A's triangles and lose its per-triangle
    colours. Same-coloured pieces then merge. With `split_colors` off, or one
    colour, everything welds into one solid."""
    renderable = to_renderable_bodies(bodies)
    has_real_solid = any(not b.flat_preview and b.role != "background" for b in renderable)
    solids, loose = [], []
    for index, cb in enumerate(renderable, 1):
        if cb.role == "background" or (cb.flat_preview and has_real_solid):
            continue
        if _is_display_only(cb):
            verts, tris = cb.raw_mesh
            if not len(tris):
                continue
            loose.append(ExportObject(np.asarray(verts, dtype=np.float32), np.asarray(tris, dtype=np.int64),
                                      cb.color or DEFAULT_EXPORT_COLOR, cb.tri_colors))
            if open_parts is not None:
                open_parts.append(index)
            continue
        if cb.body is None or cb.body.is_empty():
            continue
        tris = _mesh_arrays(cb.body)[1]
        if not len(tris):
            continue
        tri_colors = cb.tri_colors if cb.tri_colors is not None else None
        solids.append({"man": cb.body, "color": cb.color, "tri_colors": tri_colors, "tris": tris})

    def key(i):
        s = solids[i]
        # Per-triangle colour indexes its own triangle list, which any union
        # rewrites, so such a body is kept apart from everything.
        return ("tri", i) if s["tri_colors"] is not None else ("flat", s["color"])

    groups = []  # [man, color, tri_colors, source_tris, key]
    if solids:
        keys = [key(i) for i in range(len(solids))]
        if not split_colors or all(k == keys[0] for k in keys):
            groups.append([_add_all([s["man"] for s in solids]), solids[0]["color"],
                           solids[0]["tri_colors"] if split_colors else None, solids[0]["tris"], keys[0]])
        else:
            boxes = [s["man"].bounding_box() for s in solids]
            owned = []
            for n in range(len(solids) - 1, -1, -1):
                s = solids[n]
                blockers = [solids[m]["man"] for m in range(n + 1, len(solids)) if _boxes_overlap(boxes[n], boxes[m])]
                piece = s["man"] - _add_all(blockers) if blockers else s["man"]
                if piece.is_empty():
                    continue
                owned.append([piece, s["color"], s["tri_colors"], s["tris"], keys[n]])
            for g in reversed(owned):
                same = next((h for h in groups if h[4] == g[4]), None)
                if same is None:
                    groups.append(g)
                else:
                    same[0] = _add_all([same[0], g[0]])

    out = []
    for man, color, tri_colors, source_tris, _ in groups:
        parts = man.decompose() if split_components else [man]
        for part in parts:
            if part.is_empty():
                continue
            verts, tris = _mesh_arrays(part)
            if not len(tris):
                continue
            # Per-triangle colour survives only an object whose triangles came
            # through untouched; anything that cut geometry won't match.
            carried = tri_colors if (tri_colors is not None and len(tri_colors) == len(source_tris)
                                     and np.array_equal(source_tris, tris)) else None
            out.append(ExportObject(verts, tris, color or DEFAULT_EXPORT_COLOR, carried))
    return out + loose


def check_export_bodies(bodies: list[ColoredBody]) -> list[str]:
    """One message per body that is not a closed, consistently wound manifold."""
    out = []
    for n, b in enumerate(bodies, 1):
        if b.body is None or b.body.is_empty():
            continue
        d = check_mesh(*_mesh_arrays(b.body))
        if not d.ok():
            out.append(f"part {n} is not a closed manifold solid -- {d.summary()}")
    return out


def _merge_bodies(bodies: list[ColoredBody], open_parts: list):
    """The single mesh STL/OFF write: a union, not a concatenation -- where
    two bodies touch, concatenating writes both copies of the shared face,
    which a slicer rejects. Open shells can't join the union, so their
    triangles are appended and their index reported."""
    solids, loose = [], []
    for index, b in enumerate(bodies, 1):
        if b.role == "background":
            continue
        if _is_display_only(b):
            if len(b.raw_mesh[1]):
                loose.append(b.raw_mesh)
                open_parts.append(index)
            continue
        if b.body is not None and not b.body.is_empty():
            solids.append(b.body)
    parts = ([_mesh_arrays(_add_all(solids))] if solids else []) + \
        [(np.asarray(v, dtype=np.float32), np.asarray(t, dtype=np.int64)) for v, t in loose]
    if not parts:
        return None
    offsets = np.cumsum([0] + [len(v) for v, _ in parts[:-1]])
    return np.vstack([v for v, _ in parts]), np.vstack([t + o for (_, t), o in zip(parts, offsets)])


def _normals(verts, tris):
    v0, v1, v2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    n = np.cross(v1 - v0, v2 - v0).astype(np.float32)
    length = np.linalg.norm(n, axis=1, keepdims=True)
    return np.where(length > 0, n / np.where(length > 0, length, 1), 0).astype(np.float32), v0, v1, v2


def _write_stl(path: str, verts, tris) -> None:
    normals, v0, v1, v2 = _normals(verts, tris)
    data = np.zeros(len(tris), dtype=np.dtype([("normal", "<f4", (3,)), ("v0", "<f4", (3,)), ("v1", "<f4", (3,)),
                                               ("v2", "<f4", (3,)), ("attr", "<u2")]))
    data["normal"], data["v0"], data["v1"], data["v2"] = normals, v0, v1, v2
    with open(path, "wb") as f:
        f.write(b"\0" * 80)
        f.write(struct.pack("<I", len(tris)))
        f.write(data.tobytes())


def _write_off(path: str, verts, tris) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"OFF\n{len(verts)} {len(tris)} 0\n")
        f.writelines(f"{_g6(v[0])} {_g6(v[1])} {_g6(v[2])}\n" for v in verts)
        f.writelines(f"3 {t[0]} {t[1]} {t[2]}\n" for t in tris)


def _write_obj(path: str, objects: list[ExportObject]) -> None:
    """OBJ plus a .mtl beside it: `usemtl` names a material there, one per
    distinct colour, first-seen. A multi-coloured object is runs of faces
    with a usemtl wherever the colour changes, in triangle order."""
    p = Path(path)
    mtl_path = p.with_suffix(".mtl")
    materials: list[tuple] = []

    def material(c):
        if c not in materials:
            materials.append(c)
        return materials.index(c) + 1

    for o in objects:
        for c in (o.tri_colors or [o.color]):
            material(c)
    lines = [f"mtllib {mtl_path.name}\n\n"] if materials else []
    offset = 1
    for n, o in enumerate(objects, 1):
        lines.append(f"o object_{n}\n")
        lines += [f"v {_g6(v[0])} {_g6(v[1])} {_g6(v[2])}\n" for v in o.verts]
        current = None
        for t, tri in enumerate(o.tris):
            m = material(o.tri_colors[t] if o.tri_colors else o.color)
            if m != current:
                lines.append(f"usemtl color_{m}\n")
                current = m
            lines.append(f"f {tri[0] + offset} {tri[1] + offset} {tri[2] + offset}\n")
        lines.append("\n")
        offset += len(o.verts)
    p.write_text("".join(lines), encoding="utf-8")
    if materials:
        clamp = lambda c: min(max(c, 0.0), 1.0)  # noqa: E731
        mtl_path.write_text("".join(
            f"newmtl color_{i}\nKd {_g6(clamp(c[0]))} {_g6(clamp(c[1]))} {_g6(clamp(c[2]))}\n"
            + (f"d {_g6(clamp(c[3]))}\n" if c[3] < 1 else "") + "\n"  # d is opacity: 1 is solid
            for i, c in enumerate(materials, 1)), encoding="utf-8")


def _hex(rgba) -> str:
    r, g, b, a = (max(0, min(255, round(c * 255))) for c in rgba)
    return f"#{r:02X}{g:02X}{b:02X}{a:02X}"


_3MF_CONTENT_TYPES = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>'
    '</Types>')
_3MF_RELS = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Target="/3D/3dmodel.model" Id="rel0" '
    'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>')


def _write_3mf(path: str, objects: list[ExportObject]) -> None:
    """One mesh object per ExportObject, each with its own colour group. A
    multi-coloured surface is written per triangle (`p1`), which is what 3MF
    colour means -- the surface, not the material through the volume."""
    resources, build, next_id = [], [], 1
    for o in objects:
        if not len(o.tris):
            continue
        group, next_id = next_id, next_id + 1
        palette, tri_palette = [], []
        if o.tri_colors:
            for c in o.tri_colors:
                h = _hex(c)
                if h not in palette:
                    palette.append(h)
                tri_palette.append(palette.index(h))
        else:
            palette.append(_hex(o.color))
        resources.append(f'<m:colorgroup id="{group}">'
                         + "".join(f'<m:color color="{h}"/>' for h in palette) + "</m:colorgroup>")
        obj, next_id = next_id, next_id + 1
        resources.append(f'<object id="{obj}" type="model" pid="{group}" pindex="0"><mesh><vertices>')
        resources += [f'<vertex x="{_g6(v[0])}" y="{_g6(v[1])}" z="{_g6(v[2])}"/>' for v in o.verts]
        resources.append("</vertices><triangles>")
        for t, tri in enumerate(o.tris):
            extra = f' pid="{group}" p1="{tri_palette[t]}"' if tri_palette else ""
            resources.append(f'<triangle v1="{tri[0]}" v2="{tri[1]}" v3="{tri[2]}"{extra}/>')
        resources.append("</triangles></mesh></object>")
        build.append(f'<item objectid="{obj}"/>')
    if not build:
        raise ValueError("No geometry to export")
    model = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
             '<model unit="millimeter" xml:lang="en-US" '
             'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" '
             'xmlns:m="http://schemas.microsoft.com/3dmanufacturing/material/2015/02">'
             f'<resources>{"".join(resources)}</resources><build>{"".join(build)}</build></model>')
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", _3MF_CONTENT_TYPES)
        z.writestr("_rels/.rels", _3MF_RELS)
        z.writestr("3D/3dmodel.model", model)


def export_model(path: str, bodies: list[ColoredBody], fmt: str = "", strip_slivers_first: bool = True,
                 split_components: bool = False, split_colors: bool = True) -> list[str]:
    """Write `bodies` (evaluate()'s, or to_renderable_bodies()' of them) to
    `path` and return the warnings -- open shells, mesh problems, slivers
    removed. Nothing here refuses to write: a deliberately open surface is a
    legitimate export. Raises ValueError for no geometry or an unknown
    format. `split_components` gives every disconnected piece its own object
    (off: what OpenSCAD writes); `split_colors` off welds everything into
    one solid for a single-material print."""
    ext = (fmt or Path(path).suffix).lower()
    ext = ext if ext.startswith(".") else "." + ext
    if ext not in _EXTENSIONS:
        raise ValueError(f"Unsupported export format '{ext}'")
    warnings: list[str] = []

    def report_open(open_parts):
        warnings.extend(f"part {n} is not a closed solid; its surface is written as-is, and most slicers "
                        "will reject it." for n in open_parts)

    if ext in _MULTI_OBJECT:
        renderable = to_renderable_bodies(bodies)
        warnings += check_export_bodies(renderable)
        open_parts: list[int] = []
        objects = split_bodies_for_export(bodies, open_parts, split_components, split_colors)
        report_open(open_parts)
        if not objects:
            raise ValueError("No geometry to export")
        (_write_3mf if ext == ".3mf" else _write_obj)(path, objects)
        return warnings

    open_parts = []
    mesh = _merge_bodies(to_renderable_bodies(bodies), open_parts)
    report_open(open_parts)
    if mesh is None:
        raise ValueError("No geometry to export")
    verts, tris = mesh
    if strip_slivers_first:
        stripped, report = strip_slivers(verts, tris)
        if report["removed"]:
            warnings.append(f"removed {report['removed']} zero-area face(s) before writing.")
            tris = stripped
    # Checked after merging and stripping, since that is what gets written:
    # checking parts passed a Menger sponge whose file was full of duplicates.
    d = check_mesh(verts, tris)
    if not d.ok():
        warnings.append(f"exported mesh {d.summary()}")
    (_write_off if ext == ".off" else _write_stl)(path, verts, tris)
    return warnings


def export_bodies(path: str, bodies: list[ColoredBody], fmt: str | None = None) -> None:
    """export_model without the warnings, for existing callers."""
    export_model(path, bodies, fmt or "")


def write_stl(path: str, bodies: list[ColoredBody]) -> None:
    export_model(path, bodies, "stl")


def write_obj(path: str, bodies: list[ColoredBody]) -> None:
    export_model(path, bodies, "obj")


def write_off(path: str, bodies: list[ColoredBody]) -> None:
    export_model(path, bodies, "off")


def write_3mf(path: str, bodies: list[ColoredBody]) -> None:
    export_model(path, bodies, "3mf")
