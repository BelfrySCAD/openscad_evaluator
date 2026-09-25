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
_EXTENSIONS = (".3mf", ".amf", ".pdf", ".stl", ".obj", ".off", ".ply", ".svg", ".wrl", ".x3d")
_MULTI_OBJECT = {".3mf", ".amf", ".obj", ".ply", ".wrl", ".x3d"}


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


def _write_stl_ascii(path: str, verts, tris) -> None:
    normals, v0, v1, v2 = _normals(verts, tris)

    def xyz(v):
        return f"{_g6(v[0])} {_g6(v[1])} {_g6(v[2])}"
    with open(path, "w", encoding="utf-8") as f:
        f.write("solid OpenSCAD_Model\n")
        for n, a, b, c in zip(normals, v0, v1, v2):
            f.write(f"  facet normal {xyz(n)}\n    outer loop\n      vertex {xyz(a)}\n      vertex {xyz(b)}\n"
                    f"      vertex {xyz(c)}\n    endloop\n  endfacet\n")
        f.write("endsolid OpenSCAD_Model\n")


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


def _rgb255(c) -> tuple:
    return tuple(max(0, min(255, round(c[i] * 255))) for i in range(3))


def _write_ply(path: str, objects: list[ExportObject]) -> None:
    """Binary PLY with vertex colour. A vertex shared by differently coloured
    triangles has no single colour, so an object with per-triangle colour is
    unwelded: three vertices per triangle, each in that triangle's colour."""
    verts, colors, faces = [], [], []
    for o in objects:
        base = sum(len(v) for v in verts)
        if not o.tri_colors:
            verts.append(np.asarray(o.verts, dtype="<f4"))
            colors += [_rgb255(o.color)] * len(o.verts)
            faces.append(np.asarray(o.tris) + base)
        else:
            verts.append(np.asarray(o.verts, dtype="<f4")[np.asarray(o.tris).reshape(-1)])
            colors += [_rgb255(c) for c in o.tri_colors for _ in range(3)]
            faces.append(np.arange(len(o.tris) * 3).reshape(-1, 3) + base)
    v = np.vstack(verts) if verts else np.zeros((0, 3), "<f4")
    f = np.vstack(faces) if faces else np.zeros((0, 3), int)
    vrec = np.zeros(len(v), dtype=[("p", "<f4", (3,)), ("c", "u1", (3,))])
    vrec["p"], vrec["c"] = v, colors if colors else np.zeros((0, 3))
    frec = np.zeros(len(f), dtype=[("n", "u1"), ("i", "<i4", (3,))])
    frec["n"], frec["i"] = 3, f
    header = ("ply\nformat binary_little_endian 1.0\ncomment Written by BelfrySCAD\n"
              f"element vertex {len(v)}\nproperty float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              f"element face {len(f)}\nproperty list uchar int vertex_indices\nend_header\n")
    with open(path, "wb") as fh:
        fh.write(header.encode())
        fh.write(vrec.tobytes())
        fh.write(frec.tobytes())


def _face_colors(o: ExportObject):
    """(RGB palette, one index per face), empty when the object is one flat
    colour -- VRML and X3D are one scene graph in two syntaxes."""
    palette, index = [], []
    for c in o.tri_colors:
        rgb = c[:3]
        if rgb not in palette:
            palette.append(rgb)
        index.append(palette.index(rgb))
    return palette, index


def _write_vrml(path: str, objects: list[ExportObject]) -> None:
    out = ["#VRML V2.0 utf8\n# Written by BelfrySCAD\n\n"]
    for o in objects:
        palette, index = _face_colors(o)
        transparency = 1.0 - o.color[3]
        out.append(f"Shape {{\n  appearance Appearance {{\n    material Material {{\n"
                   f"      diffuseColor {_g6(o.color[0])} {_g6(o.color[1])} {_g6(o.color[2])}\n")
        # Neither VRML nor X3D carries per-triangle alpha: the base alpha applies.
        if transparency > 0:
            out.append(f"      transparency {_g6(transparency)}\n")
        out.append("    }\n  }\n  geometry IndexedFaceSet {\n    solid TRUE\n    coord Coordinate {\n      point [\n")
        out += [f"        {_g6(v[0])} {_g6(v[1])} {_g6(v[2])},\n" for v in o.verts]
        out.append("      ]\n    }\n")
        if palette:
            out.append("    colorPerVertex FALSE\n    color Color {\n      color [\n")
            out += [f"        {_g6(c[0])} {_g6(c[1])} {_g6(c[2])},\n" for c in palette]
            out.append("      ]\n    }\n    colorIndex [\n")
            out += [f"      {i},\n" for i in index]
            out.append("    ]\n")
        out.append("    coordIndex [\n")
        out += [f"      {t[0]} {t[1]} {t[2]} -1,\n" for t in o.tris]
        out.append("    ]\n  }\n}\n\n")
    Path(path).write_text("".join(out), encoding="utf-8")


def _write_x3d(path: str, objects: list[ExportObject]) -> None:
    out = ['<?xml version="1.0" encoding="UTF-8"?>\n',
           '<!DOCTYPE X3D PUBLIC "ISO//Web3D//DTD X3D 3.3//EN" "https://www.web3d.org/specifications/x3d-3.3.dtd">\n',
           '<X3D profile="Interchange" version="3.3">\n',
           '  <head>\n    <meta name="generator" content="BelfrySCAD" />\n  </head>\n  <Scene>\n']
    for o in objects:
        palette, index = _face_colors(o)
        transparency = 1.0 - o.color[3]
        out.append(f'    <Shape>\n      <Appearance>\n        <Material diffuseColor="{_g6(o.color[0])} '
                   f'{_g6(o.color[1])} {_g6(o.color[2])}"')
        if transparency > 0:
            out.append(f' transparency="{_g6(transparency)}"')
        out.append(' />\n      </Appearance>\n      <IndexedFaceSet solid="true"')
        if palette:
            out.append(f' colorPerVertex="false" colorIndex="{" ".join(map(str, index))}"')
        out.append(' coordIndex="' + " ".join(f"{t[0]} {t[1]} {t[2]} -1" for t in o.tris) + '">\n')
        out.append('        <Coordinate point="' + " ".join(f"{_g6(v[0])} {_g6(v[1])} {_g6(v[2])}" for v in o.verts)
                   + '" />\n')
        if palette:
            out.append('        <Color color="' + " ".join(f"{_g6(c[0])} {_g6(c[1])} {_g6(c[2])}" for c in palette)
                       + '" />\n')
        out.append("      </IndexedFaceSet>\n    </Shape>\n")
    out.append("  </Scene>\n</X3D>\n")
    Path(path).write_text("".join(out), encoding="utf-8")


def _write_amf(path: str, objects: list[ExportObject]) -> None:
    """AMF colours a <volume>, not a face, so an object whose triangles are
    not one colour is one volume per colour. The material table is global;
    ids start at 1 (0 is reserved)."""
    materials: list[tuple] = []

    def material(c):
        if c not in materials:
            materials.append(c)
        return materials.index(c) + 1

    per_object = []
    for o in objects:
        palette, index = _face_colors(o)
        if not palette:
            per_object.append([(material(o.color), list(o.tris))])
        else:
            vols = [(material(next(c for c in o.tri_colors if c[:3] == p)), []) for p in palette]
            for t, i in zip(o.tris, index):
                vols[i][1].append(t)
            per_object.append(vols)
    out = ['<?xml version="1.0" encoding="UTF-8"?>\n<amf unit="millimeter" version="1.1">\n',
           '  <metadata type="cad">BelfrySCAD</metadata>\n']
    for i, c in enumerate(materials, 1):
        out.append(f'  <material id="{i}">\n    <color><r>{_g6(c[0])}</r><g>{_g6(c[1])}</g><b>{_g6(c[2])}</b>'
                   + (f"<a>{_g6(c[3])}</a>" if c[3] < 1 else "") + "</color>\n  </material>\n")
    for n, (o, vols) in enumerate(zip(objects, per_object), 1):
        out.append(f'  <object id="{n}">\n    <mesh>\n      <vertices>\n')
        out += [f"        <vertex><coordinates><x>{_g6(v[0])}</x><y>{_g6(v[1])}</y><z>{_g6(v[2])}</z>"
                "</coordinates></vertex>\n" for v in o.verts]
        out.append("      </vertices>\n")
        for mat, tris in vols:
            if not tris:
                continue
            out.append(f'      <volume materialid="{mat}">\n')
            out += [f"        <triangle><v1>{t[0]}</v1><v2>{t[1]}</v2><v3>{t[2]}</v3></triangle>\n" for t in tris]
            out.append("      </volume>\n")
        out.append("    </mesh>\n  </object>\n")
    out.append("</amf>\n")
    Path(path).write_text("".join(out), encoding="utf-8")


def _collect_2d(bodies: list[ColoredBody]):
    """(contours per body, bbox) of an all-2D model -- SVG's and PDF's one
    input, so they agree about what they draw. Anything 3D refuses, in
    OpenSCAD's words: a silhouette would be a different model."""
    not_2d = "Current top level object is not a 2D object"
    per_body, pts = [], []
    for cb in bodies:
        if cb.role == "background":
            continue
        # The section decides, not the Manifold: a preview slab carries both,
        # and the contours are the real geometry.
        if cb.section is None:
            if _is_display_only(cb) or (cb.body is not None and not cb.body.is_empty()):
                raise ValueError(not_2d)
            continue
        polys = [np.asarray(p, dtype=np.float64) for p in cb.section.to_polygons()]
        per_body.append(polys)
        pts += [p for p in polys if len(p)]
    if not pts:
        raise ValueError("No geometry to export")
    allp = np.vstack(pts)
    return per_body, (*allp.min(axis=0), *allp.max(axis=0))


def _write_svg(path: str, bodies, fill=False, fill_color="white", stroke=True, stroke_color="black",
               stroke_width=0.35) -> None:
    """The model at 1:1 in millimetres, the page cut to fit: the bounding box
    padded by half the stroke (it straddles the contour), rounded outward to
    whole millimetres, Y negated so the model's top is the page's top."""
    import math
    per_body, (minx, miny, maxx, maxy) = _collect_2d(bodies)
    pad = stroke_width / 2 if stroke else 0.0
    left, top = math.floor(minx - pad), math.floor(-maxy - pad)
    width, height = math.ceil(maxx + pad) - left, math.ceil(-miny + pad) - top
    out = ['<?xml version="1.0" standalone="no"?>\n',
           '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">\n',
           f'<svg width="{width}mm" height="{height}mm" viewBox="{left} {top} {width} {height}" '
           'xmlns="http://www.w3.org/2000/svg" version="1.1">\n<title>OpenSCAD Model</title>\n']
    for polys in per_body:
        out.append('<path d="\n')
        for contour in polys:
            if not len(contour):
                continue
            out.append(f"M {contour[0][0]:g},{-contour[0][1]:g}")
            for i in range(1, len(contour)):
                out.append(f" L {contour[i][0]:g},{-contour[i][1]:g}")
                if i % 6 == 5:
                    out.append("\n")
            out.append(" z\n")
        out.append(f'" stroke="{stroke_color if stroke else "none"}" fill="{fill_color if fill else "none"}" '
                   f'stroke-width="{stroke_width:g}"/>\n')
    out.append("</svg>\n")
    Path(path).write_text("".join(out), encoding="utf-8")


_PAPER = {"a6": (298, 420), "a5": (420, 595), "a4": (595, 842), "a3": (842, 1190),
          "letter": (612, 792), "legal": (612, 1008), "tabloid": (792, 1224)}
_PDF_DEFAULTS = {"paper-size": "a4", "orientation": "portrait", "show-scale": True, "show-scale-message": True,
                 "show-grid": False, "grid-size": 10.0, "show-filename": False, "design-filename": "",
                 "fill": False, "fill-color": "black", "stroke": True, "stroke-color": "black",
                 "stroke-width": 0.35, "add-meta-data": True, "title": "", "author": "", "subject": "",
                 "keywords": ""}


def _pdf_num(v: float) -> str:
    """Short, locale-free, never an exponent: PDF has no "1e-05"."""
    import math
    s = f"{v if math.isfinite(v) else 0.0:.4f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def _pdf_string(text: str) -> str:
    # The base-14 fonts are single-byte: anything non-ASCII is dropped, not guessed.
    return "(" + "".join("\\" + c if c in "()\\" else c for c in text if 0x20 <= ord(c) < 0x7F) + ")"


def _write_pdf(path: str, bodies, options: dict | None = None) -> list[str]:
    """A 2D model at 1:1, centred on a fixed paper size, with a millimetre
    ruler labelled in MODEL coordinates -- a printed page then shows a
    printer's scaling error. Options are keyed as OpenSCAD's own
    `-O export-pdf/...`; an unknown key raises rather than being dropped."""
    import math
    import time
    import zlib
    opts = dict(_PDF_DEFAULTS)
    for k, v in (options or {}).items():
        if k not in opts:
            raise ValueError(f"unknown pdf option '{k}'")
        opts[k] = v
    paper = str(opts["paper-size"]).lower()
    if paper not in _PAPER:
        raise ValueError(f"unknown paper-size '{opts['paper-size']}' (a6, a5, a4, a3, letter, legal, tabloid)")
    orientation = str(opts["orientation"]).lower()
    if orientation not in ("portrait", "landscape", "auto"):
        raise ValueError(f"unknown orientation '{opts['orientation']}' (portrait, landscape, auto)")
    per_body, (minx, miny, maxx, maxy) = _collect_2d(bodies)
    warnings = []
    pt = 72 / 25.4
    margin, tick_mm, tick_step, label_every = 30.0, 5.0, 10.0, 2
    span_x, span_y = (maxx - minx) * pt, (maxy - miny) * pt
    landscape = orientation == "landscape" or (orientation == "auto" and span_x > span_y)
    page_w, page_h = _PAPER[paper][::-1] if landscape else _PAPER[paper]
    if span_x > page_w - 2 * margin or span_y > page_h - 2 * margin:
        warnings.append("geometry is larger than the printable area of the selected paper size; "
                        "it is drawn anyway and will run off the page.")
    # Model mm -> page pt; PDF is Y-up like the model, so no negation. Centred
    # exactly (OpenSCAD truncates the span to an int, a bug not reproduced).
    tx = page_w / 2 - (minx + maxx) / 2 * pt
    ty = page_h / 2 - (miny + maxy) / 2 * pt
    X, Y = (lambda mm: mm * pt + tx), (lambda mm: mm * pt + ty)
    ml, mr, mb, mt = margin, page_w - margin, margin, page_h - margin
    n = _pdf_num

    def color(name):
        from openscad_evaluator.evaluator import Evaluator
        return Evaluator._css_color(None, name)  # uses no state

    def line(x0, y0, x1, y1):
        return f"{n(x0)} {n(y0)} m {n(x1)} {n(y1)} l S\n"

    cs = []
    for polys in per_body:
        path_ = "".join(f"{n(X(c[0][0]))} {n(Y(c[0][1]))} m\n" + "".join(f"{n(X(p[0]))} {n(Y(p[1]))} l\n" for p in c[1:])
                        + "h\n" for c in polys if len(c))
        if not path_:
            continue
        cs.append("q\n")
        if opts["fill"]:
            r, g, b, _ = color(opts["fill-color"])
            cs.append(f"{n(r)} {n(g)} {n(b)} rg\n")
        if opts["stroke"]:
            r, g, b, _ = color(opts["stroke-color"])
            cs.append(f"{n(r)} {n(g)} {n(b)} RG\n{n(opts['stroke-width'] * pt)} w\n")
        cs.append(path_ + ("B\n" if opts["fill"] and opts["stroke"] else "f\n" if opts["fill"]
                           else "S\n" if opts["stroke"] else "n\n") + "Q\n")
    texts = []
    if opts["show-scale"]:
        cs.append("q\n0.4 G\n0.36 w\n" + line(ml, mb, ml, mt) + line(ml, mb, mr, mb))
        tick = tick_mm * pt
        # Ticks on model multiples of 10mm, not page divisions: that is what
        # makes the printed ruler measure the model.
        for i in range(math.ceil(((ml - tx) / pt) / tick_step), math.floor(((mr - tx) / pt) / tick_step) + 1):
            px = X(i * tick_step)
            cs.append(line(px, mb, px, mb - tick))
            if i % label_every == 0:
                texts.append((px + 1.0, mb - tick + 2.0, 6.0, str(int(i * tick_step))))
        for i in range(math.ceil(((mb - ty) / pt) / tick_step), math.floor(((mt - ty) / pt) / tick_step) + 1):
            py = Y(i * tick_step)
            cs.append(line(ml, py, ml - tick, py))
            if i % label_every == 0:
                texts.append((ml - tick, py + 3.0, 6.0, str(int(i * tick_step))))
        if opts["show-grid"]:
            # OpenSCAD's clamp and its deliberately asymmetric major-line rule.
            g = 2.0 if opts["grid-size"] < 1 else float(opts["grid-size"])
            major = int(g if g > 10 else int(10 / g))
            for i in range(math.ceil(((ml - tx) / pt) / g), math.floor(((mr - tx) / pt) / g) + 1):
                heavy = major > 0 and i % major == 0
                cs.append(f"{n(0.4 if heavy else 0.6)} G\n{'0.36' if heavy else '0.24'} w\n" + line(X(i * g), mb, X(i * g), mt))
            for i in range(math.ceil(((mb - ty) / pt) / g), math.floor(((mt - ty) / pt) / g) + 1):
                heavy = major > 0 and i % major == 0
                cs.append(f"{n(0.4 if heavy else 0.6)} G\n{'0.36' if heavy else '0.24'} w\n" + line(ml, Y(i * g), mr, Y(i * g)))
        cs.append("Q\n")
        if opts["show-scale-message"]:
            texts.append((ml + 1.0, mb + 2.0, 5.0, "Scale is to calibrate actual printed dimension. Check both X "
                          "and Y. Measure between tick 0 and last tick"))
    if opts["show-filename"] and opts["design-filename"]:
        texts.append((ml, mb - tick_mm * pt - 10.0, 10.0, opts["design-filename"]))
    if texts:
        cs.append("q\n0.52 g\n" + "".join(f"BT /F1 {n(s)} Tf 1 0 0 1 {n(x)} {n(y)} Tm {_pdf_string(t)} Tj ET\n"
                                           for x, y, s, t in texts) + "Q\n")
    content = "".join(cs).encode("latin-1")
    stream = zlib.compress(content)
    deflated = len(stream) < len(content)
    if not deflated:
        stream = content
    info = ""
    if opts["add-meta-data"]:
        info = (f"<< /Producer {_pdf_string('BelfrySCAD (openscad_evaluator)')} /CreationDate "
                + _pdf_string(time.strftime("D:%Y%m%d%H%M%S+00'00'", time.gmtime()))
                + "".join(f" /{key.title()} {_pdf_string(opts[key])}" for key in ("title", "author", "subject", "keywords")
                          if opts[key]) + " >>")
    objects = ["<< /Type /Catalog /Pages 2 0 R >>", "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
               f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {n(page_w)} {n(page_h)}] "
               "/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>", None,
               "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"]
    if info:
        objects.append(info)
    body = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for i, obj in enumerate(objects, 1):
        offsets.append(len(body))
        body += f"{i} 0 obj\n".encode()
        if obj is None:
            body += (f"<< /Length {len(stream)}{' /Filter /FlateDecode' if deflated else ''} >>\nstream\n").encode()
            body += stream + b"\nendstream\n"
        else:
            body += (obj + "\n").encode("latin-1")
        body += b"endobj\n"
    xref = len(body)
    body += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    body += "".join(f"{off:010d} 00000 n \n" for off in offsets).encode()
    body += (f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R"
             + (f" /Info {len(objects)} 0 R" if info else "") + f" >>\nstartxref\n{xref}\n%%EOF\n").encode()
    Path(path).write_bytes(bytes(body))
    return warnings


def export_model(path: str, bodies: list[ColoredBody], fmt: str = "", strip_slivers_first: bool = True,
                 split_components: bool = False, split_colors: bool = True, ascii_stl: bool = False,
                 svg_fill: bool = False, svg_fill_color: str = "white", svg_stroke: bool = True,
                 svg_stroke_color: str = "black", svg_stroke_width: float = 0.35,
                 pdf_options: dict | None = None) -> list[str]:
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

    if ext == ".pdf":
        return _write_pdf(path, bodies, pdf_options)
    if ext == ".svg":  # 2D: none of the mesh pipeline applies
        _write_svg(path, bodies, svg_fill, svg_fill_color, svg_stroke, svg_stroke_color, svg_stroke_width)
        return warnings

    if ext in _MULTI_OBJECT:
        renderable = to_renderable_bodies(bodies)
        warnings += check_export_bodies(renderable)
        open_parts: list[int] = []
        objects = split_bodies_for_export(bodies, open_parts, split_components, split_colors)
        report_open(open_parts)
        if not objects:
            raise ValueError("No geometry to export")
        {".3mf": _write_3mf, ".obj": _write_obj, ".amf": _write_amf, ".ply": _write_ply,
         ".wrl": _write_vrml, ".x3d": _write_x3d}[ext](path, objects)
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
    (_write_off if ext == ".off" else _write_stl_ascii if ascii_stl else _write_stl)(path, verts, tris)
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
