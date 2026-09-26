"""
AST evaluator: walks the openscad_lalr_parser AST and produces Manifold geometry.
Returns (manifold_body, id_to_node, colored_meshes) or raises EvalError.
"""
from __future__ import annotations
import functools
import math
import re
import sys
import random
import threading
import time
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field, replace

import manifold3d as m3d
import numpy as np
from fontTools.ttLib import TTFont
import uharfbuzz as hb
from fontTools.pens.basePen import BasePen
from shapely_polyskel import skeletonize

from openscad_evaluator._css_colors import CSS_COLORS

from openscad_lalr_parser import to_openscad, findLibraryFile, getASTfromFile, build_scopes
from openscad_lalr_parser.nodes import (
    ASTNode, Assignment, Identifier,
    NumberLiteral, BooleanLiteral, StringLiteral, UndefinedLiteral,
    CommentedExpr,
    ListComprehension, ListCompFor, ListCompCFor, ListCompIf, ListCompIfElse, ListCompLet, ListCompEach,
    PositionalArgument, NamedArgument,
    AdditionOp, SubtractionOp, MultiplicationOp, DivisionOp, ModuloOp, ExponentOp,
    UnaryMinusOp,
    BitwiseAndOp, BitwiseOrOp, BitwiseNotOp, BitwiseShiftLeftOp, BitwiseShiftRightOp,
    LogicalAndOp, LogicalOrOp, LogicalNotOp,
    EqualityOp, InequalityOp, GreaterThanOp, GreaterThanOrEqualOp, LessThanOp, LessThanOrEqualOp,
    TernaryOp,
    PrimaryCall, PrimaryIndex, PrimaryMember,
    RangeLiteral, RenderExpression,
    ModularCall, ModularIf, ModularIfElse, ModularFor, ModularLet,
    ModularEcho, ModularAssert, ModularIntersectionFor,
    ModularModifierShowOnly, ModularModifierHighlight,
    ModularModifierBackground, ModularModifierDisable,
    ModuleDeclaration, FunctionDeclaration, ParameterDeclaration,
    UseStatement,
    VectorElement,
    LetOp, EchoOp, AssertOp,
    FunctionLiteral,
)


_MANIFOLD_OK = m3d.Error.NoError
_NO_OPERAND = object()  # a unary operator's missing right operand

_HEX = frozenset("0123456789abcdefABCDEF")


def _vnf_from_mesh(verts: np.ndarray, tris: np.ndarray) -> tuple[list, list]:
    """Manifold (verts, CCW tris) -> VNF ([x,y,z] points, CW faces), the
    shape polyhedron() and BOSL2 take.

    Winding is reversed, since VNF faces are clockwise seen from outside; a
    reversed mesh still builds, only inside out (negative volume). Vertices
    Manifold split at seams are welded by exact position, so a cube is 8
    points, not 24 -- but only when the welded mesh stays manifold: two
    shells that touch have distinct vertices at the same place, and fusing
    them makes edges with four faces."""
    uniq, remap = np.unique(verts, axis=0, return_inverse=True)
    remap = remap.reshape(-1)
    if len(uniq) < len(verts):
        welded = remap[tris]
        # Counted, not asked of Manifold: it builds a mesh with four-face
        # edges without complaint, so its status let the fused shells through.
        if _edges_closed(welded):
            verts, tris = uniq, welded
    return verts.tolist(), tris[:, [0, 2, 1]].tolist()


def _minkowski_2d(a: m3d.CrossSection, b: m3d.CrossSection) -> m3d.CrossSection:
    """The Minkowski sum of two 2D shapes, which Manifold has no operation
    for. For a convex B containing the origin,

        A (+) B  =  A  union  (the boundary of A swept by B)

    and each edge of A sweeps to the hull of B at its two ends -- so A is
    never cut up, however concave or holed; only its edges are walked. B is
    split into convex pieces (it has to be convex for the per-edge hull),
    and each piece is moved to contain the origin and the result moved back
    (a piece off the origin would otherwise keep an unmoved copy of A).
    Minkowski distributes over union, so the pieces' sums are unioned."""
    b_polys = [np.asarray(p, dtype=np.float64) for p in b.to_polygons()]
    if len(b_polys) == 1 and abs(b.hull().area() - b.area()) <= 1e-9 * max(b.area(), 1.0):
        pieces = b_polys
    else:
        verts = np.vstack(b_polys)
        pieces = [verts[t] for t in np.asarray(m3d.triangulate(b_polys))]
    a_polys = [np.asarray(p, dtype=np.float64) for p in a.to_polygons()]
    sums = []
    for piece in pieces:
        c = piece.mean(axis=0)
        q = piece - c
        swept = [a]
        for poly in a_polys:
            for p0, p1 in zip(poly, np.roll(poly, -1, axis=0)):
                swept.append(m3d.CrossSection.hull_points(np.vstack([q + p0, q + p1])))
        sums.append(m3d.CrossSection.batch_boolean(swept, m3d.OpType.Add).translate(c.tolist()))
    return m3d.CrossSection.batch_boolean(sums, m3d.OpType.Add)


def _range_count(r: "OscRange") -> float:
    """How many elements a range has, without walking it: 0 when it is empty
    (a NaN anywhere, or a step pointing away from the end), inf when it never
    ends (a zero step toward a larger end, or an infinite end), as OpenSCAD
    counts them."""
    start, step, end = r.start, r.step, r.end
    if math.isnan(start) or math.isnan(step) or math.isnan(end):
        return 0
    if step == 0:
        return math.inf if start < end else 0
    n = (end - start) / step
    if n < -1e-10:
        return 0
    return math.inf if math.isinf(n) else math.floor(n + 1e-10) + 1


# The OpenSCAD release this evaluator tracks, as openscad_cpp_evaluator does:
# 2026.01.01, the first tagged build with hex literals and object().
# 2025.01.01, reported before, was never a release.
_OPENSCAD_VERSION = (2026, 1, 1)


def _version_num(v=None):
    """version_num(): the release as y*10000 + m*100 + d, or that of a
    [y, m, d] / [y, m] vector passed in (which was ignored); anything else
    is undef."""
    if v is None:
        v = _OPENSCAD_VERSION
    if not isinstance(v, (list, tuple)) or len(v) not in (2, 3) or \
            not all(type(x) in (int, float) for x in v):
        return None
    y, m, d = (list(v) + [0])[:3]
    return y * 10000 + m * 100 + d


def _signed_area(poly) -> float:
    """Shoelace area: positive for a counter-clockwise contour."""
    p = np.asarray(poly, dtype=np.float64)
    return 0.5 * float(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(np.roll(p[:, 0], -1), p[:, 1]))


_TRIG_HUGE = float(1 << 26) * 360.0 * float(1 << 26)
_SQRT1_2, _SQRT3_4, _SQRT1_3, _SQRT3 = 0.70710678118654752440, 0.86602540378443859659, 0.57735026918962573106, 1.73205080756887719318


def _reduce_deg(x: float, period: float):
    """x wrapped into [0, period), or None when it is too large (or not
    finite) for the wrap to mean anything."""
    if 0.0 <= x < period:
        return x
    if -_TRIG_HUGE < x < _TRIG_HUGE:
        return x - period * math.floor(x / period)
    return None


def _sin_deg(x: float) -> float:
    """OpenSCAD's sin() in degrees (degree_trig.cc): folded into [0, 90]
    and read off exact values at 30/45/60, sin below 45 and cos above -- so
    sin(45) and cos(45) are the SAME double and sin(45) - cos(45) is 0."""
    x = _reduce_deg(x, 360.0)
    if x is None:
        return math.nan
    oppose = x >= 180.0
    if oppose:
        x -= 180.0
    if x > 90.0:
        x = 180.0 - x
    if x < 45.0:
        x = 0.5 if x == 30.0 else math.sin(math.radians(x))
    elif x == 45.0:
        x = _SQRT1_2
    elif x == 60.0:
        x = _SQRT3_4
    else:
        x = math.cos(math.radians(90.0 - x))
    return -x if oppose else x


def _cos_deg(x: float) -> float:
    x = _reduce_deg(x, 360.0)
    if x is None:
        return math.nan
    oppose = x >= 180.0
    if oppose:
        x -= 180.0
    if x > 90.0:
        x = 180.0 - x
        oppose = not oppose
    if x > 45.0:
        x = 0.5 if x == 60.0 else math.sin(math.radians(90.0 - x))
    elif x == 45.0:
        x = _SQRT1_2
    elif x == 30.0:
        x = _SQRT3_4
    else:
        x = math.cos(math.radians(x))
    return -x if oppose else x


def _tan_deg(x: float) -> float:
    if not math.isfinite(x):
        return math.nan
    cycles = math.floor(x / 180.0)
    x = _reduce_deg(x, 180.0)
    if x is None:
        return math.nan
    even = cycles % 2 == 0
    oppose = x > 90.0
    if oppose:
        x = 180.0 - x
    if x == 0.0:
        x = 0.0 if even else -0.0
    elif x == 30.0:
        x = _SQRT1_3
    elif x == 45.0:
        x = 1.0
    elif x == 60.0:
        x = _SQRT3
    elif x == 90.0:
        x = math.inf if even else -math.inf
    else:
        x = math.tan(math.radians(x))
    return -x if oppose else x


def _snap_inverse(degs: float, forward, x: float) -> float:
    """An inverse's result snaps to whole degrees when the forward function
    gives back x exactly: asin(sin(30)) is 30, not 30.000000000000004."""
    if not math.isfinite(degs):
        return degs
    whole = round(degs)
    return whole if forward(whole) == x else degs


def _asin_deg(x):
    return math.nan if abs(x) > 1 else _snap_inverse(math.degrees(math.asin(x)), _sin_deg, x)


def _acos_deg(x):
    return math.nan if abs(x) > 1 else _snap_inverse(math.degrees(math.acos(x)), _cos_deg, x)


def _atan_deg(x):
    return _snap_inverse(math.degrees(math.atan(x)), _tan_deg, x)


def _atan2_deg(y, x):
    degs = math.degrees(math.atan2(y, x))
    if not math.isfinite(degs):
        return degs
    whole = round(degs)
    return whole if abs(degs - whole) < 3.0e-14 else degs


def _cos_sin_deg(deg: float) -> tuple[float, float]:
    """cos and sin of `deg` degrees, exact at multiples of 90, so a quarter
    turn leaves no 6e-17 residue (a 2D shape turned edge-on would keep a
    sliver of area instead of none)."""
    q, r = divmod(round(deg, 12), 90)
    if r == 0:
        return ((1, 0), (0, 1), (-1, 0), (0, -1))[int(q) % 4]
    return math.cos(math.radians(deg)), math.sin(math.radians(deg))


def _rot_deg(deg: float, axis: int) -> np.ndarray:
    """Rotation matrix about x/y/z (0/1/2) by `deg` degrees, exact at
    multiples of 90."""
    c, s = _cos_sin_deg(deg)
    i, j = [k for k in range(3) if k != axis]
    m = np.eye(3)
    m[i, i], m[i, j], m[j, i], m[j, j] = c, -s, s, c
    if axis == 1:  # y: z-x plane, so the sine terms swap sign
        m[i, j], m[j, i] = s, -s
    return m


@functools.lru_cache(maxsize=4096)
def _undefined_escapes(raw: str) -> int:
    """How many escapes in a literal's source OpenSCAD calls undefined --
    one it warns about, each: an unknown letter (`\\q`), a `\\x` past 7F or
    short of digits, a short `\\u`/`\\U`, a backslash before a line end."""
    n, i = 0, 0
    while True:
        i = raw.find("\\", i)
        if i < 0 or i + 1 >= len(raw):
            return n
        e = raw[i + 1]
        if e in "ntr\\\"":
            pass
        elif e in "xuU":
            width = {"x": 2, "u": 4, "U": 6}[e]
            digits = raw[i + 2:i + 2 + width]
            if not (len(digits) == width and all(d in _HEX for d in digits)
                    and (e != "x" or int(digits, 16) <= 0x7F)):
                n += 1
        else:
            n += 1
        i += 2


@functools.lru_cache(maxsize=4096)
def _unescape_string(raw: str) -> str:
    r"""A string literal's value from its source text, which the parser
    keeps verbatim so the source can be reprinted. Matches OpenSCAD:
    \n \t \r; \xNN for 01-7F only; \uXXXX and \UXXXXXX, with NUL, a
    surrogate or anything past U+10FFFF becoming a space; any other escaped
    character standing for itself (\\, \", and unknown ones like \q).
    A line ending inside the literal contributes nothing, escaped or not,
    so a string can continue on the next line."""
    if "\\" not in raw and "\n" not in raw and "\r" not in raw:
        return raw
    out = []
    i, n = 0, len(raw)
    while i < n:
        c = raw[i]
        if c == "\n" or c == "\r":
            i += 1
            continue
        if c != "\\" or i + 1 >= n:
            out.append(c)
            i += 1
            continue
        e = raw[i + 1]
        i += 2
        if e == "n":
            out.append("\n")
        elif e == "t":
            out.append("\t")
        elif e == "r":
            out.append("\r")
        elif e in "xuU":
            width = {"x": 2, "u": 4, "U": 6}[e]
            digits = raw[i:i + width]
            if len(digits) == width and all(d in _HEX for d in digits) and (e != "x" or int(digits, 16) <= 0x7F):
                cp = int(digits, 16)
                out.append(" " if cp == 0 or 0xD800 <= cp <= 0xDFFF or cp > 0x10FFFF else chr(cp))
                i += width
            else:
                out.append(e)
        elif e != "\n" and e != "\r":
            out.append(e)
    return "".join(out)


class EvalError(Exception):
    pass


# The exact message _check_debug() raises EvalError with when a debug
# hook's returned cmd == "stop" is honored -- exported (not a leading-
# underscore internal name) so cli.py can tell "the debugger itself asked
# to abort" (its own "stop"/"restart"/"quit" commands) apart from a
# genuine script error (assert()/etc, possibly also inspected via an
# error_break() pause that happens to have "stop" typed into it) without
# the two copies of this string ever risking drifting apart. Mirrors the
# C++ port's own debug_hooks.hpp::kDebuggingStoppedMessage.
DEBUGGING_STOPPED_MESSAGE = "Debugging stopped."


def _is_flat_numeric(v):
    if not v:
        return False
    for x in v:
        t = type(x)
        if t is not int and t is not float:
            return False
    return True


# numpy array creation has ~3-5µs fixed overhead; list comprehensions
# cost ~30ns/element.  Crossover is around 100-200 elements.
_NP_VEC_THRESHOLD = 128


def _scale(scalar, value):
    if type(value) is list:
        if _is_flat_numeric(value):
            if len(value) >= _NP_VEC_THRESHOLD:
                return (scalar * np.asarray(value)).tolist()
            return [scalar * x for x in value]
        return [_scale(scalar, v) for v in value]
    if type(scalar) is bool or type(value) is bool:
        return None
    try:
        return scalar * value
    except TypeError:
        return None


def _div_scale(value, divisor):
    if type(value) is list:
        if _is_flat_numeric(value):
            if len(value) >= _NP_VEC_THRESHOLD:
                arr = np.asarray(value, dtype=np.float64)
                if divisor == 0:
                    return np.where(arr == 0, np.nan, np.copysign(np.inf, arr)).tolist()
                return (arr / divisor).tolist()
            if divisor == 0:
                return [float('nan') if x == 0 else math.copysign(float('inf'), x) for x in value]
            return [x / divisor for x in value]
        return [_div_scale(v, divisor) for v in value]
    if type(value) is bool:
        return None
    try:
        if divisor == 0:
            return float('nan') if value == 0 else math.copysign(float('inf'), value)
        return value / divisor
    except TypeError:
        return None


def _vec_add(a, b):
    if type(a) is list and type(b) is list:
        if _is_flat_numeric(a) and _is_flat_numeric(b):
            if len(a) >= _NP_VEC_THRESHOLD:
                n = min(len(a), len(b))
                return (np.asarray(a[:n]) + np.asarray(b[:n])).tolist()
            return [x + y for x, y in zip(a, b)]
        return [_vec_add(x, y) for x, y in zip(a, b)]
    if type(a) is bool or type(b) is bool:
        return None
    if type(a) is str or type(b) is str:
        return None
    try:
        return a + b
    except TypeError:
        return None


def _point_seg_dist(p, a, b):
    """Euclidean distance from 2D point `p` to segment `a`-`b`."""
    ab = b - a
    denom = np.dot(ab, ab)
    t = np.dot(p - a, ab) / denom if denom else 0.0
    t = max(0.0, min(1.0, t))
    return float(np.linalg.norm(p - (a + t * ab)))


def _point_in_poly_evenodd(p, edges):
    """Even-odd ray-casting point-in-polygon test against a flat list of (a, b) edges."""
    x, y = p
    inside = False
    for a, b in edges:
        x1, y1 = a
        x2, y2 = b
        if (y1 > y) != (y2 > y):
            xint = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xint:
                inside = not inside
    return inside


# ---------------------------------------------------------------------------
# Straight-skeleton roof() helpers
# ---------------------------------------------------------------------------

_ROOF_MITER_LIMIT = 1e5


def _ccw_polygon(poly: np.ndarray) -> np.ndarray:
    """Return `poly` (Nx2) reordered to counter-clockwise winding."""
    n = len(poly)
    area2 = sum(poly[k][0] * poly[(k + 1) % n][1] - poly[(k + 1) % n][0] * poly[k][1] for k in range(n))
    return poly[::-1].copy() if area2 < 0 else poly


def _ear_clip(poly: np.ndarray) -> list[tuple[int, int, int]]:
    """Ear-clipping triangulation of a simple CCW polygon (may be concave).

    Returns CCW index triples into `poly`. Raises RuntimeError if no ear can
    be found (degenerate/self-intersecting input).
    """
    n = len(poly)
    idx = list(range(n))

    def is_convex(a, b, c):
        ax, ay = poly[a]
        bx, by = poly[b]
        cx, cy = poly[c]
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax) > 0

    def point_in_tri(p, a, b, c):
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
        d1, d2, d3 = sign(p, a, b), sign(p, b, c), sign(p, c, a)
        return not ((d1 < 0 or d2 < 0 or d3 < 0) and (d1 > 0 or d2 > 0 or d3 > 0))

    tris = []
    while len(idx) > 3:
        n = len(idx)
        for i in range(n):
            a, b, c = idx[(i - 1) % n], idx[i], idx[(i + 1) % n]
            if not is_convex(a, b, c):
                continue
            if any(point_in_tri(poly[j], poly[a], poly[b], poly[c]) for j in idx if j not in (a, b, c)):
                continue
            tris.append((a, b, c))
            idx.pop(i)
            break
        else:
            raise RuntimeError("ear clipping failed")
    tris.append((idx[0], idx[1], idx[2]))
    return tris


def _miter_vertex_velocities(poly: np.ndarray) -> np.ndarray:
    """Per-vertex velocity under `offset(-d, Miter)`: moving `poly[k]` by
    `d * v_k` reproduces the mitered inward offset by `d`.
    """
    n = len(poly)
    vel = np.zeros((n, 2))
    for k in range(n):
        prev_dir = poly[k] - poly[(k - 1) % n]
        next_dir = poly[(k + 1) % n] - poly[k]
        prev_dir = prev_dir / np.linalg.norm(prev_dir)
        next_dir = next_dir / np.linalg.norm(next_dir)
        n1 = np.array([-prev_dir[1], prev_dir[0]])
        n2 = np.array([-next_dir[1], next_dir[0]])
        denom = 1 + np.dot(n1, n2)
        vel[k] = (n1 + n2) / denom
    return vel


def _offset_collapse_distance(cs: m3d.CrossSection, d_hi: float, tol: float) -> float:
    """Binary search for the largest `d` in `[0, d_hi]` where the mitered
    inward offset of `cs` by `d` still has positive area."""
    lo, hi = 0.0, d_hi
    for _ in range(40):
        mid = (lo + hi) / 2
        area = cs.offset(-mid, m3d.JoinType.Miter, _ROOF_MITER_LIMIT).area()
        if area > tol:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _offset_is_stable(cs: m3d.CrossSection, d_max: float, n: int) -> bool:
    """True if the mitered offset of `cs` stays a single `n`-vertex polygon
    for a range of distances up to `d_max` (i.e. no intermediate
    collapse/split events)."""
    for f in (0.25, 0.5, 0.75, 0.9):
        polys = cs.offset(-d_max * f, m3d.JoinType.Miter, _ROOF_MITER_LIMIT).to_polygons()
        if len(polys) != 1 or len(polys[0]) != n:
            return False
    return True


def _skeleton_roof(cs: m3d.CrossSection) -> Optional[m3d.Manifold]:
    """Build an exact straight-skeleton roof for a simple polygon whose
    mitered offset collapses to a point/ridge with no intermediate topology
    events. Returns None if `cs` doesn't qualify (multi-contour, degenerate,
    or an unstable/multi-event collapse) or mesh construction fails.
    """
    try:
        polys = cs.to_polygons()
        if len(polys) != 1:
            return None
        p0 = _ccw_polygon(np.asarray(polys[0], dtype=np.float64))
        n = len(p0)
        if n < 3:
            return None

        minx, miny, maxx, maxy = cs.bounds()
        d_hi = max(maxx - minx, maxy - miny)
        if d_hi <= 0:
            return None
        tol = (d_hi ** 2) * 1e-12
        d_max = _offset_collapse_distance(cs, d_hi, tol)
        if d_max <= 0:
            return None
        if not _offset_is_stable(cs, d_max, n):
            return None

        vel = _miter_vertex_velocities(p0)
        p1 = p0 + d_max * vel

        raw_verts = [(p[0], p[1], 0.0) for p in p0] + [(p[0], p[1], d_max) for p in p1]
        merge_tol = 1e-4
        final_verts: list[tuple[float, float, float]] = []
        idx_map: dict[int, int] = {}
        for i, v in enumerate(raw_verts):
            matched = None
            for ridx, rv in enumerate(final_verts):
                if abs(rv[0] - v[0]) < merge_tol and abs(rv[1] - v[1]) < merge_tol and abs(rv[2] - v[2]) < merge_tol:
                    matched = ridx
                    break
            if matched is None:
                matched = len(final_verts)
                final_verts.append(v)
            idx_map[i] = matched

        tris = []
        for (i, j, k) in _ear_clip(p0):
            tris.append((idx_map[k], idx_map[j], idx_map[i]))
        for k in range(n):
            k1 = (k + 1) % n
            a, b, c, d = idx_map[k], idx_map[k1], idx_map[n + k1], idx_map[n + k]
            if c == d:
                tris.append((a, b, c))
            else:
                tris.append((a, b, c))
                tris.append((a, c, d))

        mesh = m3d.Mesh64(
            vert_properties=np.array(final_verts, dtype=np.float64),
            tri_verts=np.array(tris, dtype=np.uint64),
        )
        body = m3d.Manifold(mesh)
        if body.status() != m3d.Error.NoError or body.is_empty():
            return None
        return body
    except Exception:
        return None


def _build_skeleton_graph_with_holes(
    p0: np.ndarray,
    hole_arrays: list[np.ndarray],
) -> Optional[tuple]:
    """Build the planar straight-skeleton graph for CCW outer polygon `p0`
    with zero or more CW hole polygons.

    polyskel winding convention (y-axis down): outer must be CW-in-math
    (so pass `p0[::-1]`); holes must be CCW-in-math (so pass each
    `hole[::-1]` since holes from manifold are CW-in-math).

    Returns `(heights, adjacency, p0_keys, hole_keys_list, key_fn, degenerate_holes)`
    or None.
      heights          : position-key → offset-distance (0 on boundary)
      adjacency        : position-key → [neighbour keys]  (undirected)
      p0_keys          : keys for p0 vertices in traversal order
      hole_keys_list   : list of key lists, one list per hole in order
      key_fn           : snap function `(x, y) → position-key`
      degenerate_holes : list of bool, parallel to hole_keys_list — True for
                         holes whose skeleton was computed in isolation (no
                         connection from the main polyskel run), meaning their
                         boundary must be traced in reversed order (see
                         `_skeleton_roof_component`)
    """
    try:
        all_pts = np.vstack([p0] + hole_arrays) if hole_arrays else p0
        d_hi = max(all_pts[:, 0].max() - all_pts[:, 0].min(),
                   all_pts[:, 1].max() - all_pts[:, 1].min())
        if d_hi <= 0:
            return None
        tol = d_hi * 1e-6

        heights: dict[tuple, float] = {}

        def key(x, y):
            for k in heights:
                if abs(k[0] - x) < tol and abs(k[1] - y) < tol:
                    return k
            return (float(x), float(y))

        adjacency: dict[tuple, list] = {}

        def add_edge(a, b):
            if a != b:
                adjacency.setdefault(a, [])
                adjacency.setdefault(b, [])
                if b not in adjacency[a]:
                    adjacency[a].append(b)
                if a not in adjacency[b]:
                    adjacency[b].append(a)

        # Outer polygon boundary (CCW in math)
        n0 = len(p0)
        p0_keys = []
        for x, y in p0:
            k = key(x, y)
            heights[k] = 0.0
            adjacency.setdefault(k, [])
            p0_keys.append(k)
        for i in range(n0):
            add_edge(p0_keys[i], p0_keys[(i + 1) % n0])

        # Hole boundaries (CW in math)
        hole_keys_list: list[list] = []
        for hole in hole_arrays:
            nh = len(hole)
            hkeys = []
            for x, y in hole:
                k = key(x, y)
                heights[k] = 0.0
                adjacency.setdefault(k, [])
                hkeys.append(k)
            for i in range(nh):
                add_edge(hkeys[i], hkeys[(i + 1) % nh])
            hole_keys_list.append(hkeys)

        # polyskel: outer as CW-in-math, holes as CCW-in-math
        outer_pts = [(float(x), float(y)) for x, y in p0[::-1]]
        holes_pts = [[(float(x), float(y)) for x, y in h[::-1]] for h in hole_arrays]
        # polyskel can hang (infinite loop) on degenerate polygon configurations
        # (e.g. exact axis-aligned vertices that trigger numerical edge cases in
        # the skeleton sweep algorithm).  Run it in a daemon thread and abort if
        # it doesn't finish in time; then retry with a tiny deterministic jitter
        # to break the degeneracy, which typically lets polyskel converge.
        import threading as _threading
        import random as _random

        def _run_skeletonize(outer, holes, timeout=2.0):
            _res: list = [None]
            def _run():
                _res[0] = skeletonize(outer, holes if holes else None)
            _t = _threading.Thread(target=_run, daemon=True)
            _t.start()
            _t.join(timeout=timeout)
            return None if _t.is_alive() else _res[0]

        subtrees = _run_skeletonize(outer_pts, holes_pts)
        if subtrees is None:
            # Retry with a tiny deterministic jitter to break numerical degeneracy.
            _rng = _random.Random(0xBEEF)
            _j = tol * 0.1  # < tol so key() snaps back; still breaks numeric degeneracy
            outer_jit = [(x + _rng.uniform(-_j, _j), y + _rng.uniform(-_j, _j))
                         for x, y in outer_pts]
            holes_jit = [[(x + _rng.uniform(-_j, _j), y + _rng.uniform(-_j, _j))
                          for x, y in h] for h in holes_pts]
            subtrees = _run_skeletonize(outer_jit, holes_jit)
        if not subtrees:
            return None

        for st in subtrees:
            s = key(st.source.x, st.source.y)
            heights[s] = st.height
            adjacency.setdefault(s, [])
            # Group sinks by angle from source.  When polyskel places multiple
            # sinks on the same ray (collinear), adding all as direct edges
            # creates same-angle neighbour pairs that confuse _trace_face's
            # angle-sort.  Instead chain them: add source→closest only, then
            # closest→next, … so the path is a sequence of short hops.
            by_angle: dict[float, list] = {}
            for sink in st.sinks:
                t = key(sink.x, sink.y)
                if t == s:
                    continue  # skip self-loop sinks
                heights.setdefault(t, 0.0)
                adjacency.setdefault(t, [])
                dx, dy = t[0] - s[0], t[1] - s[1]
                ang = round(math.atan2(dy, dx), 9)
                dist2 = dx * dx + dy * dy
                by_angle.setdefault(ang, []).append((dist2, t))
            for ang, group in by_angle.items():
                group.sort()  # ascending distance
                # Connect source → closest (one short hop).
                prev = s
                for _, t in group:
                    add_edge(prev, t)
                    prev = t  # chain: each step only goes one hop further

        # Post-process: resolve same-angle neighbour pairs that arise when two
        # different subtrees share the same source vertex but each contributes a
        # sink along the same ray.  The per-subtree chain above only fixes
        # within-subtree duplicates; this pass fixes the cross-subtree case.
        # For each vertex V with two neighbours A (closer) and B (farther) at
        # the same angle, replace the V→B shortcut with a chain hop A→B and
        # remove V→B.  Repeat until the graph is stable.
        changed = True
        while changed:
            changed = False
            for v in list(adjacency):
                by_ang: dict[float, list] = {}
                for w in list(adjacency[v]):
                    dx, dy = w[0] - v[0], w[1] - v[1]
                    ang = round(math.atan2(dy, dx), 9)
                    dist2 = dx * dx + dy * dy
                    by_ang.setdefault(ang, []).append((dist2, w))
                for ang, group in by_ang.items():
                    if len(group) < 2:
                        continue
                    group.sort()
                    changed = True
                    prev = v
                    for i, (_, w) in enumerate(group):
                        if i == 0:
                            prev = w
                            continue
                        if w in adjacency.get(v, []):
                            adjacency[v].remove(w)
                        if v in adjacency.get(w, []):
                            adjacency[w].remove(v)
                        add_edge(prev, w)
                        prev = w

        # Post-process: for each hole whose boundary vertices have no interior
        # skeleton connections, compute the hole's isolated straight skeleton
        # and inject it.  polyskel sometimes fails to generate wavefront events
        # for a hole when the hole is a simple degenerate shape (e.g. a triangular
        # counter whose three bisectors converge simultaneously), producing no
        # skeleton vertices attached to the hole.  Without an interior apex, all
        # three hole-edge face traces cycle around the same flat triangle, making
        # each hole edge appear four times in the final mesh → NotManifold.
        # Running polyskel on the hole in isolation always works (e.g. gives the
        # incenter for a triangle) and produces the correct roof faces.
        #
        # The isolated skeleton is computed by treating the hole polygon as its
        # own mini *outer* polygon (CW-in-math, matching polyskel's convention),
        # so its interior apex sits on the opposite side from a normal hole's
        # skeleton (which faces the surrounding solid material). Consequently
        # `_trace_face` must walk these particular holes' boundary edges in
        # *reversed* (CCW) order — like an outer boundary trace — instead of the
        # natural CW order used for holes with a genuine connection to the main
        # skeleton. `degenerate_holes` records which holes need this treatment.
        degenerate_holes: list[bool] = [False] * len(hole_keys_list)
        for hidx, (hole_arr, hkeys) in enumerate(zip(hole_arrays, hole_keys_list)):
            hole_key_set = set(hkeys)
            has_interior = any(
                heights.get(w, 0.0) > 0.0
                for v in hkeys
                for w in adjacency.get(v, [])
                if w not in hole_key_set
            )
            if has_interior:
                continue
            degenerate_holes[hidx] = True
            # Hole vertices are CW-in-math = what polyskel expects for an outer polygon.
            hole_pts_iso = [(float(x), float(y)) for x, y in hole_arr]
            iso = _run_skeletonize(hole_pts_iso, [])
            if not iso:
                continue
            for st in iso:
                s = key(st.source.x, st.source.y)
                heights[s] = st.height
                adjacency.setdefault(s, [])
                by_angle_iso: dict[float, list] = {}
                for sink in st.sinks:
                    t = key(sink.x, sink.y)
                    if t == s:
                        continue
                    heights.setdefault(t, 0.0)
                    adjacency.setdefault(t, [])
                    dx, dy = t[0] - s[0], t[1] - s[1]
                    ang = round(math.atan2(dy, dx), 9)
                    dist2 = dx * dx + dy * dy
                    by_angle_iso.setdefault(ang, []).append((dist2, t))
                for ang, group in by_angle_iso.items():
                    group.sort()
                    prev = s
                    for _, t in group:
                        add_edge(prev, t)
                        prev = t

        return heights, adjacency, p0_keys, hole_keys_list, key, degenerate_holes
    except Exception:
        return None


def _trace_face(adjacency: dict, u: tuple, v: tuple) -> Optional[list]:
    """Trace the bounded face to the left of directed edge `(u, v)` in
    `adjacency` (a CCW polygon's boundary edge `u -> v` keeps the polygon's
    interior, and thus this roof face, on its left). At each vertex, the next
    edge is the neighbor immediately before the incoming vertex in
    angle-sorted (CCW) order, i.e. the next edge clockwise.

    Returns the ordered list of face-vertex positions, or `None` if the trace
    doesn't close within a bounded number of steps.
    """
    start = (u, v)
    face = [u]
    cur_u, cur_v = u, v
    for _ in range(2 * len(adjacency) + 4):
        face.append(cur_v)
        neighbors = adjacency.get(cur_v)
        if not neighbors or len(neighbors) < 2:
            return None
        ordered = sorted(neighbors, key=lambda w: math.atan2(w[1] - cur_v[1], w[0] - cur_v[0]))
        try:
            idx = ordered.index(cur_u)
        except ValueError:
            return None
        nxt = ordered[(idx - 1) % len(ordered)]
        cur_u, cur_v = cur_v, nxt
        if (cur_u, cur_v) == start:
            return face[:-1]
    return None


_SVG_UNIT = re.compile(r"^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*(em|ex|px|in|cm|mm|pt|pc|%)?\s*$")


def _svg_page_map(contours: list, root, dpi: float, center: bool) -> list:
    """Place SVG user-unit contours as OpenSCAD's import_svg.cc does: the
    page's width/height to millimetres (a unitless length at `dpi`, 72 by
    default; px at 96), the viewBox scaled onto it under
    preserveAspectRatio (default xMidYMid meet), and Y flipped about the
    page height -- or, with center=true, about the drawing's own centre.
    Without this a 100-unit drawing came in 2.8x too large and upside down
    below the X axis."""
    def length(attr, viewbox, valid):
        m = _SVG_UNIT.match(root.get(attr) or "")
        if m is None:  # absent: rely on the dpi, as older Illustrator files do
            return 25.4 * viewbox / dpi if valid else 0.0
        n, unit = float(m.group(1)), m.group(2)
        return {None: 25.4 * n / dpi, "px": 25.4 * n / 96, "pt": 25.4 * n / 72, "pc": 25.4 * n / 6,
                "in": 25.4 * n, "cm": 10 * n, "mm": n,
                "%": 25.4 * n / 100 * viewbox / dpi if valid else 0.0}.get(unit, viewbox if valid else 0.0)

    vb = [float(x) for x in re.split(r"[\s,]+", (root.get("viewBox") or "").strip()) if x]
    valid = len(vb) == 4 and vb[2] >= 0 and vb[3] >= 0
    width_mm = length("width", vb[2] if valid else 0.0, valid)
    height_mm = length("height", vb[3] if valid else 0.0, valid)
    sx = sy = 1.0
    vbx = vby = ax = ay = 0.0
    if valid:
        wm, hm = _SVG_UNIT.match(root.get("width") or ""), _SVG_UNIT.match(root.get("height") or "")
        vbx = vb[0] * (float(wm.group(1)) / 100 if wm and wm.group(2) == "%" else 1.0)
        vby = vb[1] * (float(hm.group(1)) / 100 if hm and hm.group(2) == "%" else 1.0)
        sx, sy = (width_mm / vb[2] if vb[2] else 0.0), (height_mm / vb[3] if vb[3] else 0.0)
        par = (root.get("preserveAspectRatio") or "").split()
        par = par[1:] if par[:1] == ["defer"] else par
        align = par[0] if par else "xMidYMid"
        if align != "none":
            sx = sy = min(sx, sy) if (par[1:2] or ["meet"])[0] != "slice" else max(sx, sy)
            where = {"Min": 0.0, "Mid": 0.5, "Max": 1.0}
            if len(align) == 8 and align[1:4] in where and align[5:8] in where:
                ax = where[align[1:4]] * (width_mm - sx * vb[2])
                ay = where[align[5:8]] * (height_mm - sy * vb[3])
            else:  # malformed: OpenSCAD keeps its xMidYMid default
                ax, ay = 0.5 * (width_mm - sx * vb[2]), 0.5 * (height_mm - sy * vb[3])
    if center:
        pts = [(sx * x, sy * y) for c in contours for x, y in c]
        cx = (min(p[0] for p in pts) + max(p[0] for p in pts)) / 2 if pts else 0.0
        cy = (min(p[1] for p in pts) + max(p[1] for p in pts)) / 2 if pts else 0.0
    else:
        cx, cy = -ax, height_mm - ay
    # -vby - y, not y - vby: OpenSCAD's own formula, kept for parity.
    return [[(sx * (x - vbx) - cx, sy * (-vby - y) + cy) for x, y in c] for c in contours]


_SPHERE_STYLES = ("orig", "aligned", "stagger", "octa", "icosa")


def _spherical_to_xyz(r: float, theta: float, phi: float) -> list[float]:
    """BOSL2's spherical_to_xyz: theta around Z from +X, phi down from +Z, degrees."""
    th, ph = math.radians(theta), math.radians(phi)
    return [r * math.sin(ph) * math.cos(th), r * math.sin(ph) * math.sin(th), r * math.cos(ph)]


def _sphere_aligned(r: float, hsides: int, vsides: int, stagger: bool):
    """sphere(style="aligned"/"stagger"): a vertex at each pole and rings on
    the latitudes between (alternate rings turned half a face for stagger).
    Vertex order and faces follow BOSL2's spheroid() triangle for triangle,
    with each face's winding reversed -- BOSL2's VNF winding is Manifold's
    inside out, which only a signed volume shows."""
    verts = [_spherical_to_xyz(r, 0, 0)]
    for i in range(1, vsides):
        for j in range(hsides):
            verts.append(_spherical_to_xyz(r, (j + (0.5 if stagger and i % 2 else 0.0)) * 360.0 / hsides,
                                           i * 180.0 / vsides))
    verts.append(_spherical_to_xyz(r, 0, 180))
    lv = len(verts)
    tris = []

    def tri(a, b, c):
        tris.append((a, c, b))

    for i in range(hsides):
        b2 = lv - 2 - hsides
        tri(i + 1, 0, (i + 1) % hsides + 1)
        tri(lv - 1, b2 + i + 1, b2 + (i + 1) % hsides + 1)
    for i in range(vsides - 2):
        base = 1 + hsides * i
        for j in range(hsides):
            if stagger and i % 2:
                tri(base + j, base + hsides + j % hsides, base + hsides + (j + hsides - 1) % hsides)
                tri(base + j, base + (j + 1) % hsides, base + hsides + j)
            else:
                tri(base + j, base + (j + 1) % hsides, base + hsides + (j + 1) % hsides)
                tri(base + j, base + hsides + (j + 1) % hsides, base + hsides + j)
    return verts, tris


def _sphere_icosa(r: float, hsides: int):
    """sphere(style="icosa"): each icosahedral face subdivided into a grid
    and pushed out to the sphere, welded along shared edges. BOSL2 rotates
    copies of one sampled face; sampling each face against its own corners
    is the same points, since the sampling is affine in them."""
    phi = (1 + math.sqrt(5)) / 2
    ico = []
    for i in (-1, 1):
        for j in (-1, 1):
            ico += [(0.0, float(i), j * phi), (float(i), j * phi, 0.0), (j * phi, 0.0, float(i))]
    ico = np.array(ico)
    faces = []
    for a in range(12):  # hull faces by brute force: 220 triples
        for b in range(a + 1, 12):
            for c in range(b + 1, 12):
                d = (ico - ico[a]) @ np.cross(ico[b] - ico[a], ico[c] - ico[a])
                d[[a, b, c]] = 0
                pos, neg = (d > 1e-9).any(), (d < -1e-9).any()
                if pos and neg:
                    continue
                faces.append((a, c, b) if not neg else (a, b, c))  # outward
    steps = max(1, round(max(5, hsides) / 5))
    n = steps - 1
    verts, weld, tris = [], {}, []

    def add(p):
        u = r * p / np.linalg.norm(p)
        key = tuple(int(round(x * 1e9)) for x in u)
        if key not in weld:
            weld[key] = len(verts)
            verts.append(u.tolist())
        return weld[key]

    for f in faces:
        p0, p1, p2 = ico[list(f)]
        grid = [[add(p0 + (p1 - p0) * (i / (n + 1)) + (p2 - p0) * (j / (n + 1))) for j in range(n + 2 - i)]
                for i in range(n + 2)]
        for i in range(n + 1):
            for j in range(n + 1 - i):
                tris.append((grid[i][j], grid[i + 1][j], grid[i][j + 1]))
                if j < n - i:
                    tris.append((grid[i + 1][j], grid[i + 1][j + 1], grid[i][j + 1]))
    return verts, tris


def _edges_closed(tris: np.ndarray) -> bool:
    """Whether every edge has exactly two faces -- no boundary, no fin.
    ponytail: skips the C++ checkMesh's pinched-vertex test; a raw mesh
    only pinches if its author reused one index across two shells."""
    if len(tris) == 0:
        return False
    edges = np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return bool((counts == 2).all())


def _triangulate_face(verts: list, loop: list[int], out: list) -> None:
    """Triangulate one polyhedron face into `out`, reversing winding
    (OpenSCAD's faces are clockwise from outside, Manifold's counter-).

    A fan is right only for a convex, planar face, and BOSL2's
    vnf_polyhedron() hands over neither: a nurbs_sheet() end cap is a
    concave 34-gon 3.5 units out of plane, and fanning it inflated the
    solid by 5% of its volume (#88). Ear clipping in the face's Newell
    best-fit plane instead, taking the fattest ear each time so a
    non-planar face folds along its surface. Port of the C++
    triangulateFace. Plain floats, not numpy: faces are a handful of
    points, where numpy's per-call overhead doubled a 40k-quad VNF's time.
    ponytail: O(n^2) per face."""
    nv = len(verts)
    loop = [i if 0 <= i < nv else 0 for i in loop]
    n = len(loop)

    def emit(a, b, c):
        if a != b and b != c and a != c:
            out.append((a, c, b))

    def fan(ids):
        for i in range(1, len(ids) - 1):
            emit(ids[0], ids[i], ids[i + 1])

    if n < 3:
        return
    if n == 3:
        return emit(*loop)
    pts = [verts[i] for i in loop]
    nx = ny = nz = 0.0
    for i in range(n):
        (x0, y0, z0), (x1, y1, z1) = pts[i], pts[(i + 1) % n]
        nx += (y0 - y1) * (z0 + z1)
        ny += (z0 - z1) * (x0 + x1)
        nz += (x0 - x1) * (y0 + y1)
    ln = math.sqrt(nx * nx + ny * ny + nz * nz)
    if not ln > 1e-12:
        return fan(loop)
    nx, ny, nz = nx / ln, ny / ln, nz / ln
    ax, ay, az = abs(nx), abs(ny), abs(nz)
    drop = (0 if ax > az else 2) if ax > ay else (1 if ay > az else 2)
    axis = [0.0, 0.0, 0.0]
    axis[(drop + 1) % 3] = 1.0
    ux, uy, uz = axis[1] * nz - axis[2] * ny, axis[2] * nx - axis[0] * nz, axis[0] * ny - axis[1] * nx
    ul = math.sqrt(ux * ux + uy * uy + uz * uz)
    if not ul > 1e-12:
        return fan(loop)
    ux, uy, uz = ux / ul, uy / ul, uz / ul
    wx, wy, wz = ny * uz - nz * uy, nz * ux - nx * uz, nx * uy - ny * ux
    flat = [(x * ux + y * uy + z * uz, x * wx + y * wy + z * wz) for x, y, z in pts]

    def cross2(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def inside(a, b, c, p):  # strictly, so a point on an edge doesn't veto an ear
        return cross2(a, b, p) > 1e-12 and cross2(b, c, p) > 1e-12 and cross2(c, a, p) > 1e-12

    def squareness(i, j, k):  # twice the area over the squared sides: 0 for a sliver
        p, q, r = pts[i], pts[j], pts[k]
        e1 = (q[0] - p[0], q[1] - p[1], q[2] - p[2])
        e2 = (r[0] - p[0], r[1] - p[1], r[2] - p[2])
        e3 = (r[0] - q[0], r[1] - q[1], r[2] - q[2])
        cx, cy, cz = e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]
        sides = sum(c * c for c in e1 + e2 + e3)
        return math.sqrt(cx * cx + cy * cy + cz * cz) / sides if sides > 1e-18 else 0.0

    twice_area = sum(flat[i][0] * flat[(i + 1) % n][1] - flat[(i + 1) % n][0] * flat[i][1]
                     for i in range(n))
    idx = list(range(n))
    if twice_area < 0:
        idx.reverse()  # work counter-clockwise
    guard = 0
    while len(idx) > 3 and guard < n * n:
        guard += 1
        best_at, best = None, -1.0
        m = len(idx)
        # Only a reflex vertex can lie inside an ear of a simple polygon, so
        # only those are tested: a 200-gon cap is otherwise O(n^3). ponytail:
        # a self-intersecting face may clip differently from the C++ port.
        reflex = [idx[i] for i in range(m)
                  if cross2(flat[idx[i - 1]], flat[idx[i]], flat[idx[(i + 1) % m]]) <= 1e-12]
        for i in range(m):
            pi, ci, ni = idx[i - 1], idx[i], idx[(i + 1) % m]
            a, b, c = flat[pi], flat[ci], flat[ni]
            if cross2(a, b, c) <= 1e-12:
                continue  # reflex, not an ear
            if any(o != pi and o != ni and inside(a, b, c, flat[o]) for o in reflex):
                continue
            score = squareness(pi, ci, ni)
            if score > best:
                best, best_at = score, i
        if best_at is None:
            break  # self-intersecting or otherwise unclippable
        emit(loop[idx[best_at - 1]], loop[idx[best_at]], loop[idx[(best_at + 1) % m]])
        del idx[best_at]
    fan([loop[i] for i in idx])


def _triangulate_planar_face(face_pts3d: np.ndarray) -> Optional[list[tuple[int, int, int]]]:
    """Triangulate a planar roof face given as 3D points (CCW order, all
    coplanar). The normal is estimated via Newell's method (a sum over all
    vertex pairs, not just the first 3): for an exactly planar CCW polygon
    this is identical in direction to `cross(p1-p0, p2-p0)`, but it stays
    numerically stable when the first few vertices happen to be near-collinear
    (common along a straight or gently-curved boundary run), which the
    3-point cross product cannot handle. The projection basis (`u` along the
    first edge projected onto the fitted plane, `v = normal x u`) makes
    `_ear_clip`'s output map directly to outward-facing 3D triangles, with no
    winding reversal.

    Returns `None` if the face is degenerate (fewer than 3 points, near-zero
    normal or first edge), not planar within tolerance, or ear-clipping fails.
    """
    n = len(face_pts3d)
    if n < 3:
        return None
    nx = ny = nz = 0.0
    for i in range(n):
        x0, y0, z0 = face_pts3d[i]
        x1, y1, z1 = face_pts3d[(i + 1) % n]
        nx += (y0 - y1) * (z0 + z1)
        ny += (z0 - z1) * (x0 + x1)
        nz += (x0 - x1) * (y0 + y1)
    normal = np.array([nx, ny, nz])
    norm_len = np.linalg.norm(normal)
    if norm_len < 1e-12:
        return None
    normal = normal / norm_len

    p0 = face_pts3d[0]
    edge = face_pts3d[1] - p0
    edge = edge - np.dot(edge, normal) * normal
    edge_len = np.linalg.norm(edge)
    if edge_len < 1e-12:
        return None
    u_axis = edge / edge_len
    v_axis = np.cross(normal, u_axis)

    span = max(float(np.linalg.norm(face_pts3d.max(axis=0) - face_pts3d.min(axis=0))), 1e-9)
    # Looser than the old 1e-4: real (but still flat, per straight-skeleton
    # theory) facets can accumulate a bit more numerical noise from polyskel
    # over many vertices than a simple 3-point check tolerated.
    tol = span * 2e-3
    pts2d = np.zeros((n, 2))
    for i, p in enumerate(face_pts3d):
        rel = p - p0
        if abs(np.dot(rel, normal)) > tol:
            return None
        pts2d[i] = (np.dot(rel, u_axis), np.dot(rel, v_axis))

    try:
        return _ear_clip(pts2d)
    except RuntimeError:
        return None


def _split_pinched_face(face: list, classify, add_face_triangles) -> bool:
    """Decompose a `_trace_face` result that spans a straight-skeleton "pinch"
    — a thin-stroke split/collision event where a ridge directly touches a
    *different* stretch of boundary — into its correct flat sub-facets, and
    add each via `add_face_triangles`.

    `face[0]` is the vertex the trace started from, so the maximal run of
    consecutive (non-SKEL) boundary points containing it is the face's "home"
    run — the boundary edge(s) actually being traced. Any *other* maximal
    boundary run found elsewhere in the cyclic face is foreign, even if it
    shares the home run's ring (`classify(pt) -> (kind, ring_id)`): two
    non-adjacent stretches of the very same outer or hole ring can end up in
    one merged trace just as easily as outer-vs-hole can (e.g. where a bowl's
    curve pinches against an unrelated stem). Each foreign run, plus the one
    skeleton point flanking it on each side, forms its own small flat facet
    (`add_face_triangles([before] + run + [after])`); the remainder of `face`
    — with each such run replaced by a direct edge between its two flanking
    points — forms the main flat facet. When only the home run is found this
    reduces to triangulating `face` unchanged (the common, non-pinched case).
    Returns False if any resulting sub-facet fails to triangulate (e.g. still
    non-planar, or degenerates below 3 points). Used as a middle fallback tier
    in `_build_roof_mesh`/`_skeleton_roof_component`, between ownership-
    constrained tracing and plain unconstrained tracing.
    """
    n = len(face)
    cats = [classify(p) for p in face]

    runs: list[tuple[int, int]] = []
    idx = 0
    while idx < n:
        cat = cats[idx]
        if cat[0] != "SKEL":
            start = idx
            j = idx
            while j + 1 < n and cats[j + 1] == cat:
                j += 1
            runs.append((start, j))
            idx = j + 1
        else:
            idx += 1

    home_run = next(r for r in runs if r[0] <= 0 <= r[1])
    foreign_runs = [r for r in runs if r != home_run]

    if not foreign_runs:
        return add_face_triangles(face)

    skip = set()
    for (s, e) in foreign_runs:
        skip.update(range(s, e + 1))
    big_face = [face[k] for k in range(n) if k not in skip]

    ok = len(big_face) >= 3 and add_face_triangles(big_face)
    for (s, e) in foreign_runs:
        before = face[(s - 1) % n]
        after = face[(e + 1) % n]
        small_face = [before] + face[s:e + 1] + [after]
        ok = add_face_triangles(small_face) and ok
    return ok


class _UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def _edge_line(a: tuple, b: tuple) -> Optional[tuple]:
    """Return `(point, unit_direction, unit_interior_normal)` for boundary
    edge `a -> b`, where the normal points to the LEFT of `a -> b` — the
    interior side for a CCW outer edge, and (per the polygon-winding
    convention used throughout this module) also the interior/solid side for
    a CW hole edge. `None` if the edge is degenerate (zero length).
    """
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy)
    if length < 1e-12:
        return None
    ux, uy = dx / length, dy / length
    return (ax, ay), (ux, uy), (-uy, ux)


def _assign_edge_ownership(
    adjacency: dict,
    heights: dict,
    p0_keys: list,
    hole_keys_list: list,
) -> tuple[dict, dict]:
    """Determine, for every graph vertex, which original boundary edge(s) its
    roof facet belongs to — the information the plain undirected-adjacency
    graph doesn't otherwise retain, and which a naive angle-sort walk
    (`_trace_face`) can silently cross between when several boundary edges'
    facets get merged by a "pass-through" vertex (no ridge to separate them).

    Two edges are first grouped together (via union-find) whenever a shared
    boundary vertex between them has no interior connection at all (a genuine
    "no ridge here, these edges' facets are contiguous" case — most commonly a
    flat or gently-curved boundary run). Then, per straight-skeleton theory, a
    point's height equals its perpendicular distance to the line of whichever
    edge(s) it belongs to — a "ridge" vertex is assigned to every edge group
    whose line it sits at height-equal distance from (usually one, but
    genuinely two or more at real ridge/valley junctions where facets meet).

    Returns `(owners, group_of)`:
      owners   : vertex-key -> set of edge-group ids that own it
      group_of : edge_id ('OUTER', i) or ('HOLE', j, i) -> its group id
    """
    all_edge_ids: list = []
    edge_lines: dict = {}
    n0 = len(p0_keys)
    for i in range(n0):
        eid = ("OUTER", i)
        all_edge_ids.append(eid)
        line = _edge_line(p0_keys[i], p0_keys[(i + 1) % n0])
        if line is not None:
            edge_lines[eid] = line
    for j, hkeys in enumerate(hole_keys_list):
        nh = len(hkeys)
        for i in range(nh):
            eid = ("HOLE", j, i)
            all_edge_ids.append(eid)
            line = _edge_line(hkeys[i], hkeys[(i + 1) % nh])
            if line is not None:
                edge_lines[eid] = line

    uf = _UnionFind(all_edge_ids)

    def union_ring(ring_keys, kind, ring_idx=None):
        n = len(ring_keys)
        for i in range(n):
            v = ring_keys[i]
            nbrs = adjacency.get(v, [])
            prev_pt, next_pt = ring_keys[(i - 1) % n], ring_keys[(i + 1) % n]
            if len(nbrs) != 2 or not all(w in (prev_pt, next_pt) for w in nbrs):
                continue
            # No ridge separates this vertex's two edges — but that only means
            # they share one flat facet if they're actually collinear. A
            # coarsely-segmented curve (e.g. low $fn) can have a real bend at
            # a vertex with no ridge event; merging across it would fold two
            # differently-angled edges' facets into one non-planar "facet".
            d1x, d1y = v[0] - prev_pt[0], v[1] - prev_pt[1]
            d2x, d2y = next_pt[0] - v[0], next_pt[1] - v[1]
            len1, len2 = math.hypot(d1x, d1y), math.hypot(d2x, d2y)
            if len1 < 1e-12 or len2 < 1e-12:
                continue
            cross = (d1x * d2y - d1y * d2x) / (len1 * len2)
            if abs(cross) > 0.02:  # ~1.1 degrees
                continue
            prev_id = ("OUTER", (i - 1) % n) if kind == "OUTER" else ("HOLE", ring_idx, (i - 1) % n)
            cur_id = ("OUTER", i) if kind == "OUTER" else ("HOLE", ring_idx, i)
            uf.union(prev_id, cur_id)

    union_ring(p0_keys, "OUTER")
    for j, hkeys in enumerate(hole_keys_list):
        union_ring(hkeys, "HOLE", j)

    group_of = {eid: uf.find(eid) for eid in all_edge_ids}

    p0_index = {k: i for i, k in enumerate(p0_keys)}
    hole_index = [{k: i for i, k in enumerate(hkeys)} for hkeys in hole_keys_list]

    owners: dict[tuple, set] = {}
    for v, h in heights.items():
        owned: set = set()
        if h <= 1e-9:
            if v in p0_index:
                i = p0_index[v]
                owned.add(group_of[("OUTER", (i - 1) % n0)])
                owned.add(group_of[("OUTER", i)])
            else:
                for j, hi in enumerate(hole_index):
                    if v in hi:
                        i = hi[v]
                        nh = len(hole_keys_list[j])
                        owned.add(group_of[("HOLE", j, (i - 1) % nh)])
                        owned.add(group_of[("HOLE", j, i)])
                        break
        else:
            best_diff = float("inf")
            best_groups: set = set()
            for eid, (pt, _ud, un) in edge_lines.items():
                dist = (v[0] - pt[0]) * un[0] + (v[1] - pt[1]) * un[1]
                if dist < -1e-6:
                    continue  # vertex is on the wrong side of this edge's line
                diff = abs(dist - h)
                gid = group_of[eid]
                if diff <= max(1e-6, h * 1e-4):
                    owned.add(gid)
                if diff < best_diff:
                    best_diff = diff
                    best_groups = {gid}
            if not owned:
                owned = best_groups
        owners[v] = owned
    return owners, group_of


def _trace_owned_face(adjacency: dict, owners: dict, group_id, u: tuple, v: tuple) -> Optional[list]:
    """Like `_trace_face`, but only follows the "immediately clockwise"
    neighbour while it's owned by `group_id` (see `_assign_edge_ownership`).
    As soon as the next vertex belongs to a different facet, that's this
    facet's natural boundary — return what's traced so far (an implicitly
    closed polygon: the caller connects its last point straight back to `u`),
    rather than blindly continuing into someone else's geometry the way a
    plain angle-sort walk would.
    """
    start = (u, v)
    face = [u]
    cur_u, cur_v = u, v
    for _ in range(2 * len(adjacency) + 4):
        face.append(cur_v)
        neighbors = adjacency.get(cur_v)
        if not neighbors or len(neighbors) < 2:
            return face if len(face) >= 3 else None
        ordered = sorted(neighbors, key=lambda w: math.atan2(w[1] - cur_v[1], w[0] - cur_v[0]))
        try:
            idx = ordered.index(cur_u)
        except ValueError:
            return face if len(face) >= 3 else None
        nxt = ordered[(idx - 1) % len(ordered)]
        if nxt != u and group_id not in owners.get(nxt, ()):
            return face if len(face) >= 3 else None
        cur_u, cur_v = cur_v, nxt
        if (cur_u, cur_v) == start:
            return face[:-1]
    return None


def _build_roof_mesh(
    p0: np.ndarray,
    hole_arrs: list[np.ndarray],
    heights: dict,
    adjacency: dict,
    p0_keys: list,
    hole_keys_list: list,
    degenerate_holes: list,
    key,
    strategy: str,
) -> Optional[m3d.Manifold]:
    """Build one candidate roof+floor mesh for a component, using one of three
    face-tracing strategies. Returns a closed Manifold or None on any failure.

    No single strategy handles every thin-stroke pinch/collision pattern a
    real font can produce, so `_skeleton_roof_component` tries all three (in
    order of how much cross-facet protection they offer) and keeps whichever
    actually produces a valid manifold:

    - `"owned"`: `_assign_edge_ownership` + `_trace_owned_face`. Computes,
      from straight-skeleton geometry alone (a vertex's height must equal its
      distance to the edge(s) it belongs to), which facet every vertex
      actually belongs to, and refuses to trace across into a different
      facet's territory. Handles most pinches (outer-vs-hole, two
      non-adjacent runs of the same ring) correctly and precisely — but when
      a ridge point is legitimately, exactly equidistant from *several*
      boundary edges at once (a real hip/valley junction, common along
      *repeatedly*-pinched thin strokes), each edge's trace can independently
      give up right there and close via a "virtual chord" back to its own
      start, with no guarantee some other trace produces the exact opposite
      chord to pair with it — leaving the mesh open.
    - `"split"`: plain `_trace_face` + `_split_pinched_face`. Traces without
      any ownership constraint (so it can merge multiple facets into one
      loop), then decomposes that loop after the fact by finding the "home"
      run of boundary points containing the trace's start and treating any
      other boundary run as a pinch to excise into its own small facet. Less
      precise than `"owned"` (a coarser, single-pass heuristic) but doesn't
      have the open-mesh failure mode, so it succeeds on some geometries
      `"owned"` doesn't.
    - `"plain"`: plain `_trace_face`, no cross-facet handling at all. Works
      whenever a component simply has no such pinch to worry about.
    """
    try:
        from shapely.geometry import Polygon as _SPoly
        from shapely import constrained_delaunay_triangles as _cdt

        n0 = len(p0_keys)
        heights = dict(heights)
        adjacency = {k: list(v) for k, v in adjacency.items()}

        final_verts: list[tuple[float, float, float]] = []
        idx_map: dict[tuple, int] = {}

        def vert_index(pos):
            if pos not in idx_map:
                idx_map[pos] = len(final_verts)
                final_verts.append((pos[0], pos[1], heights.get(pos, 0.0)))
            return idx_map[pos]

        tris = []

        # --- Floor tessellation (at z=0, normal pointing downward) ---
        if not hole_arrs:
            # No holes: ear-clip gives consistent CCW triangles → reverse for -z normal.
            for (i, j, k2) in _ear_clip(p0):
                tris.append((vert_index(p0_keys[k2]), vert_index(p0_keys[j]), vert_index(p0_keys[i])))
        else:
            # Holes: constrained Delaunay (respects polygon boundary edges) +
            # centroid filter. Winding check ensures a downward (-z) normal.
            outer_2d = [(float(p[0]), float(p[1])) for p in p0]
            holes_2d = [[(float(p[0]), float(p[1])) for p in h] for h in hole_arrs]
            shape = _SPoly(outer_2d, holes_2d)
            for tri in _cdt(shape).geoms:
                if not shape.contains(tri.centroid):
                    continue
                coords = list(tri.exterior.coords)[:3]
                ax, ay = coords[0]
                bx, by = coords[1]
                cx, cy = coords[2]
                # Signed area: positive = CCW from above (upward normal), so reverse for -z.
                signed_area2 = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
                if signed_area2 > 0:
                    coords = list(reversed(coords))
                fi = []
                for (fx, fy) in coords:
                    k = key(fx, fy)
                    heights.setdefault(k, 0.0)
                    adjacency.setdefault(k, [])
                    fi.append(vert_index(k))
                tris.append(tuple(fi))

        # --- Roof faces for outer boundary edges (CCW: interior on left) ---
        # Holes are CW-in-math; the LEFT of each natural CW directed edge is
        # the exterior (roofable) region, so trace in the natural direction.
        # Degenerate holes (isolated skeleton, no connection to the main
        # polyskel run) are the exception: their interior apex sits on the
        # opposite side of the natural direction (the isolated skeleton was
        # computed by treating the hole as its own tiny outer polygon), so
        # `_trace_face` must be called on the reversed edge to find the correct
        # face at all — but the found face's point order then comes out with
        # the base edge in the same direction as the floor triangulation's
        # (which always walks holes in their natural order), so it must be
        # reversed again before triangulating to get the opposing winding a
        # manifold mesh requires.
        if strategy == "owned":
            owners, group_of = _assign_edge_ownership(adjacency, heights, p0_keys, hole_keys_list)
        elif strategy == "split":
            p0_key_set = set(p0_keys)
            hole_key_sets = [set(hkeys) for hkeys in hole_keys_list]

            def classify(pt):
                if pt in p0_key_set:
                    return ("OUTER", -1)
                for j, hs in enumerate(hole_key_sets):
                    if pt in hs:
                        return ("HOLE", j)
                return ("SKEL", -1)

        consumed: set[tuple] = set()

        def add_face_triangles(face_pts_keys):
            face_pts3d = np.array([(p[0], p[1], heights.get(p, 0.0)) for p in face_pts_keys])
            face_tris = _triangulate_planar_face(face_pts3d)
            if face_tris is None:
                return False
            face_idx = [vert_index(p) for p in face_pts_keys]
            for (a, b, c) in face_tris:
                tris.append((face_idx[a], face_idx[b], face_idx[c]))
            nf = len(face_pts_keys)
            for k in range(nf):
                consumed.add((face_pts_keys[k], face_pts_keys[(k + 1) % nf]))
            return True

        def process_boundary_edge(edge_id, u, v):
            if (u, v) in consumed:
                return True
            if strategy == "owned":
                face = _trace_owned_face(adjacency, owners, group_of[edge_id], u, v)
                if face is None or len(face) < 3:
                    return False
                return add_face_triangles(face)
            face = _trace_face(adjacency, u, v)
            if face is None or len(face) < 3:
                return False
            if strategy == "split":
                return _split_pinched_face(face, classify, add_face_triangles)
            return add_face_triangles(face)

        for i in range(n0):
            if not process_boundary_edge(("OUTER", i), p0_keys[i], p0_keys[(i + 1) % n0]):
                return None

        for j, (hkeys, is_degenerate) in enumerate(zip(hole_keys_list, degenerate_holes)):
            nh = len(hkeys)
            if len(set(hkeys)) != nh:
                return None
            for i in range(nh):
                u, v = hkeys[i], hkeys[(i + 1) % nh]
                if is_degenerate:
                    if (u, v) in consumed:
                        continue
                    face = _trace_face(adjacency, v, u)
                    if face is not None:
                        face = list(reversed(face))
                    if face is None or len(face) < 3:
                        return None
                    if not add_face_triangles(face):
                        return None
                else:
                    if not process_boundary_edge(("HOLE", j, i), u, v):
                        return None

        if not tris or not final_verts:
            return None
        mesh = m3d.Mesh64(
            vert_properties=np.array(final_verts, dtype=np.float64),
            tri_verts=np.array(tris, dtype=np.uint64),
        )
        body = m3d.Manifold(mesh)
        if body.status() != m3d.Error.NoError or body.is_empty():
            return None
        return body
    except Exception:
        return None


def _skeleton_roof_component(
    outer_arr: np.ndarray,
    hole_arrs: list[np.ndarray],
) -> Optional[m3d.Manifold]:
    """Build a straight-skeleton roof for one connected component: a CCW outer
    polygon and zero or more CW hole polygons. Returns a closed Manifold or None.

    Tries `_build_roof_mesh`'s three tracing strategies in order — `"owned"`,
    `"split"`, `"plain"` (see its docstring) — since each fails on a different,
    non-overlapping class of thin-stroke geometry; trying all three covers far
    more real fonts/glyphs than any single one alone.
    """
    p0 = _ccw_polygon(outer_arr)
    n0 = len(p0)
    if n0 < 3:
        return None

    graph = _build_skeleton_graph_with_holes(p0, hole_arrs)
    if graph is None:
        return None
    heights, adjacency, p0_keys, hole_keys_list, key, degenerate_holes = graph
    if len(set(p0_keys)) != n0:
        return None

    for strategy in ("owned", "split", "plain"):
        body = _build_roof_mesh(
            p0, hole_arrs, heights, adjacency, p0_keys, hole_keys_list,
            degenerate_holes, key, strategy,
        )
        if body is not None:
            return body
    return None


def _skeleton_roof_general(cs: m3d.CrossSection) -> Optional[m3d.Manifold]:
    """Build an exact straight-skeleton roof for `cs`, handling any combination
    of outer contours and holes. Separates polygons into connected components
    (each outer + its direct holes), builds a skeleton roof per component via
    `_skeleton_roof_component`, and returns their union. Returns None on failure.
    """
    try:
        from shapely.geometry import Polygon as _SPoly, Point as _SPoint

        polys = cs.to_polygons()
        if not polys:
            return None

        # Separate outer (CCW, area2 > 0) from hole (CW, area2 < 0) polygons
        outer_arrs: list[np.ndarray] = []
        hole_arrs: list[np.ndarray] = []
        for poly in polys:
            arr = np.asarray(poly, dtype=np.float64)
            n = len(arr)
            area2 = float(np.sum(
                arr[:, 0] * np.roll(arr[:, 1], -1)
                - np.roll(arr[:, 0], -1) * arr[:, 1]
            ))
            if area2 > 0:
                outer_arrs.append(arr)
            elif area2 < 0:
                hole_arrs.append(arr)

        if not outer_arrs:
            return None

        # Group each hole with the smallest containing outer polygon
        outer_shapes = [_SPoly([(p[0], p[1]) for p in arr]) for arr in outer_arrs]
        components: list[tuple[np.ndarray, list[np.ndarray]]] = [
            (arr, []) for arr in outer_arrs
        ]
        for hole_arr in hole_arrs:
            cx = float(np.mean(hole_arr[:, 0]))
            cy = float(np.mean(hole_arr[:, 1]))
            pt = _SPoint(cx, cy)
            best_i, best_area = None, float('inf')
            for i, shape in enumerate(outer_shapes):
                if shape.contains(pt) and shape.area < best_area:
                    best_i, best_area = i, shape.area
            if best_i is not None:
                components[best_i][1].append(hole_arr)

        pieces: list[m3d.Manifold] = []
        for (outer_arr, comp_holes) in components:
            b = _skeleton_roof_component(outer_arr, comp_holes)
            if b is None:
                return None  # partial failure — let caller fall back to SDF
            pieces.append(b)

        if not pieces:
            return None
        body = pieces[0]
        for b in pieces[1:]:
            body = body + b
        return body
    except Exception:
        return None


def _vec_sub(a, b):
    if type(a) is list and type(b) is list:
        if _is_flat_numeric(a) and _is_flat_numeric(b):
            if len(a) >= _NP_VEC_THRESHOLD:
                n = min(len(a), len(b))
                return (np.asarray(a[:n]) - np.asarray(b[:n])).tolist()
            return [x - y for x, y in zip(a, b)]
        return [_vec_sub(x, y) for x, y in zip(a, b)]
    if type(a) is bool or type(b) is bool:
        return None
    try:
        return a - b
    except TypeError:
        return None


def _osc_type_name(v) -> str:
    """OpenSCAD's name for `v`'s type, as used in 'undefined operation (...)' warnings."""
    if v is None:
        return "undefined"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, (int, float)):
        return "number"
    if isinstance(v, str):
        return "string"
    if isinstance(v, list):
        return "vector"
    if isinstance(v, OscObject):
        return "object"
    if isinstance(v, OscRange):
        return "range"
    if isinstance(v, (Closure, FunctionLiteral, FunctionDeclaration)):
        return "function"
    return "undefined"


def _object_arg_type_name(v) -> str:
    """Type name as used in `object()`'s own argument-validation warnings
    (`<number>`, `<string>`, `<list>`, ... `<undef>`) — distinct spelling from
    `_osc_type_name()`'s `undefined`/`vector`."""
    if v is None:
        return "undef"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, (int, float)):
        return "number"
    if isinstance(v, str):
        return "string"
    if isinstance(v, list):
        return "list"
    if isinstance(v, OscRange):
        return "range"
    if isinstance(v, OscObject):
        return "object"
    if isinstance(v, (FunctionDeclaration, FunctionLiteral, Closure)):
        return "function"
    return "undef"


def _osc_equal(a, b) -> bool:
    ta, tb = type(a), type(b)
    if (ta is bool) != (tb is bool):
        return False
    if ta is list and tb is list:
        return len(a) == len(b) and all(_osc_equal(x, y) for x, y in zip(a, b))
    if ta is OscObject and tb is OscObject:
        pairs_a, pairs_b = list(a.items()), list(b.items())
        return len(pairs_a) == len(pairs_b) and all(
            ka == kb and _osc_equal(va, vb)
            for (ka, va), (kb, vb) in zip(pairs_a, pairs_b)
        )
    return a == b


def _osc_comparable(a, b) -> bool:
    ta, tb = type(a), type(b)
    if ta is bool or tb is bool:
        return ta is bool and tb is bool
    if (ta is int or ta is float) and (tb is int or tb is float):
        return True
    if ta is str and tb is str:
        return True
    if ta is list and tb is list:
        return True
    return False


def _format_number(v: float) -> str:
    """Format a number the way OpenSCAD's `echo()`/`str()` do.

    Differs from Python's `f"{v:g}"` in two ways:
    - exponents drop their leading zero (`1e+08` -> `1e+8`, `1e-07` -> `1e-7`)
    - small numbers stay in fixed notation one digit further than `%g`
      (`1e-5` -> `0.00001`, where `%g` would give `1e-05`); fixed notation
      covers exponents in `[-5, 5]`, vs. `%g`'s `[-4, 5]`.
    Both still show at most 6 significant digits, and `-0.0` -> `"0"`.
    """
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    if v == 0:
        return "0"

    neg = v < 0
    av = abs(v)
    # The exponent AFTER rounding to 6 significant digits, from Python's own
    # correctly rounded formatting. Dividing by 10**exp and rounding the
    # mantissa got the edges wrong: 99999.95 printed as 100000 (it is
    # 99999.949999... in binary; OpenSCAD prints 99999.9), a subnormal came
    # out a digit off, and 5e-324 divided by zero.
    mantissa, exp = f"{av:.5e}".split("e")
    exp = int(exp)
    if -5 <= exp <= 5:
        s = f"{av:.{5 - exp}f}"
        if "." in s:
            s = s.rstrip("0").rstrip(".")
    else:
        m = mantissa.rstrip("0").rstrip(".")
        s = f"{m}e{'+' if exp >= 0 else '-'}{abs(exp)}"
    return ("-" + s) if neg else s


def _matmul(a, b):
    if not a or not b:
        return None
    a_is_mat = bool(a) and isinstance(a[0], list)
    b_is_mat = bool(b) and isinstance(b[0], list)
    try:
        if not a_is_mat and not b_is_mat:
            n = len(a)
            if n != len(b):
                return None
            if n >= _NP_VEC_THRESHOLD:
                return np.dot(np.asarray(a), np.asarray(b)).tolist()
            s = 0
            for i in range(n):
                s += a[i] * b[i]
            return s
        na = np.asarray(a)
        nb = np.asarray(b)
        if na.dtype == object or nb.dtype == object:
            return None
        return np.dot(na, nb).tolist()
    except (TypeError, ValueError, IndexError):
        return None


def _matmul_error(a, b):
    """Why ``a * b`` on two lists is undef, in the reference's words (a port of
    multiply_visitor's messages). Only asked once _matmul has already failed."""
    def num(v):
        return type(v) in (int, float)

    def vec_mat(vec, mat):
        cols = len(mat[0]) if type(mat[0]) is list else 0
        for i in range(cols):
            for j in range(len(vec)):
                row = mat[j]
                if type(row) is not list or len(row) != cols:
                    return f"Matrix must be rectangular. Problem at row {j}"
                if not num(vec[j]):
                    return f"Vector must contain only numbers. Problem at index {j}"
                if not num(row[i]):
                    return f"Matrix must contain only numbers. Problem at row {j}, col {i}"
        return None

    if not a or not b:
        return "Multiplication is undefined on empty vectors"
    e1, e2 = a[0], b[0]
    if num(e1) and num(e2):
        if len(a) != len(b):
            return f"vector*vector requires matching lengths ({len(a)} != {len(b)})"
        for x, y in zip(a, b):
            if not num(x) or not num(y):
                return f"undefined operation ({_osc_type_name(x)} * {_osc_type_name(y)})"
    elif num(e1) and type(e2) is list:
        if len(a) != len(b):
            return f"vector*matrix requires vector length to match matrix row count ({len(a)} != {len(b)})"
        return vec_mat(a, b)
    elif type(e1) is list and num(e2):
        if len(e1) != len(b):
            return f"matrix*vector requires matrix column count to match vector length ({len(e1)} != {len(b)})"
        for i, row in enumerate(a):
            if type(row) is not list or len(row) != len(b):
                return f"Matrix must be rectangular. Problem at row {i}"
            for j, x in enumerate(row):
                if not num(x):
                    return f"Matrix must contain only numbers. Problem at row {i}, col {j}"
                if not num(b[j]):
                    return f"Vector must contain only numbers. Problem at index {j}"
    elif type(e1) is list and type(e2) is list:
        if len(e1) != len(b):
            return ("matrix*matrix requires left operand column count to match right operand "
                    f"row count ({len(e1)} != {len(b)})")
        for i, row in enumerate(a):
            if type(row) is not list or len(row) != len(b):
                n = len(row) if type(row) is list else 0
                return ("matrix*matrix left operand row length does not match right operand "
                        f"row count ({n} != {len(b)}) at row {i}")
            err = vec_mat(row, b)
            if err:
                return f"{err}: while processing left operand at row {i}"
    else:
        return ("undefined vector*vector multiplication where first elements are types "
                f"{_osc_type_name(e1)} and {_osc_type_name(e2)}")
    return None


class OscRange:
    """Lazy OpenSCAD range value — echoes as [start : step : end], iterable, indexable."""
    __slots__ = ("start", "step", "end")

    def __init__(self, start: float, step: float, end: float):
        self.start = start
        self.step = step
        self.end = end

    def __iter__(self):
        if self.step == 0:
            return
        v = self.start
        if self.step > 0:
            while v <= self.end + 1e-10:
                yield v
                v += self.step
        else:
            while v >= self.end - 1e-10:
                yield v
                v += self.step

    def __getitem__(self, idx: int):
        # OpenSCAD indexes a range as its 3 components, not its iterated values:
        # `[2:3:11][0]` -> 2 (start), `[1]` -> 3 (step), `[2]` -> 11 (end).
        return (self.start, self.step, self.end)[idx] if 0 <= idx <= 2 else None

    def __repr__(self):
        return f"OscRange({self.start}, {self.step}, {self.end})"

    def __eq__(self, other):
        # By value, as OpenSCAD compares ranges: [0:2] == [0:1:2], and two
        # empty ranges are equal. It was identity, so [5:1:0] != [5:1:0].
        if type(other) is not OscRange:
            return NotImplemented
        if (self.start, self.step, self.end) == (other.start, other.step, other.end):
            return True
        return _range_count(self) == 0 and _range_count(other) == 0

    __hash__ = None


class OscObject:
    """OpenSCAD `object()` value — an ordered string-keyed map."""
    __slots__ = ("data",)

    def __init__(self, data: dict):
        self.data = data

    def __iter__(self):
        return iter(self.data)  # keys, in insertion order

    def __len__(self):
        return len(self.data)

    def get(self, key):
        return self.data.get(key)  # missing key -> None (undef)

    def items(self):
        return self.data.items()

    def __repr__(self):
        return f"OscObject({self.data!r})"


class Closure:
    """A function literal's value: the `FunctionLiteral` plus the local
    (`let`) bindings in force where it was evaluated. Without them a
    function that outlives its defining call -- `function mk(x) =
    function(y) x + y; mk(10)(5)` -- read `x` as undef, since the call's
    frame is gone by the time the literal runs."""
    __slots__ = ("fn", "let")

    def __init__(self, fn: FunctionLiteral, let: dict):
        self.fn = fn
        self.let = let

    def __str__(self):
        return _fmt_fn(self.fn)


_BINARY_OPS = {
    "AdditionOp": "+", "SubtractionOp": "-", "MultiplicationOp": "*", "DivisionOp": "/",
    "ModuloOp": "%", "ExponentOp": "^", "BitwiseAndOp": "&", "BitwiseOrOp": "|",
    "BitwiseShiftLeftOp": "<<", "BitwiseShiftRightOp": ">>", "LogicalAndOp": "&&",
    "LogicalOrOp": "||", "EqualityOp": "==", "InequalityOp": "!=", "GreaterThanOp": ">",
    "GreaterThanOrEqualOp": ">=", "LessThanOp": "<", "LessThanOrEqualOp": "<=",
}
_UNARY_OPS = {"UnaryMinusOp": "-", "LogicalNotOp": "!", "BitwiseNotOp": "~"}


def _fmt_fn(n) -> str:
    """A function value as OpenSCAD's str()/echo() print it: every binary and
    ternary expression parenthesised, `=` spaced -- `function(x, y = 2)
    ((x * y) + 1)`. The parser's own printing (`function(x, y=2) x * y + 1`)
    is for reformatting source, not this. Port of openscad_cpp_evaluator's
    format_closure.cpp."""
    t = type(n).__name__
    j = lambda items: ", ".join(_fmt_fn(i) for i in items)  # noqa: E731
    w = lambda x: f"({_fmt_fn(x)})"  # noqa: E731 -- a list-comprehension body is wrapped
    if t in _BINARY_OPS:
        return f"({_fmt_fn(n.left)} {_BINARY_OPS[t]} {_fmt_fn(n.right)})"
    if t in _UNARY_OPS:
        return _UNARY_OPS[t] + _fmt_fn(n.expr)
    if t == "Identifier":
        return n.name
    if t == "NumberLiteral":
        return _format_number(n.val)
    if t in ("StringLiteral", "BooleanLiteral", "UndefinedLiteral"):
        return str(n)
    if t == "RangeLiteral":
        step = "" if getattr(n, "implicit_step", False) else f"{_fmt_fn(n.step)} : "
        return f"[{_fmt_fn(n.start)} : {step}{_fmt_fn(n.end)}]"
    if t == "TernaryOp":
        return f"({_fmt_fn(n.condition)} ? {_fmt_fn(n.true_expr)} : {_fmt_fn(n.false_expr)})"
    if t == "PrimaryCall":  # a callee that isn't a plain name is parenthesised: (o.f)(1)
        callee = _fmt_fn(n.left)
        return f"{callee if type(n.left).__name__ == 'Identifier' else f'({callee})'}({j(n.arguments)})"
    if t == "PrimaryIndex":
        return f"{_fmt_fn(n.left)}[{_fmt_fn(n.index)}]"
    if t == "PrimaryMember":
        return f"{_fmt_fn(n.left)}.{n.member}"
    if t == "PositionalArgument":
        return _fmt_fn(n.expr)
    if t in ("NamedArgument", "Assignment"):
        return f"{n.name.name} = {_fmt_fn(n.expr)}"
    if t == "ParameterDeclaration":
        has_default = n.default is not None and type(n.default).__name__ != "UndefinedLiteral"
        return f"{n.name.name} = {_fmt_fn(n.default)}" if has_default else n.name.name
    if t == "FunctionLiteral":
        return f"function({j(n.parameters)}) {_fmt_fn(n.body)}"
    if t in ("LetOp", "ListCompLet"):
        return f"let({j(n.assignments)}) {_fmt_fn(n.body)}"
    if t == "EchoOp":
        return f"echo({j(n.arguments)}) {_fmt_fn(n.body)}"
    if t == "AssertOp":
        return f"assert({j(n.arguments)}) {_fmt_fn(n.body)}"
    if t == "ListComprehension":
        return f"[{j(n.elements)}]"
    if t == "ListCompFor":
        return f"for({j(n.assignments)}) {w(n.body)}"
    if t == "ListCompCFor":  # no space after the semicolons, body unwrapped
        return f"for({j(n.inits)};{_fmt_fn(n.condition)};{j(n.incrs)}) {_fmt_fn(n.body)}"
    if t == "ListCompIf":
        return f"if({_fmt_fn(n.condition)}) {w(n.true_expr)}"
    if t == "ListCompIfElse":
        return f"if({_fmt_fn(n.condition)}) {w(n.true_expr)} else {w(n.false_expr)}"
    if t == "ListCompEach":
        return f"each {w(n.body)}"
    if t == "CommentedExpr":
        return _fmt_fn(n.expr)
    return str(n)


_FONT_DIR = Path(__file__).parent / "resources" / "fonts"
_FONT_PATH = _FONT_DIR / "LiberationSans-Regular.ttf"
# The bundled family's faces by lower-cased style, so "Liberation Sans:style=Bold"
# needs no fc-match (absent on macOS, and elsewhere it may know another font).
_BUNDLED_STYLES = {"regular": "Regular", "bold": "Bold", "italic": "Italic", "oblique": "Italic",
                   "bold italic": "BoldItalic", "bold oblique": "BoldItalic"}
_font_table_cache: dict[tuple[str, int], dict] = {}  # (path, ttc_index) → font tables


def _font_tables_from_path(path: str, ttc_index: int = 0) -> dict:
    """Load font tables from a file path, using a module-level cache."""
    key = (path, ttc_index)
    if key not in _font_table_cache:
        font = TTFont(path, fontNumber=ttc_index)
        # Some fonts store glyph outlines in CFF rather than glyf.
        glyf_table = font.get("glyf")
        hmtx_table = font.get("hmtx")
        name_table = font.get("name")
        family_name = name_table.getBestFamilyName() if name_table else None
        style_name = name_table.getBestSubFamilyName() if name_table else None
        _font_table_cache[key] = {
            "cmap": font.getBestCmap() or {},
            "hmtx": hmtx_table,
            "glyf": glyf_table,
            "units_per_em": font["head"].unitsPerEm,
            "head": font["head"],
            "hhea": font.get("hhea"),
            "glyph_set": font.getGlyphSet(),
            "glyph_order": font.getGlyphOrder(),  # HarfBuzz glyph index -> name
            "hb_font": hb.Font(hb.Face(hb.Blob.from_file_path(path), ttc_index)),
            "path": path,
            "ttc_index": ttc_index,
            "family_name": family_name or "Liberation Sans",
            "style_name": style_name or "Regular",
        }
    return _font_table_cache[key]


def _load_default_font() -> dict:
    """Load the bundled Liberation Sans font."""
    return _font_tables_from_path(str(_FONT_PATH), 0)


_font_spec_cache: dict[str, dict] = {}  # font-spec string → font tables


def _resolve_font(font_spec: str) -> dict:
    """Resolve an OpenSCAD/fontconfig font spec to font tables.

    Uses `fc-match` to find the best-matching system font for `font_spec`
    (e.g. `"Times New Roman:style=Bold"`).  Falls back to the bundled
    Liberation Sans if `fc-match` is not available or the spec is empty.
    """
    if not font_spec:
        return _load_default_font()
    if font_spec in _font_spec_cache:
        return _font_spec_cache[font_spec]
    family, _, rest = font_spec.partition(":")
    style = next((kv.partition("=")[2] for kv in rest.split(":") if kv.strip().lower().startswith("style=")), "")
    # A style with no family (":style=Bold") names the default family, as
    # fontconfig reads it and BOSL2's text3d() example expects.
    if family.strip() in ("", "Liberation Sans"):
        face = _BUNDLED_STYLES.get(style.strip().lower() or "regular")
        if face is not None:
            tables = _font_tables_from_path(str(_FONT_DIR / f"LiberationSans-{face}.ttf"), 0)
            _font_spec_cache[font_spec] = tables
            return tables
    try:
        import subprocess as _sp
        result = _sp.run(
            ["fc-match", "--format=%{file}:%{index}\n", font_spec],
            capture_output=True, text=True, timeout=3,
        )
        line = result.stdout.strip()
        if line and ":" in line:
            parts = line.rsplit(":", 1)
            fpath, idx_str = parts[0], parts[1]
            ttc_index = int(idx_str) if idx_str.isdigit() else 0
            if Path(fpath).exists():
                tables = _font_tables_from_path(fpath, ttc_index)
                _font_spec_cache[font_spec] = tables
                return tables
    except Exception:
        pass
    tables = _load_default_font()
    _font_spec_cache[font_spec] = tables
    return tables


def list_fonts() -> list[dict]:
    """Every font text()/textmetrics()/fontmetrics() can resolve, as dicts
    of family, style, spec and path, sorted and deduplicated by family and
    style (cpp #163). `spec` is the exact `font=` string -- the name a font
    dialog shows is often not the one OpenSCAD takes. The bundled Liberation
    Sans faces have path "<bundled>"; the rest come from `fc-list`, which is
    what _resolve_font() matches against, so none without it. Scans the
    system fonts: call it when someone asks for the list, not per render."""
    found = {("Liberation Sans", style): "<bundled>" for style in ("Regular", "Bold", "Italic", "Bold Italic")}
    try:
        import subprocess as _sp
        out = _sp.run(["fc-list", "--format", "%{family[0]}\t%{style[0]}\t%{file}\n"],
                      capture_output=True, text=True, timeout=30).stdout
    except Exception:
        out = ""
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) == 3 and parts[0] and not parts[0].startswith("."):  # dot-names are system-private
            found.setdefault((parts[0], parts[1] or "Regular"), parts[2])
    return [{"family": f, "style": st, "spec": f"{f}:style={st}", "path": path}
            for (f, st), path in sorted(found.items())]


def _glyph_bounds(gname: str, font: dict) -> tuple[float, float, float, float] | None:
    """Return (xMin, yMin, xMax, yMax) in font units for glyph `gname`, or None
    if the glyph is empty/whitespace.  Works for both TrueType (glyf table) and
    CFF (glyph_set pen draw) fonts."""
    glyf = font.get("glyf")
    if glyf is not None:
        g = glyf[gname]
        if g.numberOfContours == 0:
            return None
        return g.xMin, g.yMin, g.xMax, g.yMax
    # CFF/OTF: derive bounds by drawing the glyph contours
    glyph_set = font["glyph_set"]
    if gname not in glyph_set:
        return None
    xs: list[float] = []
    ys: list[float] = []

    class _BoundsPen(BasePen):
        def _moveTo(self, pt):
            xs.append(pt[0]); ys.append(pt[1])
        def _lineTo(self, pt):
            xs.append(pt[0]); ys.append(pt[1])
        def _curveToOne(self, bcp1, bcp2, pt):
            for p in (bcp1, bcp2, pt):
                xs.append(p[0]); ys.append(p[1])
        def _closePath(self):
            pass
        def _endPath(self):
            pass

    glyph_set[gname].draw(_BoundsPen(glyph_set))
    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _measure_text(text: str, size: float, spacing: float, font: dict | None = None,
                  direction: str = "", language: str = "", script: str = "") -> dict:
    """Shape `text` with HarfBuzz and return its ink-bbox/advance metrics
    in OpenSCAD units, scaled for `size` (see docs/evaluator.md for the
    scale factor). Kerning, ligatures, mark positioning and bidi reordering
    all apply; `direction`/`language`/`script` left empty are guessed from
    the text, as OpenSCAD's detect_properties() does (cpp #96).

    Returns a dict with `ascent`, `descent`, `ink_min_x`, `ink_max_x`,
    `advance_x`, `advance_y`, and `glyphs` (a list of `(glyph_name, x, y)`
    for each renderable glyph, used by `text()`) — aggregates are all `0`
    and `glyphs` is empty if `text` contains no measurable glyphs.
    """
    if font is None:
        font = _load_default_font()
    # The 100/72 is OpenSCAD's own text() size bug (its #4304), kept on purpose.
    scale = size * (100 / 72) / font["units_per_em"]

    buf = hb.Buffer()
    buf.add_str(text)
    if direction:
        buf.direction = direction
    if script:
        buf.script = script
    if language:
        buf.language = language
    buf.guess_segment_properties()  # fills in only what was not set above
    hb.shape(font["hb_font"], buf)

    pen_x = pen_y = 0.0
    ascent = descent = ink_min_x = ink_max_x = 0.0
    has_ink = False
    glyphs = []
    order = font["glyph_order"]
    for info, pos in zip(buf.glyph_infos or [], buf.glyph_positions or []):  # None when empty
        gname = order[info.codepoint]
        x = (pen_x + pos.x_offset) * scale
        y = (pen_y + pos.y_offset) * scale
        bounds = _glyph_bounds(gname, font)
        if bounds is not None:
            xmin, ymin, xmax, ymax = bounds
            left, right = x + xmin * scale, x + xmax * scale
            bottom, top = y + ymin * scale, y + ymax * scale
            if not has_ink:
                ink_min_x, ink_max_x, ascent, descent = left, right, top, bottom
                has_ink = True
            else:
                ink_min_x = min(ink_min_x, left)
                ink_max_x = max(ink_max_x, right)
                ascent = max(ascent, top)
                descent = min(descent, bottom)
            glyphs.append((gname, x, y))
        pen_x += pos.x_advance * spacing
        pen_y += pos.y_advance * spacing

    return {
        "ascent": ascent,
        "descent": descent,
        "ink_min_x": ink_min_x,
        "ink_max_x": ink_max_x,
        "advance_x": pen_x * scale,
        "advance_y": pen_y * scale,
        "glyphs": glyphs,
    }


def _text_align_offset(halign: str, valign: str, m: dict) -> tuple[float, float]:
    """Compute the `(offset_x, offset_y)` translation for `halign`/`valign`,
    given the dict returned by `_measure_text`. Shared by `_builtin_textmetrics`
    (which reports it) and `_builtin_text` (which applies it)."""
    advance_x, ascent, descent = m["advance_x"], m["ascent"], m["descent"]
    offset_x = -{"left": 0.0, "center": 0.5, "right": 1.0}.get(halign, 0.0) * advance_x
    offset_y = {
        "top": -ascent,
        "center": -(ascent + descent) / 2,
        "baseline": 0.0,
        "bottom": -descent,
    }.get(valign, 0.0)
    return offset_x, offset_y


class _FlattenPen(BasePen):
    """A `BasePen` that flattens glyph outlines — quadratic Bezier curves
    (TrueType `glyf` glyphs) and cubic Bezier curves (CFF/OTF glyphs) alike —
    into polygon contours, for building a `m3d.CrossSection`."""

    def __init__(self, glyphSet, segs: int):
        super().__init__(glyphSet)
        self.segs = segs
        self.contours: list[list[tuple[float, float]]] = []
        self._contour: list[tuple[float, float]] = []

    def _moveTo(self, pt):
        self._contour = [pt]

    def _lineTo(self, pt):
        self._contour.append(pt)

    def _qCurveToOne(self, pt1, pt2):
        p0 = self._contour[-1]
        for i in range(1, self.segs + 1):
            t = i / self.segs
            x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * pt1[0] + t ** 2 * pt2[0]
            y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * pt1[1] + t ** 2 * pt2[1]
            self._contour.append((x, y))

    def _curveToOne(self, pt1, pt2, pt3):
        p0 = self._contour[-1]
        for i in range(1, self.segs + 1):
            t = i / self.segs
            mt = 1 - t
            a, b, c, d = mt ** 3, 3 * mt ** 2 * t, 3 * mt * t ** 2, t ** 3
            x = a * p0[0] + b * pt1[0] + c * pt2[0] + d * pt3[0]
            y = a * p0[1] + b * pt1[1] + c * pt2[1] + d * pt3[1]
            self._contour.append((x, y))

    def _closePath(self):
        if self._contour:
            self.contours.append(self._contour)
        self._contour = []

    def _endPath(self):
        self._closePath()


# Cached as raw contour point-lists rather than `m3d.CrossSection` objects:
# nanobind-bound objects held in a module-level cache for the life of the
# process get reported as "leaked" at interpreter shutdown (finalization
# order races the manifold3d module's own teardown).
# Key: (font_path, ttc_index, glyph_name, segs)
_glyph_contour_cache: dict[tuple, list[np.ndarray]] = {}


def _glyph_cross_section(gname: str, segs: int, font: dict | None = None) -> m3d.CrossSection:
    """Return the (unscaled, font-units) `m3d.CrossSection` for glyph `gname`,
    flattening curves into `segs` segments. Contours cached per font+glyph+segs."""
    if font is None:
        font = _load_default_font()
    key = (font.get("path", ""), font.get("ttc_index", 0), gname, segs)
    contours = _glyph_contour_cache.get(key)
    if contours is None:
        glyph_set = font["glyph_set"]
        pen = _FlattenPen(glyph_set, segs)
        glyph_set[gname].draw(pen)
        contours = [np.array(c, dtype=np.float64) for c in pen.contours]
        _glyph_contour_cache[key] = contours
    if contours:
        return m3d.CrossSection(contours, m3d.FillRule.NonZero)
    return m3d.CrossSection()


@dataclass
class ColoredBody:
    """A Manifold body (3D) or CrossSection (2D) paired with an optional RGBA color."""
    body: Optional[m3d.Manifold] = None
    color: Optional[tuple[float, float, float, float]] = None  # RGBA 0-1
    section: Optional[m3d.CrossSection] = None  # set for 2D primitives
    flat_preview: bool = False  # thin extrusion standing in for a 2D shape (see to_renderable_bodies)
    # Where a 2D `section` sits along Z, applied when it is extruded for
    # display. A CrossSection has no Z, so `down(1) square(10)` lost the -1
    # and z-fought whatever it was meant to sit under (BOSL2 contour()
    # layers overlays that way). Only translation is carried; a 3D rotate
    # of a 2D shape still loses it.
    section_z: float = 0.0
    # A 2D section's originalID -- a CrossSection has no run IDs, so without
    # one a top-level 2D shape could not be picked back to source. Reserved
    # when the section is first generated, attributed in id_to_node to the
    # node that built it, and given to the preview slab (cpp #192).
    section_id: Optional[int] = None
    # The operands a union() merged into this body, unmerged, as they were
    # when merged (transforms and color() map over them since). Set only
    # under keep_minuend_color, where a difference() cuts each part on its
    # own so its cut faces take that part's colour; any other op building a
    # new body from this one drops it.
    merged_from: Optional[tuple] = None
    role: str = "normal"  # "normal" | "highlight" (#, real geom) | "highlight_ghost" (#, inside CSG) | "background" (%) | "show_only" (!)
    # Per-triangle RGBA override (shape (T, 4), aligned with body.to_mesh()'s
    # own tri_verts order), set only when a real boolean CSG merge (see
    # _generate_csg) combined children whose colors resolve to more than one
    # distinct value -- e.g. union()-ing an opaque cube with a translucent
    # sphere. None (the common case: a single-colored body) means `color`
    # alone is authoritative, same as before this field existed.
    tri_colors: Optional[np.ndarray] = None
    # An open mesh (polyhedron()/import() whose faces don't close a solid),
    # which Manifold can't represent: (verts float64 (N, 3), tris int (M, 3),
    # Manifold's CCW winding). Set only with body and section both None, so
    # everything that combines solids skips it; it is drawn and exported as
    # a surface on its own, transforms move it, and hull() uses its points.
    raw_mesh: Optional[tuple[np.ndarray, np.ndarray]] = None


@dataclass
class CSGNode:
    """One node in the persistent, coarse-grained CSG tree. Complements —
    does not replace — id_to_node (Manifold originalID -> AST node), which
    stays the fine-grained per-triangle provenance table used for WYSIWYG
    ray-cast picking.

    Built in two passes: the AST walk (Evaluator._eval_statement) resolves
    every node — plain data only, no Manifold calls — and populates `params`
    and `children`, leaving `bodies` empty; Evaluator.generate_tree() then
    walks the completed tree bottom-up and populates `bodies` by calling
    each node's generate_fn. Can be re-run on any (possibly partial) tree at
    any time — e.g. to render a live partial result at a debugger breakpoint
    (Phase 3) — since resolve never depends on any node's generated bodies.

    kind is a human-readable label (a ModularCall's call name, e.g. "cube",
    "union"; or "highlight"/"background"/"show_only"/"intersection_for" for
    the four non-ModularCall wrapper kinds). User-module calls never get a
    CSGNode of their own (Evaluator._eval_statement splices their resolved
    body directly into the enclosing node's children instead), so kind is
    always a builtin's own name or an unrecognized module name -- never a
    user module's, shadowed or not.
    """
    kind: str
    node: ASTNode
    bodies: list[ColoredBody]
    is_builtin: bool = True
    children: list["CSGNode"] = field(default_factory=list)
    params: dict = field(default_factory=dict)
    # True if this node's own resolve step (or any descendant's) called
    # rands() without accounting for global RNG state -- see
    # Evaluator._rands_call_count. Tainted nodes are never cache-hit by
    # ManifoldCache, since their resolved params aren't a pure function of
    # their own content (the actual rands() output also depends on every
    # earlier rands() call's position in the script's evaluation order).
    uncacheable: bool = False
    # The top-level call site that reached this node, for a generate-time
    # warning's ", from ..." clause: generation runs after the call stack has
    # unwound. Not part of the cache key, so two call sites share an entry.
    warn_entry: Optional[Any] = None
    # The call chain that reached this node, innermost frame first, as
    # (call position, is_module) pairs -- see Evaluator.id_to_call_chain.
    # Also out of the cache key.
    call_chain: tuple = ()


@dataclass
class CallSiteProfile:
    """Aggregated profiling data for one *call site* -- a specific source
    location that calls a specific user module/function -- not one
    declaration. Two different calls to the same function get separate
    entries; the same call expression re-executed many times (a loop body,
    recursion) aggregates into one entry with call_count > 1, since the
    AST node (and thus its Position) is identical across those
    invocations. See Evaluator's profile=True instrumentation and
    docs/evaluator.md's "Profiling" section for the self/cumulative-time
    accounting rules."""
    kind: str            # "module" | "child" (forwarded through children()) | "function"
    name: str
    caller_name: str     # enclosing module/function's name, or "<toplevel>"
    call_origin: str     # call_pos.origin ('' for the main file)
    call_line: int
    decl_origin: str
    decl_line: int
    call_count: int = 0
    call_column: int = 0  # tells two calls sharing one line apart
    self_time: float = 0.0        # seconds, own code only, never double-counted
    cumulative_time: float = 0.0  # seconds, includes children; recursion-guarded


@dataclass
class ProfileResult:
    """Whole-render profiling summary, built by Evaluator.evaluate() when
    constructed with profile=True. unattributed_time covers top-level
    script code and anything else not inside a user module/function call
    (native builtins' own resolve work, mostly) -- resolve_time always
    equals sum(s.self_time for s in call_sites) + unattributed_time, so a
    UI can show percentages that honestly add to 100%."""
    call_sites: list[CallSiteProfile]
    resolve_time: float
    generate_time: float
    total_time: float
    unattributed_time: float
    # The calling-context tree (cpp 8e54284, 709da93): a flat list of dicts,
    # paths[0] the <toplevel> root, linked by parent/children INDICES. Each
    # node is one call site on ONE path, so `cuboid` from `bracket` and from
    # `rail` are separate nodes with their own times, where call_sites sums
    # a site over every path. Keys: parent, children, kind, name,
    # call_origin, call_line, call_column, decl_origin, decl_line,
    # call_count, self_time, cumulative_time. Only direct self-recursion
    # folds onto one node; cumulative time is derived from the subtree.
    paths: list = field(default_factory=list)


# Extrusion height used to display top-level 2D results (e.g. `circle();`)
# in the 3D viewport — the renderer/exporter only know how to handle Manifold
# meshes. 1 unit, as OpenSCAD's own 2D preview is (measured at three camera
# tilts); 1e-3 gave --viewall a different box, so 2D images came out at a
# different scale from the reference's.
_TOP_LEVEL_2D_HEIGHT = 1.0

# Matches SceneRenderer._default_color (renderer.py) -- the color shown for
# geometry with no explicit color() override. ColoredBody.color normally
# stays None for uncolored geometry so the renderer can resolve it live
# against the current color theme, but a per-triangle tri_colors array (see
# Evaluator._attach_tri_colors) bakes colors in at generate time, so an
# uncolored *part* of a multi-color CSG merge needs a concrete fallback here.
_DEFAULT_GEOMETRY_COLOR = (0.9, 0.85, 0.1, 1.0)
# The faces a difference() exposes where the subtrahend carried no colour
# of its own: OpenSCAD's CGAL "back face" green (#9DCB51), which its preview
# paints and its colour-preserving render and 3MF export keep -- so a cut
# through an uncoloured part reads as a cut (cpp #173).
_CUT_FACE_COLOR = (157 / 255, 203 / 255, 81 / 255, 1.0)


def to_renderable_bodies(bodies: list[ColoredBody], height: float = _TOP_LEVEL_2D_HEIGHT) -> list[ColoredBody]:
    """Convert top-level 2D-only results (`body is None`, `section` set —
    e.g. `circle();`) into extruded Manifolds, so the renderer/exporter
    (which only handle Manifold meshes) can display them. 3D bodies pass
    through unchanged, and so do open-mesh bodies (`raw_mesh` set), which a
    renderer or exporter draws from their raw triangles.

    `height` lets a viewer thin the slab (cpp #193); keep the default, 1 as
    OpenSCAD's own preview is, for anything that frames or exports it. The
    slab carries the section's `section_id` as its one run, so a click on
    it finds the node that built the shape."""
    out = []
    for cb in bodies:
        if cb.body is not None or cb.section is None:
            out.append(cb)
            continue
        body = m3d.Manifold.extrude(cb.section, height).translate([0, 0, cb.section_z])
        if cb.section_id is not None and not body.is_empty():
            mesh = body.to_mesh64()
            tris = np.array(mesh.tri_verts, dtype=np.uint64)
            body = m3d.Manifold(m3d.Mesh64(
                vert_properties=np.array(mesh.vert_properties, dtype=np.float64), tri_verts=tris,
                run_index=np.array([0, tris.size], dtype=np.uint64),
                run_original_id=np.array([cb.section_id], dtype=np.uint32)))
        # The section stays too: it is the real geometry, which a 2D export
        # (SVG, PDF) reads instead of the slab.
        out.append(ColoredBody(body=body, color=cb.color, flat_preview=True, role=cb.role, section=cb.section))
    return out


def flatten_csg_tree(tree: list[CSGNode]) -> list[ColoredBody]:
    """Concatenate every top-level node's already-computed .bodies (NOT
    recursing into .children — a parent's .bodies already is the fully
    resolved/combined result of its children, e.g. a union node's .bodies
    is the merged CSG result, not to be added on top of its children's
    individual bodies too). Reproduces evaluate()'s returned body list
    exactly for any script with no top-level `!` (show_only) — evaluate()'s
    own post-hoc show_only filter is applied once across the whole flat
    result and is not itself represented by any single tree node."""
    return [b for node in tree for b in node.bodies]


def _summarize_param(value, max_items: int = 6, max_len: int = 40) -> str:
    """Compact one-line repr for a CSGNode.params value, used by
    format_csg_tree — collapses long lists/dicts/arrays (e.g. polyhedron
    points, imported STL verts, surface() height grids) to "<... of N>"
    instead of dumping them in full, since numpy's own repr can span
    multiple lines and would otherwise break the one-line-per-node dump.
    Collapsing is purely size-based (item count), not "contains a nested
    container" or "is a numpy array" — a small list/dict/array is shown
    in full with each element/value itself recursively summarized, so
    e.g. translate's args={0: [1.0,2.0,3.0]} (one entry whose value is a
    short flat list) displays completely instead of collapsing to an
    opaque "<dict of 1>" just because that one entry happens to be a
    list, and a small polyhedron's own points/faces (its actual
    user-authored content, unlike sphere/cylinder's auto-generated
    tessellation — see _DUMP_TESSELLATION_KEYS, which excludes the
    latter from the dump entirely by key) are visible rather than
    reduced to a bare shape."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        if len(value) > max_items:
            kind = "tuple" if isinstance(value, tuple) else "list"
            return f"<{kind} of {len(value)}>"
        return "[" + ", ".join(_summarize_param(v) for v in value) + "]"
    if isinstance(value, dict):
        if len(value) > max_items:
            return f"<dict of {len(value)}>"
        return "{" + ", ".join(f"{k!r}: {_summarize_param(v)}" for k, v in value.items()) + "}"
    text = repr(value)
    return text if len(text) <= max_len else text[:max_len - 1] + "…"


# Params keys never shown in format_csg_tree's per-node summary, regardless
# of kind -- pure internal bookkeeping (op/name duplicate the node's own
# kind; group_sizes is _generate_csg's private re-chunking data) or already
# represented structurally elsewhere in the tree (color() gets its own
# wrapping CSGNode, same as translate/rotate, so every descendant leaf
# additionally carrying its own inherited "color" param -- needed at
# generate time, not a display bug -- would otherwise be shown twice).
_DUMP_HIDDEN_PARAM_KEYS = frozenset({"color", "op", "name", "group_sizes"})

# Params keys holding auto-generated tessellation data (not user-authored
# content) for every kind except polyhedron, where the equivalent data
# *is* the user's own points/faces input and is worth seeing (still
# collapsed to <... of N> by _summarize_param if it's large, same as any
# other param -- this only controls whether the key is shown at all).
_DUMP_TESSELLATION_KEYS = frozenset({"verts", "tris", "tri_arr"})

# Params keys renamed for display only (the underlying dict key stays
# "segs" everywhere it's actually used -- _generate_cylinder/_generate_2d/
# etc. all read params["segs"]) -- "segs" is every _resolve_X's own name
# for the circular-segment count it resolved from $fn/$fa/$fs (via _fn()),
# shared by cylinder/circle/offset/text; "$fn=" reads as what it actually
# represents to someone looking at the dump, rather than exposing the
# resolve step's internal variable name.
_DUMP_KEY_RENAMES = {"segs": "$fn"}


def _format_call_args(args: dict) -> str:
    """Render a _resolve_args()-shaped dict ({0: v0, 1: v1, 'name': v, ...}
    -- positional args keyed by index, named args keyed by name) as
    OpenSCAD call-argument syntax ("v0, v1, name=v") instead of Python
    dict syntax. Used by format_csg_tree for a transform's "args" param,
    so e.g. translate([2,2,-1])'s dump reads translate([2.0, 2.0, -1.0])
    the way the user actually wrote it, not translate(args={0: [2.0,
    2.0, -1.0]})."""
    parts = []
    for k, v in args.items():
        if isinstance(k, int):
            parts.append(_summarize_param(v))
        else:
            parts.append(f"{k}={_summarize_param(v)}")
    return ", ".join(parts)


def format_csg_tree(tree: list[CSGNode], indent: int = 0) -> str:
    """Human-readable recursive dump of a resolved CSG tree — kind and a
    compact params summary (see _summarize_param and the _DUMP_*_KEYS
    filters above). Used by the Design menu's "Dump CSG Tree to Console"
    command.

    Represents geometry, not the code that produced it: neither
    children() calls nor user-module calls get a node of their own (see
    _eval_statement) -- their resolved subtree is spliced directly into
    the enclosing node's children, so e.g. a user module wrapping a cube
    shows up as just "cube(...)", not "mymodule(...) > cube(...)".

    Deliberately omits a generated-body count: once Evaluator's
    ManifoldCache (see evaluate()/generate_tree()) reuses a cached
    ancestor's result, it skips recursing into that ancestor's children
    entirely, leaving their own .bodies at the empty default from
    construction -- not because they produced no geometry, but simply
    because generate_tree never visited them on that pass. A count that
    reads "0" in that case would be actively misleading, not just
    uninformative, so this only ever describes resolved structure
    (which is always complete/reliable regardless of caching), never
    generated output.

    Indent is +1 unit for every non-root line (i.e. depth 1 gets two
    indent units, not one): the console (ConsoleWidget._append_foldable)
    displays multi-line output with a "<arrow> " prefix on the first
    line only (2 display columns) and no prefix on the rest -- without
    this compensating offset, a depth-1 child's own indent would land in
    the same column the root's text starts at (right after the arrow),
    making it look like a sibling of the root rather than its child."""
    lines = []
    pad = "  " * (indent + 1) if indent > 0 else ""
    for node in tree:
        shown = {
            k: v for k, v in node.params.items()
            if k not in _DUMP_HIDDEN_PARAM_KEYS
            and not (k in _DUMP_TESSELLATION_KEYS and node.kind != "polyhedron")
        }
        parts = [
            _format_call_args(v) if k == "args" and isinstance(v, dict)
            else f"{_DUMP_KEY_RENAMES.get(k, k)}={_summarize_param(v)}"
            for k, v in shown.items()
        ]
        params_str = ", ".join(parts)
        lines.append(f"{pad}{node.kind}({params_str})")
        if node.children:
            lines.append(format_csg_tree(node.children, indent + 1))
    return "\n".join(lines)


def _canon(value):
    """Recursively convert a CSGNode.params-shaped value (numbers, strings,
    bools, None, and nested lists/tuples/dicts/numpy arrays — params is
    documented as plain data, never Manifold objects) into a hashable
    canonical form, for use in ManifoldCache's cache key. Lists/tuples and
    numpy arrays become tuples; dict keys are sorted by str(key) (params
    dicts commonly mix int positional-arg keys with str named-arg keys —
    e.g. `cube(10, center=true)` — sorting the raw keys directly would
    raise TypeError comparing int to str). bool is tagged distinctly from
    int/float: Python's `False == 0`/`True == 1` (and matching hashes) would
    otherwise silently collide two OpenSCAD values of different type and
    meaning -- found via a real corruption where force_list(chamfer=0, 4)
    was served force_list(corner_flip=false, 4)'s cached [false,false,...]
    result, and bool*number evaluates to undef here, not 0."""
    if isinstance(value, bool):
        return ("__bool__", value)
    if isinstance(value, np.ndarray):
        return ("__ndarray__", value.shape, tuple(value.flatten().tolist()))
    if isinstance(value, (list, tuple)):
        return tuple(_canon(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted(((k, _canon(v)) for k, v in value.items()), key=lambda kv: str(kv[0])))
    return value


class ManifoldCache:
    """Content-hash cache of already-generated CSGNode subtrees, so
    generate_tree() can skip re-running Manifold work for a subtree whose
    resolved content (kind/params/children, see Evaluator._cache_key)
    hasn't changed since a previous render/debugger pause. Lives outside
    any single Evaluator/evaluate() call's lifetime — owned by MainWindow
    and passed into each new Evaluator() via its manifold_cache= kwarg, so
    it survives across the fresh Evaluator/AST/CSGNode objects every
    render creates. Thread-safe (renders and debug sessions run on
    background QThreads and can genuinely overlap)."""

    def __init__(self):
        # key -> (bodies, the warnings generating them printed, replayed on a hit)
        self._entries: dict[tuple, tuple[list[ColoredBody], list[str]]] = {}
        self._lock = threading.Lock()

    def get(self, key: tuple) -> tuple[list[ColoredBody], list[str]] | None:
        with self._lock:
            return self._entries.get(key)

    def put(self, key: tuple, entry: tuple[list[ColoredBody], list[str]]) -> None:
        with self._lock:
            self._entries[key] = entry

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


# supported_feature("name") -> the level this build implements that extension at,
# 0 for one it does not -- including names it has never heard of, so probing
# for a future feature is safe. OpenSCAD silently ignores unknown arguments
# (children(separate=true) renders the wrong shape there), so guarding on this
# is how a script refuses. Names shared with openscad_cpp_evaluator (#113, #115);
# export-name is not implemented here.
_FEATURE_LEVELS = {
    "render-expr": 1, "linear-solve": 1, "polyhedron-vnf": 1, "separate-children": 1,
    "minkowski-diff": 1, "sphere-styles": 1, "simplify-op": 1, "expr-import": 1,
    "object-function": 1, "roof-op": 1, "svg-class": 1, "levelset": 1, "mesh-repair": 1,
}


def _package_version() -> list[int]:
    try:
        from importlib.metadata import version
        return [int(p) for p in version("openscad_evaluator").split(".")[:3]]
    except Exception:  # an unusual install: still a list, so `$_BELFRYSCAD != undef` holds
        return [0, 0, 0]


_DEFAULT_DOLLAR = {"$fn": 0, "$fa": 12.0, "$fs": 2.0, "$t": 0.0, "$parent_modules": 0,
                   "$preview": False,  # a render; a previewing caller seeds true via viewport_params
                   # Undef in OpenSCAD, so `!is_undef($_SUPPORTED_FEATURE) && supported_feature(...)`
                   # is a portable, silent guard.
                   "$_SUPPORTED_FEATURE": True, "$_BELFRYSCAD": _package_version()}


class EvalContext:
    """Mutable evaluation state threaded through recursive calls."""
    __slots__ = ('scope', 'dyn', 'let', 'dyn_positions', 'dyn_explicit', 'color',
                 'children_nodes', 'children_caller_ctx')

    def __init__(self, scope, dyn=None, let=None, dyn_positions=None, dyn_explicit=None, color=None,
                 children_nodes=None, children_caller_ctx=None):
        self.scope = scope
        self.dyn = dyn if dyn is not None else dict(_DEFAULT_DOLLAR)
        self.let = let if let is not None else {}
        self.dyn_positions = dyn_positions if dyn_positions is not None else {}
        # Names the *script itself* assigned via a `$var = ...;` statement,
        # as opposed to names merely present in `dyn` because they were
        # seeded from the current viewport state (see Evaluator.evaluate's
        # viewport_params) -- lets callers distinguish "the script set
        # $vpt" from "the evaluator pre-populated $vpt from the current
        # camera," which look identical in `dyn` alone.
        self.dyn_explicit = dyn_explicit if dyn_explicit is not None else set()
        self.color = color
        self.children_nodes = children_nodes if children_nodes is not None else []
        self.children_caller_ctx = children_caller_ctx

    def child_ctx(self, scope=None, dyn=None, let=None, color=None,
                  children_nodes=None, children_caller_ctx=None, share_dyn=False):
        """Like every other field here, an unspecified children_nodes/
        children_caller_ctx means "inherit from self" -- NOT "this call has
        no deferred children," which is what defaulting to []/None would
        mean. child_ctx() is for tweaking ambient state (color, a $-var
        override) without leaving the current statement's evaluation, so a
        children() call evaluated under the result must still be able to
        forward to *this* context's caller. (Entering an actual new module
        body, where children_nodes legitimately changes to that call's own
        children, goes through call_ctx() instead, whose callers always
        pass real values explicitly.) Forgetting to thread these through
        previously broke any user module of the form `color(c) children();`
        (silently swallowing the children() call's own forwarded geometry)
        the moment _resolve_color's `ctx.child_ctx(color=rgba)` reset them —
        _resolve_transform never hit this, since it evaluates children
        against the original ctx directly rather than deriving a new one.

        `share_dyn`: only meaningful when `dyn` isn't given explicitly --
        skips the `dict(self.dyn)`/`set(self.dyn_explicit)` copies and
        shares the parent's own dict/set by reference instead. Only safe
        when the caller can guarantee this new context's dyn/dyn_explicit
        will never be mutated in place (a $-prefixed `Assignment` statement
        is the only thing that does, and those only occur in module-body
        statement evaluation, never in a bare function-call fast path) --
        see _eval_user_function's share_dyn computation for the one place
        this is actually exercised."""
        return EvalContext(
            scope=scope if scope is not None else self.scope,
            dyn=dyn if dyn is not None else (self.dyn if share_dyn else dict(self.dyn)),
            let=let if let is not None else dict(self.let),
            dyn_positions={} if dyn is None else self.dyn_positions,
            dyn_explicit=self.dyn_explicit if share_dyn else set(self.dyn_explicit),
            color=color if color is not None else self.color,
            children_nodes=children_nodes if children_nodes is not None else self.children_nodes,
            children_caller_ctx=children_caller_ctx if children_caller_ctx is not None else self.children_caller_ctx,
        )

    def let_child_ctx(self):
        ctx = EvalContext.__new__(EvalContext)
        ctx.scope = self.scope
        ctx.dyn = self.dyn
        ctx.let = dict(self.let)
        ctx.dyn_positions = self.dyn_positions
        ctx.dyn_explicit = self.dyn_explicit
        ctx.color = self.color
        ctx.children_nodes = self.children_nodes
        ctx.children_caller_ctx = self.children_caller_ctx
        return ctx

    def call_ctx(self, scope=None, color=None,
                 children_nodes=None, children_caller_ctx=None, share_dyn=False):
        """`share_dyn`: see child_ctx's docstring -- same safety contract."""
        return EvalContext(
            scope=scope if scope is not None else self.scope,
            dyn=self.dyn if share_dyn else dict(self.dyn),
            let={},
            dyn_positions={},
            dyn_explicit=self.dyn_explicit if share_dyn else set(self.dyn_explicit),
            color=color if color is not None else self.color,
            children_nodes=children_nodes if children_nodes is not None else [],
            children_caller_ctx=children_caller_ctx,
        )


def resolve_use_scopes(nodes, current_file, log_fn):
    """Resolve `use <file>` statements per OpenSCAD semantics.

    Each top-level `UseStatement` is replaced by the used file's *own*
    module and function declarations — its top-level geometry and variable
    assignments are not injected, so `current_file`'s own variable namespace
    stays isolated from (and invisible to) the used file's globals.
    Declarations that the used file itself pulled in via a nested `use` are
    not re-exported ("nested use has no effect on the base file's
    environment").

    Returns `(processed_nodes, own_nodes, root_scope)`:
    - `processed_nodes` — what `current_file` should be evaluated as: its
      own nodes plus the declarations injected via its `use` statements.
    - `own_nodes` — `current_file`'s own nodes (minus `UseStatement`s),
      excluding anything injected via `use`; this is what gets exposed to
      whoever in turn `use`s `current_file`.
    - `root_scope` — built from `processed_nodes`, then each injected
      declaration is re-anchored to its own file's root scope (computed
      recursively), giving it access to its own file's globals without
      exposing them to `current_file`.
    """
    from openscad_lalr_parser import getASTfromLibraryFile, build_scopes
    from openscad_lalr_parser.nodes import UseStatement, ModuleDeclaration, FunctionDeclaration

    injected = []
    reanchor = []
    for node in nodes:
        if not isinstance(node, UseStatement):
            continue
        try:
            fp = node.filepath.val if hasattr(node.filepath, 'val') else node.filepath
            # `include`d files are flattened into `nodes`, so a `use` statement
            # may have originated from a different file than `current_file` —
            # resolve relative paths against where it was actually written.
            origin = getattr(getattr(node, 'position', None), 'origin', None)
            lib_nodes, lib_path = getASTfromLibraryFile(origin or current_file, fp, include_comments=False)
        except Exception as e:
            msg = str(e)
            if "not found" not in msg and "No such file" not in msg:
                log_fn(f"use error: {e}")
            continue
        if not lib_nodes:
            continue
        _, lib_own_nodes, lib_root_scope = resolve_use_scopes(lib_nodes, lib_path, log_fn)
        lib_injected = [
            n for n in lib_own_nodes
            if isinstance(n, (ModuleDeclaration, FunctionDeclaration))
        ]
        injected.extend(lib_injected)
        if lib_injected:
            reanchor.append((lib_injected, lib_root_scope))

    own_nodes = [n for n in nodes if not isinstance(n, UseStatement)]
    processed_nodes = injected + own_nodes
    root_scope = build_scopes(processed_nodes)
    for lib_injected, lib_root_scope in reanchor:
        for n in lib_injected:
            n.build_scope(lib_root_scope)
    return processed_nodes, own_nodes, root_scope


class Evaluator:
    def __init__(self, echo_fn=None, debug_hook=None, error_break_fn=None, return_hook=None,
                 manifold_cache: "ManifoldCache | None" = None, profile: bool = False,
                 keep_minuend_color: bool = False, coverage: bool = False):
        self.id_to_node: dict[int, ASTNode] = {}
        # originalID -> the call chain that reached it, innermost first, as
        # (call position, is_module) pairs; () for top-level geometry.
        # id_to_node names the node that PRODUCED a body, which for library
        # geometry is inside the library -- BOSL2 overrides cube() itself --
        # so a picker walks this instead, stopping at whichever frame its
        # editor has open. Deliberately unfiltered: a cuboid() is 24 frames,
        # most of them BOSL2's, and a library author wants to step into
        # those. is_module tells a frame with geometry behind it (worth
        # dragging) from a function frame (cpp #180).
        self.id_to_call_chain: dict[int, tuple] = {}
        self._generate_call_chain: tuple = ()
        self.id_to_color: dict[int, Optional[tuple]] = {}
        self._hull_depth = 0  # hull() nodes enclosing the one being generated
        self._if_taken = False  # whether the last `if` ran a branch; see _is_operand_when_empty
        self._builtin_shadow: dict[tuple, Any] = {}  # (id(scope), builtin name) -> user decl or None
        self._escapes_checked: set[int] = set()  # StringLiterals already checked for bad escapes
        self._param_names: dict[int, tuple] = {}  # id(parameter list) -> its names, for _bind_args
        self._global_values: dict[int, Any] = {}  # id(root-scope Assignment) -> value; see _eval_identifier
        self.csg_tree: list[CSGNode] = []
        self._tree_stack: list[list[CSGNode]] = [self.csg_tree]
        # Opt-in (None by default, so every existing bare Evaluator(...)
        # call site/test is unaffected) content-hash cache shared across
        # renders/debugger pauses -- see ManifoldCache and generate_tree().
        self._manifold_cache = manifold_cache
        # difference() paints its cut faces with the minuend's colour rather
        # than the subtrahend's (or the cut green) -- what OpenCSG preview
        # cannot offer (openscad/openscad#4798; cpp #173). Off by default.
        self._keep_minuend_color = keep_minuend_color
        # coverage=True: which statements, branch arms and bodies ran, as
        # coverage_result after evaluate() -- see coverage.py (cpp #169).
        # Hits are keyed by id(node); the AST outlives the run.
        self._coverage = coverage
        self._cov_hits: dict[int, int] = {}
        self.coverage_result = None
        self._cache_producer: dict[tuple, ASTNode] = {}  # key -> node whose generate filled it
        self._warn_captures: list[list[str]] = []  # one per cacheable subtree generating now
        # Incremented by _builtin_rands -- lets _eval_statement detect
        # whether rands() was called anywhere while resolving a given
        # CSGNode, to taint it (and its ancestors) as uncacheable. See
        # CSGNode.uncacheable.
        self._rands_call_count = 0
        # id(decl)/id(func_node) -> whether any of its declared parameter
        # names starts with '$'. A purely static property of the
        # declaration (never of a particular call) -- see
        # _has_dollar_param's docstring.
        self._decl_dollar_param: dict[int, bool] = {}
        # Opt-in (False by default -- zero overhead, gated behind
        # `if self._profiling:` at every call site) per-call-site timing.
        # See CallSiteProfile/ProfileResult and _eval_user_module/
        # _eval_user_function/_eval_function_literal's instrumentation.
        self._profiling = profile
        self._profile_sites: dict[tuple, CallSiteProfile] = {}
        self._profile_paths: list[dict] = []
        self._profile_cur = 0
        self._via_children = False
        self._profile_active: set[tuple] = set()    # site_keys live on _call_stack (recursion guard)
        self._profile_child_time: list[float] = []  # parallel aux stack to _call_stack
        self.profile_result: "ProfileResult | None" = None
        # Every geometry-producing builtin kind (== CSGNode.kind) is
        # registered here (Phase 2, complete — see docs/evaluator.md "CSG
        # tree"). resolve_fn parses arguments and recursively resolves
        # children as plain data (no Manifold calls); generate_fn does the
        # actual Manifold/CrossSection work, called later by generate_tree()
        # in a separate bottom-up pass. Kinds with no entry (user-module
        # calls, unknown module names) fall back to _resolve_fallback_call /
        # generate_tree()'s default child-concatenation behavior.
        self._RESOLVE_DISPATCH = {
            "cube": self._resolve_cube,
            "sphere": self._resolve_sphere,
            "cylinder": self._resolve_cylinder,
            "polyhedron": self._resolve_polyhedron,
            "circle": self._resolve_2d,
            "square": self._resolve_2d,
            "polygon": self._resolve_2d,
            "text": self._resolve_text,
            "translate": self._resolve_transform,
            "rotate": self._resolve_transform,
            "scale": self._resolve_transform,
            "mirror": self._resolve_transform,
            "resize": self._resolve_transform,
            "multmatrix": self._resolve_transform,
            "color": self._resolve_color,
            "hull": self._resolve_hull,
            "fill": self._resolve_hull,  # the same: children only
            "minkowski": self._resolve_minkowski,
            "minkowski_difference": self._resolve_minkowski,
            "simplify": self._resolve_simplify,
            "levelset": self._resolve_levelset,
            "mesh_repair": self._resolve_simplify,  # the same one positional tolerance
            "offset": self._resolve_offset,
            "projection": self._resolve_projection,
            "union": self._resolve_csg,
            "difference": self._resolve_csg,
            "intersection": self._resolve_csg,
            "intersection_for": self._resolve_intersection_for,
            "linear_extrude": self._resolve_linear_extrude,
            "rotate_extrude": self._resolve_rotate_extrude,
            "roof": self._resolve_roof,
            "surface": self._resolve_surface,
            "import": self._resolve_import,
            "render": self._resolve_render,
            "children": self._resolve_children_call,
            "breakpoint": self._resolve_breakpoint,
            "highlight": self._resolve_modifier_child,
            "background": self._resolve_modifier_child,
            "show_only": self._resolve_modifier_child,
        }
        self._GENERATE_DISPATCH = {
            "cube": self._generate_cube,
            "sphere": self._generate_sphere,
            "cylinder": self._generate_cylinder,
            "polyhedron": self._generate_polyhedron,
            "circle": self._generate_2d,
            "square": self._generate_2d,
            "polygon": self._generate_2d,
            "text": self._generate_text,
            "translate": self._generate_transform,
            "rotate": self._generate_transform,
            "scale": self._generate_transform,
            "mirror": self._generate_transform,
            "resize": self._generate_transform,
            "multmatrix": self._generate_transform,
            "color": self._generate_color,
            "hull": self._generate_hull,
            "fill": self._generate_fill,
            "minkowski": self._generate_minkowski,
            "minkowski_difference": self._generate_minkowski_difference,
            "simplify": self._generate_simplify,
            "levelset": self._generate_levelset,
            "mesh_repair": self._generate_mesh_repair,
            "offset": self._generate_offset,
            "projection": self._generate_projection,
            "union": self._generate_csg,
            "difference": self._generate_csg,
            "intersection": self._generate_csg,
            "intersection_for": self._generate_intersection_for,
            "linear_extrude": self._generate_linear_extrude,
            "rotate_extrude": self._generate_rotate_extrude,
            "roof": self._generate_roof,
            "surface": self._generate_surface,
            "import": self._generate_import,
            "highlight": self._generate_highlight,
            "background": self._generate_background,
            "show_only": self._generate_show_only,
        }
        self._errors: list[str] = []
        # Everything printed goes through _emit, which attributes a warning
        # to the user's call site, then to _sink: the caller's echo_fn, or a
        # cache capture wrapped around it (see generate_tree).
        self._sink = echo_fn or (lambda msg: print(msg))
        self._echo_fn = self._emit
        self._generate_warn_entry = None
        self._call_stack: list = []
        self._frame_ctxs: list = []
        self._debug_hook = debug_hook
        self._debugging = debug_hook is not None
        # The statement checkpoint in progress at each call depth, for
        # _check_debug's call-site collapse. Per depth: `a = [f(1), f(2)];`
        # runs f's own checkpoints in between, one level down.
        self._last_stmt_by_depth: dict[int, tuple] = {}
        self._error_break_fn = error_break_fn
        self._return_hook = return_hook
        self._last_locals: dict = {}
        self._last_children_positions: Optional[list[tuple[Optional[str], int]]] = None
        self._last_all_frame_locals: list = []
        self._last_ctx: EvalContext | None = None
        self._root_ctx: EvalContext | None = None
        self._expr_depth: int = 0
        self._math_fns = {
            "abs": abs, "sign": lambda x: (1 if x > 0 else -1 if x < 0 else 0),
            "ceil": lambda x: x if (math.isnan(x) or math.isinf(x)) else math.ceil(x),
            "floor": lambda x: x if (math.isnan(x) or math.isinf(x)) else math.floor(x),
            "round": lambda x: x if (math.isnan(x) or math.isinf(x))
                else (math.floor(x + 0.5) if x >= 0 else math.ceil(x - 0.5)),
            "sqrt": lambda x: float('nan') if x < 0 else math.sqrt(x),
            "ln": lambda x: float('-inf') if x == 0 else (float('nan') if x < 0 else math.log(x)),
            "log": lambda x: float('-inf') if x == 0 else (float('nan') if x < 0 else math.log10(x)),
            "exp": math.exp,
            "sin": self._builtin_sin,
            "cos": self._builtin_cos,
            "tan": self._builtin_tan,
            "asin": _asin_deg,
            "acos": _acos_deg,
            "atan": _atan_deg,
            "atan2": _atan2_deg,
            "max": self._builtin_max, "min": self._builtin_min,
            "pow": self._builtin_pow,
            "norm": lambda v: math.sqrt(sum(x*x for x in v)),
            "cross": self._builtin_cross,
            "rands": self._builtin_rands,
            "concat": lambda *args: sum((list(a) if isinstance(a, list) else [a] for a in args), []),
            "len": lambda x: len(x) if isinstance(x, (list, str, OscObject)) else None,
            "str": lambda *a: "".join(x if isinstance(x, str) else self._fmt_val(x) for x in a),
            "chr": self._builtin_chr,
            "ord": lambda s: ord(s[0]) if isinstance(s, str) and len(s) >= 1 else None,
            "is_undef": lambda x: x is None,
            "is_num": lambda x: isinstance(x, (int, float)) and not isinstance(x, bool) and not math.isnan(x),
            "is_bool": lambda x: isinstance(x, bool),
            "is_string": lambda x: isinstance(x, str),
            "is_list": lambda x: isinstance(x, list),
            "is_function": lambda x: isinstance(x, (FunctionDeclaration, FunctionLiteral, Closure)),
            "is_object": lambda x: isinstance(x, OscObject),
            "search": self._builtin_search,
            "lookup": self._builtin_lookup,
            "has_key": lambda obj, key: (key in obj.data) if isinstance(obj, OscObject) else None,
            "version": lambda: list(_OPENSCAD_VERSION),
            "version_num": _version_num,
            "parent_module": self._builtin_parent_module,
            "supported_feature": lambda feature=None: _FEATURE_LEVELS.get(feature, 0) if isinstance(feature, str) else 0,
        }
        self._BUILTIN_FN_NAMES = frozenset(self._math_fns) | {"object", "textmetrics", "fontmetrics", "linear_solve",
                                                              "dxf_dim", "dxf_cross"}
        # Functions that require an actual number (or a vector of numbers)
        # and must reject a bool argument as a type error (-> undef),
        # confirmed against real OpenSCAD 2022.08.22 -- e.g. abs(true),
        # max(true, 1), norm([true, 0]) are all undef there. Needed
        # because Python's bool is a subclass of int, so every one of
        # these functions would otherwise silently treat true/false as
        # 1/0 (abs(true) -> 1, max(true, 1) -> true, norm([true, 0]) -> 1)
        # rather than raising and hitting the generic try/except below.
        self._NUMERIC_ONLY_MATH_FNS = frozenset({
            "abs", "sign", "ceil", "floor", "round", "sqrt", "ln", "log", "exp",
            "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "pow",
            "max", "min", "norm", "cross",
        })

    def _build_frame_locals(self, ctx: EvalContext):
        local_scope: dict = {}
        dyn_names: set = set()
        for k, v in ctx.let.items():
            local_scope[k] = v
            dyn_names.add(k)
        for k, v in ctx.dyn.items():
            if k.startswith('$'):
                local_scope[k] = v

        outer_scope: dict = {}
        if self._call_stack and self._root_ctx is not None:
            for k, v in self._root_ctx.let.items():
                if k not in local_scope:
                    outer_scope[k] = v

        current_frame = {"local_scope": local_scope, "outer_scope": outer_scope, "dyn_names": dyn_names,
                          "dyn_explicit": set(ctx.dyn_explicit)}
        all_frame_locals = [current_frame]
        for frame_ctx in reversed(self._frame_ctxs[:-1]):
            p_local: dict = {}
            p_dyn: set = set()
            for k, v in frame_ctx.let.items():
                p_local[k] = v
                p_dyn.add(k)
            for k, v in frame_ctx.dyn.items():
                if k.startswith('$'):
                    p_local[k] = v
            all_frame_locals.append({"local_scope": p_local, "outer_scope": {}, "dyn_names": p_dyn,
                                     "dyn_explicit": set(frame_ctx.dyn_explicit)})

        if self._call_stack:
            toplevel_frame = {
                "local_scope": dict(outer_scope),
                "outer_scope": {},
                "dyn_names": set(),
                "dyn_explicit": set(),
            }
            all_frame_locals.append(toplevel_frame)

        self._last_locals = {n: v for n, v in local_scope.items() if n in dyn_names}
        self._last_all_frame_locals = all_frame_locals
        return self._last_locals, all_frame_locals

    @staticmethod
    def _child_statement_positions(node: ASTNode, ctx: EvalContext) -> Optional[list[tuple[Optional[str], int]]]:
        """(origin, line) for each top-level, non-declaration child of
        `node` (a ModularCall's `.children` — the `{ ... }` block passed to
        a module call), if any. Used by the debugger's "Step to Child"
        command to know which statements children()/children(N) might
        forward control to — stashed on self._last_children_positions
        rather than threaded through the debug_hook callback itself, so
        adding it doesn't change that protocol's signature (every
        hand-rolled test hook would otherwise need updating).

        A `children()` call has no children of its own: it forwards the
        enclosing invocation's, which live on the context. Every forwarded
        child is a target even when an index runs only some -- a position
        never reached is never stopped at."""
        if type(node) is ModularCall and node.name.name == "children":
            node_children = ctx.children_nodes
        else:
            node_children = getattr(node, 'children', None)
        if not node_children:
            return None
        positions = []
        for c in node_children:
            if isinstance(c, (Assignment, ModuleDeclaration, FunctionDeclaration)):
                continue
            cpos = getattr(c, 'position', None)
            cline = getattr(cpos, 'line', None) if cpos else None
            if cline is not None:
                positions.append((getattr(cpos, 'origin', None), int(cline)))
        return positions or None

    def _check_debug(self, node: ASTNode, ctx: EvalContext, forced: bool = False, expr_level: bool = False,
                     call_site: bool = False):
        """`call_site`: the stop just before descending into a user function.
        It is steppable, but not a second execution of its line: when it
        lands on the line and depth of the statement checkpoint already in
        progress (`x = f(y);`) it is dropped, so a breakpoint there fires
        once, not twice. A call on its own line still stops."""
        if self._debug_hook is None:
            return
        pos = getattr(node, 'position', None)
        line = getattr(pos, 'line', None) if pos else None
        if line is None:
            return
        origin = getattr(pos, 'origin', None)
        depth = len(self._call_stack)
        if call_site and not forced and self._last_stmt_by_depth.get(depth) == (line, origin):
            return
        if not expr_level and not call_site:
            self._last_stmt_by_depth[depth] = (line, origin)
        self._last_children_positions = self._child_statement_positions(node, ctx)

        cmd, mods = self._debug_hook(
            int(line), len(self._call_stack),
            forced=forced, expr_level=expr_level,
            expr_depth=self._expr_depth, origin=origin,
            get_frames=lambda: (self._build_frame_locals(ctx), list(self._call_stack)),
        )
        for k, v in mods.items():
            ctx.let[k] = v
        if cmd == "stop":
            raise EvalError(DEBUGGING_STOPPED_MESSAGE)

    _LOC_SUFFIX = re.compile(r" in file (.*), line (\d+)$")

    def _emit(self, msg: str) -> None:
        """Print `msg`; a warning raised below the top level also names the
        user's own line that started the chain -- ", from w.scad, line 3",
        the OUTERMOST call site, since intermediate frames are in the TRACE
        lines that follow. A warning inside a library otherwise pointed only
        into the library. Generate-time warnings run after the stack has
        unwound, so they get the clause from their CSG node, and no trace.
        Deliberately more than OpenSCAD prints, as in the C++ port (3e11352);
        a top-level warning stays one line."""
        if msg.startswith("WARNING:") and (self._call_stack or self._generate_warn_entry is not None):
            entry = self._call_stack[0][2] if self._call_stack else self._generate_warn_entry
            m = self._LOC_SUFFIX.search(msg.split("\n", 1)[0])
            if entry is not None and not (m and m.group(1) == str(entry.origin) and int(m.group(2)) == entry.line):
                head, nl, rest = msg.partition("\n")
                msg = f"{head}, from {entry.origin}, line {entry.line}{nl}{rest}"
            if self._call_stack:
                msg = "\n".join([msg] + self._trace_lines())
        self._sink(msg)

    @staticmethod
    def _loc(pos) -> str:
        if pos is None:
            return ""
        return f" in file {pos.origin}, line {pos.line}"

    def _trace_lines(self, node=None, innermost_frame: str | None = None) -> list[str]:
        """Build TRACE lines matching OpenSCAD's error/warning format."""
        lines = []
        node_pos = getattr(node, 'position', None) if node is not None else None
        if innermost_frame:
            lines.append(f"TRACE: called by '{innermost_frame}'{self._loc(node_pos)}")
        for entry in reversed(self._call_stack):
            kind = entry[0]
            fname = entry[1]
            call_pos = entry[2]
            if kind == "module":
                decl_pos = entry[3] if len(entry) > 3 else None
                lines.append(f"TRACE: call of '{fname}()'{self._loc(decl_pos)}")
                lines.append(f"TRACE: called by '{fname}'{self._loc(call_pos)}")
            else:
                lines.append(f"TRACE: called by '{fname}'{self._loc(call_pos)}")
        return lines

    def error(self, msg: str, node=None, innermost_frame: str | None = None):
        pos = getattr(node, 'position', None) if node is not None else None
        header = f"ERROR: {msg}{self._loc(pos)}"
        lines = [header] + self._trace_lines(node, innermost_frame)
        full = "\n".join(lines)
        self._errors.append(full)
        if self._error_break_fn is not None:
            line = getattr(pos, 'line', 0) if pos else 0
            origin = getattr(pos, 'origin', None) if pos else None
            if self._last_ctx is not None:
                _, all_frame_locals = self._build_frame_locals(self._last_ctx)
            else:
                all_frame_locals = self._last_all_frame_locals
            self._error_break_fn(int(line), header, all_frame_locals, list(self._call_stack), origin=origin)
        raise EvalError(full)

    def _fmt_val(self, v) -> str:
        if v is None:
            return "undef"
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, OscRange):
            return f"[{_format_number(v.start)} : {_format_number(v.step)} : {_format_number(v.end)}]"
        if isinstance(v, (int, float)):  # an int too: len() of 1212201 items echoes 1.2122e+6
            return _format_number(v)
        if isinstance(v, list):
            return "[" + ", ".join(self._fmt_val(x) for x in v) + "]"
        if isinstance(v, OscObject):  # OpenSCAD's own format: { a = 1; b = 2; }, and { } when empty
            return "{ " + "".join(f"{k} = {self._fmt_val(val)}; " for k, val in v.items()) + "}"
        if isinstance(v, str):
            return f'"{v}"'
        return str(v)

    def _do_echo(self, arguments, ctx: "EvalContext"):
        parts = []
        for arg in arguments:
            val = self._eval_expr(arg.expr, ctx)
            if isinstance(arg, NamedArgument):
                parts.append(f"{arg.name.name} = {self._fmt_val(val)}")
            else:
                parts.append(self._fmt_val(val))
        self._echo_fn("ECHO: " + ", ".join(parts))

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_use_statements(nodes: list[ASTNode], root_scope) -> None:
        """Inject modules/functions from `use`d files into root_scope.

        OpenSCAD makes `use`d modules globally visible across the entire
        compilation unit.  The parser's build_scopes only hoists top-level
        declarations and skips UseStatement nodes, so we resolve them here.
        """
        seen: set[str] = set()
        for node in nodes:
            if type(node) is not UseStatement:
                continue
            filepath = node.filepath.val
            origin = getattr(node.position, 'origin', '') if node.position else ''
            lib_file = findLibraryFile(origin, filepath)
            if lib_file is None or lib_file in seen:
                continue
            seen.add(lib_file)
            used_ast = getASTfromFile(lib_file)
            if not used_ast:
                continue
            used_scope = build_scopes(used_ast)
            for name, decl in used_scope.modules.items():
                if name not in root_scope.modules:
                    root_scope.define_module(name, decl)
            for name, decl in used_scope.functions.items():
                if name not in root_scope.functions:
                    root_scope.define_function(name, decl)

    def evaluate(self, nodes: list[ASTNode], root_scope, viewport_params: dict | None = None,
                 generate: bool = True) -> tuple[list[ColoredBody], dict[int, ASTNode]]:
        """Walk top-level AST nodes to build self.csg_tree (resolve pass,
        no Manifold calls), then generate_tree() it once (generate pass,
        the only place Manifold work happens) to produce the final
        geometry. Returns (geometry, id_to_node mapping).

        `generate=False` stops after the resolve pass: the script runs in
        full, so every echo, warning and error is reported as usual, but no
        geometry is built and the body list is empty -- "does this script
        run?", as OpenSCAD's `-o out.term` asks (cpp #144)."""
        self._resolve_use_statements(nodes, root_scope)
        self._cov_hits = {}
        self._files_run: set[int] = set()
        self._call_stack.clear()
        self._frame_ctxs.clear()
        self._global_values = {}
        self._builtin_shadow = {}
        self._escapes_checked = set()
        self.csg_tree = []
        self._tree_stack = [self.csg_tree]
        self._profile_sites = {}
        self._profile_active = set()
        self._profile_child_time = []
        self._profile_paths = [{"parent": -1, "children": [], "kind": "", "name": "<toplevel>", "call_origin": "",
                                "call_line": 0, "call_column": 0, "decl_origin": "", "decl_line": 0,
                                "call_count": 0, "self_time": 0.0, "cumulative_time": 0.0}]
        self._profile_cur = 0
        self._via_children = False
        self.profile_result = None
        ctx = EvalContext(scope=root_scope)
        if viewport_params:
            ctx.dyn.update(viewport_params)
        self._root_ctx = ctx
        # OpenSCAD executes all assignments before geometry in each scope.
        assignments = [n for n in nodes if isinstance(n, Assignment)]
        others = [n for n in nodes if not isinstance(n, Assignment)]
        t_resolve_start = time.perf_counter() if self._profiling else 0.0
        for node in assignments:
            self._eval_statement(node, ctx)
            name = node.name.name
            if name[0] != '$':
                # What a function reading this global will see -- see _eval_identifier.
                self._global_values[id(root_scope.variables.get(name))] = ctx.let.get(name)
        for node in others:
            self._eval_statement(node, ctx)
        t_resolve_end = time.perf_counter() if self._profiling else 0.0
        if self._coverage:  # resolve only: the geometry pass runs no script code
            from .coverage import build_result
            self.coverage_result = build_result(*self._coverage_universe(nodes, root_scope), self._cov_hits)
        result = self.generate_tree(self._show_only_root()) if generate else []
        t_generate_end = time.perf_counter() if self._profiling else 0.0
        if self._profiling:
            resolve_time = t_resolve_end - t_resolve_start
            generate_time = t_generate_end - t_resolve_end
            self_sum = sum(s.self_time for s in self._profile_sites.values())
            # Children always sit after their parent, so one reverse sweep
            # derives every cumulative time without recursing.
            for n in reversed(self._profile_paths):
                n["cumulative_time"] = n["self_time"] + sum(self._profile_paths[c]["cumulative_time"]
                                                            for c in n["children"])
            self.profile_result = ProfileResult(
                call_sites=list(self._profile_sites.values()),
                resolve_time=resolve_time,
                generate_time=generate_time,
                total_time=resolve_time + generate_time,
                unattributed_time=max(0.0, resolve_time - self_sum),
                paths=self._profile_paths,
            )
        return result, self.id_to_node

    def _show_only_root(self) -> list[CSGNode]:
        """What to generate: the whole tree, or -- if a `!` is anywhere in it
        -- just that subtree, as OpenSCAD does. `!` makes its subtree the
        whole model, so its siblings AND every operation wrapped round it go:
        `translate([50,0,0]) !cube(5);` stays at the origin, and
        `linear_extrude(10) !circle(10);` stays a circle. Tagging the bodies
        and filtering at the end could not express that -- by then every
        ancestor had been applied. The first `!` wins; another warns, once,
        where it is (a `!` inside the root is just part of it)."""
        roots: list[CSGNode] = []

        def collect(nodes):
            for n in nodes:
                if n.kind == "show_only" and n.is_builtin:
                    roots.append(n)
                collect(n.children)
        collect(self.csg_tree)
        if not roots:
            return self.csg_tree
        if len(roots) > 1:
            self._echo_fn(f"WARNING: More than one Root Modifier (!)"
                          f"{self._loc(getattr(roots[1].node, 'position', None))}")
        return [roots[0]]

    # ------------------------------------------------------------------
    # Statement dispatch
    # ------------------------------------------------------------------

    # AST node types that get their own CSGNode in self.csg_tree. ModularCall
    # covers every primitive/transform/boolean/hull/minkowski/children()/
    # user-module call. The three tagging modifiers (#/%/!) each wrap exactly
    # one child. ModularIntersectionFor is the one control-flow-shaped node
    # that is NOT transparent like for/if/let: it combines its per-iteration
    # children into a single intersected result (see _eval_intersection_for),
    # so its iterations must nest under one tree node just like union()'s
    # children nest under a union node — otherwise flatten_csg_tree() would
    # return the pre-intersection per-iteration bodies instead of the actual
    # combined result. ModularModifierDisable (*) is deliberately excluded:
    # it never evaluates its child at all, so there is nothing to record.
    _TREE_NODE_TYPES = (
        ModularCall,
        ModularModifierHighlight,
        ModularModifierBackground,
        ModularModifierShowOnly,
        ModularIntersectionFor,
    )

    def _tree_node_kind(self, node: ASTNode, ctx: EvalContext) -> tuple[str, bool]:
        """Return (kind, is_builtin) for a node in _TREE_NODE_TYPES. is_builtin
        is only meaningful for ModularCall (False when `name` resolves to a
        user module via ctx.scope.lookup_module) — a user module can shadow a
        builtin name, so `kind` alone is not a unique discriminator."""
        if isinstance(node, ModularCall):
            name = node.name.name
            return name, ctx.scope.lookup_module(name) is None
        if isinstance(node, ModularModifierHighlight):
            return "highlight", True
        if isinstance(node, ModularModifierBackground):
            return "background", True
        if isinstance(node, ModularModifierShowOnly):
            return "show_only", True
        return "intersection_for", True  # ModularIntersectionFor

    def _eval_statement(self, node: ASTNode, ctx: EvalContext) -> list[ColoredBody]:
        """Thin wrapper around _eval_statement_impl that additionally builds
        self.csg_tree as a side effect. For the five _TREE_NODE_TYPES,
        pushes a new children accumulator before resolving, pops it in a
        finally (so a raised EvalError cleanly discards the in-progress
        node rather than corrupting a parent's accumulator), then either
        appends the completed CSGNode to whichever accumulator is now on
        top of the stack, or -- for children() calls and user-module calls,
        neither of which is itself geometry -- splices the resolved
        children directly into that accumulator with no wrapping node (see
        the kind == "children"/is_builtin check below). Every recursive
        call site already calls self._eval_statement(...), so nesting
        composes automatically with no other call site changes.

        Generation is fully deferred (Phase 2 final cutover): this only
        ever calls a resolve_fn (plain data, no Manifold calls) and always
        returns []. Real bodies are populated later, in one bottom-up pass,
        by generate_tree(self.csg_tree) — see that method and evaluate().
        Every builtin kind has a resolve_fn (registered in
        _RESOLVE_DISPATCH); anything without one (user-module calls, and
        genuinely unknown module names) falls back to
        _resolve_fallback_call, which still builds the tree correctly via
        the pre-existing _eval_modular_call dispatch.

        The _check_debug(node, ctx) call below is this method's own
        responsibility for _TREE_NODE_TYPES — _eval_statement_impl (which
        calls it for every other statement type) is never reached for these
        nodes, since the isinstance check above always routes them here
        instead. Without this, no geometry-producing statement (any
        ModularCall, plus the #/%/! modifiers and intersection_for) would
        ever pause the debugger or advance step state.
        """
        if self._coverage and type(node) is not ModuleDeclaration and type(node) is not FunctionDeclaration:
            self._cov_hit(node)  # once per execution of every statement (cpp #169)
        if not isinstance(node, self._TREE_NODE_TYPES):
            return self._eval_statement_impl(node, ctx)
        self._last_ctx = ctx
        if self._debugging:
            self._check_debug(node, ctx)
        kind, is_builtin = self._tree_node_kind(node, ctx)
        resolve_fn = (self._RESOLVE_DISPATCH.get(kind) if is_builtin else None) or self._resolve_fallback_call
        self._tree_stack.append([])
        rands_before = self._rands_call_count
        try:
            params = resolve_fn(node, ctx)
        finally:
            children = self._tree_stack.pop()
        if (kind == "children" and is_builtin) or not is_builtin:
            # Neither children() nor a user-module call is itself geometry
            # -- children() is a call-site substitution, and a user module
            # is just a named wrapper around whatever geometry statements
            # its body runs. Splice the resolved subtree directly into the
            # enclosing node's children instead of wrapping it in its own
            # node, so the CSG tree represents the geometry being
            # combined, not the code structure that produced it. But if
            # that subtree is more than one sibling, group them under a
            # "union" label for display purposes (is_builtin=False so
            # generate_tree still takes the default-concatenation path,
            # not the real _generate_csg boolean merge -- juxtaposed
            # statements with no explicit combinator keep their bodies
            # separate, same as any other module body/top-level script,
            # preserving each body's own color/provenance rather than
            # collapsing them into one Manifold the way an *explicit*
            # union() call does) so the dump reads as one shape at this
            # call site instead of N unrelated-looking siblings.
            if self._rands_call_count != rands_before:
                # rands() was called directly during *this* call's own
                # resolve -- e.g. an assignment before any geometry
                # statement in a user module's body, or within children()'s
                # own arguments -- rather than inside one of the spliced
                # children's own resolve (which already taints itself via
                # the branch below). Propagate onto every spliced child so
                # the taint isn't silently dropped by splicing away the
                # node it would otherwise have landed on.
                for c in children:
                    c.uncacheable = True
            if len(children) > 1:
                union_node = CSGNode(
                    kind="union", node=node, bodies=[], is_builtin=False,
                    children=children, params={},
                    uncacheable=any(c.uncacheable for c in children),
                    warn_entry=self._call_stack[0][2] if self._call_stack else None,
                    call_chain=self._current_call_chain(),
                )
                self._tree_stack[-1].append(union_node)
            else:
                self._tree_stack[-1].extend(children)
            return []
        # Taint this node (and thus every ancestor, since uncacheable
        # propagates via the `any(...)` below at each enclosing level) if
        # rands() was called anywhere while resolving it -- see CSGNode's
        # uncacheable docstring.
        uncacheable = (self._rands_call_count != rands_before) or any(c.uncacheable for c in children)
        # A closure in the params can't be keyed by content, only identity (cpp #136).
        uncacheable = uncacheable or (kind == "levelset" and type(params.get("field")) is Closure)
        tree_node = CSGNode(kind=kind, node=node, bodies=[],
                             is_builtin=is_builtin, children=children, params=params,
                             uncacheable=uncacheable,
                             warn_entry=self._call_stack[0][2] if self._call_stack else None,
                             call_chain=self._current_call_chain())
        self._tree_stack[-1].append(tree_node)
        return []

    def _resolve_fallback_call(self, node: ModularCall, ctx: EvalContext) -> dict:
        """Structural resolve for ModularCall kinds with no _RESOLVE_DISPATCH
        entry: user-module calls (is_builtin=False) and genuinely unknown
        module names (is_builtin=True, no matching builtin or user module).
        Reuses the existing _eval_modular_call dispatch purely for its
        tree-building side effect — its return value (real bodies) is
        unused now that generation is deferred to generate_tree(), and its
        default (no registered generate_fn) is to concatenate children's
        bodies, which matches a user module body's plain concatenation and
        an unknown module's empty children list alike."""
        self._eval_modular_call(node, ctx)
        return {}

    def _cache_key(self, node: CSGNode) -> tuple:
        """Structural content-hash key for `node`, used by generate_tree()'s
        ManifoldCache lookup: (kind, is_builtin, canonicalized params,
        recursively-hashed children). Deliberately excludes node.node (the
        AST object) — every render builds a brand-new AST via
        getASTfromFile, so keying on AST identity would defeat cross-render
        caching entirely. Pure function of already-resolved data, never
        touches .bodies, so it's always cheap/safe to compute speculatively
        even on a cache miss."""
        return (node.kind, node.is_builtin, _canon(node.params),
                tuple(self._cache_key(c) for c in node.children))

    def generate_tree(self, tree: list[CSGNode]) -> list[ColoredBody]:
        """Bottom-up second pass over an already-resolved CSG tree: for each
        node, first generates its children (so any generate_fn reading
        flatten_csg_tree(children) sees real, populated bodies), then calls
        the node's own generate_fn (or, for kinds with no registered
        generate_fn — user-module calls, render()/children()/unknown-module
        passthroughs — concatenates the children's bodies), storing the
        result on node.bodies. Can be called on any (possibly partial) tree
        at any point — e.g. the debugger calling it on self.csg_tree at a
        breakpoint to render a live partial result (Phase 3).

        If self._manifold_cache is set (opt-in — None by default, so
        existing bare Evaluator() construction/tests are unaffected), each
        node's content hash is checked before doing any Manifold work: a
        cache hit reuses the previous .bodies and skips recursing into
        node.children entirely (no wasted Manifold work re-deriving
        children that would just be discarded); a miss generates normally
        and stores the result. node.uncacheable (rands() taint) always
        forces a miss, never a hit, and never gets stored either."""
        result = []
        for node in tree:
            key = None if (self._manifold_cache is None or node.uncacheable) else self._cache_key(node)
            if key is not None and self._keep_minuend_color:
                key = ("keep_minuend_color", key)  # the two modes colour cut faces differently
            cached = self._manifold_cache.get(key) if key is not None else None
            if cached is not None:
                bodies, warnings = cached
                # A hit runs no generate_fn, which is where every generate-time
                # warning is raised; replay them, or an unchanged re-render
                # goes quiet about a defect that is still there (#186).
                for msg in warnings:
                    self._sink(msg)  # already attributed when first printed
                node.bodies = self._restamp_cached_ids(bodies, node.node, self._cache_producer.get(key),
                                                       node.call_chain)
            else:
                if key is not None:
                    self._warn_captures.append([])
                    if len(self._warn_captures) == 1:
                        real_sink = self._sink

                        def capturing_sink(msg, _real=real_sink):
                            for c in self._warn_captures:
                                c.append(msg)
                            _real(msg)
                        self._sink = capturing_sink
                in_hull = node.kind == "hull" and node.is_builtin
                self._hull_depth += in_hull
                try:
                    children_bodies = self.generate_tree(node.children)
                    generate_fn = self._GENERATE_DISPATCH.get(node.kind) if node.is_builtin else None
                    if generate_fn is not None:
                        saved = self._generate_warn_entry, self._generate_call_chain
                        self._generate_warn_entry, self._generate_call_chain = node.warn_entry, node.call_chain
                        try:
                            node.bodies = generate_fn(node.params, node.children, node.node)
                        finally:
                            self._generate_warn_entry, self._generate_call_chain = saved
                        node.bodies = self._tag_sections(node.bodies, node.node, node.call_chain)
                    else:
                        node.bodies = children_bodies
                finally:
                    self._hull_depth -= in_hull
                    if key is not None:
                        warnings = self._warn_captures.pop()
                        if not self._warn_captures:
                            self._sink = real_sink
                if key is not None:
                    self._manifold_cache.put(key, (node.bodies, warnings))
                    self._cache_producer[key] = node.node
            result.extend(node.bodies)
        return result

    def _run_file_globals(self, root) -> None:
        """A used file's globals: all of them, once per run, in source order,
        on the first read of any -- as OpenSCAD and the C++ port run them, so
        an echo in one it never reads still happens, and a read of a later
        one from an earlier one's initializer is undef. (OpenSCAD re-runs
        them on every call into the file, a known upstream bug not copied.)"""
        self._files_run.add(id(root))
        fctx = self._root_ctx.call_ctx(scope=root)
        for name, a in list(root.variables.items()):
            if type(a) is not Assignment:
                continue
            if self._coverage:
                self._cov_hit(a)  # they run here, never as statements (cpp #170)
            v = self._eval_expr(a.expr, fctx)
            fctx.let[name] = v
            self._global_values[id(a)] = v

    @staticmethod
    def _root_scope_of(scope):
        while scope.parent is not None:
            scope = scope.parent
        return scope

    def _cov_hit(self, node) -> None:
        k = id(node)
        self._cov_hits[k] = self._cov_hits.get(k, 0) + 1

    def _coverage_universe(self, nodes, root_scope) -> tuple[list, list]:
        """(roots, used-file globals) for the coverage walk: the run's own
        top-level statements, every declaration of every file it use<>s --
        shadowed or not, as the C++ port reports them -- and those files'
        own global assignments."""
        roots = [n for n in nodes if type(n) is not UseStatement]
        extra, seen_files = [], set()
        for node in nodes:
            if type(node) is not UseStatement:
                continue
            origin = getattr(node.position, "origin", "") if node.position else ""
            lib = findLibraryFile(origin, node.filepath.val)
            if lib is None or lib in seen_files:
                continue
            seen_files.add(lib)
            for n in getASTfromFile(lib) or []:
                if type(n) in (ModuleDeclaration, FunctionDeclaration):
                    roots.append(n)
                elif type(n) is Assignment:
                    extra.append(n)
        # Declarations use<> injected are the very nodes this run executed,
        # so prefer them over the fresh parse above: same span, but these
        # carry the hits. build_result dedupes by position.
        for decl in list(root_scope.modules.values()) + list(root_scope.functions.values()):
            roots.insert(0, decl)
        return roots, extra

    def _current_call_chain(self) -> tuple:
        return tuple((e[2], e[0] == "module") for e in reversed(self._call_stack))

    def _tag_sections(self, bodies: list[ColoredBody], node, chain: tuple = ()) -> list[ColoredBody]:
        """Give each new 2D section an originalID attributed to `node`; a
        transformed one keeps its own, since replace() carries it."""
        out = bodies
        for i, cb in enumerate(bodies):
            if cb.section is not None and cb.body is None and cb.section_id is None:
                sid = int(m3d.Manifold.reserve_ids(1))
                self.id_to_node[sid] = node
                self.id_to_call_chain[sid] = chain
                self.id_to_color[sid] = cb.color
                if out is bodies:
                    out = list(bodies)
                out[i] = replace(cb, section_id=sid)
        return out

    def _restamp_cached_ids(self, bodies: list[ColoredBody], node, producer, chain: tuple = ()) -> list[ColoredBody]:
        """Cached bodies with fresh originalIDs. A hit hands back the IDs of
        whichever call site first made the shape, and IDs are provenance,
        not content: two identical cylinders came back as one thing to
        select (#84). One fresh ID per run, so a cached multi-part subtree
        stays selectable part by part. An ID that stood for the reused node
        itself (`producer`) now belongs to the node reusing it; one standing
        for something deeper -- the cube inside a module called twice --
        keeps its node, which is where that part is really spelled out.
        IDs from an earlier render are unknown here and all go to `node`,
        or a re-render of unchanged source had nothing pickable (#85).
        ponytail: rebuilds each body from its mesh; 200 identical spheres
        still render faster warm than cold in the C++ port."""
        out = []
        for cb in bodies:
            if cb.section_id is not None:
                # A 2D body's one ID, by the same rule as a solid's runs.
                new = int(m3d.Manifold.reserve_ids(1))
                was = self.id_to_node.get(cb.section_id)
                self.id_to_node[new] = was if was is not None and was is not producer else node
                self.id_to_call_chain[new] = chain  # the site REUSING it: two calls are two sites
                self.id_to_color[new] = cb.color
                out.append(replace(cb, section_id=new))
                continue
            if cb.body is None or cb.body.is_empty():
                out.append(cb)
                continue
            mesh = cb.body.to_mesh64()
            ids = np.asarray(mesh.run_original_id, dtype=np.uint32)
            if not len(ids):
                out.append(cb)
                continue

            def inherit(old, new):
                was = self.id_to_node.get(old)
                self.id_to_node[new] = was if was is not None and was is not producer else node
                self.id_to_call_chain[new] = chain
                if old in self.id_to_color:
                    self.id_to_color[new] = self.id_to_color[old]

            if len(set(ids.tolist())) == 1:
                # One run: as_original() relabels it without rebuilding the
                # mesh, a tenth of the cost -- which a warm render of many
                # identical parts otherwise spends entirely here.
                body = cb.body.as_original()
                inherit(int(ids[0]), body.original_id())
                out.append(replace(cb, body=body, merged_from=self._remap_parts(
                    cb.merged_from, {int(ids[0]): body.original_id()})))
                continue
            fresh = {}
            for old in dict.fromkeys(int(i) for i in ids):
                fresh[old] = int(m3d.Manifold.reserve_ids(1))
                inherit(old, fresh[old])
            out.append(replace(cb, body=self._with_run_ids(mesh, [fresh[int(i)] for i in ids]),
                               merged_from=self._remap_parts(cb.merged_from, fresh)))
        return out

    def _remap_parts(self, parts, remap: dict):
        """A union's unmerged parts share its IDs, so a restamp relabels them
        too -- a part left on the old IDs would look uncoloured beside it."""
        if not parts:
            return parts
        out = []
        for p in parts:
            body = p.body
            if body is not None and not body.is_empty():
                mesh = body.to_mesh64()
                body = self._with_run_ids(mesh, [remap.get(int(i), int(i)) for i in mesh.run_original_id])
            out.append(replace(p, body=body, merged_from=self._remap_parts(p.merged_from, remap)))
        return tuple(out)

    def _eval_statement_impl(self, node: ASTNode, ctx: EvalContext) -> list[ColoredBody]:
        self._last_ctx = ctx
        t = type(node)
        if t is not ModuleDeclaration and t is not FunctionDeclaration and t is not ModularLet:
            if self._debugging:
                self._check_debug(node, ctx)
        if t is Assignment:
            name = node.name.name
            if name[0] == '$':
                ctx.dyn[name] = self._eval_expr(node.expr, ctx)
                ctx.dyn_explicit.add(name)
            else:
                pos = getattr(node, 'position', None)
                if name in ctx.dyn_positions:
                    first_pos = ctx.dyn_positions[name]
                    first_line = getattr(first_pos, 'line', '?') if first_pos else '?'
                    self._echo_fn(
                        f"WARNING: {name} was assigned on line {first_line}"
                        f" but was overwritten{self._loc(pos)}"
                    )
                ctx.let[name] = self._eval_expr(node.expr, ctx)
                ctx.dyn_positions[name] = pos
            return []
        if t is ModularIf:
            cond = self._eval_expr(node.condition, ctx)
            if cond:
                branch = node.true_branch
                if self._debugging:
                    self._check_debug(branch[0] if branch else node, ctx, expr_level=True)
                result = self._eval_children(branch, ctx)
                self._if_taken = True  # after the branch, whose own ifs set it too
                return result
            self._if_taken = False
            return []
        if t is ModularIfElse:
            cond = self._eval_expr(node.condition, ctx)
            branch = node.true_branch if cond else node.false_branch
            if self._debugging:
                self._check_debug(branch[0] if branch else node, ctx, expr_level=True)
            return self._eval_children(branch, ctx)
        if t is ModularFor:
            return self._eval_for(node, ctx)
        if t is ModularLet:
            return self._eval_let_block(node, ctx)
        if t is ModularEcho:
            self._do_echo(node.arguments, ctx)
            if node.children:  # echo("x") cube(1); still draws the cube
                return self._eval_children(node.children, ctx)
            return []
        if t is ModularAssert:
            args = self._resolve_args(node.arguments, ctx)
            cond = self._get_arg(args, 0, "condition", True)
            if not cond:
                raw = node.arguments
                cond_text = to_openscad([raw[0].expr]).strip() if raw else "false"
                msg = self._get_arg(args, 1, "message", None)
                err = f"Assertion '{cond_text}' failed" + (f': "{msg}"' if msg is not None else "")
                self.error(err, node, innermost_frame="assert")
                return []
            # Assertion passed — propagate any chained child geometry (e.g. assert(...) translate(...) children())
            if node.children:
                return self._eval_children(node.children, ctx)
            return []
        if isinstance(node, ModularModifierDisable):  # * — fully excluded
            return []
        if isinstance(node, (ModuleDeclaration, FunctionDeclaration)):
            return []
        return []

    @staticmethod
    def _block_ctx(children, ctx: EvalContext) -> EvalContext:
        """A braced block is its own scope: `if (c) { x = 2; }` must not
        leave x (or a `$fn = ...`) behind in the enclosing one. Copies only
        what the block's own assignments would write, and only if it has
        any -- most blocks have none."""
        names = [c.name.name for c in children if type(c) is Assignment]
        if not names:
            return ctx
        dollar = any(n[0] == '$' for n in names)
        return EvalContext(
            scope=ctx.scope,
            dyn=dict(ctx.dyn) if dollar else ctx.dyn,
            let=dict(ctx.let),
            dyn_positions={},
            dyn_explicit=set(ctx.dyn_explicit) if dollar else ctx.dyn_explicit,
            color=ctx.color,
            children_nodes=ctx.children_nodes,
            children_caller_ctx=ctx.children_caller_ctx,
        )

    def _eval_children(self, children, ctx: EvalContext, new_scope: bool = True) -> list[ColoredBody]:
        result = []
        if new_scope:
            ctx = self._block_ctx(children, ctx)
        # OpenSCAD executes all assignments before geometry in each scope.
        assignments = [c for c in children if isinstance(c, Assignment)]
        others = [c for c in children if not isinstance(c, Assignment)]
        for child in assignments + others:
            # Use the node's own scope from build_scopes when available so that
            # each node evaluates in its correct lexical scope. Share ctx.dyn
            # (not a copy) so that eager assignments in one sibling are visible
            # to subsequent siblings in the same block.
            child_scope = getattr(child, 'scope', None)
            if child_scope is not None:
                child_ctx = EvalContext(
                    scope=child_scope,
                    dyn=ctx.dyn,
                    let=ctx.let,
                    dyn_positions=ctx.dyn_positions,
                    dyn_explicit=ctx.dyn_explicit,
                    color=ctx.color,
                    children_nodes=ctx.children_nodes,
                    children_caller_ctx=ctx.children_caller_ctx,
                )
            else:
                child_ctx = ctx
            result.extend(self._eval_statement(child, child_ctx))
        return result

    # ------------------------------------------------------------------
    # Module call dispatch
    # ------------------------------------------------------------------

    def _eval_modular_call(self, node: ModularCall, ctx: EvalContext) -> list[ColoredBody]:
        name = node.name.name
        user_mod = ctx.scope.lookup_module(name)
        if user_mod is not None:
            return self._eval_user_module(user_mod, node, ctx)
        return self._eval_builtin(name, node, ctx)

    @staticmethod
    def _body_list(body: Optional[ColoredBody]) -> list[ColoredBody]:
        return [body] if body is not None else []

    @staticmethod
    def _pos_contains(outer, inner) -> bool:
        """True if `inner`'s source span is strictly contained within `outer`'s.

        Used to detect "`inner` is declared lexically inside `outer`'s body"
        (e.g. a nested `module`/`function`). Identical spans (a declaration
        calling itself — direct recursion) are NOT considered contained.
        """
        if outer is None or inner is None:
            return False
        if outer.origin != inner.origin:
            return False
        if (outer.start_offset, outer.end_offset) == (inner.start_offset, inner.end_offset):
            return False
        return outer.start_offset <= inner.start_offset and inner.end_offset <= outer.end_offset

    def _call_ctx_for(self, decl, ctx: EvalContext, scope=None,
                      children_nodes=None, children_caller_ctx=None, share_dyn=False) -> EvalContext:
        call_stack = self._call_stack
        if call_stack:
            decl_pos = decl.position
            if decl_pos is not None:
                dp_origin = decl_pos.origin
                dp_start = decl_pos.start_offset
                dp_end = decl_pos.end_offset
                for frame in call_stack:
                    outer = frame[-1]
                    if outer is not None and outer.origin == dp_origin:
                        o_start, o_end = outer.start_offset, outer.end_offset
                        if (o_start, o_end) != (dp_start, dp_end) and o_start <= dp_start and dp_end <= o_end:
                            return ctx.child_ctx(scope=scope, children_nodes=children_nodes,
                                                 children_caller_ctx=children_caller_ctx, share_dyn=share_dyn)
        return ctx.call_ctx(scope=scope, children_nodes=children_nodes,
                            children_caller_ctx=children_caller_ctx, share_dyn=share_dyn)

    def _profile_enter(self, kind: str, name: str, call_pos, decl_pos):
        """Push profiling state for a user module/function call about to
        start -- shared by _eval_user_module/_eval_user_function/
        _eval_function_literal's `if self._profiling:` blocks so the
        timing/aggregation logic lives in one place, not copy-pasted
        across all 3 call-stack push/pop sites. Returns a tuple to hand
        back to _profile_exit on the matching pop."""
        call_origin = getattr(call_pos, 'origin', None) or ''
        call_line = getattr(call_pos, 'line', 0) if call_pos else 0
        call_column = getattr(call_pos, 'column', 0) if call_pos else 0
        site_key = (kind, name, call_origin, call_line, call_column)
        site = self._profile_sites.get(site_key)
        if site is None:
            # self._call_stack still has the caller on top -- _profile_enter
            # always runs before this call's own frame is pushed. A given
            # call site is always lexically inside the same enclosing
            # module/function body regardless of which invocation this is,
            # so the caller name is a one-time-computed, structural property
            # of the site, same as decl_origin/decl_line.
            caller_name = self._call_stack[-1][1] if self._call_stack else "<toplevel>"
            site = CallSiteProfile(
                kind=kind, name=name, caller_name=caller_name,
                call_origin=call_origin, call_line=call_line, call_column=call_column,
                decl_origin=getattr(decl_pos, 'origin', None) or '',
                decl_line=getattr(decl_pos, 'line', 0) if decl_pos else 0,
            )
            self._profile_sites[site_key] = site
        site.call_count += 1
        recursive_reentry = site_key in self._profile_active
        if not recursive_reentry:
            self._profile_active.add(site_key)
        self._profile_child_time.append(0.0)
        prev_path = self._profile_cur
        node = self._profile_path_enter(kind, name, call_origin, call_line, call_column, decl_pos)
        self._profile_paths[node]["call_count"] += 1
        self._profile_cur = node
        return site, site_key, recursive_reentry, time.perf_counter(), node, prev_path

    _MAX_PROFILE_PATH_NODES = 200_000

    def _profile_path_enter(self, kind, name, call_origin, call_line, call_column, decl_pos) -> int:
        """This call's node under the current one. Only DIRECT self-recursion
        folds onto its parent -- folding a re-entry onto any ancestor
        credited the whole subtree to it and emptied every node between,
        so the tree disagreed with call_sites (cpp 709da93). Past the node
        cap the tree stops subdividing and folds onto the parent."""
        paths = self._profile_paths
        parent = self._profile_cur
        ident = (kind, name, call_origin, call_line, call_column)

        def same(n):
            return (n["kind"], n["name"], n["call_origin"], n["call_line"], n["call_column"]) == ident
        if parent > 0 and same(paths[parent]):
            return parent
        for c in paths[parent]["children"]:
            if same(paths[c]):
                return c
        if len(paths) >= self._MAX_PROFILE_PATH_NODES:
            return parent
        paths.append({"parent": parent, "children": [], "kind": kind, "name": name, "call_origin": call_origin,
                      "call_line": call_line, "call_column": call_column,
                      "decl_origin": getattr(decl_pos, "origin", None) or "",
                      "decl_line": getattr(decl_pos, "line", 0) if decl_pos else 0,
                      "call_count": 0, "self_time": 0.0, "cumulative_time": 0.0})
        paths[parent]["children"].append(len(paths) - 1)
        return len(paths) - 1

    def _profile_exit(self, site: "CallSiteProfile", site_key: tuple, recursive_reentry: bool, t_start: float,
                      node: int = -1, prev_path: int = 0):
        """Pop profiling state on the matching call-stack pop -- see
        _profile_enter. Self time is unconditional (disjoint wall-clock
        slices, never overlapping, so nothing to guard). Cumulative time
        is skipped on a recursive re-entry: the outer invocation's own
        elapsed already includes it, via the child-time propagation to
        the parent frame below -- without this guard a self-recursive
        call site's cumulative_time would balloon past total wall time."""
        elapsed = time.perf_counter() - t_start
        child_time = self._profile_child_time.pop()
        site.self_time += elapsed - child_time
        if not recursive_reentry:
            site.cumulative_time += elapsed
            self._profile_active.discard(site_key)
        if node >= 0:
            # Self time only; cumulative is derived from the subtree at the
            # end, since at a fold only the outermost entry may add elapsed.
            self._profile_paths[node]["self_time"] += elapsed - child_time
            self._profile_cur = prev_path
        if self._profile_child_time:
            self._profile_child_time[-1] += elapsed

    def _eval_user_module(self, decl: ModuleDeclaration, call: ModularCall, ctx: EvalContext) -> list[ColoredBody]:
        if self._coverage:
            self._cov_hit(decl)
        # Bind parameters
        child_scope = getattr(decl, 'scope', None) or ctx.scope
        params = getattr(decl, 'parameters', None) or []
        args = self._bind_args(params, call.arguments, ctx, call)

        # Expanded, so a children(separate=true) in the block is one real
        # statement per child it forwards: $children counts them, and
        # children(i) indexes them.
        children_nodes = self._expand_child_statements(call.children, ctx)
        child_ctx = self._call_ctx_for(
            decl, ctx,
            scope=child_scope,
            children_nodes=children_nodes,
            children_caller_ctx=ctx,
        )
        # $children is the number of module-instantiation children passed in
        # `{}`, not the number of geometries they produced — e.g. `children()`
        # counts as one child even if the caller passed it none to forward.
        child_ctx.dyn["$children"] = len([
            c for c in children_nodes
            if not isinstance(c, (Assignment, ModuleDeclaration, FunctionDeclaration))
        ])
        for k, v in args.items():
            if k[0] == '$':
                child_ctx.dyn[k] = v
            else:
                child_ctx.let[k] = v
        # Apply defaults for missing params
        self._apply_defaults(params, args, child_ctx)

        name = call.name.name
        call_pos = getattr(call, 'position', None)
        decl_pos = getattr(decl, 'position', None)
        # Counting this module too: 1 in a module called from top level, as in OpenSCAD.
        child_ctx.dyn["$parent_modules"] = 1 + sum(1 for e in self._call_stack if e[0] == "module")
        # A module handed in through children() profiles as "child": both
        # `module foo() foo();` and `foo() foo();` make a foo->foo edge, and
        # only the first is recursion. Inside the body it is off again.
        via_children, self._via_children = self._via_children, False
        prof = self._profile_enter("child" if via_children else "module", name, call_pos, decl_pos) \
            if self._profiling else None
        self._call_stack.append(("module", name, call_pos, decl_pos))
        self._frame_ctxs.append(child_ctx)
        try:
            module_body = getattr(decl, 'children', None) or getattr(decl, 'body', None) or []
            return self._eval_children(module_body, child_ctx, new_scope=False)  # child_ctx is already fresh
        finally:
            self._call_stack.pop()
            self._frame_ctxs.pop()
            self._via_children = via_children
            if prof is not None:
                self._profile_exit(*prof)

    def _bind_args(self, params, arguments, ctx: EvalContext, call=None) -> dict[str, Any]:
        result = {}
        positional_idx = 0
        nparams = len(params)
        _eval = self._eval_expr
        suspicious = False
        for arg in arguments:
            if type(arg) is NamedArgument:
                result[arg.name.name] = _eval(arg.expr, ctx)
                suspicious = True  # checked below, off the common path
            else:
                if positional_idx < nparams:
                    result[params[positional_idx].name.name] = _eval(arg.expr, ctx)
                positional_idx += 1
        if positional_idx > nparams or suspicious:
            names = self._param_names.get(id(params))
            if names is None:
                names = self._param_names[id(params)] = tuple(p.name.name for p in params)
            self._warn_unexpected_args(names, arguments, call)
        return result

    # Builtin modules' parameters, as OpenSCAD declares them, for the same
    # warnings a user module gives. Builtin FUNCTIONS read their arguments
    # positionally and never warn (`sin(bogus=30)` is sin(30)), except these two.
    _BUILTIN_PARAMS = {
        "cube": ("size", "center"),
        "sphere": ("r", "d", "style"),
        "cylinder": ("h", "r1", "r2", "center", "r", "d", "d1", "d2"),
        "polyhedron": ("points", "faces", "convexity", "triangles"),
        "square": ("size", "center"),
        "circle": ("r", "d"),
        "polygon": ("points", "paths", "convexity"),
        "translate": ("v",), "rotate": ("a", "v"), "scale": ("v",), "mirror": ("v",),
        "multmatrix": ("m",), "resize": ("newsize", "auto", "convexity"),
        "color": ("c", "alpha"),
        "union": (), "difference": (), "intersection": (), "hull": (), "fill": (),
        "minkowski": ("convexity",), "minkowski_difference": (), "simplify": ("tolerance",), "mesh_repair": ("tolerance",),
        "levelset": ("field", "bounds", "isovalue", "invert", "edge"),
        "children": ("index", "separate"), "render": ("convexity",),
        "import": ("file", "layer", "convexity", "origin", "scale", "width", "height",
                   "filename", "layername", "center", "dpi", "id", "class", "repair", "tolerance"),
        "linear_extrude": ("height", "v", "scale", "center", "twist", "slices", "segments", "convexity"),
        "rotate_extrude": ("angle", "start", "convexity"),
        "projection": ("cut", "convexity"),
        "roof": ("method", "convexity"),
        "offset": ("r", "delta", "chamfer"),
        "surface": ("file", "center", "convexity", "invert"),
        "text": ("text", "size", "font", "direction", "language", "script", "halign", "valign", "spacing"),
        "breakpoint": ("condition",),  # this package's debugger extension
        "textmetrics": ("text", "size", "font", "direction", "language", "script", "halign", "valign", "spacing"),
        "fontmetrics": ("size", "font"),
        "linear_solve": ("A", "b"),
    }

    def _warn_unexpected_args(self, declared: tuple, arguments, call, builtin: bool = False) -> None:
        """OpenSCAD's warnings for arguments a callee doesn't declare, which
        were silently dropped -- a misspelt argument name went unnoticed,
        the parameter keeping its default. $-names pass (they set a dynamic
        variable), except $children, which is not settable that way."""
        loc = self._loc(getattr(call, "position", None))
        positional = 0
        seen = set()
        for arg in arguments:
            if type(arg) is not NamedArgument:
                positional += 1
        if positional > len(declared):
            self._echo_fn(f"WARNING: Too many unnamed arguments supplied{loc}")
        for arg in arguments:
            if type(arg) is not NamedArgument:
                continue
            name = arg.name.name
            if builtin:
                pass  # OpenSCAD's builtins word these their own way, per builtin
            elif name in seen:
                self._echo_fn(f'WARNING: argument "{name}" supplied more than once{loc}')
            elif name in declared[:positional]:
                self._echo_fn(f'WARNING: argument "{name}" overrides positional argument{loc}')
            seen.add(name)
            if name not in declared and (name[0] != "$" or name == "$children"):
                self._echo_fn(f'WARNING: variable "{name}" not specified as parameter{loc}')

    # ------------------------------------------------------------------
    # Built-in modules
    # ------------------------------------------------------------------

    def _resolve_call_args(self, node: ModularCall, ctx: EvalContext) -> tuple[dict, EvalContext]:
        """Resolve a ModularCall's arguments and apply any $-prefixed
        named-arg dynamic-context overrides (e.g. sphere(r=2, $fn=64)).
        Shared by _eval_builtin (for not-yet-migrated builtins) and every
        migrated _resolve_* method (Phase 2), which bypass _eval_builtin's
        dispatch entirely and so need this logic themselves."""
        args = self._resolve_args(node.arguments, ctx)
        declared = self._BUILTIN_PARAMS.get(node.name.name) if type(node) is ModularCall else None
        if declared is not None:
            self._warn_unexpected_args(declared, node.arguments, node, builtin=True)
        dyn_overrides = {k: v for k, v in args.items() if isinstance(k, str) and k.startswith("$")}
        if dyn_overrides:
            ctx = ctx.child_ctx(dyn={**ctx.dyn, **dyn_overrides})
        return args, ctx

    def _eval_builtin(self, name: str, node: ModularCall, ctx: EvalContext) -> list[ColoredBody]:
        args, ctx = self._resolve_call_args(node, ctx)

        if name == "echo":
            self._do_echo(node.arguments, ctx)
            return []
        if name == "assert":
            return []
        # Unknown module — warn with call stack, matching OpenSCAD's WARNING format
        pos = getattr(node, 'position', None)
        warn = f"WARNING: Ignoring unknown module '{name}'{self._loc(pos)}"
        trace = self._trace_lines(node)
        self._echo_fn("\n".join([warn] + trace))
        return []

    def _resolve_render(self, node: ModularCall, ctx: EvalContext) -> dict:
        # render() is a display hint; just pass through children — no
        # generate_fn is registered, so generate_tree()'s default
        # (concatenate children's bodies) reproduces the passthrough.
        args, ctx = self._resolve_call_args(node, ctx)
        self._eval_children(node.children, ctx)  # side effect only, see _resolve_transform
        return {}

    def _resolve_children_call(self, node: ModularCall, ctx: EvalContext) -> dict:
        args, ctx = self._resolve_call_args(node, ctx)
        self._builtin_children(args, ctx, node)  # side effect only; return (real bodies) unused now
        return {}

    def _resolve_breakpoint(self, node: ModularCall, ctx: EvalContext) -> dict:
        args, ctx = self._resolve_call_args(node, ctx)
        self._builtin_breakpoint(args, node, ctx)  # side effect only (debug hook); never had children
        return {}

    def _resolve_modifier_child(self, node, ctx: EvalContext) -> dict:
        """Shared resolve for the #/%/! modifiers (ModularModifierHighlight/
        Background/ShowOnly), which each wrap exactly one child (node.child,
        not a ModularCall's node.children list). Builds the child into the
        tree for its side effect only; the actual role tagging happens in
        the matching _generate_highlight/_generate_background/_generate_show_only."""
        self._eval_statement(node.child, ctx)
        return {}

    def _generate_highlight(self, params: dict, children: list[CSGNode], node: ASTNode) -> list[ColoredBody]:
        return [replace(b, role="highlight") for b in flatten_csg_tree(children)]

    def _generate_background(self, params: dict, children: list[CSGNode], node: ASTNode) -> list[ColoredBody]:
        return [replace(b, role="background") for b in flatten_csg_tree(children)]

    def _generate_show_only(self, params: dict, children: list[CSGNode], node: ASTNode) -> list[ColoredBody]:
        # Being the root is all `!` does -- see _show_only_root.
        return flatten_csg_tree(children)

    def _resolve_args(self, arguments, ctx: EvalContext) -> dict:
        result = {}
        pos = 0
        _eval = self._eval_expr
        for arg in arguments:
            if type(arg) is PositionalArgument:
                result[pos] = _eval(arg.expr, ctx)
                pos += 1
            else:
                result[arg.name.name] = _eval(arg.expr, ctx)
        return result

    def _get_arg(self, args: dict, pos: int, name: str, default=None):
        if name in args:
            return args[name]
        if pos in args:
            return args[pos]
        return default

    # --- primitives ---

    def _tag(self, body: m3d.Manifold, node: ASTNode, ctx: EvalContext) -> ColoredBody:
        chain = self._current_call_chain()
        for orig_id in body.to_mesh().run_original_id:
            self.id_to_node[int(orig_id)] = node
            self.id_to_call_chain[int(orig_id)] = chain
            self.id_to_color[int(orig_id)] = ctx.color
        return ColoredBody(body=body, color=ctx.color)

    def _tag_generated(self, body: m3d.Manifold, node: ASTNode, color) -> ColoredBody:
        """Generate-phase equivalent of _tag(): takes an already-resolved
        color instead of ctx, since ctx isn't available once a builtin has
        been migrated to the resolve/generate split (Phase 2)."""
        for orig_id in body.to_mesh().run_original_id:
            self.id_to_node[int(orig_id)] = node
            self.id_to_call_chain[int(orig_id)] = self._generate_call_chain
            self.id_to_color[int(orig_id)] = color
        return ColoredBody(body=body, color=color)

    def _fn(self, ctx: EvalContext, r: float = 0.0) -> int:
        return self._fn_segments(ctx.dyn.get("$fn", 0), ctx.dyn.get("$fa", 12.0),
                                  ctx.dyn.get("$fs", 2.0), r)

    @staticmethod
    def _fn_segments(fn, fa, fs, r: float = 0.0) -> int:
        """Pure segment-count formula, split out of _fn() so generate-phase
        code (e.g. rotate_extrude, which needs the merged children's bounds
        — unavailable until generate — for its radius) can compute segments
        from cached $fn/$fa/$fs values without a live ctx."""
        if isinstance(fn, (int, float)) and fn > 0:
            return max(3, int(fn))
        if not isinstance(fa, (int, float)) or fa <= 0:
            fa = 12.0
        if not isinstance(fs, (int, float)) or fs <= 0:
            fs = 2.0
        r = abs(r) if isinstance(r, (int, float)) and math.isfinite(r) else 0.0
        return int(math.ceil(max(5, min(360.0 / fa, r * 2.0 * math.pi / fs))))

    # --- cube (resolve/generate — Phase 2) ---

    def _resolve_cube(self, node: ModularCall, ctx: EvalContext) -> dict:
        args, ctx = self._resolve_call_args(node, ctx)
        size = self._get_arg(args, 0, "size", None)
        center = bool(self._get_arg(args, 1, "center", False))
        # As OpenSCAD: a number or three numbers; undef is the default; any
        # other size warns and falls back to 1. It raised a TypeError here.
        def num(v):
            return type(v) in (int, float)
        if size is None:
            size = [1.0] * 3
        elif num(size):
            size = [float(size)] * 3
        elif type(size) is list and len(size) == 3 and all(num(v) for v in size):
            size = [float(v) for v in size]
        else:
            self._echo_fn(f"WARNING: Unable to convert cube(size={self._fmt_val(size)}, ...) parameter to a "
                          f"number or a vec3 of numbers{self._loc(getattr(node, 'position', None))}")
            size = [1.0] * 3
        return {"size": size, "center": center, "color": ctx.color}

    def _generate_cube(self, params: dict, children: list[CSGNode], node: ASTNode) -> list[ColoredBody]:
        body = m3d.Manifold.cube(params["size"], params["center"])
        return [self._tag_generated(body, node, params["color"])]

    # --- sphere (resolve/generate — Phase 2) ---

    def _resolve_sphere(self, node: ModularCall, ctx: EvalContext) -> dict:
        args, ctx = self._resolve_call_args(node, ctx)
        r = self._get_arg(args, 0, "r", None)
        d = self._get_arg(args, None, "d", None)
        if d is not None:
            r = d / 2
        if r is None:
            r = 1.0
        r = float(r)
        n = self._fn(ctx, r)  # longitude segments
        stacks = max(2, int(math.ceil(n / 2)))  # number of latitude rings (no single-point poles)

        # style= names the tessellation, BOSL2 spheroid()'s five (cpp #102).
        # Only "orig", the default, is what OpenSCAD builds.
        style = self._get_arg(args, None, "style", None)
        if isinstance(style, str) and style not in _SPHERE_STYLES:
            self._echo_fn(f'WARNING: sphere: unknown style "{style}"; expected one of '
                          f'orig, aligned, stagger, octa, icosa{self._loc(getattr(node, "position", None))}')
        elif style is not None and not isinstance(style, str):
            self._echo_fn(f"WARNING: sphere: style must be a string{self._loc(getattr(node, 'position', None))}")
        if style in ("aligned", "stagger", "icosa", "octa"):
            if style == "octa":  # Manifold's sphere IS a subdivided octahedron
                verts, tris = np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint64)
            elif style == "icosa":
                verts, tris = _sphere_icosa(r, n)
            else:
                verts, tris = _sphere_aligned(r, n, stacks, style == "stagger")
            return {"r": r, "segs": n, "style": style, "color": ctx.color,
                    "verts": np.array(verts, dtype=np.float64), "tris": np.array(tris, dtype=np.uint64)}

        # OpenSCAD-compatible sphere: polygon caps at top/bottom (no triangulated poles),
        # quad belts between rings. Rings evenly spaced excluding the actual poles.
        step = math.pi / stacks  # latitude step in radians
        verts = []
        rings = []  # rings[i] = list of vertex indices

        for s in range(stacks):
            lat = -math.pi / 2 + (s + 0.5) * step
            ring_r = r * math.cos(lat)
            z = r * math.sin(lat)
            ring = []
            for seg in range(n):
                angle = 2 * math.pi * seg / n
                ring.append(len(verts))
                verts.append([ring_r * math.cos(angle), ring_r * math.sin(angle), z])
            rings.append(ring)

        tris = []

        # Bottom polygon cap: fan with reversed winding → outward normal points down
        bot = rings[0]
        for i in range(1, n - 1):
            tris.append([bot[0], bot[i + 1], bot[i]])

        # Quad belts between adjacent rings
        for s in range(stacks - 1):
            lo, hi = rings[s], rings[s + 1]
            for seg in range(n):
                a, b = lo[seg], lo[(seg + 1) % n]
                c, d_ = hi[seg], hi[(seg + 1) % n]
                tris.append([a, b, d_])
                tris.append([a, d_, c])

        # Top polygon cap: forward-winding fan → outward normal points up
        top = rings[-1]
        for i in range(1, n - 1):
            tris.append([top[0], top[i], top[i + 1]])

        verts_arr = np.array(verts, dtype=np.float64)
        tris_arr = np.array(tris, dtype=np.uint64)
        return {"r": r, "segs": n, "verts": verts_arr, "tris": tris_arr, "color": ctx.color}

    def _generate_sphere(self, params: dict, children: list[CSGNode], node: ASTNode) -> list[ColoredBody]:
        if params.get("style") == "octa":
            return [self._tag_generated(m3d.Manifold.sphere(params["r"], params["segs"]), node, params["color"])]
        mesh = m3d.Mesh64(vert_properties=params["verts"], tri_verts=params["tris"])
        body = m3d.Manifold(mesh)
        return [self._tag_generated(body, node, params["color"])]

    # --- cylinder (resolve/generate — Phase 2) ---

    def _resolve_cylinder(self, node: ModularCall, ctx: EvalContext) -> dict:
        args, ctx = self._resolve_call_args(node, ctx)
        h = float(self._get_arg(args, 0, "h", 1.0))
        # Positional order is (h, r1, r2, center); r/d/d1/d2 are named only,
        # so cylinder(10, 5, 2) is a cone. Applied in OpenSCAD's order, each
        # overriding the last: r, d, r1, r2, d1, d2 -- d beats r, and each
        # end defaults to 1 on its own. Only numbers count; a stray `true`
        # in a radius slot is ignored rather than read as 1.
        def num(v):
            return type(v) in (int, float)
        r1 = r2 = 1.0
        r = self._get_arg(args, None, "r")
        d = self._get_arg(args, None, "d")
        if num(r):
            r1 = r2 = r
        if num(d):
            r1 = r2 = d / 2
        v = self._get_arg(args, 1, "r1")
        if num(v):
            r1 = v
        v = self._get_arg(args, 2, "r2")
        if num(v):
            r2 = v
        v = self._get_arg(args, None, "d1")
        if num(v):
            r1 = v / 2
        v = self._get_arg(args, None, "d2")
        if num(v):
            r2 = v / 2
        center = bool(self._get_arg(args, 3, "center", False))
        segs = self._fn(ctx, max(float(r1), float(r2)))

        return {"h": h, "r1": float(r1), "r2": float(r2), "center": center,
                "segs": segs, "color": ctx.color}

    def _generate_cylinder(self, params: dict, children: list[CSGNode], node: ASTNode) -> list[ColoredBody]:
        body = m3d.Manifold.cylinder(params["h"], params["r1"], params["r2"],
                                      circular_segments=params["segs"], center=params["center"])
        return [self._tag_generated(body, node, params["color"])]

    # --- transforms ---

    def _resolve_transform(self, node: ModularCall, ctx: EvalContext) -> dict:
        name = node.name.name
        args, ctx = self._resolve_call_args(node, ctx)
        # Evaluate children for the side effect of building their CSGNodes
        # (pushed onto self._tree_stack) — the returned bodies themselves
        # are discarded here; generate reads them back via each child
        # CSGNode's own .bodies, which is correct regardless of whether a
        # given child is itself migrated or still eager.
        self._eval_children(node.children, ctx)
        return {"name": name, "args": args}

    def _one_dimension(self, bodies: list[ColoredBody], node) -> list[ColoredBody]:
        """What a node combining its children keeps of them: the dimension of
        the first, as OpenSCAD renders it, dropping the other with its two
        warnings. `union() { cube(2); square(3); }` crashed (None +
        CrossSection); OpenSCAD keeps the cube. Used by the booleans, the
        transforms and color(); the top level keeps both."""
        dims = [b.section is not None for b in bodies
                if b.body is not None or b.section is not None or b.raw_mesh is not None]
        if not dims or all(d == dims[0] for d in dims):
            return bodies
        keep_2d = dims[0]
        loc = self._loc(getattr(node, "position", None))
        kept, dropped = ("2D", "3D") if keep_2d else ("3D", "2D")
        self._echo_fn(f"WARNING: Mixing 2D and 3D objects is not supported{loc}")
        self._echo_fn(f"WARNING: Ignoring {dropped} child object for {kept} operation{loc}")
        return [b for b in bodies if (b.section is not None) == keep_2d
                or (b.body is None and b.section is None and b.raw_mesh is None)]

    def _generate_transform(self, params: dict, children: list[CSGNode], node: ASTNode) -> list[ColoredBody]:
        name, args = params["name"], params["args"]
        result = []
        for b in self._one_dimension(flatten_csg_tree(children), node):
            if b.section is not None:
                z = b.section_z + (self._to_vec3(self._get_arg(args, 0, "v", [0, 0, 0]))[2]
                                   if name == "translate" else 0.0)
                result.append(replace(b, section=self._apply_transform_2d(name, args, b.section), section_z=z))
            elif b.body is not None:
                result.append(replace(b, body=self._apply_transform_3d(name, args, b.body),
                                      merged_from=self._transform_parts(name, args, b.merged_from)))
            elif b.raw_mesh is not None:
                result.append(replace(b, raw_mesh=self._transform_raw_mesh(name, args, b.raw_mesh)))
            else:
                result.append(b)
        return result

    def _transform_parts(self, name: str, args: dict, parts):
        """A union's unmerged parts move with it (see ColoredBody.merged_from)."""
        if not parts:
            return parts
        return tuple(replace(p, body=self._apply_transform_3d(name, args, p.body) if p.body is not None else None,
                             merged_from=self._transform_parts(name, args, p.merged_from)) for p in parts)

    def _transform_raw_mesh(self, name: str, args: dict, raw):
        """_apply_transform_3d for a raw_mesh, which has no Manifold to call:
        the same transform as a matrix, flipping the winding when it mirrors."""
        verts, tris = raw
        span = verts.max(axis=0) - verts.min(axis=0) if len(verts) else np.zeros(3)
        m = self._transform_matrix(name, args, span)
        verts = verts @ m[:3, :3].T + m[:3, 3]
        if np.linalg.det(m[:3, :3]) < 0:
            tris = tris[:, [0, 2, 1]]
        return verts, tris

    def _resize_scales(self, args: dict, span) -> list[float]:
        """resize()'s scale per axis: newsize/current where newsize is given;
        where it is 0 and `auto` is set for that axis, the largest of the
        given scales, so resize([10,0,0], auto=true) scales uniformly (auto
        was ignored); otherwise 1."""
        newsize = [float(x) for x in self._get_arg(args, 0, "newsize", [0, 0, 0])]
        newsize += [0.0] * (3 - len(newsize))
        auto = self._get_arg(args, 1, "auto", False)
        auto = [bool(a) for a in auto][:3] + [False] * 3 if isinstance(auto, list) else [bool(auto)] * 3
        scales = [ns / sp if ns and sp else None for ns, sp in zip(newsize, span)]
        given = [sc for sc in scales if sc is not None]
        fill = max(given) if given else 1.0
        return [sc if sc is not None else (fill if auto[i] else 1.0) for i, sc in enumerate(scales)]

    def _transform_matrix(self, name: str, args: dict, span) -> np.ndarray:
        """The 4x4 matrix of a transform; `span` is the extent resize() scales to."""
        m = np.eye(4)
        if name == "translate":
            m[:3, 3] = self._to_vec3(self._get_arg(args, 0, "v", [0, 0, 0]))
        elif name == "rotate":
            a = self._get_arg(args, 0, "a", 0)
            v = self._get_arg(args, 1, "v", None)
            if isinstance(a, (list, tuple)):
                ax, ay, az = (self._to_vec3(a) + [0.0])[:3]
                m[:3, :3] = _rot_deg(az, 2) @ _rot_deg(ay, 1) @ _rot_deg(ax, 0)
            else:
                m[:3, :] = np.array(self._axis_angle_matrix(self._to_vec3(v if v is not None else [0, 0, 1]),
                                                            math.radians(float(a))))
        elif name == "scale":
            v = self._get_arg(args, 0, "v", [1, 1, 1])
            m[:3, :3] = np.diag([float(v)] * 3 if isinstance(v, (int, float)) else [float(x) for x in v][:3])
        elif name == "mirror":
            n = np.array(self._to_vec3(self._get_arg(args, 0, "v", [1, 0, 0])))
            if n @ n:
                m[:3, :3] -= 2 * np.outer(n, n) / (n @ n)
        elif name == "resize":
            m[:3, :3] = np.diag(self._resize_scales(args, span))
        elif name == "multmatrix":
            mat = self._get_arg(args, 0, "m", None)
            if mat is not None:
                m[:3, :] = np.array(self._to_matrix4x3(mat), dtype=np.float64)
        return m

    def _apply_transform_2d(self, name: str, args: dict, cs: "m3d.CrossSection") -> "m3d.CrossSection":
        if name == "resize" or (name == "rotate" and self._rotates_out_of_plane(args)) or (
                name == "mirror" and self._to_vec3(self._get_arg(args, 0, "v", [1, 0, 0]))[2] != 0):
            # A 2D shape takes the in-plane part of a 3D transform, as in
            # OpenSCAD: rotate([60,0,0]) square(10) is 10 x 5, and
            # mirror([0,0,1]) leaves it as it was. It was left flat
            # (x/y rotation dropped) or, for a z mirror, degenerate.
            x0, y0, x1, y1 = cs.bounds()
            m = self._transform_matrix(name, args, np.array([x1 - x0, y1 - y0, 0.0]))
            return cs.transform([[m[0, 0], m[0, 1], m[0, 3]], [m[1, 0], m[1, 1], m[1, 3]]])
        if name == "translate":
            v = self._get_arg(args, 0, "v", [0, 0])
            cs = cs.translate([float(v[0]), float(v[1])])
        elif name == "rotate":
            a = self._get_arg(args, 0, "a", 0)
            # 2D rotation: scalar angle (Z), or [x,y,z] list → use Z component
            if isinstance(a, list):
                angle = float(a[2]) if len(a) > 2 else 0.0
            else:
                angle = float(a)
            cs = cs.rotate(angle)
        elif name == "scale":
            v = self._get_arg(args, 0, "v", [1, 1])
            if isinstance(v, (int, float)):
                v = [float(v), float(v)]
            cs = cs.scale([float(v[0]), float(v[1])])
        elif name == "mirror":
            v = self._get_arg(args, 0, "v", [1, 0])
            cs = cs.mirror([float(v[0]), float(v[1])])
        elif name == "multmatrix":
            m = self._get_arg(args, 0, "m", None)
            if m is not None:
                # Extract 2×3 affine matrix from 4×4: rows 0,1, cols 0,1,3
                mat2x3 = [
                    [float(m[0][0]), float(m[0][1]), float(m[0][3])],
                    [float(m[1][0]), float(m[1][1]), float(m[1][3])],
                ]
                cs = cs.transform(mat2x3)
        return cs

    def _apply_transform_3d(self, name: str, args: dict, body: "m3d.Manifold") -> "m3d.Manifold":
        if name == "translate":
            v = self._get_arg(args, 0, "v", [0, 0, 0])
            v = self._to_vec3(v)
            body = body.translate(v)
        elif name == "rotate":
            a = self._get_arg(args, 0, "a", 0)
            v = self._get_arg(args, 1, "v", None)
            body = self._apply_rotate(body, a, v)
        elif name == "scale":
            v = self._get_arg(args, 0, "v", [1, 1, 1])
            if isinstance(v, (int, float)):
                v = [v, v, v]
            v = [float(x) for x in v]
            body = body.scale(v)
        elif name == "mirror":
            v = self._get_arg(args, 0, "v", [1, 0, 0])
            v = self._to_vec3(v)
            body = body.mirror(v)
        elif name == "resize":
            bb = body.bounding_box()  # (xmin,ymin,zmin,xmax,ymax,zmax)
            body = body.scale(self._resize_scales(args, [bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2]]))
        elif name == "multmatrix":
            m = self._get_arg(args, 0, "m", None)
            if m is not None:
                mat = self._to_matrix4x3(m)
                body = body.transform(mat)
        return body

    def _rotates_out_of_plane(self, args: dict) -> bool:
        a = self._get_arg(args, 0, "a", 0)
        if isinstance(a, list):
            return any(float(x) != 0 for x in a[:2])
        v = self._get_arg(args, 1, "v", None)
        return v is not None and any(float(x) != 0 for x in self._to_vec3(v)[:2])

    def _apply_rotate(self, body: m3d.Manifold, a, v) -> m3d.Manifold:
        if isinstance(a, (list, tuple)):
            # rotate([x,y,z]) — Euler angles in degrees, applied Z then Y then X
            ax, ay, az = float(a[0]), float(a[1]), float(a[2]) if len(a) > 2 else 0.0
            body = body.rotate([ax, ay, az])
            return body
        else:
            # rotate(a, v) — angle around axis
            angle = float(a)
            if v is None:
                v = [0, 0, 1]
            v = self._to_vec3(v)
            # Rodrigues rotation via matrix
            mat = self._axis_angle_matrix(v, math.radians(angle))
            body = body.transform(mat)
            return body

    def _axis_angle_matrix(self, axis, angle_rad: float) -> list:
        ax, ay, az = axis
        length = math.sqrt(ax*ax + ay*ay + az*az)
        if length == 0:
            return [[1,0,0,0],[0,1,0,0],[0,0,1,0]]
        ax, ay, az = ax/length, ay/length, az/length
        c, s = _cos_sin_deg(math.degrees(angle_rad))
        t = 1 - c
        return [
            [t*ax*ax+c,    t*ax*ay-s*az, t*ax*az+s*ay, 0],
            [t*ax*ay+s*az, t*ay*ay+c,    t*ay*az-s*ax, 0],
            [t*ax*az-s*ay, t*ay*az+s*ax, t*az*az+c,    0],
        ]

    def _to_vec3(self, v) -> list[float]:
        if isinstance(v, (int, float)):
            return [float(v), 0.0, 0.0]
        result = [float(x) for x in v]
        while len(result) < 3:
            result.append(0.0)
        return result[:3]

    def _to_matrix4x3(self, m) -> list:
        """Convert 4x4 or 4x3 matrix to manifold's 4x3 row-major transform."""
        rows = []
        for row in m[:3]:
            r = [float(x) for x in row]
            while len(r) < 4:
                r.append(0.0)
            rows.append(r[:4])
        return rows

    # --- color ---

    def _resolve_color(self, node: ModularCall, ctx: EvalContext) -> dict:
        args, ctx = self._resolve_call_args(node, ctx)
        c = self._get_arg(args, 0, "c", [1, 1, 1, 1])
        alpha = float(self._get_arg(args, 1, "alpha", 1.0))
        if isinstance(c, str):
            rgba = self._css_color(c, alpha)
        elif isinstance(c, (list, tuple)):
            rgba = tuple(float(x) for x in c) + (alpha,) if len(c) == 3 else tuple(float(x) for x in c[:4])
        else:
            rgba = (1.0, 1.0, 1.0, 1.0)

        child_ctx = ctx.child_ctx(color=rgba)
        self._eval_children(node.children, child_ctx)  # side effect only, see _resolve_transform
        return {"rgba": rgba}

    def _generate_color(self, params: dict, children: list[CSGNode], node: ASTNode) -> list[ColoredBody]:
        # Recorded against the bodies' run IDs as well, because a later
        # union() recovers colour only from those (_attach_tri_colors). A
        # primitive records the colour it is born with, but forwarded
        # children -- `module c(x) { color(x) children(); }`, as BOSL2 does --
        # are born in the caller's colourless context.
        rgba = params["rgba"]
        bodies = self._one_dimension(flatten_csg_tree(children), node)
        for b in bodies:
            if b.body is not None:
                oid = b.body.original_id()
                ids = (oid,) if oid >= 0 else b.body.to_mesh().run_original_id
                for rid in ids:
                    self.id_to_color[int(rid)] = rgba
            if b.section_id is not None:
                self.id_to_color[b.section_id] = rgba
        return [replace(b, color=rgba, tri_colors=None, merged_from=self._recolor_parts(b.merged_from, rgba))
                for b in bodies]

    @classmethod
    def _recolor_parts(cls, parts, rgba):
        """color() over a union() colours every part it was merged from; the
        parts' runs are the body's, already recorded."""
        if not parts:
            return parts
        return tuple(replace(p, color=rgba, merged_from=cls._recolor_parts(p.merged_from, rgba)) for p in parts)

    def _css_color(self, name: str, alpha: float = 1.0) -> tuple:
        if name.startswith("#"):
            h = name.lstrip("#")
            if len(h) == 6:
                rgb = (int(h[0:2],16)/255, int(h[2:4],16)/255, int(h[4:6],16)/255)
            elif len(h) == 3:
                rgb = (int(h[0],16)/15, int(h[1],16)/15, int(h[2],16)/15)
            else:
                rgb = (1, 1, 1)
            return rgb + (alpha,)

        rgb = CSS_COLORS.get(name.lower(), (1.0, 1.0, 1.0))
        return rgb + (alpha,)

    # --- CSG ---

    def _resolve_csg(self, node: ModularCall, ctx: EvalContext) -> dict:
        # Evaluate each top-level geometry statement separately so their body groups are
        # preserved.  For difference(), all bodies from the FIRST statement form the
        # positive operand (unioned implicitly, as OpenSCAD does within a scope); bodies
        # from each subsequent statement are unioned and then subtracted.  A flat
        # evaluation loses this grouping and produces wrong results when BOSL2's
        # attachable() returns multiple bodies (parent + attached children) as the first
        # operand of difference().
        #
        # group_sizes records, per top-level statement, how many CSGNode
        # children it contributed to self._tree_stack[-1] — needed because
        # for/if/let are "transparent" in the tree (Phase 1), so one
        # statement can contribute a variable, unmarked number of tree
        # children (e.g. a for loop's iterations) with no boundary marker
        # otherwise. Measured as a stack-length delta: pure bookkeeping, no
        # Manifold calls, safe here.
        #
        # Every statement is always resolved (no short-circuiting): with
        # generation fully deferred (Phase 2 final cutover), resolve can no
        # longer tell whether a statement's geometry is empty — that's only
        # knowable once it's actually generated. _generate_csg re-derives
        # the discard-vs-skip short-circuit semantics itself, from real
        # generated bodies, using these same group_sizes to re-chunk children.
        op = node.name.name
        args, ctx = self._resolve_call_args(node, ctx)
        ctx = self._block_ctx(node.children, ctx)
        block = self._expand_child_statements(node.children, ctx)
        assign_nodes = [c for c in block if isinstance(c, Assignment)]
        geo_nodes = [c for c in block
                     if not isinstance(c, (Assignment, ModuleDeclaration, FunctionDeclaration))]

        # Process assignments first for side-effects (they update ctx.dyn in-place)
        if assign_nodes:
            self._eval_children(assign_nodes, ctx, new_scope=False)

        group_sizes: list[int] = []
        for geo_node in geo_nodes:
            before = len(self._tree_stack[-1])
            self._eval_children([geo_node], ctx, new_scope=False)
            size = len(self._tree_stack[-1]) - before
            if size or self._is_operand_when_empty(geo_node):
                group_sizes.append(size)
        return {"op": op, "group_sizes": group_sizes}

    def _is_operand_when_empty(self, node) -> bool:
        """Whether a statement that produced nothing still counts as an
        (empty) operand of union/difference/intersection -- the difference
        between `intersection() { cube(2); if (false) cube(1); }` keeping the
        cube and emptying it. As in OpenSCAD, a module call or loop builds a
        node whatever it contains, and an empty one annihilates; an `if` that
        took no branch, a `*`-disabled statement, and a bare echo()/assert()
        build none and are skipped. Checked case by case against OpenSCAD."""
        t = type(node)
        if t is ModularIf:
            return self._if_taken
        return t not in (ModularModifierDisable, ModularEcho, ModularAssert)

    def _generate_csg(self, params: dict, children: list[CSGNode], node: ASTNode) -> list[ColoredBody]:
        op = params["op"]
        all_bg: list[ColoredBody] = []
        all_hi: list[ColoredBody] = []
        all_so: list[ColoredBody] = []
        csg_result: Optional[ColoredBody] = None
        # 2D keeps colour geometrically: one part per colour, since a
        # CrossSection has no per-edge provenance to recover it from after
        # a merge the way _attach_tri_colors does for 3D.
        parts_2d: Optional[list[ColoredBody]] = None
        idx = 0
        # The first child's dimension wins; a child of the other dimension is
        # dropped (and, dropped, is an empty operand -- intersection() with
        # one is empty, as in OpenSCAD).
        kept = {id(b) for b in self._one_dimension(flatten_csg_tree(children), node)}
        keeping = op == "difference" and self._keep_minuend_color
        remember = op == "union" and self._keep_minuend_color
        keep_parts: list[list] = []  # [part, the run IDs it was born with]
        union_parts: list[ColoredBody] = []

        for size in params["group_sizes"]:
            group_nodes = children[idx:idx + size]
            idx += size
            stmt_bodies = [b for b in flatten_csg_tree(group_nodes) if id(b) in kept]

            bg, fg, hi, so = self._split_by_role(stmt_bodies)
            all_bg.extend(bg)
            all_hi.extend(hi)
            all_so.extend(so)

            # A Manifold-invalid operand (an open polyhedron, say) would poison
            # the whole boolean -- one bad part emptied a 65-part union --
            # so it is dropped instead, as OpenSCAD does.
            bodies_3d = [c for c in fg if c.body is not None and c.body.status() == _MANIFOLD_OK]
            sections_2d = [c for c in fg if c.section is not None]

            if not bodies_3d and not sections_2d:
                # Empty statement: intersection(∅, B)=∅ discards any csg_result
                # already built from prior statements (matches resolve's own
                # short-circuit — group_sizes never has entries past this
                # point for intersection). difference(∅, B)=∅ only applies
                # while no positive operand has been established yet; union
                # just skips the empty contributor and keeps going.
                if op == "intersection":
                    csg_result = parts_2d = None
                    break
                if op == "difference" and csg_result is None and parts_2d is None:
                    break
                continue

            if bodies_3d:
                # Union all 3D bodies from this statement before applying the op
                grp = bodies_3d[0].body
                for c in bodies_3d[1:]:
                    grp = grp + c.body
                if op == "difference" and csg_result is not None:
                    for c in bodies_3d:
                        if c.color is None and c.tri_colors is None:
                            self._record_run_colors(c.body, _CUT_FACE_COLOR)
                if csg_result is None:
                    csg_result = ColoredBody(body=grp, color=bodies_3d[0].color)
                    for leaf in (self._keep_parts(bodies_3d) if keeping else ()):
                        keep_parts.append([leaf, set(self._run_ids(leaf.body))])
                    if remember:
                        union_parts += self._keep_parts(bodies_3d)
                elif op == "union":
                    csg_result = replace(csg_result, body=csg_result.body + grp)
                    if remember:
                        union_parts += self._keep_parts(bodies_3d)
                elif op == "difference":
                    csg_result = replace(csg_result, body=csg_result.body - grp)
                    for part in keep_parts:
                        part[0] = replace(part[0], body=part[0].body - grp)
                elif op == "intersection":
                    csg_result = replace(csg_result, body=csg_result.body ^ grp)
            elif parts_2d is None or op == "union":
                parts_2d = parts_2d or []
                for c in sections_2d:
                    self._paint_2d(parts_2d, c)
            else:
                grp = sections_2d[0].section
                for c in sections_2d[1:]:
                    grp = grp + c.section
                parts_2d = [replace(p, section=p.section - grp if op == "difference" else p.section ^ grp)
                            for p in parts_2d]
                parts_2d = [p for p in parts_2d if not p.section.is_empty()]

        if csg_result is not None and csg_result.body is not None:
            if remember and len(union_parts) > 1:
                csg_result = replace(csg_result, merged_from=tuple(union_parts))
            if keep_parts:
                csg_result = replace(csg_result, body=self._finish_keep_minuend(keep_parts))
        # Return: CSG result + background ghosts + highlight overlays + show_only bodies (all separate from CSG result)
        if csg_result is not None and csg_result.body is not None:
            csg_result = self._attach_tri_colors(csg_result)
        result = [csg_result] if csg_result is not None else []
        if parts_2d:
            result.extend(parts_2d)
        return result + all_bg + all_hi + all_so

    @staticmethod
    def _run_ids(body: m3d.Manifold) -> list[int]:
        oid = body.original_id()
        return [oid] if oid >= 0 else [int(r) for r in body.to_mesh().run_original_id]

    @classmethod
    def _keep_parts(cls, bodies) -> list[ColoredBody]:
        """keep_minuend_color: the leaf parts of these operands -- each one,
        or what a union() merged it from, recursively."""
        out = []
        for b in bodies:
            if b.merged_from:
                out += cls._keep_parts(b.merged_from)
            elif b.body is not None:
                out.append(b)
        return out

    def _finish_keep_minuend(self, parts: list[list]) -> m3d.Manifold:
        """Union the per-part differences, first re-minting each part's cut
        faces -- the subtrahend's runs, shared by every part's result --
        under fresh IDs carrying that part's colour, so _attach_tri_colors
        tells one part's cut faces from another's. Their node stays the
        subtrahend's: clicking a cut face still finds the tool.
        ponytail: a part that is itself a multi-colour merge gives its cut
        faces its first child's colour, not the one the cut passed through."""
        out = None
        for part, own in parts:
            if part.body.is_empty():
                continue
            mesh = part.body.to_mesh64()
            ids = [int(r) for r in mesh.run_original_id]
            fresh = {}
            for old in dict.fromkeys(ids):
                if old in own:
                    continue
                new = fresh[old] = int(m3d.Manifold.reserve_ids(1))
                if old in self.id_to_node:
                    self.id_to_node[new] = self.id_to_node[old]
                if old in self.id_to_call_chain:
                    self.id_to_call_chain[new] = self.id_to_call_chain[old]
                self.id_to_color[new] = part.color
            body = part.body if not fresh else self._with_run_ids(mesh, [fresh.get(i, i) for i in ids])
            out = body if out is None else out + body
        return out if out is not None else m3d.Manifold()

    @staticmethod
    def _with_run_ids(mesh, run_ids: list[int]) -> m3d.Manifold:
        """A Manifold rebuilt from `mesh` (a to_mesh64()) with its runs relabelled."""
        def arr(name, dtype):
            a = np.array(getattr(mesh, name), dtype=dtype)  # a copy: the binding rejects its own read-only arrays
            return a if a.size else None  # and an empty one
        return m3d.Manifold(m3d.Mesh64(
            vert_properties=arr("vert_properties", np.float64), tri_verts=arr("tri_verts", np.uint64),
            merge_from_vert=arr("merge_from_vert", np.uint64), merge_to_vert=arr("merge_to_vert", np.uint64),
            run_index=arr("run_index", np.uint64), run_original_id=np.array(run_ids, dtype=np.uint32),
            run_transform=arr("run_transform", np.float64), face_id=arr("face_id", np.uint64)))

    def _record_run_colors(self, body: m3d.Manifold, rgba) -> None:
        """Colour `body`'s runs in id_to_color, which is all _attach_tri_colors
        has to go on once a merge has thrown the bodies away."""
        oid = body.original_id()
        for rid in ((oid,) if oid >= 0 else body.to_mesh().run_original_id):
            self.id_to_color[int(rid)] = rgba

    @staticmethod
    def _paint_2d(parts: list[ColoredBody], c: ColoredBody) -> None:
        """Add 2D body `c` to a union's per-colour parts in painter's order:
        it notches what it covers out of every differently-coloured part and
        merges into the part of its own colour, so a union of red and blue
        squares stays red and blue, and two red squares are one red shape."""
        same = None
        for i, p in enumerate(parts):
            if p.color == c.color:
                same = i
            else:
                parts[i] = replace(p, section=p.section - c.section)
        if same is None:
            parts.append(ColoredBody(section=c.section, color=c.color))
        else:
            parts[same] = replace(parts[same], section=parts[same].section + c.section)
        parts[:] = [p for p in parts if not p.section.is_empty()]

    def _attach_tri_colors(self, cb: ColoredBody) -> ColoredBody:
        """After a real boolean merge, per-input color is otherwise lost --
        `cb.color` is just one arbitrary child's color (see _generate_csg).
        manifold3d preserves per-triangle provenance through boolean ops via
        each merged mesh's run_original_id/run_index (already relied on for
        WYSIWYG ray-cast picking, self.id_to_node); reuse the same mechanism
        here to recover each triangle's real originating color from
        self.id_to_color (populated by _tag/_tag_generated when each child
        was itself first generated, before being merged away). If every
        triangle resolves to the same color, this is a no-op (leaves
        tri_colors None) -- the common single-material case pays no extra
        cost and keeps following live color-theme changes for uncolored
        geometry, same as before this existed."""
        mesh = cb.body.to_mesh()
        run_ids = mesh.run_original_id
        # run_index counts flattened vertex corners (3 per triangle), not
        # triangles -- e.g. a 302-triangle mesh's run_index might read
        # [0, 138, 906], where 906 (> 302) only makes sense as 3*302.
        run_idx = [i // 3 for i in mesh.run_index]
        T = len(mesh.tri_verts)
        if T == 0 or len(run_ids) <= 1:
            return cb
        per_run_color = [self.id_to_color.get(int(rid), cb.color) for rid in run_ids]
        if len(set(per_run_color)) <= 1:
            return cb
        tri_colors = np.empty((T, 4), dtype=np.float32)
        for i in range(len(run_idx) - 1):
            s, e = int(run_idx[i]), min(int(run_idx[i + 1]), T)
            if s < T:
                tri_colors[s:e] = per_run_color[i] if per_run_color[i] is not None else _DEFAULT_GEOMETRY_COLOR
        return replace(cb, tri_colors=tri_colors)

    @staticmethod
    def _split_by_role(bodies: list[ColoredBody]) -> tuple[list[ColoredBody], list[ColoredBody], list[ColoredBody], list[ColoredBody]]:
        """Split a flat body list into (background, foreground,
        highlight_ghost, show_only) per the %/#/! modifier convention —
        shared by hull/minkowski's generate steps (previously duplicated
        identically in both, plus a per-statement-group variant in
        _generate_csg). show_only (!) bodies must come out of `fg` just
        like background/highlight do: they were being unioned/hulled/
        minkowski-summed together with ordinary bodies, which silently
        strips their role (the operation's result has no role at all),
        so ! stopped isolating its subtree the moment it was nested inside
        any boolean op/hull/minkowski rather than sitting at the top
        level — evaluate()'s "any show_only body anywhere -> show only
        those + highlights" check at the very end never saw one."""
        bg = [c for c in bodies if c.role == "background"]
        fg = [c for c in bodies if c.role not in ("background", "show_only")]
        hi = [replace(c, role="highlight_ghost") for c in fg if c.role == "highlight"]
        so = [c for c in bodies if c.role == "show_only"]
        return bg, fg, hi, so

    def _resolve_hull(self, node: ModularCall, ctx: EvalContext) -> dict:
        args, ctx = self._resolve_call_args(node, ctx)
        self._eval_children(node.children, ctx)  # side effect only, see _resolve_transform
        return {}

    def _generate_hull(self, params: dict, children: list[CSGNode], node: ASTNode) -> list[ColoredBody]:
        bodies = flatten_csg_tree(children)
        if not bodies:
            return []
        bg, fg, hi, so = self._split_by_role(bodies)
        hull_result: Optional[ColoredBody] = None
        if fg:
            bodies_3d = [c.body for c in fg if c.body is not None]
            raw_pts = [c.raw_mesh[0] for c in fg if c.raw_mesh is not None]
            if raw_pts:
                # A hull needs only points, so an open mesh takes part too --
                # BOSL2's hull_points() feeds hull() a polyhedron of arbitrary faces.
                pts = np.vstack(raw_pts + [np.asarray(b.to_mesh().vert_properties[:, :3], dtype=np.float64)
                                           for b in bodies_3d])
                hull_result = ColoredBody(body=m3d.Manifold.hull_points(pts), color=fg[0].color)
            elif bodies_3d:
                hull_result = ColoredBody(body=m3d.Manifold.batch_hull(bodies_3d), color=fg[0].color)
            else:
                sections = [c.section for c in fg if c.section is not None]
                if sections:
                    hull_result = ColoredBody(section=m3d.CrossSection.batch_hull(sections), color=fg[0].color)
        return ([hull_result] if hull_result is not None else []) + bg + hi + so

    def _generate_fill(self, params: dict, children: list[CSGNode], node: ASTNode) -> list[ColoredBody]:
        """fill(): its 2D children's union with every hole filled -- only the
        outer (counter-clockwise) contours kept. It was an unknown module."""
        bodies = flatten_csg_tree(children)
        cs = self._to_cross_section(bodies)
        if cs is None:
            return []
        outers = [p for p in cs.to_polygons() if _signed_area(p) > 0]
        filled = m3d.CrossSection([np.asarray(p, dtype=np.float64) for p in outers], m3d.FillRule.Positive) \
            if outers else m3d.CrossSection()
        return [ColoredBody(section=filled, color=bodies[0].color if bodies else None)]

    def _resolve_polyhedron(self, node: ModularCall, ctx: EvalContext) -> dict:
        args, ctx = self._resolve_call_args(node, ctx)
        points = self._get_arg(args, 0, "points", None)
        faces = self._get_arg(args, 1, "faces", None)
        if faces is None:
            faces = self._get_arg(args, 1, "triangles", None)  # legacy alias
        if isinstance(points, OscObject):
            # polyhedron(obj): any object with vertices (or points) and faces
            # -- a render() result round-trips in one call.
            verts = points.get("vertices")
            if verts is None:
                verts = points.get("points")
            if verts is None:
                self.error("polyhedron: object has no 'vertices' (or 'points') key", node)
            if points.get("faces") is None:
                self.error("polyhedron: object has no 'faces' key", node)
            points, faces = verts, points.get("faces")
        elif (faces is None and isinstance(points, list) and len(points) == 2
              and all(isinstance(half, list) and half and isinstance(half[0], list) for half in points)):
            # polyhedron(vnf): BOSL2's [vertices, faces] 2-list, told apart as
            # BOSL2's is_vnf() does -- its second element is a list of LISTS,
            # where a 2-point list would have a point there. Only when faces
            # wasn't given separately, so the two-argument form always wins.
            points, faces = points
        if points is None or faces is None:
            self.error("polyhedron: 'points' and 'faces' are required", node)
        if not isinstance(points, list) or not isinstance(faces, list):
            self.error("polyhedron: 'points' and 'faces' must be lists", node)
        for i, p in enumerate(points):
            if not isinstance(p, list) or len(p) != 3 or any(c is None for c in p):
                self.error(f"polyhedron: point[{i}] is not a valid [x,y,z] coordinate", node)
        try:
            # float64 throughout: float32 moved every vertex by up to half an
            # ulp, enough to open seams in a large model (#94).
            verts = np.array([[float(c) for c in p] for p in points], dtype=np.float64)
            tris = []
            vlist = verts.tolist()
            for face in faces:
                _triangulate_face(vlist, [int(x) for x in face], tris)
            tri_arr = np.array(tris, dtype=np.uint64) if tris else np.zeros((0, 3), dtype=np.uint64)
            # Welding coincident vertices repairs BOSL2 VNFs, whose seams and
            # poles repeat points. It only ever repairs: applied to a mesh
            # that is already sound, it fuses two shells that merely touch
            # into edges with four faces (#105). So weld only when needed.
            _, unique_idx, remap = np.unique(np.round(verts, decimals=6), axis=0,
                                             return_index=True, return_inverse=True)
            if len(unique_idx) != len(verts) and not _edges_closed(tri_arr):
                verts, tri_arr = verts[unique_idx], remap.reshape(-1)[tri_arr].astype(np.uint64)
                tri_arr = tri_arr[(tri_arr[:, 0] != tri_arr[:, 1]) & (tri_arr[:, 1] != tri_arr[:, 2])
                                  & (tri_arr[:, 0] != tri_arr[:, 2])]
        except Exception as e:
            self.error(f"polyhedron: {e}", node)
        return {"verts": verts, "tri_arr": tri_arr, "color": ctx.color}

    def _generate_polyhedron(self, params: dict, children: list[CSGNode], node: ASTNode) -> list[ColoredBody]:
        try:
            mesh = m3d.Mesh64(vert_properties=params["verts"], tri_verts=params["tri_arr"])
            body = m3d.Manifold(mesh)
        except Exception as e:
            self.error(f"polyhedron: {e}", node)
        return [self._mesh_body(body, params["verts"], params["tri_arr"], node, params["color"], "polyhedron")]

    def _mesh_body(self, body: m3d.Manifold, verts, tris, node, color, what: str) -> ColoredBody:
        """The body for a mesh built from user data. Closed: the Manifold.
        Open: the triangles themselves as a display-only raw_mesh, since
        Manifold gives back an empty body for them -- the object vanished
        without a word before. Any other failure (NaN coordinates, say)
        keeps the empty body, drawing nothing, but now says so."""
        status = body.status()
        if status == _MANIFOLD_OK:
            return self._tag_generated(body, node, color)
        pos = getattr(node, "position", None)
        if status != m3d.Error.NotManifold:
            self._echo_fn(f"WARNING: {what}: mesh could not be built ({status.name}); "
                          f"nothing is drawn{self._loc(pos)}")
            return self._tag_generated(body, node, color)
        verts = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
        tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
        if not self._hull_depth:  # hull() only needs the points, and OpenSCAD is silent there
            count, first = self._boundary_edges(tris)
            where = ""
            if first is not None:
                a, b = (self._fmt_val([float(c) for c in verts[i]]) for i in first)
                where = f", first at {a} - {b}"
            self._echo_fn(
                f"WARNING: {what}: mesh is not closed -- {count} boundary edge(s){where}; "
                "drawing the object as an open surface rather than a solid -- nothing is "
                "patched. hull() can still use its points, but it cannot take part in "
                f"union/difference/intersection{self._loc(pos)}")
        return ColoredBody(color=color, raw_mesh=(verts, tris))

    @staticmethod
    def _boundary_edges(tris: np.ndarray) -> tuple[int, Optional[tuple[int, int]]]:
        """How many edges a closed mesh would need another face on (used an
        odd number of times), and the lowest-numbered one -- deterministic,
        so the warning names the same edge every render."""
        if len(tris) == 0:
            return 0, None
        edges = np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1)
        uniq, counts = np.unique(edges, axis=0, return_counts=True)
        odd = uniq[counts % 2 == 1]
        return len(odd), (tuple(int(i) for i in odd[0]) if len(odd) else None)

    def _resolve_surface(self, node: ModularCall, ctx: EvalContext) -> dict:
        args, ctx = self._resolve_call_args(node, ctx)
        file_arg = self._get_arg(args, 0, "file", None)
        center = bool(self._get_arg(args, None, "center", False))
        invert = bool(self._get_arg(args, None, "invert", False))
        color = ctx.color

        if file_arg is None:
            self.error("surface: 'file' parameter is required", node)
            return {"heights": None, "center": center, "color": color}

        # Resolve path relative to the source file
        base_dir = None
        pos = getattr(node, 'position', None)
        if pos and getattr(pos, 'origin', None):
            import os as _os
            base_dir = _os.path.dirname(pos.origin)
        if base_dir:
            import os as _os
            file_path = _os.path.join(base_dir, str(file_arg)) if not _os.path.isabs(str(file_arg)) else str(file_arg)
        else:
            file_path = str(file_arg)

        try:
            heights = self._surface_load(file_path, invert)
        except Exception as e:
            self.error(f"surface: {e}", node)
            return {"heights": None, "center": center, "color": color}

        if heights is None or len(heights) == 0 or len(heights[0]) == 0:
            self.error("surface: empty height data", node)
            return {"heights": None, "center": center, "color": color}

        return {"heights": heights, "center": center, "color": color}

    def _generate_surface(self, params: dict, children: list[CSGNode], node: ASTNode) -> list[ColoredBody]:
        heights = params["heights"]
        if heights is None:
            return []
        center = params["center"]

        rows = len(heights)
        cols = len(heights[0])

        x_off = -(cols - 1) / 2.0 if center else 0.0
        y_off = -(rows - 1) / 2.0 if center else 0.0

        # Build vertex grid: (cols) * (rows) top vertices + same for bottom (z=0)
        # top verts: index = row * cols + col
        # bottom verts: index = rows*cols + row * cols + col
        n = rows * cols
        verts = []
        for r in range(rows):
            for c in range(cols):
                verts.append([c + x_off, r + y_off, float(heights[r][c])])
        for r in range(rows):
            for c in range(cols):
                verts.append([c + x_off, r + y_off, 0.0])

        tris = []

        def top(r, c):
            return r * cols + c

        def bot(r, c):
            return n + r * cols + c

        # Top surface (CCW from above = outward upward normal)
        for r in range(rows - 1):
            for c in range(cols - 1):
                tl, tr, bl, br = top(r+1, c), top(r+1, c+1), top(r, c), top(r, c+1)
                tris.append([tl, bl, br])
                tris.append([tl, br, tr])

        # Bottom face (CCW from below = outward downward normal)
        for r in range(rows - 1):
            for c in range(cols - 1):
                tl, tr, bl, br = bot(r+1, c), bot(r+1, c+1), bot(r, c), bot(r, c+1)
                tris.append([tl, tr, br])
                tris.append([tl, br, bl])

        # Side walls (outward normals: front=-Y, back=+Y, left=-X, right=+X)
        for c in range(cols - 1):  # front (r=0, outward=-Y)
            tris.append([top(0, c), bot(0, c), bot(0, c+1)])
            tris.append([top(0, c), bot(0, c+1), top(0, c+1)])
        for c in range(cols - 1):  # back (r=rows-1, outward=+Y)
            tris.append([top(rows-1, c), top(rows-1, c+1), bot(rows-1, c+1)])
            tris.append([top(rows-1, c), bot(rows-1, c+1), bot(rows-1, c)])
        for r in range(rows - 1):  # left (c=0, outward=-X)
            tris.append([top(r, 0), top(r+1, 0), bot(r+1, 0)])
            tris.append([top(r, 0), bot(r+1, 0), bot(r, 0)])
        for r in range(rows - 1):  # right (c=cols-1, outward=+X)
            tris.append([top(r, cols-1), bot(r+1, cols-1), top(r+1, cols-1)])
            tris.append([top(r, cols-1), bot(r, cols-1), bot(r+1, cols-1)])

        try:
            verts_arr = np.array(verts, dtype=np.float64)
            tris_arr = np.array(tris, dtype=np.uint64)
            mesh = m3d.Mesh64(vert_properties=verts_arr, tri_verts=tris_arr)
            body = m3d.Manifold(mesh)
            return [self._tag_generated(body, node, params["color"])]
        except Exception as e:
            self.error(f"surface: mesh construction failed: {e}", node)
            return []

    def _surface_load(self, file_path: str, invert: bool):
        """Load height data from a .dat text file or a PNG image."""
        import os as _os
        ext = _os.path.splitext(file_path)[1].lower()
        if ext in (".png", ".jpg", ".jpeg", ".bmp", ".gif"):
            return self._surface_load_image(file_path, invert)
        return self._surface_load_dat(file_path)

    def _surface_load_dat(self, file_path: str):
        heights = []
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                heights.append([float(v) for v in line.split()])
        heights.reverse()  # first row in file = highest Y (OpenSCAD convention)
        return heights

    def _surface_load_image(self, file_path: str, invert: bool):
        try:
            from PIL import Image
        except ImportError:
            raise RuntimeError("Pillow is required for image-based surface() — install it with: uv add Pillow")
        img = Image.open(file_path).convert("RGB")
        w, h = img.size
        pixels = img.load()
        heights = []
        for row in range(h - 1, -1, -1):  # bottom row of image = Y=0
            r_vals = []
            for col in range(w):
                r, g, b = pixels[col, row]
                gray = 0.2126 * r + 0.7152 * g + 0.0722 * b  # linear luminance
                val = (255.0 - gray) / 255.0 * 100.0 if invert else gray / 255.0 * 100.0
                r_vals.append(val)
            heights.append(r_vals)
        return heights

    # ------------------------------------------------------------------
    # import() — 3D mesh, 2D geometry, JSON
    # ------------------------------------------------------------------

    def _resolve_import_path(self, file_arg: Any, node) -> str:
        import os as _os
        pos = getattr(node, "position", None)
        base_dir = _os.path.dirname(pos.origin) if pos and getattr(pos, "origin", None) else None
        path = str(file_arg) if file_arg is not None else ""
        if base_dir and not _os.path.isabs(path):
            path = _os.path.join(base_dir, path)
        return path

    def _resolve_import(self, node: ModularCall, ctx: EvalContext) -> dict:
        """Loads the file's raw data (verts/tris, or 2D contours) as plain
        data during resolve — all pure file I/O and numpy math, no Manifold
        calls — so generate only needs to build the Manifold/CrossSection
        from already-parsed data, matching the caching approach used for
        surface(). Mirrors _builtin_import's per-extension dispatch and
        each removed _import_X_geometry wrapper's own exception handling
        exactly, so observable errors/warnings are unchanged."""
        import os as _os
        args, ctx = self._resolve_call_args(node, ctx)
        file_arg = self._get_arg(args, 0, "file", None)
        layer    = self._get_arg(args, None, "layer", None)
        color = ctx.color
        if file_arg is None:
            self.error("import: 'file' parameter is required", node)
            return {"color": color}
        path = self._resolve_import_path(file_arg, node)
        ext  = _os.path.splitext(path)[1].lower()
        if not _os.path.exists(path):
            # A missing file warns and imports nothing, as in OpenSCAD; it
            # aborted the whole render. Warned at generate, where OpenSCAD does.
            return {"kind": "missing", "path": path, "ext": ext, "color": color}
        try:
            if ext in (".stl", ".obj", ".off", ".3mf"):
                loader = {".stl": self._load_stl, ".obj": self._load_obj,
                          ".off": self._load_off, ".3mf": self._load_3mf}[ext]
                try:
                    verts, tris = loader(path)
                except Exception as e:
                    self.error(f"import: {e}", node)
                    return {"color": color}
                # Not OpenSCAD parameters, so a script using them won't run
                # upstream -- which is why repair is asked for, not automatic.
                return {"kind": "mesh", "verts": verts, "tris": tris, "color": color,
                        "repair": bool(self._get_arg(args, None, "repair", False)),
                        "tolerance": self._get_arg(args, None, "tolerance")}
            elif ext == ".dxf":
                contours = self._load_dxf_contours(path, layer, node)
                return {"kind": "dxf", "contours": contours, "color": color}
            elif ext in (".svg", ".pdf"):
                filtered = any(isinstance(self._get_arg(args, None, k), str) for k in ("id", "class"))
                try:
                    contours = self._load_svg_contours(path, node, self._get_arg(args, None, "id"),
                                                       self._get_arg(args, None, "class"),
                                                       self._get_arg(args, None, "dpi", 72.0),
                                                       bool(self._get_arg(args, None, "center", False)))
                except Exception as e:
                    self.error(f"import: {e}", node)
                    return {"color": color}
                return {"kind": "svg", "contours": contours, "color": color, "filtered": filtered}
            elif ext == ".json":
                self.error("import: .json returns data, not geometry — use as an expression", node)
                return {"color": color}
            else:
                self.error(f"import: unsupported file type '{ext}'", node)
                return {"color": color}
        except OSError as e:
            self.error(f"import: {e}", node)
            return {"color": color}

    def _generate_import(self, params: dict, children: list[CSGNode], node: ASTNode) -> list[ColoredBody]:
        kind = params.get("kind")
        color = params["color"]
        if kind == "missing":
            # OpenSCAD's own wording, which names a line but not the file.
            pos = getattr(node, "position", None)
            where = f", import() at line {pos.line}" if pos else ""
            self._echo_fn(f"WARNING: Can't open DXF file '{params['path']}'." if params["ext"] == ".dxf"
                          else f"WARNING: Can't open import file '{params['path']}'{where}")
            return []
        if kind == "mesh":
            return self._import_mesh(params, node, color)
        if kind == "dxf":
            contours = params["contours"]
            if contours is None:
                return []
            if not contours:
                self.error("import: no closed contours found in DXF file", node)
                return []
            polys = [np.array(c, dtype=np.float64) for c in contours]
            cs = m3d.CrossSection(polys, m3d.FillRule.EvenOdd)
            return [ColoredBody(section=cs, color=color)]
        if kind == "svg":
            contours = params["contours"]
            if not contours:
                if params.get("filtered"):
                    return []  # a filter that missed already warned; it imports nothing
                self.error("import: no shapes found in SVG file", node)
                return []
            polys = [np.array(c, dtype=np.float64) for c in contours]
            cs = m3d.CrossSection(polys, m3d.FillRule.EvenOdd)
            return [ColoredBody(section=cs, color=color)]
        return []

    def _import_as_value(self, args: dict, node) -> Any:
        import os as _os
        file_arg = self._get_arg(args, 0, "file", None)
        layer    = self._get_arg(args, None, "layer", None)
        if file_arg is None:
            self.error("import: 'file' parameter is required", node)
            return None
        path = self._resolve_import_path(file_arg, node)
        ext  = _os.path.splitext(path)[1].lower()
        if not _os.path.exists(path):
            self._echo_fn(f"WARNING: Could not read file '{path}'{self._loc(getattr(node, 'position', None))}")
            return None
        try:
            if ext == ".json":
                import json as _json
                with open(path, "r", encoding="utf-8") as f:
                    return self._json_to_osc(_json.load(f))
            elif ext in (".stl", ".obj", ".off", ".3mf"):
                return self._import_as_vnf(path, ext, node)
            elif ext in (".dxf", ".svg"):
                return self._import_as_region(path, ext, layer, node,
                                              self._get_arg(args, None, "id"), self._get_arg(args, None, "class"),
                                              self._get_arg(args, None, "dpi", 72.0),
                                              bool(self._get_arg(args, None, "center", False)))
            else:
                self.error(f"import: unsupported file type '{ext}'", node)
                return None
        except OSError as e:
            self.error(f"import: {e}", node)
            return None

    def _import_as_vnf(self, path: str, ext: str, node) -> Any:
        """Load a mesh file and return a VNF: [[verts], [faces]]."""
        try:
            if ext == ".stl":
                raw_verts, raw_tris = self._load_stl(path)
            elif ext == ".obj":
                raw_verts, raw_tris = self._load_obj(path)
            elif ext == ".off":
                raw_verts, raw_tris = self._load_off(path)
            else:
                raw_verts, raw_tris = self._load_3mf(path)
        except Exception as e:
            self.error(f"import: {e}", node)
            return None
        vert_map: dict[tuple, int] = {}
        verts_out: list[list[float]] = []
        faces_out: list[list[int]] = []
        raw_verts_list = list(raw_verts)  # handle numpy arrays
        for face in raw_tris:
            fi = []
            for vi in face:
                v = raw_verts_list[int(vi)]
                key = (float(v[0]), float(v[1]), float(v[2]))
                if key not in vert_map:
                    vert_map[key] = len(verts_out)
                    verts_out.append(list(key))
                fi.append(vert_map[key])
            faces_out.append(fi)
        return [verts_out, faces_out]

    def _import_as_region(self, path: str, ext: str, layer: Any, node, id_=None, cls=None,
                          dpi=72.0, center: bool = False) -> Any:
        """Load a 2D file and return a Region: [[[x,y],...], ...]."""
        try:
            if ext == ".dxf":
                contours = self._load_dxf_contours(path, layer, node)
            else:
                contours = self._load_svg_contours(path, node, id_, cls, dpi, center)
        except Exception as e:
            self.error(f"import: {e}", node)
            return None
        if contours is None:
            return None
        return [[[pt[0], pt[1]] for pt in c] for c in contours]

    def _json_to_osc(self, v: Any) -> Any:
        """Recursively convert JSON-parsed Python value to evaluator-native types.
        JSON objects → OscObject; arrays/scalars pass through as-is."""
        if isinstance(v, dict):
            return OscObject({k: self._json_to_osc(val) for k, val in v.items()})
        if isinstance(v, list):
            return [self._json_to_osc(x) for x in v]
        return v  # str, int, float, bool, None — all native

    def _import_mesh(self, params: dict, node, color) -> list[ColoredBody]:
        """An imported mesh, repaired on request (import(..., repair=true,
        tolerance=)), and when it is still not a solid, named for what is
        wrong -- "4 boundary edges" rather than Manifold's NotManifold --
        and drawn as the open surface it is (cpp 72ca136, #191)."""
        from .mesh_check import DEFAULT_WELD_TOLERANCE, check_mesh, repair_mesh
        verts = np.asarray(params["verts"], dtype=np.float64).reshape(-1, 3)
        tris = np.asarray(params["tris"], dtype=np.int64).reshape(-1, 3)
        if len(tris) == 0:
            self.error("import: mesh has no triangles", node)
            return []
        pos = getattr(node, "position", None)
        if params.get("repair"):
            tol = params.get("tolerance")
            if type(tol) in (int, float) and tol < 0:
                self._echo_fn(f"WARNING: import: tolerance must not be negative; using the default{self._loc(pos)}")
                tol = None
            elif tol is not None and type(tol) not in (int, float):
                self._echo_fn(f"WARNING: import: tolerance must be a number{self._loc(pos)}")
                tol = None
            before = check_mesh(verts, tris)
            verts, tris, report = repair_mesh(verts, tris, DEFAULT_WELD_TOLERANCE if tol is None else float(tol))
            after = check_mesh(verts, tris)
            if report.did_anything():
                self._echo_fn(f"WARNING: import: repaired the mesh -- {report.summary()}{self._loc(pos)}")
            if not after.ok() and not before.ok():
                self._echo_fn(f"WARNING: import: still not manifold after repair -- {after.summary()}{self._loc(pos)}")
        body = m3d.Manifold(m3d.Mesh64(vert_properties=verts, tri_verts=tris.astype(np.uint64)))
        if body.status() == _MANIFOLD_OK:
            return [self._tag_generated(body, node, color)]
        if not self._hull_depth:
            why = check_mesh(verts, tris).summary() or body.status().name
            self._echo_fn(
                f"WARNING: import: mesh is not a closed solid ({why}); drawing the object as an open "
                "surface rather than a solid -- nothing is patched. hull() can still use its points, "
                "but it cannot take part in union/difference/intersection"
                f"{'' if params.get('repair') else '. Try import(..., repair=true)'}{self._loc(pos)}")
        return [ColoredBody(color=color, raw_mesh=(verts, tris))]

    def _generate_mesh_repair(self, params: dict, children: list[CSGNode], node: ASTNode) -> list[ColoredBody]:
        """mesh_repair(tolerance): make an almost-closed mesh a solid -- weld,
        drop degenerate and duplicate faces, orient, fill holes, flip
        outward, strip zero-area faces. Not in OpenSCAD (cpp #191). Reads an
        open body's raw_mesh, since it has no Manifold to repair. Welding
        discards vertices, so the default tolerance is tight; a gap closed by
        filling rather than welding can leave a needle, which is thin, not
        zero-area, and stays -- a larger tolerance is the fix for that."""
        from .mesh_check import DEFAULT_WELD_TOLERANCE, repair_mesh
        bodies = flatten_csg_tree(children)
        pos = getattr(node, "position", None)
        tol = params["tolerance"]
        if type(tol) in (int, float):
            if tol < 0:
                self._echo_fn(f"WARNING: mesh_repair: tolerance must not be negative{self._loc(pos)}")
                return bodies
        elif tol is not None:
            self._echo_fn(f"WARNING: mesh_repair: tolerance must be a number{self._loc(pos)}")
            tol = None
        tol = DEFAULT_WELD_TOLERANCE if tol is None else float(tol)
        out = []
        for b in bodies:
            if b.role != "normal" or (b.body is None and b.raw_mesh is None):
                out.append(b)
                continue
            if b.raw_mesh is not None:
                verts, tris = b.raw_mesh
            else:
                mesh = b.body.to_mesh64()
                verts, tris = np.array(mesh.vert_properties)[:, :3], np.array(mesh.tri_verts)
            verts, tris, report = repair_mesh(verts, tris, tol)
            if report.did_anything():
                rebuilt = m3d.Manifold(m3d.Mesh64(vert_properties=verts, tri_verts=tris.astype(np.uint64)))
                if rebuilt.status() == _MANIFOLD_OK:
                    # A solid now: drop the display-only soup, or it stays out of
                    # every boolean; and tri_colors, since the count changed.
                    # ponytail: tagged to this node so it stays pickable, which
                    # the C++ port does not do.
                    b = replace(self._tag_generated(rebuilt, node, b.color), role=b.role)
                    self._echo_fn(f"WARNING: mesh_repair: {report.summary()}{self._loc(pos)}")
                else:
                    self._echo_fn(f"WARNING: mesh_repair: could not rebuild the repaired mesh "
                                  f"({rebuilt.status().name}); geometry left unchanged{self._loc(pos)}")
            if report.unfilled_holes:
                self._echo_fn(f"WARNING: mesh_repair: {report.unfilled_holes} hole(s) left open -- a crack "
                              f"too narrow to fill needs a larger tolerance to weld shut instead{self._loc(pos)}")
            out.append(b)
        return out

    def _load_stl(self, path: str):
        """Return (verts, tris) from binary or ASCII STL."""
        import struct as _struct
        with open(path, "rb") as f:
            header = f.read(80)
            rest   = f.read()
        try:
            sample = (header + rest[:256]).decode("ascii", errors="ignore")
            is_ascii = "facet normal" in sample
        except Exception:
            is_ascii = False
        if is_ascii:
            text = (header + rest).decode("ascii", errors="replace")
            verts: list = []; tris: list = []; tri_verts: list = []
            for line in text.splitlines():
                line = line.strip()
                if line.startswith("vertex "):
                    parts = line.split()
                    tri_verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    if len(tri_verts) == 3:
                        base = len(verts)
                        verts.extend(tri_verts)
                        tris.append([base, base + 1, base + 2])
                        tri_verts = []
        else:
            count = _struct.unpack_from("<I", rest, 0)[0]
            dtype = np.dtype([("normal", np.float32, (3,)),
                              ("v0", np.float32, (3,)), ("v1", np.float32, (3,)),
                              ("v2", np.float32, (3,)), ("attr", np.uint16)])
            data  = np.frombuffer(rest[4:4 + count * 50], dtype=dtype)
            verts = np.empty((count * 3, 3), dtype=np.float64)
            verts[0::3] = data["v0"]; verts[1::3] = data["v1"]; verts[2::3] = data["v2"]
            tris = np.arange(count * 3, dtype=np.uint32).reshape(-1, 3)
        return self._weld_stl_vertices(verts, tris)

    def _weld_stl_vertices(self, verts, tris):
        """STL has no vertex-index concept -- each triangle carries its own
        private copy of its 3 corner positions -- so a naive load produces a
        "vertex soup" with no shared indices at shared edges. manifold3d
        requires welded/shared indices to recognize a mesh as a closed
        manifold: confirmed empirically that the exact same cube topology
        already validated via _UNIT_CUBE_OBJ (volume 1, Error.NoError) comes
        back as Error.NotManifold / volume 0 once expanded into a vertex
        soup, even though it's a perfectly valid closed solid. Merging
        coincident vertices (exact-match, which is what matters for STL's
        own repeated-corner floats) and remapping triangle indices through
        the merge fixes this for every STL, not just malformed ones."""
        verts_arr = np.asarray(verts, dtype=np.float64)
        if len(verts_arr) == 0:
            return verts_arr, np.asarray(tris, dtype=np.uint32)
        unique_verts, inverse = np.unique(verts_arr, axis=0, return_inverse=True)
        welded_tris = np.asarray(inverse, dtype=np.uint32).reshape(-1)[np.asarray(tris, dtype=np.int64)]
        return unique_verts, welded_tris

    def _load_obj(self, path: str):
        verts: list[list[float]] = []; tris: list[list[int]] = []
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line.startswith("v "):
                    p = line.split()
                    verts.append([float(p[1]), float(p[2]), float(p[3])])
                elif line.startswith("f "):
                    idx = [int(p.split("/")[0]) - 1 for p in line.split()[1:]]
                    for i in range(1, len(idx) - 1):
                        tris.append([idx[0], idx[i], idx[i + 1]])
        return verts, tris

    def _load_off(self, path: str):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]
        idx = 0
        if lines[idx].upper().startswith("OFF"):
            idx += 1
        n_v, n_f, _ = (int(x) for x in lines[idx].split()); idx += 1
        verts = []
        for _ in range(n_v):
            p = lines[idx].split(); verts.append([float(p[0]), float(p[1]), float(p[2])]); idx += 1
        tris: list[list[int]] = []
        for _ in range(n_f):
            p = [int(x) for x in lines[idx].split()]; idx += 1
            cnt, face_idx = p[0], p[1:p[0] + 1]
            for i in range(1, cnt - 1):
                tris.append([face_idx[0], face_idx[i], face_idx[i + 1]])
        return verts, tris

    def _load_3mf(self, path: str):
        import zipfile as _zf
        import xml.etree.ElementTree as _ET
        NS = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
        verts_all: list[list[float]] = []; tris_all: list[list[int]] = []
        with _zf.ZipFile(path) as z:
            model_name = next((n for n in z.namelist() if n.lower().endswith("3dmodel.model")), None)
            if model_name is None:
                raise ValueError("No 3dmodel.model found in 3MF archive")
            with z.open(model_name) as f:
                tree = _ET.parse(f)
        for mesh_el in tree.iter(f"{{{NS}}}mesh"):
            verts_el = mesh_el.find(f"{{{NS}}}vertices")
            tris_el  = mesh_el.find(f"{{{NS}}}triangles")
            if verts_el is None or tris_el is None:
                continue
            base = len(verts_all)
            for v in verts_el:
                verts_all.append([float(v.get("x", 0)), float(v.get("y", 0)), float(v.get("z", 0))])
            for t in tris_el:
                tris_all.append([base + int(t.get("v1")), base + int(t.get("v2")), base + int(t.get("v3"))])
        return verts_all, tris_all

    def _load_dxf_contours(self, path: str, layer: Any, node) -> Optional[list]:
        try:
            import ezdxf as _ezdxf
        except ImportError:
            self.error("import: DXF requires the 'ezdxf' library (pip install ezdxf)", node)
            return None
        doc = _ezdxf.readfile(path)
        msp = doc.modelspace()
        contours: list[list[tuple[float, float]]] = []
        for entity in msp:
            if layer is not None and entity.dxf.layer != str(layer):
                continue
            dtype = entity.dxftype()
            if dtype == "LWPOLYLINE":
                pts = [(p[0], p[1]) for p in entity.get_points()]
                if pts and entity.is_closed:
                    contours.append(pts)
            elif dtype == "POLYLINE" and entity.is_2d_polyline:
                pts = [(v.dxf.location.x, v.dxf.location.y) for v in entity.vertices]
                if pts and entity.is_closed:
                    contours.append(pts)
        return contours

    def _load_svg_contours(self, path: str, node=None, id_=None, cls=None, dpi=72.0,
                           center: bool = False) -> list[list[tuple[float, float]]]:
        """The SVG's filled outlines. `id_` (upstream's) and `cls` (this
        port's, supported_feature("svg-class")) select elements: a match is
        taken whole, so id= on a <g> means that group, with the transforms
        above it still applied so it lands where it does in the drawing. A
        miss imports nothing and warns -- falling back to the whole drawing
        would hand a cut layer every layer, in silence (cpp #182). `layer=`
        stays DXF's: upstream reads it from Inkscape's inkscape:label."""
        import xml.etree.ElementTree as _ET
        import re as _re
        import math as _math

        SEGS = 32

        def _parse_transform(t_str: str) -> np.ndarray:
            m = np.eye(3, dtype=np.float64)
            if not t_str:
                return m
            for cmd, args_s in _re.findall(r'(\w+)\(([^)]*)\)', t_str):
                ns = [float(x) for x in _re.split(r'[,\s]+', args_s.strip()) if x]
                if cmd == "matrix" and len(ns) >= 6:
                    a, b, c, d, e, f = ns[:6]
                    m = np.array([[a, c, e], [b, d, f], [0, 0, 1]], dtype=np.float64) @ m
                elif cmd == "translate":
                    tx, ty = ns[0], ns[1] if len(ns) > 1 else 0.0
                    m = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64) @ m
                elif cmd == "scale":
                    sx, sy = ns[0], ns[1] if len(ns) > 1 else ns[0]
                    m = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64) @ m
                elif cmd == "rotate":
                    a  = _math.radians(ns[0])
                    cx = ns[1] if len(ns) > 1 else 0.0
                    cy = ns[2] if len(ns) > 2 else 0.0
                    ca, sa = _math.cos(a), _math.sin(a)
                    t1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float64)
                    r  = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]], dtype=np.float64)
                    t2 = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], dtype=np.float64)
                    m  = t2 @ r @ t1 @ m
            return m

        def _apply(pt: tuple, mat: np.ndarray) -> tuple:
            v = mat @ np.array([pt[0], pt[1], 1.0])
            return (float(v[0]), float(v[1]))  # SVG user units; _svg_page_map places them

        def _cubic(p0, p1, p2, p3):
            pts = []
            for i in range(1, SEGS + 1):
                t = i / SEGS; mt = 1 - t
                pts.append((mt**3*p0[0]+3*mt**2*t*p1[0]+3*mt*t**2*p2[0]+t**3*p3[0],
                             mt**3*p0[1]+3*mt**2*t*p1[1]+3*mt*t**2*p2[1]+t**3*p3[1]))
            return pts

        def _quad(p0, p1, p2):
            pts = []
            for i in range(1, SEGS + 1):
                t = i / SEGS; mt = 1 - t
                pts.append((mt**2*p0[0]+2*mt*t*p1[0]+t**2*p2[0],
                             mt**2*p0[1]+2*mt*t*p1[1]+t**2*p2[1]))
            return pts

        def _arc(x1, y1, rx, ry, x_rot, large, sweep, x2, y2):
            if rx == 0 or ry == 0:
                return [(x2, y2)]
            cos_r = _math.cos(_math.radians(x_rot)); sin_r = _math.sin(_math.radians(x_rot))
            dx, dy = (x1 - x2) / 2, (y1 - y2) / 2
            x1p =  cos_r*dx + sin_r*dy; y1p = -sin_r*dx + cos_r*dy
            lam = (x1p/rx)**2 + (y1p/ry)**2
            if lam > 1:
                rx *= _math.sqrt(lam); ry *= _math.sqrt(lam)
            sq = max(0.0, (rx*ry)**2 - (rx*y1p)**2 - (ry*x1p)**2)
            sq = _math.sqrt(sq / max(1e-12, (rx*y1p)**2 + (ry*x1p)**2))
            if large == sweep:
                sq = -sq
            cxp = sq*rx*y1p/ry; cyp = -sq*ry*x1p/rx
            cx = cos_r*cxp - sin_r*cyp + (x1+x2)/2
            cy = sin_r*cxp + cos_r*cyp + (y1+y2)/2
            def _angle(ux, uy, vx, vy): return _math.atan2(ux*vy - uy*vx, ux*vx + uy*vy)
            th1 = _angle(1, 0, (x1p-cxp)/rx, (y1p-cyp)/ry)
            dth = _angle((x1p-cxp)/rx, (y1p-cyp)/ry, (-x1p-cxp)/rx, (-y1p-cyp)/ry)
            if sweep == 0 and dth > 0: dth -= 2*_math.pi
            if sweep == 1 and dth < 0: dth += 2*_math.pi
            n = max(4, int(abs(dth)/(2*_math.pi)*SEGS*4))
            pts = []
            for i in range(1, n + 1):
                th = th1 + dth*i/n
                pts.append((cos_r*rx*_math.cos(th) - sin_r*ry*_math.sin(th) + cx,
                             sin_r*rx*_math.cos(th) + cos_r*ry*_math.sin(th) + cy))
            return pts

        def _parse_d(d: str, mat: np.ndarray) -> list:
            toks = _re.findall(
                r'[MmZzLlHhVvCcSsQqTtAa]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?', d)
            contours: list = []; contour: list = []
            cur = (0.0, 0.0); start = (0.0, 0.0); last_ctrl = None; cmd = "M"; ti = 0

            def nx():
                nonlocal ti; v = float(toks[ti]); ti += 1; return v

            while ti < len(toks):
                t = toks[ti]
                if t in "MmZzLlHhVvCcSsQqTtAa":
                    cmd = t; ti += 1; last_ctrl = None; continue
                rel = cmd.islower(); ox, oy = cur if rel else (0.0, 0.0); lc = cmd.upper()
                if lc == "M":
                    if contour: contours.append(contour)
                    cur = (nx()+ox, nx()+oy); start = cur
                    contour = [_apply(cur, mat)]; cmd = "l" if rel else "L"
                elif lc == "Z":
                    if contour: contours.append(contour)
                    cur = start; contour = []
                elif lc == "L":
                    cur = (nx()+ox, nx()+oy); contour.append(_apply(cur, mat))
                elif lc == "H":
                    cur = (nx()+ox, cur[1]); contour.append(_apply(cur, mat))
                elif lc == "V":
                    cur = (cur[0], nx()+oy); contour.append(_apply(cur, mat))
                elif lc == "C":
                    p1 = (nx()+ox, nx()+oy); p2 = (nx()+ox, nx()+oy); p3 = (nx()+ox, nx()+oy)
                    last_ctrl = p2
                    for pt in _cubic(cur, p1, p2, p3): contour.append(_apply(pt, mat))
                    cur = p3
                elif lc == "S":
                    refl = (2*cur[0]-last_ctrl[0], 2*cur[1]-last_ctrl[1]) if last_ctrl else cur
                    p2 = (nx()+ox, nx()+oy); p3 = (nx()+ox, nx()+oy); last_ctrl = p2
                    for pt in _cubic(cur, refl, p2, p3): contour.append(_apply(pt, mat))
                    cur = p3
                elif lc == "Q":
                    p1 = (nx()+ox, nx()+oy); p2 = (nx()+ox, nx()+oy); last_ctrl = p1
                    for pt in _quad(cur, p1, p2): contour.append(_apply(pt, mat))
                    cur = p2
                elif lc == "T":
                    refl = (2*cur[0]-last_ctrl[0], 2*cur[1]-last_ctrl[1]) if last_ctrl else cur
                    p2 = (nx()+ox, nx()+oy); last_ctrl = refl
                    for pt in _quad(cur, refl, p2): contour.append(_apply(pt, mat))
                    cur = p2
                elif lc == "A":
                    rx2, ry2, xr, lg, sw = nx(), nx(), nx(), int(nx()), int(nx())
                    ex, ey = nx()+ox, nx()+oy
                    for pt in _arc(cur[0], cur[1], rx2, ry2, xr, lg, sw, ex, ey):
                        contour.append(_apply(pt, mat))
                    cur = (ex, ey)
            if contour:
                contours.append(contour)
            return contours

        def _shape_contours(el, mat: np.ndarray) -> list:
            tag = el.tag.split("}")[-1] if "}" in el.tag else el.tag
            if tag == "path":
                return _parse_d(el.get("d", ""), mat)
            if tag in ("polygon", "polyline"):
                # Both treated as closed fill contours here (this import
                # path only ever produces closed CrossSection polygons,
                # same as the DXF loader) -- <polyline> is nominally an
                # open SVG shape, but there's no "open path" concept
                # downstream to preserve, so closing it is the only useful
                # interpretation. Previously restricted to tag ==
                # "polygon", which made every <polyline> a silent no-op.
                nums = [float(x) for x in _re.split(r'[,\s]+', el.get("points", "").strip()) if x]
                pts = list(zip(nums[::2], nums[1::2]))
                return [[_apply(p, mat) for p in pts]] if pts else []
            if tag == "rect":
                x = float(el.get("x", 0)); y = float(el.get("y", 0))
                w = float(el.get("width", 0)); h = float(el.get("height", 0))
                pts = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
                return [[_apply(p, mat) for p in pts]]
            if tag == "circle":
                cx = float(el.get("cx", 0)); cy = float(el.get("cy", 0)); r = float(el.get("r", 0))
                pts = [(cx+r*_math.cos(2*_math.pi*i/SEGS), cy+r*_math.sin(2*_math.pi*i/SEGS))
                       for i in range(SEGS)]
                return [[_apply(p, mat) for p in pts]]
            if tag == "ellipse":
                cx = float(el.get("cx", 0)); cy = float(el.get("cy", 0))
                rx = float(el.get("rx", 0)); ry = float(el.get("ry", 0))
                pts = [(cx+rx*_math.cos(2*_math.pi*i/SEGS), cy+ry*_math.sin(2*_math.pi*i/SEGS))
                       for i in range(SEGS)]
                return [[_apply(p, mat) for p in pts]]
            return []

        def _walk(el, mat: np.ndarray) -> list:
            tag = el.tag.split("}")[-1] if "}" in el.tag else el.tag
            if tag in ("defs", "symbol"):
                return []
            m = _parse_transform(el.get("transform", "")) @ mat
            out = _shape_contours(el, m)
            for child in el:
                out.extend(_walk(child, m))
            return out

        tree = _ET.parse(path)
        root = tree.getroot()
        dpi = float(dpi) if type(dpi) in (int, float) and dpi > 0 else 72.0
        id_ = id_ if isinstance(id_, str) else None
        cls = cls if isinstance(cls, str) else None
        if id_ is None and cls is None:
            return _svg_page_map(_walk(root, np.eye(3, dtype=np.float64)), root, dpi, center)
        matched = False

        def _walk_filtered(el, mat: np.ndarray) -> list:
            nonlocal matched
            tag = el.tag.split("}")[-1] if "}" in el.tag else el.tag
            if tag in ("defs", "symbol"):
                return []
            if (id_ is not None and el.get("id") == id_) or (cls is not None and cls in el.get("class", "").split()):
                matched = True
                return _walk(el, mat)  # taken whole, its own transform included
            m = _parse_transform(el.get("transform", "")) @ mat
            return [c for child in el for c in _walk_filtered(child, m)]

        out = _walk_filtered(tree.getroot(), np.eye(3, dtype=np.float64))
        if not matched:
            what = ", ".join(f'{k} = "{v}"' for k, v in (("id", id_), ("class", cls)) if v is not None)
            self._echo_fn(f"WARNING: import() filter {what} did not match anything"
                          f"{self._loc(getattr(node, 'position', None))}")
        return _svg_page_map(out, root, dpi, center)

    def _resolve_offset(self, node: ModularCall, ctx: EvalContext) -> dict:
        args, ctx = self._resolve_call_args(node, ctx)
        self._eval_children(node.children, ctx)  # side effect only, see _resolve_transform
        # offset(2) is offset(r=2), and a bare offset() is r=1, as in OpenSCAD
        # (both did nothing).
        r = self._get_arg(args, 0, "r", None)
        delta = self._get_arg(args, None, "delta", None)
        if r is None and delta is None:
            r = 1.0
        chamfer = bool(self._get_arg(args, None, "chamfer", False))
        segs = self._fn(ctx, abs(float(r))) if r is not None else None
        return {"r": r, "delta": delta, "chamfer": chamfer, "segs": segs, "color": ctx.color}

    def _generate_offset(self, params: dict, children: list[CSGNode], node: ASTNode) -> list[ColoredBody]:
        bodies = flatten_csg_tree(children)
        cs = self._to_cross_section(bodies)
        if cs is None:
            return []
        r, delta, chamfer = params["r"], params["delta"], params["chamfer"]
        if r is not None:
            result = cs.offset(float(r), m3d.JoinType.Round, circular_segments=params["segs"])
        elif delta is not None:
            # Sharp corners unless chamfered -- these were the wrong way round:
            # offset(delta=2) square(10) is 14 x 14 (196), not cut at the corners.
            jt = m3d.JoinType.Square if chamfer else m3d.JoinType.Miter
            result = cs.offset(float(delta), jt)
        else:
            return [bodies[0]] if bodies else []
        return [ColoredBody(section=result, color=params["color"])]

    def _resolve_projection(self, node: ModularCall, ctx: EvalContext) -> dict:
        args, ctx = self._resolve_call_args(node, ctx)
        self._eval_children(node.children, ctx)  # side effect only, see _resolve_transform
        return {"cut": bool(self._get_arg(args, None, "cut", False))}

    def _generate_projection(self, params: dict, children: list[CSGNode], node: ASTNode) -> list[ColoredBody]:
        bodies = flatten_csg_tree(children)
        bodies_3d = [c for c in bodies if c.body is not None]
        if not bodies_3d:
            return []
        combined = self._combine(bodies_3d).body
        try:
            if params["cut"]:
                cs = combined.slice(0.0)
            else:
                raw = combined.project()
                # project() may produce self-intersecting polygons; re-fill to clean up
                polys = raw.to_polygons()
                cs = m3d.CrossSection(polys, m3d.FillRule.Positive) if polys else raw
            return [ColoredBody(section=cs, color=bodies_3d[0].color)]
        except Exception as e:
            self.error(f"projection: {e}", node)

    def _resolve_2d(self, node: ModularCall, ctx: EvalContext) -> dict:
        """circle/square/polygon share one dispatch entry, matching
        _builtin_2d's own name-based if/elif structure (kind == name for a
        ModularCall, so name is re-derived from node.name.name here)."""
        name = node.name.name
        args, ctx = self._resolve_call_args(node, ctx)
        try:
            if name == "circle":
                r = self._get_arg(args, 0, "r", None)
                d = self._get_arg(args, None, "d", None)
                if d is not None:
                    r = d / 2
                if r is None:
                    r = 1.0
                r = float(r)
                segs = self._fn(ctx, r)
                return {"name": name, "r": r, "segs": segs, "color": ctx.color}
            if name == "square":
                size = self._get_arg(args, 0, "size", 1.0)
                center = bool(self._get_arg(args, 1, "center", False))
                if isinstance(size, (int, float)):
                    size = [size, size]
                return {"name": name, "size": [float(size[0]), float(size[1])],
                        "center": center, "color": ctx.color}
            # polygon
            points = self._get_arg(args, 0, "points", None)
            paths = self._get_arg(args, 1, "paths", None)
            if isinstance(points, OscObject):
                # polygon(obj): the 2D counterpart of polyhedron(obj). A missing
                # `paths` means one contour, as it does for the list form.
                verts = points.get("vertices")
                if verts is None:
                    verts = points.get("points")
                if verts is None:
                    raise ValueError("object has no 'vertices' (or 'points') key")
                points, paths = verts, points.get("paths")
            if points is None:
                self.error("polygon: 'points' is required", node)
            pts = [[float(p[0]), float(p[1])] for p in points]
            path_indices = None if paths is None else [[int(i) for i in path] for path in paths]
            return {"name": name, "pts": pts, "paths": path_indices, "color": ctx.color}
        except Exception as e:
            self.error(f"{name}: {e}", node)

    def _generate_2d(self, params: dict, children: list[CSGNode], node: ASTNode) -> list[ColoredBody]:
        name = params["name"]
        try:
            if name == "circle":
                cs = m3d.CrossSection.circle(params["r"], params["segs"])
            elif name == "square":
                cs = m3d.CrossSection.square(params["size"], params["center"])
            else:  # polygon
                pts, paths = params["pts"], params["paths"]
                if paths is None:
                    contour = np.array(pts, dtype=np.float64)
                    cs = m3d.CrossSection([contour], m3d.FillRule.EvenOdd)
                else:
                    contours = [np.array([pts[i] for i in path], dtype=np.float64) for path in paths]
                    cs = m3d.CrossSection(contours, m3d.FillRule.EvenOdd)
            return [ColoredBody(section=cs, color=params["color"])]
        except Exception as e:
            self.error(f"{name}: {e}", node)

    def _resolve_text(self, node: ModularCall, ctx: EvalContext) -> dict:
        """`text(text=.., size=.., font=.., halign=.., valign=.., spacing=..)`.

        Renders `text` as 2D glyph outlines, using the font specified by `font=`
        (an OpenSCAD/fontconfig pattern such as `"Times New Roman:style=Bold"`).
        Resolved via `fc-match` when available; falls back to bundled Liberation
        Sans if the font cannot be found.  Shaped by HarfBuzz; `direction`,
        `language` and `script` are honoured, and guessed from the text when unset.
        """
        args, ctx = self._resolve_call_args(node, ctx)
        text = self._get_arg(args, 0, "text", "")
        size = self._get_arg(args, 1, "size", 10)
        font_spec = self._get_arg(args, 2, "font", "") or ""  # positional, but nothing after it is
        halign = self._get_arg(args, None, "halign", "left")
        valign = self._get_arg(args, None, "valign", "baseline")
        spacing = self._get_arg(args, None, "spacing", 1)
        shaping = [self._get_arg(args, None, k, "") or "" for k in ("direction", "language", "script")]

        try:
            font = _resolve_font(str(font_spec))
            scale = size * (100 / 72) / font["units_per_em"]
            segs = max(2, self._fn(ctx) // 2)
            m = _measure_text(text, size, spacing, font, *map(str, shaping))
            offset_x, offset_y = _text_align_offset(halign, valign, m)
        except Exception as e:
            self.error(f"text: {e}", node)
        # font_spec (not the font dict itself) is cached: _resolve_font()
        # memoizes per spec string, so re-resolving it in generate is cheap
        # and doesn't need the (possibly large) font dict carried through.
        return {"font_spec": str(font_spec), "glyphs": m["glyphs"], "scale": scale,
                "segs": segs, "offset": (offset_x, offset_y), "color": ctx.color}

    def _generate_text(self, params: dict, children: list[CSGNode], node: ASTNode) -> list[ColoredBody]:
        try:
            font = _resolve_font(params["font_spec"])
            scale = params["scale"]
            sections = []
            for gname, x, y in params["glyphs"]:
                glyph_cs = _glyph_cross_section(gname, params["segs"], font)
                sections.append(glyph_cs.scale([scale, scale]).translate([x, y]))
            cs = m3d.CrossSection.batch_boolean(sections, m3d.OpType.Add) if sections else m3d.CrossSection()
            offset_x, offset_y = params["offset"]
            cs = cs.translate([offset_x, offset_y])
            return [ColoredBody(section=cs, color=params["color"])]
        except Exception as e:
            self.error(f"text: {e}", node)

    def _resolve_linear_extrude(self, node: ModularCall, ctx: EvalContext) -> dict:
        args, ctx = self._resolve_call_args(node, ctx)
        self._eval_children(node.children, ctx)  # side effect only, see _resolve_transform
        height = float(self._get_arg(args, 0, "height", 1.0))
        center = bool(self._get_arg(args, None, "center", False))
        twist = float(self._get_arg(args, None, "twist", 0.0))
        slices = int(self._get_arg(args, None, "slices", 0))
        scale = self._get_arg(args, None, "scale", None)
        if scale is None:
            scale_top = (1.0, 1.0)
        elif isinstance(scale, (int, float)):
            scale_top = (float(scale), float(scale))
        else:
            scale_top = (float(scale[0]), float(scale[1]))
        return {"height": height, "center": center, "twist": twist, "slices": slices,
                "scale_top": scale_top, "color": ctx.color}

    def _generate_linear_extrude(self, params: dict, children: list[CSGNode], node: ASTNode) -> list[ColoredBody]:
        cs = self._to_cross_section(flatten_csg_tree(children))
        if cs is None or cs.is_empty():
            return []
        try:
            body = m3d.Manifold.extrude(cs, params["height"], params["slices"],
                                         -params["twist"], params["scale_top"])
            if params["center"]:
                body = body.translate([0, 0, -params["height"] / 2])
            return [self._tag_generated(body, node, params["color"])]
        except Exception as e:
            self.error(f"linear_extrude: {e}", node)
            return []

    def _resolve_rotate_extrude(self, node: ModularCall, ctx: EvalContext) -> dict:
        args, ctx = self._resolve_call_args(node, ctx)
        self._eval_children(node.children, ctx)  # side effect only, see _resolve_transform
        angle = float(self._get_arg(args, 0, "angle", 360.0))
        return {"angle": angle, "fn": ctx.dyn.get("$fn", 0), "fa": ctx.dyn.get("$fa", 12.0),
                "fs": ctx.dyn.get("$fs", 2.0), "color": ctx.color}

    def _generate_rotate_extrude(self, params: dict, children: list[CSGNode], node: ASTNode) -> list[ColoredBody]:
        cs = self._to_cross_section(flatten_csg_tree(children))
        if cs is None or cs.is_empty():
            return []
        # max_x depends on the merged children's bounds, which doesn't exist
        # until generate — segment count can't be precomputed in resolve the
        # way e.g. offset's can, so it's derived here from cached $fn/$fa/$fs.
        bounds = cs.bounds()
        max_x = max(abs(bounds[0]), abs(bounds[2])) if bounds else 0.0
        segs = self._fn_segments(params["fn"], params["fa"], params["fs"], max_x)
        try:
            body = cs.revolve(segs, params["angle"])
            return [self._tag_generated(body, node, params["color"])]
        except Exception as e:
            self.error(f"rotate_extrude: {e}", node)
            return []

    def _resolve_roof(self, node: ModularCall, ctx: EvalContext) -> dict:
        args, ctx = self._resolve_call_args(node, ctx)
        self._eval_children(node.children, ctx)  # side effect only, see _resolve_transform
        method = self._get_arg(args, None, "method", "voronoi")
        if method not in ("voronoi", "straight"):
            self._echo_fn(f"WARNING: Unknown roof method '{method}'. Using 'voronoi'.")
            method = "voronoi"
        return {"method": method, "color": ctx.color}

    def _generate_roof(self, params: dict, children: list[CSGNode], node: ASTNode) -> list[ColoredBody]:
        cs = self._to_cross_section(flatten_csg_tree(children))
        if cs is None:
            return []
        try:
            if not cs.to_polygons():
                return []
            body = _skeleton_roof(cs)
            if body is None:
                body = _skeleton_roof_general(cs)
            if body is None:
                body = self._roof_sdf_fallback(cs)
            if body is None:
                return []
            return [self._tag_generated(body, node, params["color"])]
        except Exception as e:
            self.error(f"roof: {e}", node)
            return []

    def _roof_sdf_fallback(self, cs: m3d.CrossSection) -> Optional[m3d.Manifold]:
        """Signed-distance-field/`level_set` approximation of a roof, used
        when `_skeleton_roof` doesn't apply (holes, multi-contour, or a
        mitered-offset collapse with intermediate topology events)."""
        polys = cs.to_polygons()
        if not polys:
            return None
        edge_a_list, edge_b_list = [], []
        for poly in polys:
            n = len(poly)
            for i in range(n):
                edge_a_list.append(poly[i])
                edge_b_list.append(poly[(i + 1) % n])
        edge_a = np.array(edge_a_list, dtype=np.float64)  # (E, 2)
        edge_b = np.array(edge_b_list, dtype=np.float64)  # (E, 2)
        # Precompute per-edge AB and squared-length for fast per-voxel SDF.
        ab = edge_b - edge_a  # (E, 2)
        ab_sq = np.einsum('ij,ij->i', ab, ab)  # (E,)
        raw_edges = list(zip(edge_a, edge_b))  # for even-odd test

        minx, miny, maxx, maxy = cs.bounds()
        width, height = maxx - minx, maxy - miny

        # Scan a coarse grid to find the true maximum interior distance (= roof
        # height). Bounding-box heuristics badly overestimate for thin glyphs.
        _n = 40
        max_sdf = 0.0
        for xi in range(_n):
            for yi in range(_n):
                x = minx + width * xi / (_n - 1)
                y = miny + height * yi / (_n - 1)
                p = np.array([x, y])
                pa = p - edge_a
                t = np.einsum('ij,ij->i', pa, ab) / np.where(ab_sq > 0, ab_sq, 1.0)
                t = np.clip(t, 0.0, 1.0)
                d = float(np.min(np.linalg.norm(pa - t[:, None] * ab, axis=1)))
                if _point_in_poly_evenodd(p, raw_edges):
                    max_sdf = max(max_sdf, d)
        if max_sdf <= 0:
            return None
        z_max = max_sdf * 1.02
        edge_length = z_max / 5
        eps = edge_length / 2

        def sdf(x, y, z):
            p = np.array([x, y])
            pa = p - edge_a
            t = np.einsum('ij,ij->i', pa, ab) / np.where(ab_sq > 0, ab_sq, 1.0)
            t = np.clip(t, 0.0, 1.0)
            d = float(np.min(np.linalg.norm(pa - t[:, None] * ab, axis=1)))
            d2 = d if _point_in_poly_evenodd(p, raw_edges) else -d
            return d2 - z

        bounds = [minx - eps, miny - eps, 0.0, maxx + eps, maxy + eps, z_max + eps]
        body = m3d.Manifold.level_set(sdf, bounds, edge_length)
        if body.is_empty():
            return None
        return body.simplify(edge_length * 0.05)

    def _resolve_minkowski(self, node: ModularCall, ctx: EvalContext) -> dict:
        args, ctx = self._resolve_call_args(node, ctx)
        self._eval_children(node.children, ctx)  # side effect only, see _resolve_transform
        return {}

    def _generate_minkowski(self, params: dict, children: list[CSGNode], node: ASTNode) -> list[ColoredBody]:
        bodies = flatten_csg_tree(children)
        bg, fg, hi, so = self._split_by_role(bodies)
        bodies_3d = [c for c in fg if c.body is not None]
        if not bodies_3d:
            sections = [c for c in fg if c.section is not None]
            if len(sections) > 1:
                # 2D children were dropped: linear_extrude(6) minkowski() {
                # square(...); circle(4); } drew nothing.
                result = sections[0].section
                for c in sections[1:]:
                    result = _minkowski_2d(result, c.section)
                return [ColoredBody(section=result, color=sections[0].color)] + bg + hi + so
            return sections + bg + hi + so
        if len(bodies_3d) == 1:
            return bodies_3d + bg + hi + so
        try:
            result = bodies_3d[0].body
            for c in bodies_3d[1:]:
                result = result.minkowski_sum(c.body)
            return [ColoredBody(body=result, color=bodies_3d[0].color)] + bg + hi + so
        except Exception as e:
            self.error(f"minkowski: {e}", node)
            return bg + hi + so

    def _generate_minkowski_difference(self, params: dict, children: list[CSGNode], node: ASTNode) -> list[ColoredBody]:
        """Erosion, which minkowski() has no inverse for: the first child
        eroded by each later one in turn. Not in OpenSCAD (cpp #101). 3D
        only -- 2D already has offset(r=-N), and Manifold no 2D erosion."""
        bg, fg, hi, so = self._split_by_role(flatten_csg_tree(children))
        solids = [c for c in fg if c.body is not None]
        rest = bg + hi + so
        if len(solids) < 2:
            return solids[:1] + rest  # nothing to erode with, as minkowski() of one child
        result = solids[0].body
        for c in solids[1:]:
            result = result.minkowski_difference(c.body)
        if result.status() != _MANIFOLD_OK:
            self._echo_fn(f"WARNING: minkowski_difference: result is not manifold"
                          f"{self._loc(getattr(node, 'position', None))}")
        # Per-triangle colour can't survive: the surface it indexed is gone.
        return [ColoredBody(body=result, color=solids[0].color)] + rest

    def _resolve_simplify(self, node: ModularCall, ctx: EvalContext) -> dict:
        args, ctx = self._resolve_call_args(node, ctx)
        self._eval_children(node.children, ctx)
        return {"tolerance": self._get_arg(args, 0, "tolerance")}

    def _resolve_levelset(self, node: ModularCall, ctx: EvalContext) -> dict:
        args, ctx = self._resolve_call_args(node, ctx)
        self._eval_children(node.children, ctx)
        return {"field": self._get_arg(args, 0, "field"), "bounds": self._get_arg(args, 1, "bounds"),
                "isovalue": self._get_arg(args, 2, "isovalue"), "invert": self._get_arg(args, 3, "invert", False),
                "edge": self._get_arg(args, 4, "edge"), "color": ctx.color}

    def _generate_levelset(self, params: dict, children: list[CSGNode], node: ASTNode) -> list[ColoredBody]:
        """levelset(field, bounds, isovalue, invert, edge): a solid, or a
        section, from an implicit surface -- 2D or 3D decided by `bounds`,
        `field` a grid (field[i][j] / [i][j][k]) or a function(x,y[,z]).
        Not in OpenSCAD; see levelset.py and openscad_cpp_evaluator #124-#127.
        The sign follows distance fields (smaller is inside); Manifold's is
        the other way, so the band flips it and invert flips back."""
        from . import levelset as ls
        pos = getattr(node, "position", None)

        def warn(msg):
            self._echo_fn(f"WARNING: levelset(): {msg}{self._loc(pos)}")
            return []

        def nums(v):
            return v if type(v) is list and all(type(x) in (int, float) and x == x for x in v) else None

        field, fn = params["field"], None
        if type(field) is Closure:
            fn = field
            if len(fn.fn.parameters or []) < 2:
                return warn("the field function needs function(x,y) for 2D or function(x,y,z) for 3D")
        b = params["bounds"]
        lo, hi = (nums(b[0]), nums(b[1])) if type(b) is list and len(b) == 2 else (None, None)
        if lo is None or hi is None or len(lo) != len(hi) or len(lo) not in (2, 3):
            return warn("bounds must be [[x0,y0],[x1,y1]] or [[x0,y0,z0],[x1,y1,z1]]")
        if not all(h > l for l, h in zip(lo, hi)):
            return warn("bounds must be increasing along every axis")
        iso, iso_lo, iso_hi = params["isovalue"], -math.inf, 0.0
        if type(iso) in (int, float):
            iso_hi = float(iso)
        elif iso is not None:  # undef is absent (cpp #125): BOSL2's wrapper forwards it
            pair = nums(iso)
            if pair is None or len(pair) != 2:
                return warn("isovalue must be a number or a [low, high] range")
            iso_lo, iso_hi = pair
            if not iso_hi > iso_lo:
                return warn("isovalue range must be increasing")
        invert = bool(params["invert"])

        def band(v):
            return ls.band_distance(v, iso_lo, iso_hi, invert)

        edge = params["edge"]
        if type(edge) in (int, float) and not edge > 0:
            warn("edge must be positive")
        edge = float(edge) if type(edge) in (int, float) and edge > 0 else 0.0
        if fn is not None and edge <= 0:
            return warn("a function field needs edge= (the sample spacing)")

        call = None
        if fn is not None:
            # No live context at generate time: a root from the closure's own
            # scope, so $-variables sit at their defaults inside it.
            fctx = EvalContext(scope=fn.fn.scope or self._root_ctx.scope)
            fctx.let = dict(fn.let)
            names = [p.name.name for p in fn.fn.parameters]

            def call(*xyz):
                for n, x in zip(names, xyz):
                    fctx.let[n] = x
                v = self._eval_expr(fn.fn.body, fctx)
                return band(float(v) if type(v) in (int, float) and math.isfinite(v) else sys.float_info.max)

        color = params["color"]
        if len(lo) == 2:
            if fn is None:
                grid = field if (type(field) is list and len(field) >= 2 and all(
                    nums(r) is not None and len(r) == len(field[0]) >= 2 for r in field)) else None
                if grid is None:
                    if type(field) is list and field and type(field[0]) is list and field[0] and type(field[0][0]) is list:
                        return warn("a 2D field must be field[i][j]; a 3D array was given")
                    return warn("2D field must be a rectangular field[i][j] of numbers")
                cs = ls.section_2d(lambda i, j: band(grid[i][j]), len(grid), len(grid[0]), lo, hi)
            else:
                nx = int(math.floor((hi[0] - lo[0]) / edge)) + 1
                ny = int(math.floor((hi[1] - lo[1]) / edge)) + 1
                if nx < 2 or ny < 2:
                    return warn("edge is larger than the bounds")
                sx, sy = (hi[0] - lo[0]) / (nx - 1), (hi[1] - lo[1]) / (ny - 1)
                cs = ls.section_2d(lambda i, j: call(lo[0] + i * sx, lo[1] + j * sy), nx, ny, lo, hi)
            return [ColoredBody(section=cs, color=color)] if cs is not None else []

        if fn is None:
            ok = (type(field) is list and field and all(type(r) is list and len(r) == len(field[0]) and r for r in field)
                  and all(type(c) is list and len(c) == len(field[0][0]) and nums(c) is not None
                          for r in field for c in r))
            if not ok:
                return warn("field must be a function(x,y,z) or a rectangular field[i][j][k] of numbers")
            n = (len(field), len(field[0]), len(field[0][0]))
            if min(n) < 2:
                return warn("the field needs at least 2 samples along each axis")
            spacing = [(hi[a] - lo[a]) / (n[a] - 1) for a in range(3)]
            if edge <= 0:
                edge = min(spacing)  # finer than the grid buys nothing
            sample = ls.grid_sampler(field, lo, spacing, band, invert)
        else:
            if len(fn.fn.parameters) < 3:
                return warn("a 3D field function needs three parameters, as in function(x,y,z) ...")
            sample = call
        body = ls.solid_3d(sample, lo, hi, edge)
        return [self._tag_generated(body, node, color)] if body is not None else []

    _SIMPLIFY_FRACTION = 0.001  # of the bounding-box diagonal, when no tolerance is given

    def _generate_simplify(self, params: dict, children: list[CSGNode], node: ASTNode) -> list[ColoredBody]:
        """Decimate within a tolerance (Manifold/CrossSection simplify). Not
        in OpenSCAD (cpp #103). The default is 0.1% of each body's own
        diagonal: simplify(0) falls back to Manifold's epsilon and changes
        nothing, and an absolute default right in mm is wrong in metres."""
        bodies = flatten_csg_tree(children)
        tol = params["tolerance"]
        explicit = type(tol) in (int, float)
        pos = getattr(node, "position", None)
        if tol is not None and not explicit:
            self._echo_fn(f"WARNING: simplify: tolerance must be a number{self._loc(pos)}")
        if explicit and tol < 0:
            self._echo_fn(f"WARNING: simplify: tolerance must not be negative{self._loc(pos)}")
            return bodies
        out = []
        for b in bodies:
            if b.role != "normal":
                out.append(b)
            elif b.body is not None:
                lo, hi_ = np.array(b.body.bounding_box()).reshape(2, 3)
                t = tol if explicit else self._SIMPLIFY_FRACTION * float(np.linalg.norm(hi_ - lo))
                # tri_colors is indexed by triangle, and decimation changes the count.
                out.append(replace(b, body=b.body.simplify(t), tri_colors=None) if t > 0 else b)
            elif b.section is not None:
                x0, y0, x1, y1 = b.section.bounds()
                t = tol if explicit else self._SIMPLIFY_FRACTION * math.hypot(x1 - x0, y1 - y0)
                out.append(replace(b, section=b.section.simplify(t)) if t > 0 else b)
            else:
                out.append(b)
        return out

    @staticmethod
    def _copy_body(b: ColoredBody) -> ColoredBody:
        return ColoredBody(body=b.body, color=b.color, section=b.section,
                           flat_preview=b.flat_preview, role=b.role)

    def _eval_children_lazy(self, ctx: EvalContext) -> list[ColoredBody]:
        """Evaluate deferred children nodes with current $-variables injected."""
        if not ctx.children_nodes:
            return []
        caller_ctx = ctx.children_caller_ctx
        if caller_ctx is None:
            return []
        eval_ctx = caller_ctx.child_ctx(
            children_nodes=caller_ctx.children_nodes,
            children_caller_ctx=caller_ctx.children_caller_ctx,
        )
        for k, v in ctx.dyn.items():
            # $children is the count of the block being forwarded -- the
            # caller's own, not that of whichever module forwards it.
            if k[0] == '$' and k != '$children':
                eval_ctx.dyn[k] = v
        for k, v in ctx.let.items():
            if k.startswith('$'):
                eval_ctx.let[k] = v
        via, self._via_children = self._via_children, True
        try:
            return self._eval_children(ctx.children_nodes, eval_ctx)
        finally:
            self._via_children = via

    @staticmethod
    def _is_separating_children_call(stmt) -> bool:
        """Syntactically `children(..., separate=...)` or `children(i, s)` --
        gated on the text first, so an ordinary children() never has its
        arguments resolved twice."""
        if type(stmt) is not ModularCall or stmt.name.name != "children":
            return False
        positional = 0
        for a in stmt.arguments:
            if type(a) is NamedArgument:
                if a.name.name == "separate":
                    return True
            else:
                positional += 1
                if positional == 2:
                    return True
        return False

    def _expand_child_statements(self, block: list, ctx: EvalContext) -> list:
        """`block` with each `children(..., separate=true)` replaced by one
        synthetic `children(k)` per child it selects (none if it selects
        nothing). They are real statements, so an operator gives each its
        own operand -- `difference() children(separate=true)` subtracts
        children 1..n from child 0 -- while a `for` inside one child stays
        one operand, and nothing leaks into another module's operator.
        Not in OpenSCAD, which silently ignores `separate` (cpp #110-#112).
        The synthetic calls carry the author's position and no scope."""
        if not any(self._is_separating_children_call(stmt) for stmt in block):
            return block
        out = []
        for stmt in block:
            if not self._is_separating_children_call(stmt):
                out.append(stmt)
                continue
            args, eff_ctx = self._resolve_call_args(stmt, ctx)
            if not self._get_arg(args, 1, "separate", False):
                out.append(stmt)  # written but false: ponytail, re-resolves its args once
                continue
            pos = stmt.position
            out.extend(ModularCall(position=pos, scope=None, name=Identifier(position=pos, scope=None, name="children"),
                                   arguments=[PositionalArgument(position=pos, scope=None,
                                                                 expr=NumberLiteral(position=pos, scope=None, val=i))],
                                   children=[])
                       for i in self._children_indices(args, eff_ctx, stmt))
        return out

    def _children_indices(self, args: dict, ctx: EvalContext, node) -> list[int]:
        """Which of ctx's forwarded children a children() call selects, with
        the reference's out-of-bounds warning."""
        geo_nodes = [c for c in (ctx.children_nodes or [])
                     if not isinstance(c, (Assignment, ModuleDeclaration, FunctionDeclaration))]
        idx = self._get_arg(args, 0, "index", None)
        if idx is None:
            return list(range(len(geo_nodes)))
        indices = [idx] if type(idx) in (int, float) else self._loop_values(idx, node)
        picked = []
        for i in indices:
            if type(i) not in (int, float):
                continue
            i = int(i)
            if 0 <= i < len(geo_nodes):
                picked.append(i)
            else:
                self._echo_fn(f"WARNING: Children index ({i}) out of bounds ({len(geo_nodes)} children)"
                              f"{self._loc(getattr(node, 'position', None))}")
        return picked

    def _builtin_children(self, args: dict, ctx: EvalContext, node=None) -> list[ColoredBody]:
        idx = self._get_arg(args, 0, "index", None)
        if idx is None:
            return self._eval_children_lazy(ctx)
        # children(N) must index into child STATEMENTS, not output bodies.
        # A filtered statement may produce 0 bodies, shifting all subsequent
        # body-index lookups — so we evaluate only the Nth statement directly.
        # N may be a number, a vector or a range (those crashed in int()).
        indices = [idx] if type(idx) in (int, float) else self._loop_values(idx, node)
        if not ctx.children_nodes:
            return []
        caller_ctx = ctx.children_caller_ctx
        if caller_ctx is None:
            return []
        geo_nodes = [c for c in ctx.children_nodes
                     if not isinstance(c, (Assignment, ModuleDeclaration, FunctionDeclaration))]
        picked = []
        for i in indices:
            if type(i) not in (int, float):
                continue
            i = int(i)
            if 0 <= i < len(geo_nodes):
                picked.append(geo_nodes[i])
            else:
                self._echo_fn(f"WARNING: Children index ({i}) out of bounds ({len(geo_nodes)} children)"
                              f"{self._loc(getattr(node, 'position', None))}")
        if not picked:
            return []
        eval_ctx = caller_ctx.child_ctx(
            children_nodes=caller_ctx.children_nodes,
            children_caller_ctx=caller_ctx.children_caller_ctx,
        )
        for k, v in ctx.dyn.items():
            # $children is the count of the block being forwarded -- the
            # caller's own, not that of whichever module forwards it.
            if k[0] == '$' and k != '$children':
                eval_ctx.dyn[k] = v
        for k, v in ctx.let.items():
            if k.startswith('$'):
                eval_ctx.let[k] = v
        result = []
        via, self._via_children = self._via_children, True
        try:
            for child in picked:
                result.extend(self._eval_children([child], eval_ctx))
        finally:
            self._via_children = via
        return result

    def _builtin_breakpoint(self, args: dict, node, ctx: EvalContext):
        cond = self._get_arg(args, 0, "condition", default=None)
        if cond is not None and not cond:
            return None
        if self._debugging:
            self._check_debug(node, ctx, forced=True)
        return None

    # --- for loops ---

    def _eval_for(self, node: ModularFor, ctx: EvalContext) -> list[ColoredBody]:
        # The parser puts body-level assignments into node.assignments alongside the actual
        # loop variables. Skip any assignment that also appears as a body node — those are
        # per-iteration let-like definitions, not loop variables.
        body_ids = {id(b) for b in node.body}
        # Each range is evaluated inside the loops before it, so
        # `for (i = [0:2], j = [0:i])` sees i.
        _av_pairs = [(assign, assign.name.name) for assign in node.assignments
                     if id(assign) not in body_ids]

        result = []
        _debugging = self._debugging

        def _nested(depth: int, parent_ctx: EvalContext) -> None:
            if depth == len(_av_pairs):
                if _debugging and node.body:
                    self._check_debug(node.body[0], parent_ctx, expr_level=True)
                result.extend(self._eval_children(node.body, parent_ctx))
                return
            assign_node, name = _av_pairs[depth]
            for val in self._loop_values(self._eval_expr(assign_node.expr, parent_ctx), assign_node):
                child = parent_ctx.child_ctx(children_nodes=ctx.children_nodes,
                                             children_caller_ctx=ctx.children_caller_ctx)
                child.let[name] = val
                if _debugging:
                    self._check_debug(assign_node, child)
                _nested(depth + 1, child)

        _nested(0, ctx)
        return result

    _MAX_RANGE_ELEMENTS = 1_000_000

    def _loop_values(self, values, at=None) -> list:
        """What a `for` iterates over: a range's elements, an object's keys,
        a string's characters, a list as is, undef as nothing, and any other
        single value once.

        A range of a million elements or more is refused, with a warning and
        no iterations, as OpenSCAD does -- `[0:1:1/0]` iterated forever. The
        limit is per range (two 1100-element ranges make 1.2M iterations
        fine); `at` is the node the warning names."""
        if values is None:
            return []
        t = type(values)
        if t is list:
            return values
        if t is OscRange:
            n = _range_count(values)
            if n >= self._MAX_RANGE_ELEMENTS:
                self._echo_fn(f"WARNING: Bad range parameter in for statement: too many elements "
                              f"({min(n, 4294967295)}){self._loc(getattr(at, 'position', None))}")
                return []
            return list(values) if n else []
        if t is OscObject or t is str:
            return list(values)
        return [values]

    def _resolve_intersection_for(self, node: ModularIntersectionFor, ctx: EvalContext) -> dict:
        # group_sizes records, per loop iteration, how many CSGNode children
        # it contributed — same rationale as _resolve_csg's group_sizes
        # (the loop body can itself contain for/if/let, which are
        # transparent in the tree, so one iteration can contribute a
        # variable number of tree children). Combining each iteration's
        # children into one body (_combine, a real Manifold call) is
        # deferred to generate — only the plain-data grouping happens here.
        body_node = node.body if isinstance(node.body, list) else [node.body]
        body_node = self._expand_child_statements(body_node, ctx)
        _debugging = self._debugging
        group_sizes: list[int] = []
        assigns = node.assignments

        # Nested like _eval_for, so a later range can read an earlier variable.
        def _nested(depth: int, parent_ctx: EvalContext) -> None:
            if depth == len(assigns):
                if _debugging and body_node:
                    self._check_debug(body_node[0], parent_ctx, expr_level=True)
                before = len(self._tree_stack[-1])
                self._eval_children(body_node, parent_ctx)
                group_sizes.append(len(self._tree_stack[-1]) - before)
                return
            assign = assigns[depth]
            for val in self._loop_values(self._eval_expr(assign.expr, parent_ctx), assign):
                loop_ctx = parent_ctx.child_ctx(children_nodes=ctx.children_nodes,
                                                children_caller_ctx=ctx.children_caller_ctx)
                loop_ctx.let[assign.name.name] = val
                _nested(depth + 1, loop_ctx)

        _nested(0, ctx)
        return {"group_sizes": group_sizes}

    def _generate_intersection_for(self, params: dict, children: list[CSGNode], node: ASTNode) -> list[ColoredBody]:
        idx = 0
        iterations = []
        for size in params["group_sizes"]:
            group_nodes = children[idx:idx + size]
            idx += size
            stmt_bodies = flatten_csg_tree(group_nodes)
            if stmt_bodies:
                iterations.append(self._combine(stmt_bodies))

        if not iterations:
            return []
        # Intersect all iteration results
        bodies_3d = [c for c in iterations if c.body is not None]
        if bodies_3d:
            result = bodies_3d[0].body
            for c in bodies_3d[1:]:
                result = result ^ c.body  # intersection
            return [ColoredBody(body=result, color=bodies_3d[0].color)]
        # 2D intersection
        sections = [c.section for c in iterations if c.section is not None]
        if sections:
            result = sections[0]
            for s in sections[1:]:
                result = result ^ s
            return [ColoredBody(section=result, color=iterations[0].color)]
        return []

    # --- let ---

    def _eval_let_block(self, node: ModularLet, ctx: EvalContext) -> list[ColoredBody]:
        child_ctx = ctx.child_ctx(children_nodes=ctx.children_nodes,
                                 children_caller_ctx=ctx.children_caller_ctx)
        for assign in node.assignments:
            if self._debugging:
                self._check_debug(assign, ctx)
            v = self._eval_expr(assign.expr, child_ctx)  # sequential: b=a+1 sees a
            # dyn/dyn_explicit are already a fresh copy (plain child_ctx(),
            # not let_child_ctx()) -- dyn_copied=True skips the redundant copy.
            self._bind_let_name(child_ctx, assign.name.name, v, True)
        body = getattr(node, 'children', None) or getattr(node, 'body', None) or []
        return self._eval_children(body, child_ctx)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _combine(self, bodies: list[ColoredBody]) -> ColoredBody:
        bodies_3d = [b for b in bodies if b.body is not None]
        if bodies_3d:
            if len(bodies_3d) == 1:
                return bodies_3d[0]
            composed = m3d.Manifold.compose([b.body for b in bodies_3d])
            return ColoredBody(body=composed, color=bodies_3d[0].color)
        # Pure 2D — union all cross sections
        sections = [b.section for b in bodies if b.section is not None]
        if not sections:
            return ColoredBody(body=m3d.Manifold())
        cs = sections[0]
        for s in sections[1:]:
            cs = cs + s
        return ColoredBody(section=cs, color=bodies[0].color)

    def _to_cross_section(self, children: list[ColoredBody]) -> Optional[m3d.CrossSection]:
        """Union all 2D children into a single CrossSection. Returns None if no 2D children."""
        sections = [c.section for c in children if c.section is not None]
        if not sections:
            return None
        cs = sections[0]
        for s in sections[1:]:
            cs = cs + s
        return cs

    # ------------------------------------------------------------------
    # Expression evaluator
    # ------------------------------------------------------------------

    def _eval_expr(self, node, ctx: EvalContext):
        t = type(node)
        if t is NumberLiteral or t is BooleanLiteral:
            return node.val
        if t is StringLiteral:
            raw = node.val
            if "\\" in raw and id(node) not in self._escapes_checked:
                self._escapes_checked.add(id(node))  # once per literal, as OpenSCAD warns at parse time
                for _ in range(_undefined_escapes(raw)):
                    self._echo_fn(f"WARNING: Undefined escape sequence{self._loc(getattr(node, 'position', None))}")
            return _unescape_string(raw)
        if t is Identifier:
            name = node.name
            let = ctx.let
            v = let.get(name)
            if v is not None:
                return v
            if name in let:
                return None
            if name[0] == '$':
                dyn = ctx.dyn
                v = dyn.get(name)
                if v is not None:
                    return v
                if name in dyn:
                    return v
            return self._eval_identifier(node, ctx)
        if t is UndefinedLiteral:
            return None
        if t is CommentedExpr:
            return self._eval_expr(node.expr, ctx)
        handler = _EXPR_DISPATCH.get(t)
        if handler is not None:
            return handler(self, node, ctx)
        return None

    # _expr_listcomp and _expr_range removed — dispatch table points directly

    def _undefined_op(self, node, op: str, a, b=_NO_OPERAND):
        """OpenSCAD's warning for an operator applied to types it has no
        meaning for -- "undefined operation (number + string)" -- which was
        missing; the result was already undef."""
        types = f"{op}{_osc_type_name(a)}" if b is _NO_OPERAND else \
            f"{_osc_type_name(a)} {op} {_osc_type_name(b)}"
        self._echo_fn(f"WARNING: undefined operation ({types}){self._loc(getattr(node, 'position', None))}")

    def _expr_add(self, node, ctx):
        a, b = self._eval_expr(node.left, ctx), self._eval_expr(node.right, ctx)
        ta, tb = type(a), type(b)
        if (ta is int or ta is float) and (tb is int or tb is float):
            return a + b
        r = _vec_add(a, b)
        if r is None:
            self._undefined_op(node, "+", a, b)
        return r

    def _expr_sub(self, node, ctx):
        a, b = self._eval_expr(node.left, ctx), self._eval_expr(node.right, ctx)
        ta, tb = type(a), type(b)
        if (ta is int or ta is float) and (tb is int or tb is float):
            return a - b
        r = _vec_sub(a, b)
        if r is None:
            self._undefined_op(node, "-", a, b)
        return r

    def _expr_mul(self, node, ctx):
        a, b = self._eval_expr(node.left, ctx), self._eval_expr(node.right, ctx)
        r = self._mul_value(a, b)
        if r is None:
            err = _matmul_error(a, b) if type(a) is list and type(b) is list else None
            if err:
                self._echo_fn(f"WARNING: {err}{self._loc(getattr(node, 'position', None))}")
            else:
                self._undefined_op(node, "*", a, b)
        return r

    @staticmethod
    def _mul_value(a, b):
        ta, tb = type(a), type(b)
        if (ta is int or ta is float) and (tb is int or tb is float):
            return a * b
        if ta is list and tb is list:
            return _matmul(a, b)
        if ta is list and tb in (int, float):
            return [_scale(b, x) for x in a]
        if tb is list and ta in (int, float):
            return [_scale(a, x) for x in b]
        if ta is bool or tb is bool:
            return None
        try:
            return a * b
        except TypeError:
            return None

    def _expr_div(self, node, ctx):
        a, b = self._eval_expr(node.left, ctx), self._eval_expr(node.right, ctx)
        r = self._div_value(a, b)
        if r is None:
            self._undefined_op(node, "/", a, b)
        return r

    @staticmethod
    def _div_value(a, b):
        ta, tb = type(a), type(b)
        if (ta is int or ta is float) and (tb is int or tb is float):
            if b == 0:
                return float('nan') if a == 0 else math.copysign(float('inf'), a)
            return a / b
        if ta is bool or tb is bool:
            return None
        if ta is list and tb in (int, float):
            return _div_scale(a, b)
        if ta not in (int, float) or tb not in (int, float):
            return None
        if b == 0:
            return float('nan') if a == 0 else math.copysign(float('inf'), a)
        return a / b

    def _expr_mod(self, node, ctx):
        a, b = self._eval_expr(node.left, ctx), self._eval_expr(node.right, ctx)
        if type(a) not in (int, float) or type(b) not in (int, float):
            self._undefined_op(node, "%", a, b)
            return None
        # C's fmod, as OpenSCAD: the sign follows the dividend (-7 % 3 is -1,
        # Python's % gave 2) and x % 0 is nan (it was undef).
        try:
            return math.fmod(a, b)
        except ValueError:  # an infinite dividend, or a zero divisor
            return math.nan

    def _expr_exp(self, node, ctx):
        a, b = self._eval_expr(node.left, ctx), self._eval_expr(node.right, ctx)
        if type(a) not in (int, float) or type(b) not in (int, float):
            self._undefined_op(node, "^", a, b)
            return None
        return self._builtin_pow(a, b)  # 0^-1 is inf, as for pow() (it was undef)

    def _expr_unary_minus(self, node, ctx):
        v = self._eval_expr(node.expr, ctx)
        if type(v) is list:
            return self._negate_list(v)
        if type(v) in (int, float):
            return -v
        self._undefined_op(node, "-", v)
        return None

    # Bitwise/shift operators, added by real OpenSCAD in PR #4833 (merged
    # 2025-03-14, "Bitwise operators. Fixes #3345."). No real integer type:
    # operands truncate-to-int64 (real OpenSCAD's own Value::toInt64(),
    # Value.cc: `trunc(toDouble())` cast to int64_t), operate in int64
    # two's-complement arithmetic, then cast back to a plain OpenSCAD
    # number (float here) -- matching real OpenSCAD's own "operates on
    # ordinary OpenSCAD numbers, no new integer type" design.
    @staticmethod
    def _to_bitwise_int64(v) -> int:
        """Truncate-to-int64 two's-complement conversion for bitwise ops.
        Python ints are arbitrary precision and already implement
        two's-complement semantics for negative numbers natively (`~x` is
        already `-x-1`, `-1 & 5` already gives `5`), so the mask below only
        matters for values that would overflow a real bounded int64 -- e.g.
        re-wrapping a left-shift's result the way int64_t's own bounded
        arithmetic would (`1 << 32 << 32 == 0`)."""
        n = int(math.trunc(v)) & 0xFFFFFFFFFFFFFFFF
        return n - 0x10000000000000000 if n >= 0x8000000000000000 else n

    def _expr_bitor(self, node, ctx):
        a, b = self._eval_expr(node.left, ctx), self._eval_expr(node.right, ctx)
        ta, tb = type(a), type(b)
        if (ta is int or ta is float) and (tb is int or tb is float):
            return float(self._to_bitwise_int64(a) | self._to_bitwise_int64(b))
        self._echo_fn(f"WARNING: undefined operation ({_osc_type_name(a)} | {_osc_type_name(b)}){self._loc(getattr(node, 'position', None))}")
        return None

    def _expr_bitand(self, node, ctx):
        a, b = self._eval_expr(node.left, ctx), self._eval_expr(node.right, ctx)
        ta, tb = type(a), type(b)
        if (ta is int or ta is float) and (tb is int or tb is float):
            return float(self._to_bitwise_int64(a) & self._to_bitwise_int64(b))
        self._echo_fn(f"WARNING: undefined operation ({_osc_type_name(a)} & {_osc_type_name(b)}){self._loc(getattr(node, 'position', None))}")
        return None

    def _expr_bitnot(self, node, ctx):
        v = self._eval_expr(node.expr, ctx)
        if type(v) is int or type(v) is float:
            return float(~self._to_bitwise_int64(v))
        self._echo_fn(f"WARNING: undefined operation (~{_osc_type_name(v)}){self._loc(getattr(node, 'position', None))}")
        return None

    def _expr_shl(self, node, ctx):
        a, b = self._eval_expr(node.left, ctx), self._eval_expr(node.right, ctx)
        ta, tb = type(a), type(b)
        if not ((ta is int or ta is float) and (tb is int or tb is float)):
            self._echo_fn(f"WARNING: undefined operation ({_osc_type_name(a)} << {_osc_type_name(b)}){self._loc(getattr(node, 'position', None))}")
            return None
        rhs = math.trunc(b)
        if rhs < 0:
            self._echo_fn(f"WARNING: negative shift{self._loc(getattr(node, 'position', None))}")
            return None
        if rhs >= 64:
            self._echo_fn(f"WARNING: shift too large{self._loc(getattr(node, 'position', None))}")
            return None
        return float(self._to_bitwise_int64(self._to_bitwise_int64(a) << rhs))

    def _expr_shr(self, node, ctx):
        a, b = self._eval_expr(node.left, ctx), self._eval_expr(node.right, ctx)
        ta, tb = type(a), type(b)
        if not ((ta is int or ta is float) and (tb is int or tb is float)):
            self._echo_fn(f"WARNING: undefined operation ({_osc_type_name(a)} >> {_osc_type_name(b)}){self._loc(getattr(node, 'position', None))}")
            return None
        rhs = math.trunc(b)
        if rhs < 0:
            self._echo_fn(f"WARNING: negative shift{self._loc(getattr(node, 'position', None))}")
            return None
        if rhs >= 64:
            self._echo_fn(f"WARNING: shift too large{self._loc(getattr(node, 'position', None))}")
            return None
        # Python's native >> on an int already sign-extends (arithmetic
        # shift), matching int64_t's own C++20-guaranteed arithmetic right
        # shift for a negative left-hand side.
        return float(self._to_bitwise_int64(a) >> rhs)

    def _expr_and(self, node, ctx):
        if not self._eval_expr(node.left, ctx):
            return False
        if self._coverage:
            self._cov_hit(node.right)
        return bool(self._eval_expr(node.right, ctx))

    def _expr_or(self, node, ctx):
        if self._eval_expr(node.left, ctx):
            return True
        if self._coverage:
            self._cov_hit(node.right)
        return bool(self._eval_expr(node.right, ctx))

    def _expr_not(self, node, ctx):
        return not bool(self._eval_expr(node.expr, ctx))

    def _expr_eq(self, node, ctx):
        return _osc_equal(self._eval_expr(node.left, ctx), self._eval_expr(node.right, ctx))

    def _expr_neq(self, node, ctx):
        return not _osc_equal(self._eval_expr(node.left, ctx), self._eval_expr(node.right, ctx))

    def _expr_gt(self, node, ctx):
        a, b = self._eval_expr(node.left, ctx), self._eval_expr(node.right, ctx)
        if not _osc_comparable(a, b):
            self._echo_fn(f"WARNING: undefined operation ({_osc_type_name(a)} > {_osc_type_name(b)}){self._loc(getattr(node, 'position', None))}")
            return None
        try:
            return a > b
        except TypeError:
            return None

    def _expr_gte(self, node, ctx):
        a, b = self._eval_expr(node.left, ctx), self._eval_expr(node.right, ctx)
        if not _osc_comparable(a, b):
            self._echo_fn(f"WARNING: undefined operation ({_osc_type_name(a)} >= {_osc_type_name(b)}){self._loc(getattr(node, 'position', None))}")
            return None
        try:
            return a >= b
        except TypeError:
            return None

    def _expr_lt(self, node, ctx):
        a, b = self._eval_expr(node.left, ctx), self._eval_expr(node.right, ctx)
        if not _osc_comparable(a, b):
            self._echo_fn(f"WARNING: undefined operation ({_osc_type_name(a)} < {_osc_type_name(b)}){self._loc(getattr(node, 'position', None))}")
            return None
        try:
            return a < b
        except TypeError:
            return None

    def _expr_lte(self, node, ctx):
        a, b = self._eval_expr(node.left, ctx), self._eval_expr(node.right, ctx)
        if not _osc_comparable(a, b):
            self._echo_fn(f"WARNING: undefined operation ({_osc_type_name(a)} <= {_osc_type_name(b)}){self._loc(getattr(node, 'position', None))}")
            return None
        try:
            return a <= b
        except TypeError:
            return None

    def _expr_ternary(self, node, ctx):
        if self._debugging:
            self._check_debug(node, ctx)
        cond = self._eval_expr(node.condition, ctx)
        branch = node.true_expr if cond else node.false_expr
        if self._coverage:
            self._cov_hit(branch)
        if self._debugging:
            self._check_debug(branch, ctx, expr_level=True)
        return self._eval_expr(branch, ctx)

    # _expr_call removed — dispatch table points directly to _eval_function_call

    _SWIZZLE = {"x": 0, "y": 1, "z": 2, "w": 3}

    def _expr_index(self, node, ctx):
        obj = self._eval_expr(node.left, ctx)
        idx = self._eval_expr(node.index, ctx)
        tobj, tidx = type(obj), type(idx)
        if tobj is list or tobj is str:
            if tidx is int or tidx is float:
                i = int(idx)
                if i < 0:
                    return None
                try:
                    return obj[i]
                except IndexError:
                    return None
        tobj2 = type(obj)
        if tobj2 is OscRange and (tidx is int or tidx is float):
            return obj[int(idx)]
        if tobj2 is OscObject and tidx is str:
            return obj.get(idx)
        return None

    def _expr_member(self, node, ctx):
        obj = self._eval_expr(node.left, ctx)
        member = getattr(node.member, 'name', None) or str(node.member)
        tobj = type(obj)
        if tobj is list or tobj is tuple:
            idx = self._SWIZZLE.get(member)
            if idx is not None and idx < len(obj):
                return obj[idx]
        if tobj is OscObject:
            return obj.get(member)
        return None

    @staticmethod
    def _bind_let_name(child_ctx: EvalContext, name: str, v, dyn_copied: bool) -> bool:
        """Write one let()-clause binding into the right dict. $-prefixed
        names are special variables -- real OpenSCAD scopes them
        dynamically, so a let($fn=99) must remain visible to anything
        called from inside the let, not just the let's own body -- so they
        go into .dyn, not .let (verified against real OpenSCAD: a called
        function/module reading $fn sees the let()'s override). Everything
        else is an ordinary lexical binding, local to this let. child_ctx's
        dyn/dyn_explicit may start out shared by reference with the parent
        (see let_child_ctx) -- copy them on the first $-write so the
        override doesn't leak back out once the let returns; dyn_copied
        tracks whether that copy has already happened."""
        if name[0] == '$':
            if not dyn_copied:
                child_ctx.dyn = dict(child_ctx.dyn)
                child_ctx.dyn_explicit = set(child_ctx.dyn_explicit)
                dyn_copied = True
            child_ctx.dyn[name] = v
            child_ctx.dyn_explicit.add(name)
        else:
            child_ctx.let[name] = v
        return dyn_copied

    def _expr_let(self, node, ctx):
        child_ctx = ctx.let_child_ctx()
        dyn_copied = False
        for assign in node.assignments:
            if self._debugging:
                self._check_debug(assign, child_ctx)
            v = self._eval_expr(assign.expr, child_ctx)
            dyn_copied = self._bind_let_name(child_ctx, assign.name.name, v, dyn_copied)
        return self._eval_expr(node.body, child_ctx)

    def _expr_echo(self, node, ctx):
        if self._debugging:
            self._check_debug(node, ctx)
        self._do_echo(node.arguments, ctx)
        return self._eval_expr(node.body, ctx)

    def _expr_assert(self, node, ctx):
        self._check_assert(node, ctx)
        return self._eval_expr(node.body, ctx)

    def _expr_render(self, node: RenderExpression, ctx: EvalContext) -> OscObject:
        """`obj = render() { ... };`: builds the children's geometry, measures
        it, and throws it away -- nothing is drawn, from any context, which is
        what keeps a function using it pure. The only way a script can inspect
        its own geometry. See openscad_cpp_evaluator's CLAUDE.md for the
        contract this follows (the key sets and their order are part of it)."""
        _, ctx = self._resolve_call_args(node, ctx)  # $fn etc. reach the children; convexity is ignored
        self._tree_stack.append([])
        try:
            self._eval_children(node.children, ctx)
        finally:
            sub = self._tree_stack.pop()
        bodies = self.generate_tree(sub)
        return self._measure(bodies, node)

    def _measure(self, bodies: list[ColoredBody], node) -> OscObject:
        """The render() object for a generated subtree, taken as one implicit
        union. 3D: vertices, faces, volume, area, genus, boundingbox, dim, vnf.
        2D: vertices, paths, area, perimeter, boundingbox, dim. Nothing: the 3D
        keys, zero, with boundingbox undef rather than an infinite box."""
        pos = getattr(node, "position", None)
        _, fg, _, _ = self._split_by_role(bodies)
        solids = [b.body for b in fg if b.body is not None and b.body.status() == _MANIFOLD_OK
                  and not b.body.is_empty()]
        sections = [b.section for b in fg if b.section is not None]
        if solids:
            body = solids[0] if len(solids) == 1 else m3d.Manifold.batch_boolean(solids, m3d.OpType.Add)
            mesh = body.to_mesh64()
            verts, faces = _vnf_from_mesh(np.asarray(mesh.vert_properties[:, :3], dtype=np.float64),
                                          np.asarray(mesh.tri_verts, dtype=np.int64))
            bb = body.bounding_box()
            return OscObject({
                "vertices": verts, "faces": faces,
                "volume": body.volume(), "area": body.surface_area(), "genus": float(body.genus()),
                "boundingbox": [list(bb[:3]), list(bb[3:])], "dim": 3.0, "vnf": [verts, faces],
            })
        if sections:
            cs = sections[0]
            for s in sections[1:]:
                cs = cs + s
            if not cs.is_empty():
                verts, paths, perimeter = [], [], 0.0
                for poly in cs.to_polygons():
                    pts = [[float(x), float(y)] for x, y in poly]
                    paths.append(list(range(len(verts), len(verts) + len(pts))))
                    verts += pts
                    perimeter += sum(math.dist(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts)))
                x0, y0, x1, y1 = cs.bounds()
                return OscObject({
                    "vertices": verts, "paths": paths, "area": cs.area(), "perimeter": perimeter,
                    "boundingbox": [[x0, y0], [x1, y1]], "dim": 2.0,
                })
        raw = [b.raw_mesh for b in fg if b.raw_mesh is not None]
        if raw:
            # An open surface has no solid to measure, but the script still
            # gets its mesh back. The polyhedron/import warning already said
            # where the holes are.
            v, t = raw[0]
            verts, faces = _vnf_from_mesh(v, t)
            self._echo_fn(f"WARNING: render(): result is not a closed solid; volume and genus "
                          f"are unavailable{self._loc(pos)}")
            return OscObject({
                "vertices": verts, "faces": faces, "volume": 0.0, "area": 0.0, "genus": None,
                "boundingbox": [v.min(axis=0).tolist(), v.max(axis=0).tolist()] if len(v) else None,
                "dim": 3.0, "vnf": [verts, faces],
            })
        return OscObject({"vertices": [], "faces": [], "volume": 0.0, "area": 0.0, "genus": 0.0,
                          "boundingbox": None, "dim": 0.0, "vnf": [[], []]})

    def _expr_function_literal(self, node, ctx):
        return Closure(node, dict(ctx.let))

    _CONSTANTS = {"PI": math.pi}

    def _eval_identifier(self, node: Identifier, ctx: EvalContext, warn_if_undef: bool = True) -> Any:
        name = node.name
        v = ctx.let.get(name)
        if v is not None:
            return v
        if name in ctx.let:
            return None
        if name[0] == '$':
            v = ctx.dyn.get(name)
            if v is not None:
                return v
            if name in ctx.dyn:
                return v
        if name in self._CONSTANTS:
            return self._CONSTANTS[name]
        decl = ctx.scope.lookup_variable(name)
        if decl is None:
            if warn_if_undef:
                pos = getattr(node, 'position', None)
                self._echo_fn(f'WARNING: Ignoring unknown variable "{name}"{self._loc(pos)}')
            return None
        if type(decl) is ParameterDeclaration:
            return None
        # A file's globals are evaluated once per run, not once per read: a
        # function reading `r = rands(...)` must see the same r every call.
        # Only root-scope assignments qualify -- a module body's locals
        # differ per call.
        key = id(decl)
        globals_ = self._global_values
        if key in globals_:
            return globals_[key]
        if self._is_global(ctx.scope, name, decl):
            root = self._root_scope_of(ctx.scope)
            if root is not self._root_ctx.scope and id(root) not in self._files_run:
                self._run_file_globals(root)
                if key in globals_:
                    return globals_[key]
            # Read before its own assignment has run -- from a function an
            # earlier global's initializer called, in this file or a used one.
            # OpenSCAD evaluates a file's globals in order, so it is unknown
            # there; evaluating it on demand gave `c = d + 1; d = 5;` c = 6.
            if warn_if_undef:
                self._echo_fn(f'WARNING: Ignoring unknown variable "{name}"{self._loc(getattr(node, "position", None))}')
            return None
        return self._eval_expr(decl.expr, ctx)

    @staticmethod
    def _is_global(scope, name: str, decl) -> bool:
        while scope.parent is not None:
            if scope.variables.get(name) is decl:
                return False
            scope = scope.parent
        return scope.variables.get(name) is decl

    def _eval_list_comp(self, node: ListComprehension, ctx: EvalContext) -> list:
        result = []
        for elem in node.elements:
            te = type(elem)
            if te is ListCompFor:
                result.extend(self._eval_listcomp_for(elem, ctx))
            elif te is ListCompCFor:
                result.extend(self._eval_listcomp_cfor(elem, ctx))
            elif te is ListCompIf:
                if self._debugging:
                    self._check_debug(elem, ctx)
                if self._eval_expr(elem.condition, ctx):
                    if self._coverage:
                        self._cov_hit(elem.true_expr)
                    self._expr_depth += 1
                    if self._debugging:
                        self._check_debug(elem.true_expr, ctx, expr_level=True)
                    result.extend(self._eval_list_comp_body(elem.true_expr, ctx))
                    self._expr_depth -= 1
            elif te is ListCompIfElse:
                if self._debugging:
                    self._check_debug(elem, ctx)
                branch = elem.true_expr if self._eval_expr(elem.condition, ctx) else elem.false_expr
                if self._coverage:
                    self._cov_hit(branch)
                self._expr_depth += 1
                if self._debugging:
                    self._check_debug(branch, ctx, expr_level=True)
                result.extend(self._eval_list_comp_body(branch, ctx))
                self._expr_depth -= 1
            elif te is ListCompLet:
                let_ctx = ctx.let_child_ctx()
                dyn_copied = False
                for assign in elem.assignments:
                    if self._debugging:
                        self._check_debug(assign, let_ctx)
                    v = self._eval_expr(assign.expr, let_ctx)
                    dyn_copied = self._bind_let_name(let_ctx, assign.name.name, v, dyn_copied)
                result.extend(self._eval_list_comp_body(elem.body, let_ctx))
            elif te is ListCompEach:
                self._expr_depth += 1
                if self._debugging:
                    self._check_debug(elem, ctx, expr_level=True)
                inner = elem.body
                ti = type(inner)
                if ti is ListCompIf or ti is ListCompIfElse or ti is ListCompFor or ti is ListCompCFor or ti is ListCompLet or ti is ListCompEach:
                    for item in self._eval_list_comp_body(inner, ctx):
                        if type(item) is list:
                            result.extend(item)
                        elif item is not None:
                            result.append(item)
                else:
                    v = self._eval_expr(inner, ctx)
                    if type(v) is list:
                        result.extend(v)
                    elif type(v) is OscRange or type(v) is str:
                        result.extend(self._loop_values(v, elem))  # elements / characters
                    elif v is not None:
                        result.append(v)
                self._expr_depth -= 1
            else:
                if self._debugging:
                    self._check_debug(elem, ctx, expr_level=True)
                result.append(self._eval_expr(elem, ctx))
        return result

    def _eval_list_comp_body(self, body, ctx: EvalContext) -> list:
        t = type(body)
        if t is ListComprehension:
            self._expr_depth += 1
            result = [self._eval_list_comp(body, ctx)]
            self._expr_depth -= 1
            return result
        if t is ListCompFor:
            return self._eval_listcomp_for(body, ctx)
        if t is ListCompCFor:
            return self._eval_listcomp_cfor(body, ctx)
        if t is ListCompLet:
            let_ctx = ctx.let_child_ctx()
            dyn_copied = False
            for assign in body.assignments:
                if self._debugging:
                    self._check_debug(assign, let_ctx)
                v = self._eval_expr(assign.expr, let_ctx)
                dyn_copied = self._bind_let_name(let_ctx, assign.name.name, v, dyn_copied)
            return self._eval_list_comp_body(body.body, let_ctx)
        if t is ListCompIf:
            if self._debugging:
                self._check_debug(body, ctx)
            if self._eval_expr(body.condition, ctx):
                if self._coverage:
                    self._cov_hit(body.true_expr)
                self._expr_depth += 1
                if self._debugging:
                    self._check_debug(body.true_expr, ctx, expr_level=True)
                result = self._eval_list_comp_body(body.true_expr, ctx)
                self._expr_depth -= 1
                return result
            return []
        if t is ListCompIfElse:
            if self._debugging:
                self._check_debug(body, ctx)
            branch = body.true_expr if self._eval_expr(body.condition, ctx) else body.false_expr
            if self._coverage:
                self._cov_hit(branch)
            self._expr_depth += 1
            if self._debugging:
                self._check_debug(branch, ctx, expr_level=True)
            result = self._eval_list_comp_body(branch, ctx)
            self._expr_depth -= 1
            return result
        if t is ListCompEach:
            self._expr_depth += 1
            if self._debugging:
                self._check_debug(body, ctx, expr_level=True)
            inner = body.body
            ti = type(inner)
            if ti is ListCompIf or ti is ListCompIfElse or ti is ListCompFor or ti is ListCompCFor or ti is ListCompLet or ti is ListCompEach:
                result = []
                for item in self._eval_list_comp_body(inner, ctx):
                    if type(item) is list:
                        result.extend(item)
                    elif item is not None:
                        result.append(item)
                self._expr_depth -= 1
                return result
            v = self._eval_expr(inner, ctx)
            self._expr_depth -= 1
            if type(v) is list:
                return v
            if type(v) is OscRange or type(v) is str:
                return self._loop_values(v, body)  # `each` expands a range, splits a string
            return [v] if v is not None else []
        if self._debugging:
            self._check_debug(body, ctx, expr_level=True)
        v = self._eval_expr(body, ctx)
        return [v]

    def _eval_listcomp_for(self, node: ListCompFor, ctx: EvalContext) -> list:
        # Ranges are evaluated inside the loops before them -- see _eval_for.
        _av_pairs = [(assign, assign.name.name) for assign in node.assignments]
        _loop_values = self._loop_values  # bound: it warns

        result = []
        _debugging = self._debugging
        is_lc = type(node.body) is ListComprehension

        def _nested(depth: int, parent_ctx: EvalContext) -> None:
            if depth == len(_av_pairs):
                self._expr_depth += 1
                if is_lc:
                    result.append(self._eval_list_comp(node.body, parent_ctx))
                else:
                    result.extend(self._eval_list_comp_body(node.body, parent_ctx))
                self._expr_depth -= 1
                return
            assign_node, name = _av_pairs[depth]
            values = self._eval_expr(assign_node.expr, parent_ctx)
            if type(values) is not list:
                values = _loop_values(values, assign_node)
            for val in values:
                child = parent_ctx.let_child_ctx()
                child.let[name] = val
                if _debugging:
                    self._check_debug(assign_node, child)
                _nested(depth + 1, child)

        _nested(0, ctx)
        return result

    _MAX_CFOR_ITERATIONS = 1_000_000

    def _eval_listcomp_cfor(self, node: ListCompCFor, ctx: EvalContext) -> list:
        loop_ctx = ctx.let_child_ctx()
        _debugging = self._debugging
        for assign in node.inits:
            if _debugging:
                self._check_debug(assign, loop_ctx)
            loop_ctx.let[assign.name.name] = self._eval_expr(assign.expr, loop_ctx)

        result = []
        iterations = 0
        is_lc = type(node.body) is ListComprehension
        while True:
            if _debugging:
                self._check_debug(node.condition, loop_ctx, expr_level=True)
            if not self._eval_expr(node.condition, loop_ctx):
                break
            iterations += 1
            if iterations > self._MAX_CFOR_ITERATIONS:
                self.error("C-style for loop exceeded maximum iteration count", node)
            self._expr_depth += 1
            if _debugging:
                self._check_debug(node, loop_ctx)
            if is_lc:
                result.append(self._eval_list_comp(node.body, loop_ctx))
            else:
                result.extend(self._eval_list_comp_body(node.body, loop_ctx))
            self._expr_depth -= 1
            for assign in node.incrs:
                if _debugging:
                    self._check_debug(assign, loop_ctx)
                loop_ctx.let[assign.name.name] = self._eval_expr(assign.expr, loop_ctx)
        return result

    def _eval_range(self, node: RangeLiteral, ctx: EvalContext) -> OscRange:
        start = self._eval_expr(node.start, ctx)

        stop = self._eval_expr(node.end, ctx)
        increment = self._eval_expr(node.step, ctx)

        start = float(start) if start is not None else 0.0
        stop = float(stop) if stop is not None else 0.0
        increment = float(increment) if increment is not None else 1.0
        # [5:0] iterates nothing and is almost always [5:-1:0] mistyped.
        # Reported where the range is written, as OpenSCAD does, so `r = [5:0];`
        # warns even if r is never iterated. Only for an implicit step, unlike
        # OpenSCAD: writing [5:1:0] out says it is meant. The epsilon lets
        # float arithmetic landing a hair past the end go unremarked.
        if node.implicit_step and start - stop > 1e-10:
            pos = getattr(node, 'position', None)
            self._echo_fn(f"WARNING: begin is greater than the end, but step is positive{self._loc(pos)}")
        return OscRange(start, increment, stop)

    def _eval_function_call(self, node: PrimaryCall, ctx: EvalContext) -> Any:
        left = node.left
        name = left.name if type(left) is Identifier else None

        if name:
            if name == "import":
                args = self._resolve_args(node.arguments, ctx)
                return self._import_as_value(args, node)
            # A user function shadows a builtin of the same name, as in OpenSCAD.
            if name in self._BUILTIN_FN_NAMES:
                # Cached, since builtins are called constantly and hardly
                # ever shadowed; scopes don't change during a run.
                key = (id(ctx.scope), name)
                cache = self._builtin_shadow
                decl = cache.get(key, cache)
                if decl is cache:
                    decl = cache[key] = ctx.scope.lookup_function(name)
            else:
                decl = ctx.scope.lookup_function(name)
            if decl is not None:
                if self._debugging:
                    self._check_debug(node, ctx, call_site=True)
                return self._eval_user_function(name, decl, node.arguments, ctx, node)
            if name in self._BUILTIN_FN_NAMES:
                if name == "is_undef" and len(node.arguments) == 1 \
                        and type(node.arguments[0]) is PositionalArgument \
                        and type(node.arguments[0].expr) is Identifier:
                    # Asking whether a name is defined must not warn that it isn't.
                    return self._eval_identifier(node.arguments[0].expr, ctx, warn_if_undef=False) is None
                args = self._resolve_args(node.arguments, ctx)
                if name in ("textmetrics", "fontmetrics"):
                    self._warn_unexpected_args(self._BUILTIN_PARAMS[name], node.arguments, node, builtin=True)
                if name == "object":
                    return self._builtin_object(args, node)
                if name == "search":
                    positional = [args[i] for i in range(len(args)) if i in args]
                    try:
                        return self._builtin_search(*positional, node=node)
                    except Exception:
                        return None
                if name == "textmetrics":
                    return self._builtin_textmetrics(args, node)
                if name == "fontmetrics":
                    return self._builtin_fontmetrics(args, node)
                if name in ("dxf_dim", "dxf_cross"):
                    return self._builtin_dxf(name, args, node)
                if name == "linear_solve":
                    self._warn_unexpected_args(self._BUILTIN_PARAMS[name], node.arguments, node, builtin=True)
                    # An explicit undef b is absent, not bad: BOSL2's fixed-signature
                    # wrapper `function _linear_solve(A, b) = linear_solve(A, b);`
                    # always forwards it.
                    return self._builtin_linear_solve(self._get_arg(args, 0, "A"), self._get_arg(args, 1, "b"),
                                                      node, nargs=len(args))
                fn = self._math_fns.get(name)
                if fn is not None:
                    positional = [args[i] for i in range(len(args)) if i in args]
                    if not positional:
                        positional = [args[k] for k in args if type(k) is str]
                    if name in self._NUMERIC_ONLY_MATH_FNS or name in ("len", "ord"):
                        if not self._check_builtin_args(name, positional, node):
                            return None
                    try:
                        return fn(*positional)
                    except Exception:
                        return None

        if type(left) is Identifier:
            func_node = self._eval_identifier(left, ctx, warn_if_undef=False)
        else:
            func_node = self._eval_expr(left, ctx)
        if type(func_node) is Closure:
            if self._debugging:
                self._check_debug(node, ctx, call_site=True)
            return self._eval_function_literal(func_node, node.arguments, ctx, node, name=name)

        if name and func_node is None:
            pos = getattr(node, 'position', None)
            self._echo_fn(f"WARNING: Ignoring unknown function '{name}'{self._loc(pos)}")

        return None

    # How many leading arguments of each numeric builtin must be numbers.
    _NUMBER_ARGS = {
        "abs": 1, "sign": 1, "ceil": 1, "floor": 1, "round": 1, "sqrt": 1, "exp": 1, "ln": 1,
        "log": 2, "sin": 1, "cos": 1, "tan": 1, "asin": 1, "acos": 1, "atan": 1, "atan2": 2, "pow": 2,
    }

    def _check_builtin_args(self, name: str, args: list, node) -> bool:
        """Whether a numeric builtin's arguments have the types it needs; if
        not, OpenSCAD's warning -- "cos() parameter could not be converted:
        argument 0: expected number, found string ("a")" -- and the caller
        returns undef. The value was already undef; only the warning was
        missing. A bool is not a number here."""
        def is_num(v):
            return type(v) in (int, float)

        loc = self._loc(getattr(node, "position", None))

        def bad(i, expected, v=None, what="argument"):
            v = args[i] if v is None and what == "argument" else v
            self._echo_fn(f"WARNING: {name}() parameter could not be converted: {what} {i}: "
                          f"expected {expected}, found {_osc_type_name(v)} ({self._fmt_val(v)}){loc}")
            return False

        def warn(msg):
            self._echo_fn(f"WARNING: {msg}{loc}")
            return False

        if name in self._NUMBER_ARGS:
            for i in range(min(len(args), self._NUMBER_ARGS[name])):
                if not is_num(args[i]):
                    return bad(i, "number")
            return True
        if name in ("max", "min"):
            if len(args) == 1 and type(args[0]) is list:
                for i, v in enumerate(args[0]):
                    if not is_num(v):
                        return bad(i, "number", v, "vector element")
                return True
            for i, v in enumerate(args):
                if not is_num(v):
                    return bad(i, "number")
            return True
        if name == "norm":
            if not args or type(args[0]) is not list:
                return bool(args) and bad(0, "vector")
            return all(is_num(v) for v in args[0]) or warn("Incorrect arguments to norm()")
        if name == "ord":
            return bool(args) and (type(args[0]) is str or bad(0, "string"))
        if name == "len":
            return bool(args) and (type(args[0]) in (list, str, OscObject) or bad(0, "string"))
        if name == "cross":
            ok = (len(args) == 2 and all(type(v) is list and len(v) in (2, 3) for v in args)
                  and len(args[0]) == len(args[1]))
            if not ok:
                return warn("Invalid vector size of parameter for cross()")
            for x in (x for v in args for x in v):
                if type(x) not in (int, float):
                    return warn("Invalid value in parameter vector for cross()")
                if not math.isfinite(x):
                    return warn(f"Invalid value ({'NaN' if x != x else 'INF'}) in parameter vector for cross()")
            return True
        return True

    def _builtin_minmax(self, op, args):
        """Shared logic for OpenSCAD's `min`/`max`.

        A single vector argument returns `op` of its elements; multiple
        arguments must all be scalars (mixing in a vector is `undef`, like
        real OpenSCAD); a single scalar argument returns itself.
        """
        if len(args) == 1:
            v = args[0]
            return op(v) if isinstance(v, list) else v
        if any(isinstance(a, list) for a in args):
            return None
        return op(args)

    def _builtin_max(self, *args):
        return self._builtin_minmax(max, args)

    def _builtin_min(self, *args):
        return self._builtin_minmax(min, args)

    def _builtin_pow(self, a, b):
        if math.isnan(a) or math.isnan(b):
            return math.nan
        if a < 0 and not float(b).is_integer() and not math.isinf(b):
            return float('nan')
        if a == 0 and b < 0:
            # 0 ** negative is +inf in OpenSCAD; Python's pow()/math.pow() raise.
            return float('inf')
        try:
            return pow(a, b)
        except OverflowError:  # 10^400: inf, as in C (negative for an odd power of a negative)
            return -math.inf if a < 0 and float(b).is_integer() and int(b) % 2 else math.inf

    def _negate_list(self, v):
        if _is_flat_numeric(v):
            if len(v) >= _NP_VEC_THRESHOLD:
                return (-np.asarray(v)).tolist()
            return [-x for x in v]
        result = []
        for x in v:
            if isinstance(x, list):
                result.append(self._negate_list(x))
            elif isinstance(x, bool) or x is None:
                result.append(None)
            else:
                try:
                    result.append(-x)
                except TypeError:
                    result.append(None)
        return result

    def _builtin_sin(self, x):
        return _sin_deg(x)

    def _builtin_cos(self, x):
        return _cos_deg(x)

    def _builtin_tan(self, x):
        return _tan_deg(x)

    def _builtin_cross(self, a, b):
        # Real OpenSCAD validates every component up front and returns
        # undef (with a WARNING naming the offending value) rather than
        # computing through: nan/inf components would otherwise propagate
        # unevenly (inf*0 is nan, not 0), producing a mixed
        # finite/nan/inf result instead of a clean undef -- confirmed
        # against real OpenSCAD 2022.08.22.
        for v in (a, b):
            for c in v:
                if isinstance(c, float) and not math.isfinite(c):
                    return None
        if len(a) == 2 and len(b) == 2:
            return a[0]*b[1] - a[1]*b[0]
        return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]

    def _builtin_chr(self, x):
        # Real OpenSCAD silently skips non-finite/non-numeric code points
        # (in a list) and returns "" (rather than undef) for a scalar
        # non-finite/undef argument -- confirmed against real OpenSCAD
        # 2022.08.22 (chr(1/0), chr(0/0), chr(undef) all -> "";
        # chr([65, 1/0, 66]) -> "AB", skipping the invalid element).
        def valid(c):
            return isinstance(c, (int, float)) and not isinstance(c, bool) and math.isfinite(c)
        if isinstance(x, list):
            return "".join(chr(int(c)) for c in x if valid(c))
        return chr(int(x)) if valid(x) else ""

    def _builtin_rands(self, minval, maxval, n, seed=None):
        self._rands_call_count += 1
        if seed is not None:
            random.seed(int(seed))
        return [random.uniform(float(minval), float(maxval)) for _ in range(int(n))]

    def _builtin_search(self, match, vector, num_returns=1, index_col=0, node=None):
        """OpenSCAD search(): find positions of match value(s) in vector.

        Strings are treated as character arrays — each character is searched
        independently, mirroring OpenSCAD semantics.
        """
        num_returns = int(num_returns)
        col = int(index_col)

        def _find_all(val):
            results = []
            for i, item in enumerate(vector):
                # A vector match value (e.g. searching for a coordinate like
                # [0,0,1]) is compared directly against each whole element,
                # not column-indexed — index_col only applies to scalar matches.
                if isinstance(val, list):
                    target = item
                else:
                    target = item[col] if isinstance(item, list) else item
                if target == val:
                    results.append(i)
            return results

        def _result_for(val):
            """Result for one element in a list/string match context."""
            matches = _find_all(val)
            if num_returns == 1:
                return matches[0] if matches else []
            elif num_returns == 0:
                return matches
            else:
                return matches[:num_returns]

        if isinstance(match, str) and isinstance(vector, list):
            # Characters are looked for in a column of each entry, so every
            # entry must be a list that long; OpenSCAD refuses the whole
            # search at the first that isn't (a list of strings matched).
            for i, item in enumerate(vector):
                if not isinstance(item, list) or len(item) <= col:
                    self._echo_fn(
                        f"WARNING: Invalid entry in search vector at index {i}, required number of "
                        f"values in the entry: {col + 1}. Invalid entry: {self._fmt_val(item)}"
                        f"{self._loc(getattr(node, 'position', None))}")
                    return []
        if isinstance(match, str):
            # String → character array: search for each char independently.
            # With num_returns=1: not-found chars are dropped (not included as []).
            # With num_returns=0: all chars included, not-found → [].
            results = []
            for c in match:
                r = _result_for(c)
                if num_returns != 1 or r != []:
                    results.append(r)
            return results
        elif isinstance(match, list):
            return [_result_for(m) for m in match]
        else:
            # Scalar number: always return a list of matching indices
            matches = _find_all(match)
            if num_returns == 1:
                return matches[:1]      # [idx] or []
            elif num_returns == 0:
                return matches
            else:
                return matches[:num_returns]

    def _builtin_parent_module(self, idx=0):
        """Return the name of the module idx levels up from the current module."""
        modules = [e[1] for e in self._call_stack if e[0] == "module"]
        rev_idx = len(modules) - 1 - int(idx)
        return modules[rev_idx] if 0 <= rev_idx < len(modules) else None

    def _builtin_lookup(self, key, table):
        """Linear interpolation lookup in a [[key, value], ...] table."""
        if not table:
            return None
        pairs = sorted(table, key=lambda p: p[0])
        if key <= pairs[0][0]:
            return pairs[0][1]
        if key >= pairs[-1][0]:
            return pairs[-1][1]
        for i in range(len(pairs) - 1):
            k0, v0 = pairs[i]
            k1, v1 = pairs[i + 1]
            if k0 <= key <= k1:
                t = (key - k0) / (k1 - k0)
                return v0 + t * (v1 - v0)
        return 0

    def _builtin_object(self, args: dict, node) -> Optional[OscObject]:
        """`object(a=1, b=2, ...)` — an ordered string-keyed map.

        Positional arguments merge an existing `OscObject`'s entries, or a
        list of `[key, value]` pairs, into the result (in their own order);
        named arguments set/override entries in call order. Any other
        positional argument type is invalid and the whole call is `undef`.
        """
        result: dict = {}
        for key, val in args.items():
            if isinstance(key, str):
                result[key] = val
                continue
            if isinstance(val, OscObject):
                for k, v in val.items():
                    result[k] = v
            elif isinstance(val, list):
                for entry in val:
                    if isinstance(entry, list) and len(entry) == 2 and isinstance(entry[0], str):
                        result.pop(entry[0], None)  # re-set moves it last, as in OpenSCAD
                        result[entry[0]] = entry[1]
                    elif isinstance(entry, list) and len(entry) == 1 and isinstance(entry[0], str):
                        result.pop(entry[0], None)  # [key] deletes; an absent key is fine
                    else:
                        self._echo_fn(
                            f"WARNING: object(Argument {key}) malformed [key,value] entry in "
                            f"unnamed list argument{self._loc(getattr(node, 'position', None))}"
                        )
                        return None
            else:
                tname = _object_arg_type_name(val)
                self._echo_fn(
                    f"WARNING: object(Argument {key} <{tname}>) An unnamed argument must be "
                    f"either <object> or <list>, it is <{tname}>. "
                    f"{self._loc(getattr(node, 'position', None))}"
                )
                return None
        return OscObject(result)

    def _builtin_textmetrics(self, args: dict, node) -> OscObject:
        """`textmetrics(text=.., size=.., halign=.., valign=.., spacing=.., font=..)`.

        Measures `text` against the font resolved by `font=` (an OpenSCAD/
        fontconfig pattern, e.g. `"Times New Roman:style=Bold"`, via
        `_resolve_font()` — same resolution `text()` uses) and returns an
        `OscObject` with `position`, `size`, `ascent`, `descent`, `offset`,
        `advance` — matching real OpenSCAD's key order. Falls back to the
        bundled Liberation Sans if `font=` is unset, `fc-match` is
        unavailable, or the font can't be found. Shaped by HarfBuzz, with
        `direction`/`language`/`script` honoured as in `text()`.
        """
        text = self._get_arg(args, 0, "text", "")
        # All nine positional, unlike text(), which takes only font that way.
        size = self._get_arg(args, 1, "size", 10)
        halign = self._get_arg(args, 6, "halign", "left")
        valign = self._get_arg(args, 7, "valign", "baseline")
        spacing = self._get_arg(args, 8, "spacing", 1)
        font_spec = self._get_arg(args, 2, "font", "") or ""
        shaping = [self._get_arg(args, i, k, "") or "" for i, k in ((3, "direction"), (4, "language"), (5, "script"))]

        font = _resolve_font(str(font_spec))
        m = _measure_text(text, size, spacing, font, *map(str, shaping))
        ascent, descent = m["ascent"], m["descent"]
        advance_x = m["advance_x"]

        offset_x, offset_y = _text_align_offset(halign, valign, m)

        position = [offset_x + m["ink_min_x"], offset_y + descent]
        size_vec = [m["ink_max_x"] - m["ink_min_x"], ascent - descent]

        return OscObject({
            "position": position,
            "size": size_vec,
            "ascent": ascent,
            "descent": descent,
            "offset": [offset_x, offset_y],
            "advance": [advance_x, m["advance_y"]],
        })

    def _builtin_dxf(self, name: str, args: dict, node):
        """dxf_dim(file, name, layer, origin, scale) / dxf_cross(file, layer,
        origin, scale): see dxf_dim.py."""
        from .dxf_dim import dxf_cross, dxf_dim
        file_arg = self._get_arg(args, 0, "file")
        raw = file_arg if isinstance(file_arg, str) else self._fmt_val(file_arg)
        path = self._resolve_import_path(raw, node)
        layer = self._get_arg(args, None, "layer")
        layer = layer if isinstance(layer, str) else ""
        origin = self._get_arg(args, None, "origin")
        origin = [float(origin[0]), float(origin[1])] if (
            type(origin) is list and len(origin) >= 2
            and all(type(v) in (int, float) for v in origin[:2])) else [0.0, 0.0]
        scale = self._get_arg(args, None, "scale", 1.0)
        scale = float(scale) if type(scale) in (int, float) else 1.0
        if name == "dxf_dim":
            dim = self._get_arg(args, None, "name")
            value, warning = dxf_dim(path, raw, layer, origin, scale, dim if isinstance(dim, str) else "")
        else:
            value, warning = dxf_cross(path, raw, layer, origin, scale)
        if warning:
            self._echo_fn(f"WARNING: {warning}{self._loc(getattr(node, 'position', None))}")
        return value

    def _builtin_linear_solve(self, a, b, node, nargs: int = 2) -> Optional[OscObject]:
        """`linear_solve(A, b)` -> object(x, det, singular). Port of the C++
        builtin: pivoted LU for a square A (the determinant falls out of the
        same pass; BOSL2's determinant() is a cofactor expansion, O(n!)),
        least squares via QR for a tall A, minimum norm via QR of A^T for a
        wide one, where det is undef. Singularity is RELATIVE to the largest
        entry -- BOSL2's fixed 1e-9 calls a well-conditioned A*1e-10
        singular. Without column pivoting a rank test cannot tell rank
        deficiency from severe ill-conditioning; it is a heuristic."""
        loc = self._loc(getattr(node, "position", None))

        def warn(msg):
            self._echo_fn(f"WARNING: linear_solve() {msg}{loc}")

        def matrix(v):
            if type(v) is not list or not v or any(type(r) is not list or not r for r in v):
                return None
            if len({len(r) for r in v}) != 1 or not all(type(x) in (int, float) for r in v for x in r):
                return None
            return np.array(v, dtype=np.float64)

        if not nargs:
            warn("number of parameters does not match: expected 1 or 2, found 0")
            return None
        if type(a) is not list:
            self._echo_fn(f"WARNING: linear_solve() parameter could not be converted: argument 0: "
                          f"expected vector, found {_osc_type_name(a)} ({self._fmt_val(a)}){loc}")
            return None
        A = matrix(a)
        if A is None:
            warn("requires a matrix of numbers")
            return None
        if not np.isfinite(A).all():
            warn("matrix contains a non-finite value")
            return None
        m, n = A.shape
        B, vector = None, False
        if b is not None:
            if type(b) is list and len(b) == m and all(type(x) in (int, float) for x in b):
                B, vector = np.array(b, dtype=np.float64).reshape(m, 1), True
            elif (B := matrix(b)) is None or B.shape[0] != m:
                warn(f"right-hand side must be a vector of {m} numbers, or a matrix with that many rows")
                return None
            if not np.isfinite(B).all():
                warn("right-hand side contains a non-finite value")
                return None
        tol = np.finfo(np.float64).eps * max(m, n) * max(float(np.abs(A).max()), 1.0)

        def shaped(x):
            return x.reshape(-1).tolist() if vector else x.tolist()

        if m != n:
            # Householder QR (LAPACK's, as the C++ port hand-rolls); |R_jj| is
            # the column norm the C++ tests against tol.
            tall = A if m > n else A.T
            q, r = np.linalg.qr(tall)
            if (np.abs(np.diag(r)) <= tol).any():
                return OscObject({"x": None, "det": None, "singular": True})
            if B is None:
                return OscObject({"x": None, "det": None, "singular": False})
            if m > n:
                x = np.linalg.solve(r, q.T @ B)
            else:
                x = q @ np.linalg.solve(r.T, B)  # minimum norm: Q [y; 0]
            return OscObject({"x": shaped(x), "det": None, "singular": False})

        lu = A.copy()
        rhs = B.copy() if B is not None else np.zeros((n, 0))
        det = 1.0
        for col in range(n):
            pivot = col + int(np.argmax(np.abs(lu[col:, col])))
            if abs(lu[pivot, col]) <= tol:
                return OscObject({"x": None, "det": 0.0, "singular": True})
            if pivot != col:
                lu[[col, pivot]] = lu[[pivot, col]]
                rhs[[col, pivot]] = rhs[[pivot, col]]
                det = -det
            p = lu[col, col]
            det *= p
            f = lu[col + 1:, col] / p
            lu[col + 1:, col:] -= np.outer(f, lu[col, col:])
            rhs[col + 1:] -= np.outer(f, rhs[col])
        if B is None:
            return OscObject({"x": None, "det": float(det), "singular": False})
        for col in range(n - 1, -1, -1):
            rhs[col] = (rhs[col] - lu[col, col + 1:] @ rhs[col + 1:]) / lu[col, col]
        return OscObject({"x": shaped(rhs), "det": float(det), "singular": False})

    def _builtin_fontmetrics(self, args: dict, node) -> OscObject:
        """`fontmetrics(size=.., font=..)` — global metrics of the font
        resolved by `font=` (via `_resolve_font()`, same resolution `text()`
        and `textmetrics()` use), scaled for `size`. Returns a nested
        `OscObject` with `nominal`/`max`/`interline`/`font`; `font.family`/
        `font.style` report the *actually resolved* font's real name (read
        from its `name` table via `getBestFamilyName()`/`getBestSubFamilyName()`),
        not just an echo of the request — e.g. `font="Times New Roman:style=Bold"`
        yields `family="Times New Roman"`, `style="Bold"`. Falls back to the
        bundled Liberation Sans if `font=` is unset, `fc-match` is
        unavailable, or the font can't be found."""
        size = self._get_arg(args, 0, "size", 10)
        font_spec = self._get_arg(args, 1, "font", "") or ""

        font = _resolve_font(str(font_spec))
        head, hhea = font["head"], font["hhea"]
        scale = size * (100 / 72) / font["units_per_em"]

        return OscObject({
            "nominal": OscObject({
                "ascent": hhea.ascent * scale,
                "descent": hhea.descent * scale,
            }),
            "max": OscObject({
                "ascent": head.yMax * scale,
                "descent": head.yMin * scale,
            }),
            "interline": (hhea.ascent - hhea.descent + hhea.lineGap) * scale,
            "font": OscObject({
                "family": font["family_name"],
                "style": font["style_name"],
            }),
        })

    def _apply_defaults(self, params, bound, child_ctx: EvalContext):
        """Fill in any param not already present in `bound` (_bind_args' own
        return value -- the authoritative "did the caller actually supply
        this name" record) from its default expression. Matches real
        OpenSCAD (verified directly against /Applications/OpenSCAD.app): a
        default expression is evaluated purely lexically against the
        function/module's own declaration scope (child_ctx.scope) -- it
        sees neither the caller's local variables nor this same call's
        other (sibling) parameters, though $-vars remain dynamically scoped
        as usual (child_ctx.dyn is already the correctly-threaded dynamic
        environment, so it's reused as-is). A default that reads a variable
        the caller shadows via let() resolves to the function's own
        enclosing scope, not the caller's shadow; a default referencing an
        earlier sibling parameter is an unknown variable (warning + undef),
        not a forward reference.

        A $-prefixed parameter's result (bound or defaulted) goes into
        child_ctx.dyn, everything else into child_ctx.let -- matching the
        bound-argument loop at each call site exactly. `bound`, not
        child_ctx.let/dyn, must be the "already bound" source of truth:
        child_ctx.dyn is always pre-seeded with $fn/$fa/$fs/$t/
        $parent_modules regardless of what the caller actually passed, so
        checking dyn's mere presence could never distinguish "the caller
        explicitly passed $fn" from "$fn merely exists because of that
        ambient seed" -- and checking only child_ctx.let (the original
        shape of this function) missed a $-prefixed name bound into dyn
        just as badly in the other direction: it always looked unbound,
        clobbering the real value with a fresh undef written into .let,
        which then shadowed the correct .dyn value the moment the body
        referenced that name as a plain identifier (_eval_identifier checks
        .let before .dyn)."""
        let_dict = child_ctx.let
        dyn_dict = child_ctx.dyn
        default_ctx = None
        _eval = self._eval_expr
        for param in params:
            pname = param.name.name
            if pname not in bound:
                is_dyn = pname[0] == '$'
                default = param.default
                if default is None:
                    if is_dyn:
                        dyn_dict[pname] = None
                    else:
                        let_dict[pname] = None
                else:
                    if default_ctx is None:
                        default_ctx = child_ctx.child_ctx(let={}, share_dyn=True)
                    value = _eval(default, default_ctx)
                    if is_dyn:
                        dyn_dict[pname] = value
                    else:
                        let_dict[pname] = value

    def _has_dollar_param(self, decl_id: int, params) -> bool:
        """Whether any of `params`' declared names starts with '$' --
        memoized by declaration identity, since this is a purely static
        property of the declaration, never of any particular call. IS
        _eval_user_function/_eval_function_literal's own share_dyn value
        directly (not just a fast-path skip): every declared $-param
        writes to child_ctx.dyn exactly once per call, either via the
        bound-argument loop (if the caller supplied it) or via
        _apply_defaults (if not) -- so a declaration with no $-param at all
        is the ONLY case where dyn is guaranteed untouched regardless of
        the call's actual arguments, and share_dyn can be True
        unconditionally; any $-param at all means False unconditionally,
        no per-call bound-keys check needed."""
        cached = self._decl_dollar_param.get(decl_id)
        if cached is None:
            cached = any(p.name.name[0] == '$' for p in params)
            self._decl_dollar_param[decl_id] = cached
        return cached

    @staticmethod
    def _bound_has_dollar_key(bound: dict) -> bool:
        """See _has_dollar_param's docstring -- the per-call half of the
        share_dyn safety check it can't provide by itself. _bind_args
        writes ANY named call-site argument into `bound` unconditionally,
        including one that doesn't match any declared parameter -- that's
        how an undeclared `$fn=64`-style dynamic-scope override works for
        a call in general. share_dyn must also stay False whenever this
        particular call's own `bound` contains such a key, or the
        bound-argument loop mutates the ALIASED (shared, not copied) dyn
        dict in place, leaking the override into the caller's own scope
        once the call returns."""
        return any(k[0] == '$' for k in bound)

    def _eval_user_function(self, name: str, decl: FunctionDeclaration, arguments, ctx: EvalContext, call_node=None) -> Any:
        if self._coverage:
            self._cov_hit(decl)
        child_ctx = self._bind_user_function(decl, arguments, ctx, call_node)
        return self._run_function_body(name, decl.expr, decl.position, child_ctx, call_node)

    def _bind_user_function(self, decl: FunctionDeclaration, arguments, ctx: EvalContext, call_node) -> EvalContext:
        params = decl.parameters or []
        bound = self._bind_args(params, arguments, ctx, call_node)
        fn_scope = decl.scope or ctx.scope
        # No $-prefixed parameter is declared AND this call's own bound
        # arguments include no $-prefixed key either -- the common case,
        # and the only case where child_ctx.dyn is guaranteed untouched by
        # either the bound-argument loop below or _apply_defaults -- safe
        # to skip copying dyn/dyn_explicit and share ctx's own dict/set by
        # reference instead. A real, measured optimization: on a
        # BOSL2-heavy script (Anklet.scad, ~1.1M user function calls),
        # context-creation machinery was ~9.6% of total evaluate() time,
        # and this is its single biggest piece.
        share_dyn = not self._has_dollar_param(id(decl), params) and not self._bound_has_dollar_key(bound)
        child_ctx = self._call_ctx_for(decl, ctx, scope=fn_scope, share_dyn=share_dyn)
        for k, v in bound.items():
            if k[0] == '$':
                child_ctx.dyn[k] = v
            else:
                child_ctx.let[k] = v
        self._apply_defaults(params, bound, child_ctx)
        return child_ctx

    _TAIL_CALL_CAP = 1_000_000  # OpenSCAD runs 999,999 hops and stops at 2,000,000; so does the C++ port

    def _run_function_body(self, name: str, body, decl_pos, child_ctx: EvalContext, call_node) -> Any:
        """Evaluate a function body with tail calls as a loop, as OpenSCAD's
        own trampoline does: a ternary, let, echo or assert in tail position
        hands over its tail expression, and a call there to a user function
        (named or literal) rebinds and carries on in this frame instead of
        recursing. `function acc(n, a=0) = n <= 0 ? a : acc(n - 1, a + 1);`
        otherwise died at a few hundred deep on Python's stack. Anything
        else is evaluated normally.

        The call-stack frame is replaced on each hop, so a TRACE names the
        latest; the return hook fires once, for the whole chain.
        ponytail: not while profiling, which should show the real call tree
        (the C++ port lumps a chain into its first call instead) -- so a
        profiled script can still run out of stack on very deep recursion."""
        pos = call_node.position if call_node is not None else None
        prof = self._profile_enter("function", name, pos, decl_pos) if self._profiling else None
        self._call_stack.append(("function", name, pos, decl_pos))
        self._frame_ctxs.append(child_ctx)
        ctx, expr, hops = child_ctx, body, 0
        try:
            if self._debugging:
                self._check_debug(expr, ctx)
            while True:
                t = type(expr)
                if t is TernaryOp:
                    if self._debugging:
                        self._check_debug(expr, ctx)
                    expr = expr.true_expr if self._eval_expr(expr.condition, ctx) else expr.false_expr
                    if self._coverage:
                        self._cov_hit(expr)  # the trampoline bypasses _expr_ternary
                    if self._debugging:
                        self._check_debug(expr, ctx, expr_level=True)
                elif t is LetOp:
                    let_ctx = ctx.let_child_ctx()
                    dyn_copied = False
                    for assign in expr.assignments:
                        if self._debugging:
                            self._check_debug(assign, let_ctx)
                        v = self._eval_expr(assign.expr, let_ctx)
                        dyn_copied = self._bind_let_name(let_ctx, assign.name.name, v, dyn_copied)
                    ctx, expr = let_ctx, expr.body
                elif t is EchoOp:
                    if self._debugging:
                        self._check_debug(expr, ctx)
                    self._do_echo(expr.arguments, ctx)
                    expr = expr.body
                elif t is AssertOp:
                    self._check_assert(expr, ctx)
                    expr = expr.body
                elif t is CommentedExpr:
                    expr = expr.expr
                elif t is PrimaryCall and not self._profiling and (hop := self._tail_callee(expr, ctx)) is not None:
                    hops += 1
                    if hops >= self._TAIL_CALL_CAP:
                        self.error(f"Recursion detected calling function '{hop[0]}'", expr)
                    hop_name, target = hop
                    call = expr
                    if self._debugging:
                        self._check_debug(call, ctx, call_site=True)
                    if self._coverage:
                        # A hop is the only way into a tail-called body (cpp #175).
                        self._cov_hit(target.fn if type(target) is Closure else target)
                    if type(target) is Closure:
                        ctx = self._bind_closure(target, call.arguments, ctx, call)
                        expr, decl_pos = target.fn.body, target.fn.position
                    else:
                        ctx = self._bind_user_function(target, call.arguments, ctx, call)
                        expr, decl_pos = target.expr, target.position
                    self._call_stack[-1] = ("function", hop_name, call.position, decl_pos)
                    self._frame_ctxs[-1] = ctx
                    if self._debugging:
                        self._check_debug(expr, ctx)
                else:
                    result = self._eval_expr(expr, ctx)
                    break
            if self._return_hook is not None:
                self._return_hook(name, result, len(self._call_stack))
            return result
        finally:
            self._call_stack.pop()
            self._frame_ctxs.pop()
            if prof is not None:
                self._profile_exit(*prof)

    def _tail_callee(self, node: PrimaryCall, ctx: EvalContext):
        """(name, FunctionDeclaration or Closure) if `node` calls a user
        function, resolved exactly as _eval_function_call would, else None."""
        left = node.left
        if type(left) is not Identifier:
            return None
        name = left.name
        if name == "import":
            return None
        if name in self._BUILTIN_FN_NAMES:
            key = (id(ctx.scope), name)
            cache = self._builtin_shadow
            decl = cache.get(key, cache)
            if decl is cache:
                decl = cache[key] = ctx.scope.lookup_function(name)
        else:
            decl = ctx.scope.lookup_function(name)
        if decl is not None:
            return name, decl
        if name in self._BUILTIN_FN_NAMES:
            return None
        value = self._eval_identifier(left, ctx, warn_if_undef=False)
        return (name, value) if type(value) is Closure else None

    def _check_assert(self, node, ctx) -> None:
        if self._debugging:
            self._check_debug(node, ctx)
        raw = node.arguments
        condition = self._eval_expr(raw[0].expr, ctx) if raw else True
        if not condition:
            cond_text = to_openscad([raw[0].expr]).strip() if raw else "false"
            msg = self._eval_expr(raw[1].expr, ctx) if len(raw) > 1 else None
            err = f"Assertion '{cond_text}' failed" + (f': "{msg}"' if msg is not None else "")
            self.error(err, node, innermost_frame="assert")

    def _eval_function_literal(self, closure: Closure, arguments, ctx: EvalContext, call_node=None, name: str | None = None) -> Any:
        if self._coverage:
            self._cov_hit(closure.fn)
        child_ctx = self._bind_closure(closure, arguments, ctx, call_node)
        return self._run_function_body(name or "<function>", closure.fn.body, closure.fn.position, child_ctx, call_node)

    def _bind_closure(self, closure: Closure, arguments, ctx: EvalContext, call_node) -> EvalContext:
        func_node = closure.fn
        params = func_node.parameters
        bound = self._bind_args(params, arguments, ctx, call_node)
        fn_scope = func_node.scope or ctx.scope
        # See _eval_user_function's matching comment -- same optimization.
        share_dyn = not self._has_dollar_param(id(func_node), params) and not self._bound_has_dollar_key(bound)
        child_ctx = ctx.call_ctx(scope=fn_scope, share_dyn=share_dyn)
        child_ctx.let = dict(closure.let)
        for k, v in bound.items():
            if k[0] == '$':
                child_ctx.dyn[k] = v
            else:
                child_ctx.let[k] = v
        self._apply_defaults(params, bound, child_ctx)
        return child_ctx


_EXPR_DISPATCH: dict[type, callable] = {
    ListComprehension: Evaluator._eval_list_comp,
    RangeLiteral: Evaluator._eval_range,
    AdditionOp: Evaluator._expr_add,
    SubtractionOp: Evaluator._expr_sub,
    MultiplicationOp: Evaluator._expr_mul,
    DivisionOp: Evaluator._expr_div,
    ModuloOp: Evaluator._expr_mod,
    ExponentOp: Evaluator._expr_exp,
    UnaryMinusOp: Evaluator._expr_unary_minus,
    BitwiseOrOp: Evaluator._expr_bitor,
    BitwiseAndOp: Evaluator._expr_bitand,
    BitwiseNotOp: Evaluator._expr_bitnot,
    BitwiseShiftLeftOp: Evaluator._expr_shl,
    BitwiseShiftRightOp: Evaluator._expr_shr,
    LogicalAndOp: Evaluator._expr_and,
    LogicalOrOp: Evaluator._expr_or,
    LogicalNotOp: Evaluator._expr_not,
    EqualityOp: Evaluator._expr_eq,
    InequalityOp: Evaluator._expr_neq,
    GreaterThanOp: Evaluator._expr_gt,
    GreaterThanOrEqualOp: Evaluator._expr_gte,
    LessThanOp: Evaluator._expr_lt,
    LessThanOrEqualOp: Evaluator._expr_lte,
    TernaryOp: Evaluator._expr_ternary,
    PrimaryCall: Evaluator._eval_function_call,
    PrimaryIndex: Evaluator._expr_index,
    PrimaryMember: Evaluator._expr_member,
    LetOp: Evaluator._expr_let,
    EchoOp: Evaluator._expr_echo,
    AssertOp: Evaluator._expr_assert,
    FunctionLiteral: Evaluator._expr_function_literal,
    RenderExpression: Evaluator._expr_render,
}
