"""levelset(): a solid or section from an implicit surface. A port of
openscad_cpp_evaluator's topology.cpp levelset (cpp #124, #126, #127) --
see that file and its CLAUDE.md for the measurements behind each choice.
The pieces here are pure; the evaluator glue lives in Evaluator."""
from __future__ import annotations

import math
from typing import Callable, Optional

import manifold3d as m3d

INF = math.inf


def band_distance(v: float, lo: float, hi: float, invert: bool) -> float:
    """Positive inside the band [lo, hi], zero on either surface (Manifold
    takes positive as inside). A scalar isovalue v is [-INF, v], "at or
    below"; [lo, INF] is BOSL2's "at or above"; a finite pair is between."""
    lo_f, hi_f = math.isfinite(lo), math.isfinite(hi)
    if lo_f and hi_f:
        d = min(v - lo, hi - v)
    elif hi_f:
        d = hi - v
    elif lo_f:
        d = v - lo
    else:
        d = 1.0
    return -d if invert else d


def marching_squares(v: list[list[float]], ox: float, oy: float, dx: float, dy: float) -> list:
    """Closed contours of the positive region of padded grid v[i][j] (x, y).
    Saddles (cases 5/10) take the cell-centre average, or the topology is a
    coin flip; vertices are keyed on EDGE identity, so loops close exactly."""
    nx, ny = len(v), len(v[0])
    h_count = (nx - 1) * ny
    pts: dict[int, tuple[float, float]] = {}

    def lerp(i0, j0, i1, j1, eid):
        if eid not in pts:
            a, b = v[i0][j0], v[i1][j1]
            t = 0.5 if a == b else a / (a - b)
            pts[eid] = (ox + (i0 + t * (i1 - i0)) * dx, oy + (j0 + t * (j1 - j0)) * dy)
        return eid

    segs = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            d0, d1, d2, d3 = v[i][j], v[i + 1][j], v[i + 1][j + 1], v[i][j + 1]
            mask = (d0 > 0) | (d1 > 0) << 1 | (d2 > 0) << 2 | (d3 > 0) << 3
            if mask in (0, 15):
                continue
            e_b, e_t = j * (nx - 1) + i, (j + 1) * (nx - 1) + i
            e_l, e_r = h_count + j * nx + i, h_count + j * nx + i + 1
            B = lambda: lerp(i, j, i + 1, j, e_b)
            T = lambda: lerp(i, j + 1, i + 1, j + 1, e_t)
            L = lambda: lerp(i, j, i, j + 1, e_l)
            R = lambda: lerp(i + 1, j, i + 1, j + 1, e_r)
            if mask in (1, 14):
                segs.append((L(), B()))
            elif mask in (2, 13):
                segs.append((B(), R()))
            elif mask in (3, 12):
                segs.append((L(), R()))
            elif mask in (4, 11):
                segs.append((R(), T()))
            elif mask in (6, 9):
                segs.append((B(), T()))
            elif mask in (8, 7):
                segs.append((T(), L()))
            else:  # saddle
                L(), B(), R(), T()
                if (mask == 5) == (0.25 * (d0 + d1 + d2 + d3) > 0):
                    segs += [(e_l, e_b), (e_r, e_t)]
                else:
                    segs += [(e_l, e_t), (e_b, e_r)]
    adj: dict[int, list[int]] = {}
    for a, b in segs:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    used, loops = set(), []
    for start in adj:
        if start in used:
            continue
        loop, cur, prev = [], start, None
        while cur not in used:
            used.add(cur)
            loop.append(pts[cur])
            nxt = next((c for c in adj[cur] if c != prev and c not in used), None)
            if nxt is None:
                break
            prev, cur = cur, nxt
        if len(loop) >= 3:
            loops.append(loop)
    return loops


def section_2d(sample: Callable[[int, int], float], nx: int, ny: int, lo, hi) -> Optional[m3d.CrossSection]:
    """Contour an nx x ny sampling of [lo, hi] (sample(i, j) already a band
    distance), padded with a ring firmly outside so every contour closes,
    then clipped back to the box: the pad otherwise leaves a contour that
    runs off the box 0.9 spacings too large on every side it touches."""
    sx, sy = (hi[0] - lo[0]) / (nx - 1), (hi[1] - lo[1]) / (ny - 1)
    grid = [[-1.0] * (ny + 2) for _ in range(nx + 2)]
    for i in range(nx):
        for j in range(ny):
            grid[i + 1][j + 1] = sample(i, j)
    loops = marching_squares(grid, lo[0] - sx, lo[1] - sy, sx, sy)
    if not loops:
        return None
    cs = m3d.CrossSection(loops, m3d.FillRule.EvenOdd) ^ m3d.CrossSection.square([hi[0] - lo[0], hi[1] - lo[1]]).translate(lo)
    return None if cs.is_empty() else cs


def grid_sampler(field: list, lo, spacing, band: Callable[[float], float], invert: bool):
    """Trilinear sampling of field[i][j][k]; outside the block the answer is
    "outside", falling away with distance -- clamping would smear the edge
    values outward and extend the surface rather than close it."""
    n = (len(field), len(field[0]), len(field[0][0]))

    def sample(x, y, z):
        pos = (x, y, z)
        g, i0, t, outside = [0.0] * 3, [0] * 3, [0.0] * 3, 0.0
        for a in range(3):
            raw = (pos[a] - lo[a]) / spacing[a]
            maxg = n[a] - 1
            if raw < 0:
                outside = max(outside, -raw)
            elif raw > maxg:
                outside = max(outside, raw - maxg)
            ga = min(max(raw, 0.0), maxg)
            i0[a] = min(int(ga), n[a] - 2)
            t[a] = ga - i0[a]
        if outside > 0:
            return 1.0 + outside if invert else -(1.0 + outside)
        i, j, k = i0
        tx, ty, tz = t
        f0, f1 = field[i], field[i + 1]
        c00 = f0[j][k] * (1 - tx) + f1[j][k] * tx
        c10 = f0[j + 1][k] * (1 - tx) + f1[j + 1][k] * tx
        c01 = f0[j][k + 1] * (1 - tx) + f1[j][k + 1] * tx
        c11 = f0[j + 1][k + 1] * (1 - tx) + f1[j + 1][k + 1] * tx
        return band((c00 * (1 - ty) + c10 * ty) * (1 - tz) + (c01 * (1 - ty) + c11 * ty) * tz)
    return sample


def solid_3d(sample: Callable[[float, float, float], float], lo, hi, edge: float) -> Optional[m3d.Manifold]:
    """Mesh a padded box and cut back to the one asked for: Manifold closes
    the mesh where the surface meets the box, following the sample lattice,
    which leaves a staircase on anything reaching the boundary. Tolerance
    -1: extra evaluations would only re-interpolate data already used."""
    pad = 2 * edge
    solid = m3d.Manifold.level_set(sample, [lo[0] - pad, lo[1] - pad, lo[2] - pad,
                                            hi[0] + pad, hi[1] + pad, hi[2] + pad], edge, 0.0, -1)
    if solid.is_empty():
        return None
    solid = solid ^ m3d.Manifold.cube([hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]]).translate(list(lo))
    return None if solid.is_empty() else solid
