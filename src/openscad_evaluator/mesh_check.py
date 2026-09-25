"""Strict mesh diagnosis and best-effort repair. A port of
openscad_cpp_evaluator's mesh_check.cpp (72ca136, #191): the same checks,
the same repair order, the same wording. Meshes are (verts (N, 3) float64,
tris (M, 3) int) pairs."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# OpenSCAD's GRID_FINE, 2^-20: the constant its Grid.h uses to decide two
# points are one, so a mesh that welds there welds here. Deliberately tight:
# welding discards vertices, and hole filling closes what it declines to.
DEFAULT_WELD_TOLERANCE = 0.00000095367431640625


def _plural(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


@dataclass
class MeshDiagnosis:
    """Four independent conditions, since they fail independently: two boxes
    fused along a face are watertight but have four-face edges; two cones
    tip to tip pass every edge test and still pinch at the apex."""
    boundary_edges: int = 0      # one face: a hole
    non_manifold_edges: int = 0  # three or more faces: a seam
    pinched_vertices: int = 0    # incident faces form more than one fan
    inconsistent_edges: int = 0  # both faces traverse it the same way round
    degenerate_faces: int = 0    # zero area, or a vertex named twice
    duplicate_faces: int = 0     # same three vertices as another face
    unwelded_vertices: int = 0   # distinct indices at one position

    def watertight(self) -> bool:
        return self.boundary_edges == 0

    def manifold(self) -> bool:
        # Topology only: slivers are reported but a mesh full of them can be
        # a perfectly good closed manifold, and CSG routinely emits them.
        return self.boundary_edges == 0 and self.non_manifold_edges == 0 and self.pinched_vertices == 0

    def orientable(self) -> bool:
        return self.inconsistent_edges == 0

    def ok(self) -> bool:
        return self.manifold() and self.orientable()

    def summary(self) -> str:
        parts = [_plural(n, one, many) for n, one, many in (
            (self.boundary_edges, "boundary edge", "boundary edges"),
            (self.non_manifold_edges, "non-manifold edge", "non-manifold edges"),
            (self.pinched_vertices, "pinched vertex", "pinched vertices"),
            (self.inconsistent_edges, "inconsistently wound edge", "inconsistently wound edges"),
            (self.degenerate_faces, "degenerate face", "degenerate faces"),
            (self.duplicate_faces, "duplicate face", "duplicate faces"),
            (self.unwelded_vertices, "unwelded vertex", "unwelded vertices")) if n]
        return ", ".join(parts)


@dataclass
class MeshRepairReport:
    welded_vertices: int = 0
    dropped_degenerate: int = 0
    dropped_duplicate: int = 0
    reversed_faces: int = 0
    filled_holes: int = 0
    filled_triangles: int = 0
    split_vertices: int = 0
    unfilled_holes: int = 0
    stripped_slivers: int = 0

    def did_anything(self) -> bool:
        return bool(self.welded_vertices or self.dropped_degenerate or self.dropped_duplicate
                    or self.reversed_faces or self.filled_holes or self.split_vertices or self.stripped_slivers)

    def summary(self) -> str:
        parts = []
        for n, one, many in ((self.welded_vertices, "vertex welded", "vertices welded"),
                             (self.dropped_degenerate, "degenerate face dropped", "degenerate faces dropped"),
                             (self.dropped_duplicate, "duplicate face dropped", "duplicate faces dropped"),
                             (self.reversed_faces, "face re-wound", "faces re-wound"),
                             (self.split_vertices, "pinched vertex split", "pinched vertices split")):
            if n:
                parts.append(_plural(n, one, many))
        if self.filled_holes:
            parts.append(f"{self.filled_holes} {'hole filled' if self.filled_holes == 1 else 'holes filled'} "
                         f"({self.filled_triangles} triangles)")
        for n, one, many in ((self.stripped_slivers, "zero-area face stripped", "zero-area faces stripped"),
                             (self.unfilled_holes, "hole left open", "holes left open")):
            if n:
                parts.append(_plural(n, one, many))
        return ", ".join(parts)


def _twice_area_sq(verts: np.ndarray, tris: np.ndarray) -> np.ndarray:
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    cr = np.cross(b - a, c - a)
    return np.einsum("ij,ij->i", cr, cr)


def _repeats(tris: np.ndarray) -> np.ndarray:
    return (tris[:, 0] == tris[:, 1]) | (tris[:, 1] == tris[:, 2]) | (tris[:, 0] == tris[:, 2])


def _count_pinched(tris: np.ndarray, n_verts: int) -> int:
    """Vertices whose link -- the opposite edges of their faces -- falls into
    more than one connected piece. Undirected, so a reversed neighbour does
    not also count as a pinch; winding is checked separately."""
    links: list[list[tuple[int, int]]] = [[] for _ in range(n_verts)]
    for a, b, c in tris.tolist():
        links[a].append((b, c))
        links[b].append((c, a))
        links[c].append((a, b))
    pinched = 0
    for edges in links:
        if len(edges) < 2:
            continue
        parent: dict[int, int] = {}

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        for a, b in edges:
            parent.setdefault(a, a)
            parent.setdefault(b, b)
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
        if sum(1 for i in parent if find(i) == i) > 1:
            pinched += 1
    return pinched


def check_mesh(verts, tris) -> MeshDiagnosis:
    """Diagnose a mesh against every condition. Read-only."""
    verts = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    d = MeshDiagnosis()
    if len(tris):
        rep = _repeats(tris)
        d.degenerate_faces = int((rep | (_twice_area_sq(verts, tris) < 1e-24)).sum())
        # A zero-area face stays in the topology (dropping it would open a
        # hole that is not there); a repeated vertex cannot, its self-edge
        # matches nothing.
        live = tris[~rep]
        if len(live):
            _, counts = np.unique(np.sort(live, axis=1), axis=0, return_counts=True)
            d.duplicate_faces = int((counts - 1).sum())
            a = live.reshape(-1)
            b = np.roll(live, -1, axis=1).reshape(-1)
            key = np.minimum(a, b) * (len(verts) + 1) + np.maximum(a, b)
            order = np.argsort(key, kind="stable")
            key, fwd = key[order], (a < b)[order]
            starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
            total = np.diff(np.r_[starts, len(key)])
            forward = np.add.reduceat(fwd.astype(np.int64), starts)
            d.boundary_edges = int((total == 1).sum())
            d.non_manifold_edges = int((total > 2).sum())
            d.inconsistent_edges = int(((total == 2) & (forward != 1)).sum())
            d.pinched_vertices = _count_pinched(live, len(verts))
    if len(verts):
        _, counts = np.unique(np.round(verts * 1e6).astype(np.int64), axis=0, return_counts=True)
        d.unwelded_vertices = int((counts - 1).sum())
    return d


def _weld_map(verts: np.ndarray, tolerance: float) -> tuple[np.ndarray, int]:
    """Each vertex's representative: the lowest-numbered vertex on its grid
    point. A tolerance of 0 or less welds exact duplicates only."""
    keys = np.round(verts / tolerance).astype(np.int64) if tolerance > 0 else verts
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    remap = first[inverse.reshape(-1)]
    return remap, int(len(verts) - len(first))


def _boundary_loops(tris: list) -> list[list[int]]:
    counts: dict[tuple[int, int], int] = {}
    for t in tris:
        for i in range(3):
            e = (min(t[i], t[(i + 1) % 3]), max(t[i], t[(i + 1) % 3]))
            counts[e] = counts.get(e, 0) + 1
    nxt: dict[int, list[int]] = {}
    for t in tris:
        for i in range(3):
            a, b = t[i], t[(i + 1) % 3]
            if counts[(min(a, b), max(a, b))] == 1:
                nxt.setdefault(b, []).append(a)  # walked backwards: fill faces wind against the owner
    loops = []
    while nxt:
        start = min(nxt)
        cur, loop = start, []
        while cur in nxt:
            n = nxt[cur].pop(0)
            if not nxt[cur]:
                del nxt[cur]
            loop.append(cur)
            cur = n
            if cur == start or len(loop) > 100000:
                break
        if len(loop) >= 3:
            loops.append(loop)
    return loops


def repair_mesh(verts, tris, tolerance: float = DEFAULT_WELD_TOLERANCE):
    """(verts, tris, MeshRepairReport): weld, drop degenerate and duplicate
    faces, orient consistently, fill holes, flip each component outward,
    strip zero-area faces -- in the only order that works. A mesh it cannot
    close comes back improved as far as it got."""
    verts = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    tris_in = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    report = MeshRepairReport()
    remap, report.welded_vertices = _weld_map(verts, tolerance)
    remapped = remap[tris_in] if len(tris_in) else tris_in
    seen, faces = set(), []
    for f in remapped.tolist():
        if f[0] == f[1] or f[1] == f[2] or f[0] == f[2]:
            report.dropped_degenerate += 1
            continue
        key = tuple(sorted(f))
        if key in seen:
            report.dropped_duplicate += 1
            continue
        seen.add(key)
        faces.append(f)

    # Orientation: flood-fill each component, reversing any neighbour that
    # traverses a shared edge the same way.
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for i, f in enumerate(faces):
        for e in range(3):
            a, b = f[e], f[(e + 1) % 3]
            edge_faces.setdefault((min(a, b), max(a, b)), []).append(i)
    component = [-1] * len(faces)
    n_comp = 0
    for seed in range(len(faces)):
        if component[seed] >= 0:
            continue
        component[seed] = n_comp
        stack = [seed]
        while stack:
            i = stack.pop()
            for e in range(3):
                a, b = faces[i][e], faces[i][(e + 1) % 3]
                for j in edge_faces[(min(a, b), max(a, b))]:
                    if j == i or component[j] >= 0:
                        continue
                    fj = faces[j]
                    if any(fj[k] == a and fj[(k + 1) % 3] == b for k in range(3)):
                        fj[1], fj[2] = fj[2], fj[1]
                    component[j] = n_comp
                    stack.append(j)
        n_comp += 1

    for loop in _boundary_loops(faces):
        for i in range(1, len(loop) - 1):
            faces.append([loop[0], loop[i], loop[i + 1]])
            report.filled_triangles += 1
        report.filled_holes += 1

    # Outward, by signed volume -- only meaningful once the holes are closed.
    if n_comp:
        component += [0] * (len(faces) - len(component))
        f = np.array(faces, dtype=np.int64)
        a, b, c = verts[f[:, 0]], verts[f[:, 1]], verts[f[:, 2]]
        vol = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6
        per = np.bincount(np.array(component), weights=vol, minlength=n_comp)
        for i in range(len(faces)):
            if per[component[i]] < 0:
                faces[i][1], faces[i][2] = faces[i][2], faces[i][1]

    def canon(f):
        m = f.index(min(f))
        return tuple(f[m:] + f[:m])
    known = {canon(list(f)) for f in remapped.tolist()}
    report.reversed_faces = sum(1 for f in faces if canon(f) not in known and canon([f[0], f[2], f[1]]) in known)

    out_tris = np.array(faces, dtype=np.int64).reshape(-1, 3)
    out_tris, strip = strip_slivers(verts, out_tris)
    report.stripped_slivers = strip["removed"]
    return verts, out_tris, report


def strip_slivers(verts, tris):
    """Remove zero-area faces and restitch the T-joints their removal
    exposes: a sliver's middle vertex sits inside its neighbour's edge, so
    the neighbour is split there. A needle (two corners at one point) needs
    no restitching; its coincident pair is merged. Repeats, since a split
    can itself leave a sliver. Returns (tris, report dict)."""
    verts = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    tris = [list(t) for t in np.asarray(tris, dtype=np.int64).reshape(-1, 3).tolist()]
    report = {"removed": 0, "restitched": 0, "needles": 0, "left_behind": 0, "passes": 0}
    key6 = np.round(verts * 1e6).astype(np.int64)
    for _ in range(12):
        if not tris:
            break
        t = np.array(tris, dtype=np.int64)
        sliver = (~_repeats(t)) & (_twice_area_sq(verts, t) < 1e-24)
        slivers = np.flatnonzero(sliver).tolist()
        if not slivers:
            break
        report["passes"] += 1
        info = {}
        wanted = set()
        for si in slivers:
            f = tris[si]
            pair = None
            for i in range(3):
                a, b = f[i], f[(i + 1) % 3]
                if a == b or (key6[a] == key6[b]).all():
                    pair = (min(a, b), max(a, b))
                    break
            if pair is not None and pair[0] != pair[1]:
                info[si] = ("needle", pair)
                continue
            p = verts[f]
            opp = [np.sum((p[1] - p[2]) ** 2), np.sum((p[0] - p[2]) ** 2), np.sum((p[0] - p[1]) ** 2)]
            mid = int(np.argmax(opp))
            a, b = f[(mid + 1) % 3], f[(mid + 2) % 3]
            info[si] = ("sliver", mid)
            wanted.add((min(a, b), max(a, b)))
        owners: dict[tuple[int, int], list[int]] = {}
        if wanted:
            for i, f in enumerate(tris):
                for e in range(3):
                    k = (min(f[e], f[(e + 1) % 3]), max(f[e], f[(e + 1) % 3]))
                    if k in wanted:
                        owners.setdefault(k, []).append(i)
        is_sliver = set(slivers)
        dead, added, merge = set(), [], {}
        for si in slivers:
            if si in dead:
                continue
            kind, val = info[si]
            if kind == "needle":
                keep, drop = val
                dead.add(si)
                merge[drop] = keep
                report["removed"] += 1
                report["needles"] += 1
                continue
            f = tris[si]
            m, a, b = f[val], f[(val + 1) % 3], f[(val + 2) % 3]
            nb = fallback = None
            for cand in owners.get((min(a, b), max(a, b)), []):
                if cand == si or cand in dead:
                    continue
                if cand in is_sliver:
                    fallback = cand if fallback is None else fallback
                    continue
                nb = cand
                break
            nb = nb if nb is not None else fallback
            if nb is None:
                report["left_behind"] += 1
                continue
            n = tris[nb]
            e = next((i for i in range(3) if {n[i], n[(i + 1) % 3]} == {a, b}), -1)
            if e < 0:
                report["left_behind"] += 1
                continue
            x, y, apex = n[e], n[(e + 1) % 3], n[(e + 2) % 3]
            dead.update((si, nb))
            added += [[x, m, apex], [m, y, apex]]
            report["removed"] += 1
            report["restitched"] += 1

        def resolve(v):
            for _ in range(8):
                if v not in merge:
                    break
                v = merge[v]
            return v
        nxt = [[resolve(v) for v in f] for i, f in enumerate(tris) if i not in dead]
        nxt += [[resolve(v) for v in f] for f in added]
        if len(nxt) == len(tris) and not added:
            break
        tris = nxt
    return np.array(tris, dtype=np.int64).reshape(-1, 3), report
