"""dxf_dim() / dxf_cross(): read a measurement out of a DXF file rather than
out of the model. A port of openscad_cpp_evaluator's dxf_dim.cpp, itself a
port of OpenSCAD's io/dxfdim.cc and the DIMENSION/LINE half of io/DxfData.cc
(cpp #100). Each returns (value, warning); the caller prints the warning."""
from __future__ import annotations

import math
import os


def _entities(path: str, xorigin: float, yorigin: float, scale: float) -> list[dict]:
    """One entity per group-0 marker, coordinates shifted and scaled as the
    reference does while parsing -- except groups 11/12/16 (and 21/22/26),
    extents rather than positions, which are scaled but not shifted."""
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    out, cur = [], None

    def num(s):
        try:
            return float(s)
        except ValueError:
            return 0.0

    for i in range(0, len(lines) - 1, 2):
        code_s, data = lines[i].strip(), lines[i + 1].strip()
        if not code_s:
            continue
        try:
            code = int(code_s)
        except ValueError:
            break
        if code == 0:
            if cur is not None:
                out.append(cur)
            cur = {"type": data, "layer": "", "name": "", "dim_type": 0, "angle": 0.0, "angle2": 0.0, "radius": 0.0,
                   "coords": [[0.0, 0.0] for _ in range(7)], "xs": [], "ys": []}
            continue
        if cur is None:
            continue
        if 10 <= code <= 16:
            cur["coords"][code - 10][0] = num(data) * scale if code in (11, 12, 16) else (num(data) - xorigin) * scale
        elif 20 <= code <= 26:
            cur["coords"][code - 20][1] = num(data) * scale if code in (21, 22, 26) else (num(data) - yorigin) * scale
        if code == 1:
            cur["name"] = data
        elif code == 8:
            cur["layer"] = data
        elif code in (10, 11):
            cur["xs"].append((num(data) - xorigin) * scale)
        elif code in (20, 21):
            cur["ys"].append((num(data) - yorigin) * scale)
        elif code == 50:
            cur["angle"] = num(data)
        elif code == 51:
            cur["angle2"] = num(data)
        elif code == 40:
            cur["radius"] = num(data) * scale
        elif code == 70:
            try:
                cur["dim_type"] = int(data)
            except ValueError:
                pass
    if cur is not None:
        out.append(cur)
    return out


def dxf_dim(path: str, raw_file: str, layer: str, origin, scale: float, name: str):
    if not os.path.exists(path):
        return None, f"Can't open DXF file '{raw_file}'!"
    for e in _entities(path, origin[0], origin[1], scale):
        if e["type"] != "DIMENSION" or (layer and layer != e["layer"]) or (name and e["name"] != name):
            continue
        c, t = e["coords"], e["dim_type"] & 7
        if t == 0:  # rotated, horizontal or vertical
            x, y = c[4][0] - c[3][0], c[4][1] - c[3][1]
            a = math.radians(e["angle"])
            return abs(x * math.cos(a) + y * math.sin(a)), None
        if t == 1:  # aligned
            return math.hypot(c[4][0] - c[3][0], c[4][1] - c[3][1]), None
        if t == 2:  # angular
            a1 = math.degrees(math.atan2(c[0][0] - c[5][0], c[0][1] - c[5][1]))
            a2 = math.degrees(math.atan2(c[4][0] - c[3][0], c[4][1] - c[3][1]))
            return abs(a1 - a2), None
        if t in (3, 4):  # diameter or radius
            return math.hypot(c[5][0] - c[0][0], c[5][1] - c[0][1]), None
        if t == 6:  # ordinate
            return (c[3][0] if e["dim_type"] & 64 else c[3][1]), None
        # type 5 (angular 3-point) is unsupported, as in the reference.
        return None, f"Dimension '{name}' in '{raw_file}', layer '{layer}' has unsupported type!"
    return None, f"Can't find dimension '{name}' in '{raw_file}', layer '{layer}'!"


def dxf_cross(path: str, raw_file: str, layer: str, origin, scale: float):
    """The intersection of the first two 2-point paths on the layer, as the
    reference walks them: a LINE sharing an endpoint with another joins it
    into a longer path and is no stroke of a cross. (openscad_cpp_evaluator
    takes the first two LINEs outright, and finds a "cross" in any two lines
    of an outline.) ponytail: LINE and ARC ends join; other entity types
    (polylines, splines) do not disqualify a LINE here."""
    if not os.path.exists(path):
        return None, f"Can't open DXF file '{raw_file}'!"
    ents = [e for e in _entities(path, origin[0], origin[1], scale) if not layer or layer == e["layer"]]
    segs = [((e["xs"][0], e["ys"][0]), (e["xs"][1], e["ys"][1]))
            for e in ents if e["type"] == "LINE" and len(e["xs"]) >= 2 and len(e["ys"]) >= 2]
    ends = {}

    def touch(p):
        key = (round(p[0], 6), round(p[1], 6))
        ends[key] = ends.get(key, 0) + 1

    for seg in segs:
        for p in seg:
            touch(p)
    for e in ents:
        if e["type"] == "ARC":
            (cx, cy), r = e["coords"][0], e["radius"]
            for a in (e["angle"], e["angle2"]):
                touch((cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a))))
    pts = []
    for seg in segs:
        if any(ends[(round(p[0], 6), round(p[1], 6))] > 1 for p in seg):
            continue
        pts += list(seg)
        if len(pts) == 4:
            (x1, y1), (x2, y2), (x3, y3), (x4, y4) = pts
            dem = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
            if dem == 0:
                break  # parallel: no cross
            ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / dem
            return [x1 + ua * (x2 - x1), y1 + ua * (y2 - y1)], None
    return None, f"Can't find cross in '{raw_file}', layer '{layer}'!"
