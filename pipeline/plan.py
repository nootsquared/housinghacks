import argparse
import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from layout import rot, wall_after, wall_point

WALL = 0.12
TOUCH = 0.03
DOOR_W = 0.9
MARGIN = 0.08
MIN_DOOR = 0.75
FT2 = 0.092903
OPP = {"n": "s", "s": "n", "w": "e", "e": "w"}
WALLS = ("n", "e", "s", "w")

ALIAS = {
    "hallway": "hall", "corridor": "hall", "foyer": "entry", "living_room": "living", "lounge": "living",
    "bath": "bathroom", "toilet": "bathroom", "ensuite": "bathroom", "study": "office", "den": "office",
    "dining_room": "dining", "utility": "laundry",
}
COMMON = {"hall", "entry", "living", "kitchen", "dining"}
LINKS = {
    "hall": {"hall", "entry", "living", "kitchen", "dining", "bedroom", "bathroom", "office", "laundry"},
    "entry": {"hall", "entry", "living", "kitchen", "dining", "bedroom", "bathroom", "office", "laundry"},
    "living": {"hall", "entry", "living", "kitchen", "dining", "bedroom", "bathroom", "office"},
    "kitchen": {"hall", "entry", "living", "kitchen", "dining", "laundry"},
    "dining": {"hall", "entry", "living", "kitchen"},
    "bedroom": {"hall", "entry", "living", "bathroom"},
    "bathroom": {"hall", "entry", "living", "bedroom"},
    "office": {"hall", "entry", "living"},
    "laundry": {"hall", "entry", "kitchen"},
}
HUB_ORDER = ("living", "kitchen", "dining", "hall", "entry")

W = {
    "door": 1.0, "virtual": 1.2, "blocked": 2.5, "dangling": 0.8, "forbidden": 4.0, "closet": 0.5,
    "fill": 8.0, "concave": 0.6, "void": 0.4, "window": 2.0, "sqft": 3.0, "path": 6.0, "bath_access": 3.0,
    "bath_through": 3.0, "hall_dead_end": 1.5, "entry": 1.5, "prior": 4.0, "flex": 3.0, "kitchen": 1.5, "daylight": 3.0,
}


def canon(t):
    t = (t or "").lower()
    return ALIAS.get(t, t)


def allowed(a, b):
    return b in LINKS.get(a, set()) and a in LINKS.get(b, set())


@dataclass(frozen=True)
class Slot:
    wall: str
    a: float
    b: float
    kind: str
    conf: float = 1.0

    @property
    def flexible(self):
        return self.kind in ("wild", "virtual")


@dataclass
class Shape:
    rect: tuple
    slots: list
    windows: list
    box: list
    k: int
    variant: int


@dataclass
class Room:
    id: str
    type: str
    flex: bool
    shapes: dict = field(default_factory=dict)


@dataclass
class Placed:
    id: str
    shape: Shape
    off: tuple

    @property
    def rect(self):
        x0, x1, z0, z1 = self.shape.rect
        dx, dz = self.off
        return (x0 + dx, x1 + dx, z0 + dz, z1 + dz)

    def world(self, s):
        d = self.off[0] if s.wall in "ns" else self.off[1]
        return replace(s, a=s.a + d, b=s.b + d)

    def slots(self):
        return [self.world(s) for s in self.shape.slots]

    def windows(self):
        return [self.world(s) for s in self.shape.windows]


def wall_span(rect, wall):
    x0, x1, z0, z1 = rect
    return (x0, x1) if wall in "ns" else (z0, z1)


def wall_pos(rect, wall):
    x0, x1, z0, z1 = rect
    return {"n": z0, "s": z1, "w": x0, "e": x1}[wall]


def turn_interval(wall, a, b, box, k):
    R = rot(k)
    p = R @ np.array(wall_point(wall, a, box), float)
    q = R @ np.array(wall_point(wall, b, box), float)
    w = wall_after(wall, k)
    ax = 0 if w in "ns" else 2
    lo, hi = sorted((float(p[ax]), float(q[ax])))
    return w, lo, hi


def turned_rect(box, k):
    R = rot(k)
    (x0, x1), (z0, z1), _ = box
    pts = np.array([[x, 0, z] for x in (x0, x1) for z in (z0, z1)]) @ R.T
    return (float(pts[:, 0].min()), float(pts[:, 0].max()), float(pts[:, 2].min()), float(pts[:, 2].max()))


def flex_boxes(box):
    (x0, x1), (z0, z1), h = box
    cx, cz, wx, wz = (x0 + x1) / 2, (z0 + z1) / 2, x1 - x0, z1 - z0
    out = []
    for sx in (0.5, 1.0, 1.6):
        for sz in (0.5, 1.0, 1.6):
            nx, nz = max(wx * sx, 1.0), max(wz * sz, 1.0)
            b = [[cx - nx / 2, cx + nx / 2], [cz - nz / 2, cz + nz / 2], h]
            if all(abs(b[0][1] - b[0][0] - (o[0][1] - o[0][0])) > 0.05 or abs(b[1][1] - b[1][0] - (o[1][1] - o[1][0])) > 0.05 for o in out):
                out.append(b)
    out.sort(key=lambda b: (abs((b[0][1] - b[0][0]) - wx) + abs((b[1][1] - b[1][0]) - wz)))
    return out


def room_doors(room, doors):
    if room["id"] in doors:
        return [d for d in doors[room["id"]] if d.get("kind", "door") in ("door", "closet")]
    return [{**o, "confidence": o.get("confidence", 0.6)} for o in room.get("openings", []) if o["kind"] == "door"]


def make_shape(room, box, k, v=0):
    flex, ds, real, windows = room["_flex"], room["_doors"], room["_real"], room["_windows"]
    rect = turned_rect(box, k)
    slots = []
    for d in ds:
        w, a, b = turn_interval(d["wall"], d["from"], d["to"], box, k)
        slots.append(Slot(w, a, b, d.get("kind", "door"), float(d.get("confidence", 0.6))))
    if flex or not real or room["_relax"]:
        kind = "wild" if flex else "virtual"
        glazed = {o["wall"] for o in windows}
        for wall in WALLS:
            if not flex and wall in glazed:
                continue
            w = wall_after(wall, k)
            lo, hi = wall_span(rect, w)
            if hi - lo >= DOOR_W + 2 * MARGIN:
                slots.append(Slot(w, lo + MARGIN, hi - MARGIN, kind, 1.0))
    wins = [Slot(*turn_interval(o["wall"], o["from"], o["to"], box, k), "window") for o in windows]
    return Shape(rect, slots, wins, box, k, v)


def widen(d, box, min_w=MIN_DOOR):
    lo, hi = box[0] if d["wall"] in "ns" else box[1]
    a, b = d["from"], d["to"]
    if b - a >= min_w:
        return d
    c = min(max((a + b) / 2, lo + min_w / 2), hi - min_w / 2)
    return {**d, "from": c - min_w / 2, "to": c + min_w / 2}


def annotate(room, doors, relax=False):
    ds = [widen(d, room["box"]) if d.get("kind", "door") == "door" else d for d in room_doors(room, doors)]
    real = [d for d in ds if d.get("kind", "door") == "door"]

    def doorway(o):
        return any(d["wall"] == o["wall"] and overlap(d["from"], d["to"], o["from"], o["to"]) > 0.5 * (o["to"] - o["from"]) for d in ds)

    windows = [o for o in room.get("openings", []) if o["kind"] == "window" and not doorway(o)]
    return {**room, "_flex": not room.get("shell", True), "_doors": ds, "_real": real, "_windows": windows, "_relax": relax}


def prepare(room, doors, relax=False):
    room = annotate(room, doors, relax)
    boxes = flex_boxes(room["box"]) if room["_flex"] else [room["box"]]
    r = Room(room["id"], canon(room["type"]), room["_flex"])
    for v, box in enumerate(boxes):
        for k in range(4):
            r.shapes[(v, k)] = make_shape(room, box, k, v)
    return r


def realize(rooms, doors, solution):
    out = {}
    for room in rooms:
        sol = solution[room["id"]]
        shape = make_shape(annotate(room, doors, solution.get("relaxed", False)), sol.get("box", room["box"]), sol["turn"])
        out[room["id"]] = Placed(room["id"], shape, (sol["offset"][0], sol["offset"][2]))
    return out


def overlap(a0, a1, b0, b1):
    return min(a1, b1) - max(a0, b0)


def separated(r, q):
    gx = max(q[0] - r[1], r[0] - q[1])
    gz = max(q[2] - r[3], r[2] - q[3])
    return max(gx, gz) >= WALL - TOUCH


def contact(r, wall, q):
    """Overlap interval of r's wall with q when q sits flush (one wall thickness) beyond it."""
    gap = {"e": q[0] - r[1], "w": r[0] - q[1], "s": q[2] - r[3], "n": r[2] - q[3]}[wall]
    if abs(gap - WALL) > TOUCH:
        return None
    lo, hi = wall_span(r, wall)
    qlo, qhi = wall_span(q, wall)
    a, b = max(lo, qlo), min(hi, qhi)
    return (a, b) if b - a > 0.05 else None


def facing_distance(rect, wall, a, b, others):
    """Distance from the wall segment [a,b] to the nearest room straight outside it."""
    p = wall_pos(rect, wall)
    best = math.inf
    for q in others:
        qlo, qhi = wall_span(q, wall)
        if overlap(a, b, qlo, qhi) <= 0.05:
            continue
        d = {"e": q[0] - p, "w": p - q[1], "s": q[2] - p, "n": p - q[3]}[wall]
        if d >= -TOUCH:
            best = min(best, d)
    return best


def shift_candidates(sa, sb, a_span, b_span):
    """Offsets t (added to b's along-wall coords) that line a door on A's wall up with one on B's."""
    ts = [a_span[0] - b_span[0], a_span[1] - b_span[1], (a_span[0] + a_span[1] - b_span[0] - b_span[1]) / 2]
    if not sa.flexible and not sb.flexible:
        ts.append((sa.a + sa.b - sb.a - sb.b) / 2)
    elif not sa.flexible:
        ts += [sa.a - MARGIN - sb.a, sa.b + MARGIN - sb.b, (sa.a + sa.b - sb.a - sb.b) / 2]
    elif not sb.flexible:
        ts += [sa.a + MARGIN - sb.a, sa.b - MARGIN - sb.b, (sa.a + sa.b - sb.a - sb.b) / 2]
    seen, out = set(), []
    for t in ts:
        key = round(t, 3)
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


def door_fit(sa, sb, t, a_span, b_span):
    """Door interval and match quality for a given shift, or None if the doors cannot meet."""
    lo = max(a_span[0], b_span[0] + t)
    hi = min(a_span[1], b_span[1] + t)
    if hi - lo < DOOR_W * 0.8:
        return None
    if not sa.flexible and not sb.flexible:
        ov = overlap(sa.a, sa.b, sb.a + t, sb.b + t)
        width = min(sa.b - sa.a, sb.b - sb.a)
        if ov < 0.6 * width:
            return None
        off = abs((sa.a + sa.b) / 2 - (sb.a + sb.b + 2 * t) / 2)
        q = sa.conf * sb.conf * ov * (0.5 / width + 0.5 / max(sa.b - sa.a, sb.b - sb.a)) - 0.4 * off
        return (max(sa.a, sb.a + t), min(sa.b, sb.b + t)), q, 0
    if not sa.flexible or not sb.flexible:
        fixed = sa if not sa.flexible else replace(sb, a=sb.a + t, b=sb.b + t)
        flex = sb if not sa.flexible else sa
        fa, fb = (flex.a + t, flex.b + t) if flex is sb else (flex.a, flex.b)
        if fixed.a < max(fa, lo) - 0.02 or fixed.b > min(fb, hi) + 0.02:
            return None
        return (fixed.a, fixed.b), fixed.conf, int(flex.kind == "virtual")
    fa, fb = max(sa.a, sb.a + t), min(sa.b, sb.b + t)
    if fb - fa < DOOR_W:
        return None
    c = (fa + fb) / 2
    return (c - DOOR_W / 2, c + DOOR_W / 2), 0.5, int(sa.kind == "virtual") + int(sb.kind == "virtual")


@dataclass
class Edge:
    a: str
    b: str
    wall: str
    door: tuple
    quality: float
    virtual: int
    slot_a: Slot
    slot_b: Slot


@dataclass
class State:
    placed: dict
    edges: list
    cost: float = 0.0

    def key(self):
        return tuple(sorted((p.id, p.shape.k, p.shape.variant, round(p.off[0], 2), round(p.off[1], 2)) for p in self.placed.values()))


def used_on(state, rid, wall, a, b, gap=0.0):
    for e in state.edges:
        if (rid == e.a and e.wall == wall) or (rid == e.b and OPP[e.wall] == wall):
            if overlap(e.door[0], e.door[1], a, b) > -gap:
                return True
    return False


def windows_clear(p, others):
    for s in p.windows():
        if facing_distance(p.rect, s.wall, s.a, s.b, others) < 0.6:
            return False
    return True


def expand(state, rooms):
    children = []
    unplaced = [r for r in rooms.values() if r.id not in state.placed]
    for A in sorted(state.placed.values(), key=lambda p: p.id):
        ta = rooms[A.id].type
        for sa in A.slots():
            if sa.kind == "closet":
                continue
            if not sa.flexible and used_on(state, A.id, sa.wall, sa.a, sa.b):
                continue
            a_span = wall_span(A.rect, sa.wall)
            for B in unplaced:
                if not allowed(ta, B.type):
                    continue
                for (v, k), shape in sorted(B.shapes.items()):
                    for sb in shape.slots:
                        if sb.kind == "closet" or sb.wall != OPP[sa.wall]:
                            continue
                        b_span = wall_span(shape.rect, sb.wall)
                        perp = wall_pos(A.rect, sa.wall) + (WALL if sa.wall in "es" else -WALL) - wall_pos(shape.rect, sb.wall)
                        for t in shift_candidates(sa, sb, a_span, b_span):
                            fit = door_fit(sa, sb, t, a_span, b_span)
                            if fit is None:
                                continue
                            door, q, virt = fit
                            if sa.flexible and used_on(state, A.id, sa.wall, door[0], door[1], 0.15):
                                continue
                            off = (t, perp) if sa.wall in "ns" else (perp, t)
                            P = Placed(B.id, shape, off)
                            others = [p.rect for p in state.placed.values()]
                            if not all(separated(P.rect, o) for o in others):
                                continue
                            if not windows_clear(P, others):
                                continue
                            if not all(windows_clear(p, [P.rect]) for p in state.placed.values()):
                                continue
                            sb_world = P.world(sb)
                            e = Edge(A.id, B.id, sa.wall, door, q, virt, sa, sb_world)
                            children.append(State({**state.placed, B.id: P}, state.edges + [e]))
    return children


def raster(rects, res=0.05):
    xs = [r[0] - WALL / 2 for r in rects] + [r[1] + WALL / 2 for r in rects]
    zs = [r[2] - WALL / 2 for r in rects] + [r[3] + WALL / 2 for r in rects]
    x0, z0 = min(xs) - 2 * res, min(zs) - 2 * res
    nx = int(math.ceil((max(xs) - x0) / res)) + 3
    nz = int(math.ceil((max(zs) - z0) / res)) + 3
    g = np.zeros((nz, nx), bool)
    for r in rects:
        i0, i1 = int(round((r[2] - WALL / 2 - z0) / res)), int(round((r[3] + WALL / 2 - z0) / res))
        j0, j1 = int(round((r[0] - WALL / 2 - x0) / res)), int(round((r[1] + WALL / 2 - x0) / res))
        g[i0:i1, j0:j1] = True
    return g, res


def shape_metrics(rects):
    g, res = raster(rects)
    k = max(1, int(round(0.35 / res)))
    closed = ndimage.binary_closing(np.pad(g, k), structure=np.ones((k, k)))[k:-k, k:-k]
    filled = ndimage.binary_fill_holes(closed)
    voids = filled & ~g
    zi, xi = np.nonzero(filled)
    bbox = (zi.max() - zi.min() + 1) * (xi.max() - xi.min() + 1)
    p = np.pad(filled, 1).astype(int)
    s = p[:-1, :-1] + p[1:, :-1] + p[:-1, 1:] + p[1:, 1:]
    diag = (p[:-1, :-1] == p[1:, 1:]) & (p[1:, :-1] == p[:-1, 1:]) & (p[:-1, :-1] != p[1:, :-1])
    concave = int((s == 3).sum() + 2 * diag.sum())
    return {
        "fill": filled.sum() / bbox, "concave": concave, "void": voids.sum() * res * res,
        "bbox_area": bbox * res * res, "area": g.sum() * res * res,
        "bbox": ((xi.max() - xi.min() + 1) * res, (zi.max() - zi.min() + 1) * res),
    }


def links(state, rooms):
    """All connections: tree edges plus doors that happen to line up with a neighbour."""
    graph = {r: set() for r in state.placed}
    for e in state.edges:
        graph[e.a].add(e.b)
        graph[e.b].add(e.a)
    issues = {"blocked": 0.0, "forbidden": 0.0, "closet": 0.0, "virtual": 0.0}
    exits = []
    placed = state.placed
    for rid, P in sorted(placed.items()):
        for s in P.slots():
            if s.flexible:
                continue
            if s.kind == "door" and used_on(state, rid, s.wall, s.a, s.b):
                continue
            hit = None
            for qid, Q in sorted(placed.items()):
                if qid == rid:
                    continue
                c = contact(P.rect, s.wall, Q.rect)
                if c and overlap(s.a, s.b, *c) > 0.3 * (s.b - s.a):
                    hit = qid
                    break
            if s.kind == "closet":
                if hit is not None:
                    issues["closet"] += s.conf
                continue
            if hit is None:
                others = [q.rect for q in placed.values() if q.id != rid]
                d = facing_distance(P.rect, s.wall, s.a, s.b, others)
                exits.append((rid, s, d))
                continue
            Q = placed[hit]
            match = None
            for t in Q.slots():
                if t.wall != OPP[s.wall] or t.kind == "closet":
                    continue
                ov = overlap(s.a, s.b, t.a, t.b)
                need = DOOR_W * 0.8 if t.flexible else 0.5 * min(s.b - s.a, t.b - t.a)
                if t.flexible:
                    ov = overlap(s.a, s.b, t.a, t.b) if t.a <= s.a + 0.02 and t.b >= s.b - 0.02 else 0
                    need = 0.1
                if ov >= need:
                    match = t
                    break
            if match is None:
                issues["blocked"] += s.conf
            elif not allowed(rooms[rid].type, rooms[hit].type):
                issues["forbidden"] += s.conf
            else:
                if hit not in graph[rid]:
                    graph[rid].add(hit)
                    graph[hit].add(rid)
                issues["virtual"] += int(match.kind == "virtual")
    return graph, issues, exits


def exterior_runs(state, rid, min_run=1.0, glass_ok=False):
    P = state.placed[rid]
    others = [q.rect for q in state.placed.values() if q.id != rid]
    wins = P.windows()
    runs = []
    for wall in WALLS:
        lo, hi = wall_span(P.rect, wall)
        ts = np.arange(lo + 0.05, hi - 0.05, 0.1)
        ok = []
        for t in ts:
            free = facing_distance(P.rect, wall, t - 0.05, t + 0.05, others) > 1.5
            glazed = any(w.wall == wall and w.a - 0.1 <= t <= w.b + 0.1 for w in wins)
            ok.append(free and (glass_ok or not glazed))
        run, best = 0, (0, 0)
        for i, v in enumerate(ok):
            run = run + 1 if v else 0
            best = max(best, (run, i))
        if best[0] * 0.1 >= min_run:
            end = ts[best[1]] + 0.05
            runs.append((wall, end - best[0] * 0.1, end))
    return runs


def reach(graph, start, through):
    seen, todo = {start}, [start]
    while todo:
        u = todo.pop()
        for v in graph[u]:
            if v not in seen:
                seen.add(v)
                if through(v):
                    todo.append(v)
    return seen


def prior_cost(state, rooms, prior):
    allr = [P.rect for P in state.placed.values()]
    X0, X1 = min(r[0] for r in allr), max(r[1] for r in allr)
    Z0, Z1 = min(r[2] for r in allr), max(r[3] for r in allr)
    norm = {}
    for rid, P in state.placed.items():
        x0, x1, z0, z1 = P.rect
        norm[rid] = (((x0 + x1) / 2 - X0) / max(X1 - X0, 1e-6), ((z0 + z1) / 2 - Z0) / max(Z1 - Z0, 1e-6))
    scores = {}
    for mirror in (False, True):
        for g in range(4):
            total, n = 0.0, 0
            for t in sorted({rooms[r].type for r in norm}):
                ids = sorted(r for r in norm if rooms[r].type == t)
                cands = [p for p in prior["rooms"] if canon(p["type"]) == t]
                if not cands:
                    continue
                D = np.array([[math.dist(dihedral(*norm[r], g, mirror), (p["u"], p["v"])) for p in cands] for r in ids])
                if len(cands) < len(ids):
                    D = np.hstack([D] * math.ceil(len(ids) / len(cands)))
                i, j = linear_sum_assignment(D)
                total += D[i, j].sum()
                n += len(i)
            scores[(mirror, g)] = total / n if n else 0.0
    upright = min(range(4), key=lambda g: (scores[(False, g)], g))
    return min(scores.values()), (4 - upright) % 4


def dihedral(u, v, g, mirror):
    if mirror:
        u = 1 - u
    for _ in range(g):
        u, v = 1 - v, u
    return u, v


def evaluate(state, rooms, sqft=None, prior=None):
    b = {}
    graph, issues, exits = links(state, rooms)
    b["door"] = -sum(e.quality for e in state.edges) * W["door"]
    b["virtual"] = (sum(e.virtual for e in state.edges) + issues["virtual"]) * W["virtual"]
    b["blocked"] = issues["blocked"] * W["blocked"]
    b["forbidden"] = issues["forbidden"] * W["forbidden"]
    b["closet"] = issues["closet"] * W["closet"]
    m = shape_metrics([P.rect for P in state.placed.values()])
    b["fill"] = (1 - m["fill"]) * W["fill"]
    b["concave"] = m["concave"] * W["concave"]
    b["void"] = m["void"] * W["void"]
    win = 0.0
    for rid, P in state.placed.items():
        others = [q.rect for q in state.placed.values() if q.id != rid]
        for s in P.windows():
            d = facing_distance(P.rect, s.wall, s.a, s.b, others)
            win += max(0.0, (4.0 - d) / 4.0)
    b["window"] = win * W["window"]
    if sqft:
        target = sqft * FT2
        b["sqft"] = (max(0, m["bbox_area"] / target - 1.15) + max(0, m["area"] / target - 1.0)
                     + max(0, 0.75 - m["area"] / target)) * W["sqft"]
    b["flex"] = flex_cost(state, rooms)
    hubs = [rid for rid in state.placed if rooms[rid].type in COMMON]
    root = min(hubs, key=lambda r: (HUB_ORDER.index(rooms[r].type), r)) if hubs else sorted(state.placed)[0]
    common = reach(graph, root, lambda v: rooms[v].type in COMMON)
    path = 0.0
    for rid in state.placed:
        if rooms[rid].type in ("bedroom", "office") and rid not in common:
            path += 1
    b["path"] = path * W["path"]
    baths = [rid for rid in state.placed if rooms[rid].type == "bathroom"]
    if baths and not any(rid in common for rid in baths):
        b["bath_access"] = W["bath_access"]
    b["bath_through"] = sum(max(0, len(graph[r]) - 1) for r in baths) * W["bath_through"]
    dark = [r for r in state.placed if rooms[r].type in ("bedroom", "living")
            and not state.placed[r].windows() and not exterior_runs(state, r, glass_ok=True)]
    if dark:
        b["daylight"] = len(dark) * W["daylight"]
    living = [r for r in state.placed if rooms[r].type in ("living", "dining")]
    for rid in state.placed:
        if rooms[rid].type == "kitchen" and living and not graph[rid] & set(living):
            b["kitchen"] = b.get("kitchen", 0) + W["kitchen"]
    dangling, entry = 0.0, None
    for rid, s, d in exits:
        if entry is None and rooms[rid].type in COMMON and d > 1.5:
            entry = (rid, s)
            continue
        dangling += s.conf * (0.5 if d > 1.5 else 1.0)
    b["dangling"] = dangling * W["dangling"]
    if entry is None:
        order = ("entry", "hall", "living", "kitchen", "dining")
        for rid in sorted(hubs, key=lambda r: (order.index(rooms[r].type), r)):
            runs = exterior_runs(state, rid)
            if runs:
                wall, a, c = max(runs, key=lambda r: (r[2] - r[1], r[0]))
                mid = (a + c) / 2
                entry = (rid, Slot(wall, mid - DOOR_W / 2, mid + DOOR_W / 2, "entry"))
                break
        else:
            b["entry"] = W["entry"]
    for rid in state.placed:
        if rooms[rid].type in ("hall", "entry") and len(graph[rid]) + (entry is not None and entry[0] == rid) < 2:
            b["hall_dead_end"] = b.get("hall_dead_end", 0) + W["hall_dead_end"]
    g_turn = 0
    if prior:
        pc, g_turn = prior_cost(state, rooms, prior)
        b["prior"] = pc * W["prior"]
    total = sum(b.values())
    return total, b, {"graph": {k: sorted(v) for k, v in graph.items()}, "entry": entry, "metrics": m, "global_turn": g_turn}


def flex_cost(state, rooms):
    cost = 0.0
    for rid, P in state.placed.items():
        if rooms[rid].flex:
            (x0, x1), (z0, z1), _ = P.shape.box
            (a0, a1), (c0, c1), _ = rooms[rid].shapes[(0, 0)].box
            ratio = (x1 - x0) * (z1 - z0) / ((a1 - a0) * (c1 - c0))
            cost += max(ratio - 1, 0) + max(1 / ratio - 1, 0) * 0.3
    return cost * W["flex"]


def partial_cost(state, rooms):
    rects = [P.rect for P in state.placed.values()]
    pad = WALL / 2
    area = sum((r[1] - r[0] + 2 * pad) * (r[3] - r[2] + 2 * pad) for r in rects)
    bbox = (max(r[1] for r in rects) - min(r[0] for r in rects) + 2 * pad) * (max(r[3] for r in rects) - min(r[2] for r in rects) + 2 * pad)
    cost = W["fill"] * (1 - area / bbox)
    cost += sum(-e.quality * W["door"] + e.virtual * W["virtual"] for e in state.edges)
    for rid, P in state.placed.items():
        others = [q.rect for q in state.placed.values() if q.id != rid]
        for s in P.windows():
            cost += W["window"] * max(0.0, (4.0 - facing_distance(P.rect, s.wall, s.a, s.b, others)) / 4.0)
    return cost + flex_cost(state, rooms)


def topology(state):
    return frozenset(frozenset((e.a, e.b)) for e in state.edges)


def diverse(states, beam, per_topology=None):
    per_topology = per_topology or max(4, beam // 20)
    count, out = {}, []
    for s in states:
        t = topology(s)
        if count.get(t, 0) < per_topology:
            count[t] = count.get(t, 0) + 1
            out.append(s)
        if len(out) >= beam:
            break
    return out


def search(rooms, sqft=None, prior=None, beam=400, keep=12):
    hubs = [r for r in rooms.values() if r.type in COMMON]
    root = min(hubs, key=lambda r: (HUB_ORDER.index(r.type), r.id)) if hubs else sorted(rooms.values(), key=lambda r: r.id)[0]
    frontier = [State({root.id: Placed(root.id, root.shapes[(v, 0)], (0.0, 0.0))}, [])
                for v in sorted({v for v, _ in root.shapes})]
    for _ in range(len(rooms) - 1):
        seen = {}
        for s in frontier:
            for c in expand(s, rooms):
                c.cost = partial_cost(c, rooms)
                key = c.key()
                if key not in seen or c.cost < seen[key].cost:
                    seen[key] = c
        frontier = diverse(sorted(seen.values(), key=lambda s: (s.cost, s.key())), beam)
        if not frontier:
            break
    complete = [s for s in frontier if len(s.placed) == len(rooms)]
    scored = []
    for s in complete:
        total, b, info = evaluate(s, rooms, sqft, prior)
        scored.append((total, s.key(), s, b, info))
    scored.sort(key=lambda x: (x[0], x[1]))
    out, shapes = [], set()
    for total, key, s, b, info in scored:
        coarse = (topology(s), tuple(sorted((p.id, *(round(v / 0.3) for v in p.rect)) for p in s.placed.values())))
        if coarse in shapes:
            continue
        shapes.add(coarse)
        out.append((total, s, b, info))
        if len(out) >= keep:
            break
    return out


def placement(state, global_turn=0, entry=None):
    R = rot(global_turn)
    res = {}
    for rid, P in state.placed.items():
        off = R @ np.array([P.off[0], 0.0, P.off[1]])
        res[rid] = {"turn": (P.shape.k + global_turn) % 4, "offset": [round(float(off[0]), 4), 0.0, round(float(off[2]), 4)]}
        if P.shape.variant:
            res[rid]["box"] = P.shape.box
    conns = []
    for e in state.edges:
        A = state.placed[e.a]
        p = wall_pos(A.rect, e.wall) + (WALL / 2 if e.wall in "es" else -WALL / 2)
        ends = [(t, p) if e.wall in "ns" else (p, t) for t in e.door]
        ends = [(R @ np.array([x, 0.0, z]))[[0, 2]].round(4).tolist() for x, z in ends]
        kind = "virtual" if e.virtual else ("wild" if e.slot_a.flexible or e.slot_b.flexible else "door")
        conns.append({"a": e.a, "b": e.b, "door": ends, "kind": kind, "quality": round(e.quality, 3)})
    if entry:
        rid, sl = entry
        rect = state.placed[rid].rect
        p = wall_pos(rect, sl.wall)
        ends = [(t, p) if sl.wall in "ns" else (p, t) for t in (sl.a, sl.b)]
        res["entrance"] = {"room": rid, "door": [(R @ np.array([x, 0.0, z]))[[0, 2]].round(4).tolist() for x, z in ends]}
    return res, conns


def candidates(rooms, doors, sqft=None, prior=None, beam=400, keep=12):
    for relax in (False, True):
        prepared = {r["id"]: prepare(r, doors, relax) for r in rooms}
        ranked = search(prepared, sqft, prior, beam=beam, keep=keep)
        if ranked:
            return relax, ranked
    raise ValueError("no layout satisfies the hard constraints")


def solution(state, b, info, total, relaxed):
    out, conns = placement(state, info["global_turn"], info["entry"])
    out["connections"] = conns
    out["relaxed"] = relaxed
    out["score_breakdown"] = {"total": round(total, 3), **{k: round(v, 3) for k, v in b.items()}}
    return out


def solve(rooms: list[dict], doors: dict, sqft: float | None = None, prior: dict | None = None, beam: int = 400) -> dict:
    relaxed, ranked = candidates(rooms, doors, sqft, prior, beam=beam, keep=1)
    total, state, b, info = ranked[0]
    return solution(state, b, info, total, relaxed)


def load_listing(name, doors_path=None):
    root = Path(__file__).resolve().parent
    rooms = json.loads((root / "out" / name / "rooms_prelayout.json").read_text())
    dp = Path(doors_path) if doors_path else root / "out" / name / "doors.json"
    doors = json.loads(dp.read_text()) if dp.exists() else {}
    listing_dir = root.parent / "data" / "listings" / name
    listing = json.loads((listing_dir / "listing.json").read_text())
    prior_path = listing_dir / "floorplan_prior.json"
    prior = json.loads(prior_path.read_text()) if prior_path.exists() else None
    return rooms, doors, listing.get("sqft"), prior


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("listing")
    ap.add_argument("--doors")
    ap.add_argument("--prior")
    ap.add_argument("--out")
    args = ap.parse_args()
    rooms, doors, sqft, prior = load_listing(args.listing, args.doors)
    if args.prior:
        prior = json.loads(Path(args.prior).read_text())
    text = json.dumps(solve(rooms, doors, sqft, prior), indent=1)
    if args.out:
        Path(args.out).write_text(text)
    print(text)
