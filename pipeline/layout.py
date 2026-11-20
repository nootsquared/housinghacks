import numpy as np

WALL = 0.12
NEXT = {"n": "w", "w": "s", "s": "e", "e": "n"}


def rot(k):
    t = np.radians(90 * k)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]).round(12)


def wall_after(wall, k):
    for _ in range(k % 4):
        wall = NEXT[wall]
    return wall


def turned_box(box, R):
    (x0, x1), (z0, z1), h = box
    pts = np.array([[x, 0, z] for x in (x0, x1) for z in (z0, z1)]) @ R.T
    return [float(pts[:, 0].min()), float(pts[:, 0].max())], [float(pts[:, 2].min()), float(pts[:, 2].max())], h


def wall_point(wall, t, box):
    (x0, x1), (z0, z1), _ = box
    return {"n": [t, 0, z0], "s": [t, 0, z1], "w": [x0, 0, t], "e": [x1, 0, t]}[wall]


def turn_room(r, k):
    k %= 4
    r["turn"] = k
    if k == 0:
        return r
    R = rot(k)
    old = r["box"]
    r["box"] = turned_box(old, R)
    for pv in r["pending"]:
        pv["C"] = R @ pv["C"]
        pv["Rc2w"] = R @ pv["Rc2w"]
    if r.get("fused"):
        r["fused"]["pos"] = (r["fused"]["pos"] @ R.T).astype(np.float32)
    if r.get("shell"):
        r["shell"]["x"], r["shell"]["z"] = r["box"][0], r["box"][1]
        r["shell"]["surfaces"] = []
    m = r.get("model")
    if m:
        m["walls"] = {wall_after(w, k): c for w, c in m["walls"].items()}
        m["fills"] = {(f if f == "floor" else wall_after(f, k)): v for f, v in m.get("fills", {}).items()}
        for o in m["openings"]:
            a = R @ np.array(wall_point(o["wall"], o["from"], old))
            b = R @ np.array(wall_point(o["wall"], o["to"], old))
            o["wall"] = wall_after(o["wall"], k)
            ax = 0 if o["wall"] in ("n", "s") else 2
            o["from"], o["to"] = sorted([round(float(a[ax]), 3), round(float(b[ax]), 3)])
        for f in m["furniture"]:
            f["center"] = (R @ np.array(f["center"])).round(3).tolist()
            if k % 2:
                f["size"] = [f["size"][2], f["size"][1], f["size"][0]]
    return r


def window_wall(model):
    if not model:
        return None
    width = {}
    for o in model["openings"]:
        if o["kind"] == "window":
            width[o["wall"]] = width.get(o["wall"], 0) + o["to"] - o["from"]
    return max(width, key=width.get) if width else None


def facing_north(wall):
    for k in range(4):
        if wall_after(wall, k) == "n":
            return k
    return 0


def lay_out(built):
    for r in built:
        spec = r["room"].get("turn")
        wall = window_wall(r.get("model"))
        turn_room(r, spec if spec is not None else (facing_north(wall) if wall else 0))
    facade = [r for r in built if window_wall(r.get("model")) == "n" and r["room"].get("at") is None]
    inner = [r for r in built if r not in facade and r["room"].get("at") is None]
    offsets = {}
    x = 0.0
    depth = 0.0
    for r in facade:
        (x0, x1), (z0, z1), _ = r["box"]
        offsets[id(r)] = np.array([x - x0, 0, -z0])
        x += x1 - x0 + WALL
        depth = max(depth, z1 - z0)
    x = 0.0
    for r in inner:
        (x0, x1), (z0, z1), _ = r["box"]
        offsets[id(r)] = np.array([x - x0, 0, depth + WALL - z0])
        x += x1 - x0 + WALL
    for r in built:
        at = r["room"].get("at")
        if at is not None:
            (x0, _), (z0, _), _ = r["box"]
            offsets[id(r)] = np.array([at[0] - x0, 0, at[1] - z0])
    return [offsets[id(r)] for r in built]
