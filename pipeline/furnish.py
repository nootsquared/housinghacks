"""Replace each room's fused object mesh with one solid, photo-textured model per piece of furniture.

Runs after build_scene on the web scene it wrote: every view there already has its camera and depth in apartment
coordinates. For each piece of furniture this cuts it out of its clearest photo, generates a complete shape from
that cutout, fits the shape to the measured depth and the photo silhouette, then textures it from the photos.
Texels no photo saw take real pixels from the object's mirror side, and failing that, patches of the object's
own photographed texture.
"""

import argparse
import json
import pickle
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from cutout import detect, matte, square_on_white

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "viewer" / "public" / "data"
WORK = ROOT / "pipeline" / "out"
SHAPE_PY = ROOT / "pipeline" / "third_party" / "hunyuan3d-mlx" / ".venv" / "bin" / "python"

FURNITURE = ["bed", "pillow", "headboard", "nightstand", "side table", "table lamp", "floor lamp", "sofa", "armchair",
             "chair", "stool", "bench", "ottoman", "coffee table", "dining table", "desk", "dresser", "cabinet",
             "bookshelf", "tv stand", "potted plant", "toilet", "bathtub", "sink", "vanity", "refrigerator", "stove",
             "oven", "microwave", "dishwasher", "kitchen island", "washing machine", "dryer"]
ON_FLOOR = {"bed", "nightstand", "side table", "floor lamp", "sofa", "armchair", "chair", "stool", "bench", "ottoman",
            "coffee table", "dining table", "desk", "dresser", "cabinet", "bookshelf", "tv stand", "potted plant",
            "toilet", "bathtub", "sink", "vanity", "refrigerator", "stove", "oven", "dishwasher", "kitchen island",
            "washing machine", "dryer", "headboard"}
BOXY = {"bed", "nightstand", "dresser", "cabinet", "bookshelf", "tv stand", "desk", "bench", "kitchen island",
        "refrigerator", "stove", "oven", "dishwasher", "bathtub", "sofa", "dining table", "coffee table", "vanity",
        "washing machine", "dryer", "headboard", "microwave"}
PARTS = {"pillow"}
SKIP_TYPES = {"hallway"}  # its photos mostly look through doorways into other rooms
MIN_AREA = 0.004
FACES = 40000
SAMPLES = 5000
TEXEL_M = 0.004


def quat_to_R(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class View:
    def __init__(self, scene_dir: Path, v: dict):
        self.id, self.room = v["id"], v["room"]
        self.image = np.asarray(Image.open(scene_dir / v["file"]).convert("RGB"))
        self.H, self.W = self.image.shape[:2]
        self.R, self.C = quat_to_R(v["quaternion"]), np.array(v["position"], float)
        t = np.tan(np.radians(v["fov"]) / 2)
        self.fy = self.H / 2 / t
        self.fx = self.W / 2 / (v["aspect"] * t)
        raw = (scene_dir / v["depth"]).read_bytes()
        w, h = np.frombuffer(raw[:4], np.uint16)
        d = np.frombuffer(raw[4:4 + w * h * 2], np.uint16).reshape(h, w).astype(np.float32) / 1000
        self.depth = cv2.resize(d, (self.W, self.H), interpolation=cv2.INTER_NEAREST)

    def project(self, X):
        pc = (X - self.C) @ self.R
        z = -pc[:, 2]
        zs = np.maximum(z, 1e-6)
        return self.W / 2 + self.fx * pc[:, 0] / zs, self.H / 2 - self.fy * pc[:, 1] / zs, z

    def unproject(self, ys, xs):
        d = self.depth[ys, xs]
        pc = np.stack([(xs + 0.5 - self.W / 2) / self.fx * d, -(ys + 0.5 - self.H / 2) / self.fy * d, -d], -1)
        return pc @ self.R.T + self.C, d > 0.05

    def forward(self):
        return -self.R[:, 2]


def dedupe(dets, iou=0.7):
    keep = []
    for d in sorted(dets, key=lambda d: -d[1]):
        b = np.array(d[2])
        if all(box_iou(b, np.array(k[2])) < iou or k[0] != d[0] for k in keep):
            keep.append(d)
    return keep


def box_iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def canonical(label):
    return {"side table": "nightstand", "table lamp": "lamp", "floor lamp": "lamp", "oven": "stove",
            "vanity": "sink", "dryer": "washing machine"}.get(label, label)


def observe(view: View, img: Image.Image, room_box):
    """Every piece of furniture in one photo: label, alpha matte, and its measured surface points."""
    dets = []
    for i in range(0, len(FURNITURE), 9):
        dets += detect(img, FURNITURE[i:i + 9], box_thr=0.32)
    found = []
    for label, score, box in dedupe([d for d in dets if d[0] in FURNITURE]):
        a = matte(img, box)
        m = a > 127
        x0, y0, x1, y1 = [int(v) for v in box]
        inside = np.zeros_like(m)
        inside[max(0, y0):y1, max(0, x0):x1] = True
        m &= inside
        if m.sum() < MIN_AREA * m.size:
            continue
        found.append({"label": label, "score": score, "box": box, "mask": m})
    order = []
    for f in found:
        ys, xs = np.nonzero(f["mask"])
        f["z"] = float(np.median(view.depth[ys, xs]))
        order.append(f)
    for f in order:
        for g in order:
            if g is f or g["z"] >= f["z"] - 0.1 or g["label"] in PARTS:
                continue
            f["mask"] = f["mask"] & ~(g["mask"] & (view.depth < f["z"] - 0.1))
    out = []
    (bx0, bx1), (bz0, bz1), bh = room_box
    for f in order:
        m = cv2.erode(f["mask"].astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        ys, xs = np.nonzero(m)
        if len(ys) < 200:
            continue
        sel = np.random.default_rng(0).choice(len(ys), min(len(ys), 6000), replace=False)
        P, ok = view.unproject(ys[sel], xs[sel])
        P = P[ok]
        P = P[(P[:, 0] > bx0 - 0.1) & (P[:, 0] < bx1 + 0.1) & (P[:, 2] > bz0 - 0.1) & (P[:, 2] < bz1 + 0.1)]
        if len(P) < 100:
            continue
        med = np.median(P, 0)
        spread = np.quantile(np.linalg.norm(P - med, axis=1), 0.9)
        P = P[np.linalg.norm(P - med, axis=1) < spread * 1.5]
        ext = np.quantile(P, 0.95, 0) - np.quantile(P, 0.05, 0)
        if min(ext[0], ext[2]) < 0.1 and "lamp" not in f["label"]:
            continue
        edge = m[:3].any() or m[-3:].any() or m[:, :3].any() or m[:, -3:].any()
        out.append({**f, "view": view.id, "points": P, "truncated": bool(edge),
                    "lo": np.quantile(P, 0.03, 0), "hi": np.quantile(P, 0.97, 0)})
    return out


def merge(obs):
    """One object per physical item, whatever each photo's detector called it: overlapping measured volumes join,
    and the kind is the area-weighted vote of its detections."""
    groups = []
    for o in sorted(obs, key=lambda o: -o["mask"].sum()):
        for g in groups:
            lo, hi = np.maximum(g["lo"], o["lo"]), np.minimum(g["hi"], o["hi"])
            inter = np.prod(np.clip(hi - lo, 0, None))
            small = min(np.prod(np.maximum(g["hi"] - g["lo"], 1e-3)), np.prod(np.maximum(o["hi"] - o["lo"], 1e-3)))
            if inter > 0.45 * small:
                g["obs"].append(o)
                g["lo"], g["hi"] = np.minimum(g["lo"], o["lo"]), np.maximum(g["hi"], o["hi"])
                break
        else:
            groups.append({"obs": [o], "lo": o["lo"].copy(), "hi": o["hi"].copy()})
    for g in groups:
        votes = {}
        for o in g["obs"]:
            k = canonical(o["label"])
            votes[k] = votes.get(k, 0) + o["score"] * o["mask"].sum()
        g["kind"] = max(votes, key=votes.get)
        g["points"] = np.concatenate([o["points"] for o in g["obs"]])
        named = [o for o in g["obs"] if canonical(o["label"]) == g["kind"]]
        g["best"] = max(named, key=lambda o: o["mask"].sum())
    return [g for g in groups if g["kind"] not in PARTS and not part_of(g, groups)]


def completeness(view: View, lo, hi):
    """Share of the object's measured box that falls inside this photo's frame."""
    c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    S = lo + (hi - lo) * np.random.default_rng(0).random((400, 3))
    u, w, z = view.project(np.concatenate([c, S]))
    return float(np.mean((z > 0.1) & (u >= 0) & (u < view.W) & (w >= 0) & (w < view.H)))


def pick_views(groups, views, keep=0.85):
    out = []
    for g in groups:
        named = [o for o in g["obs"] if canonical(o["label"]) == g["kind"]]
        score = [(completeness(views[o["view"]], g["lo"], g["hi"]), o) for o in named]
        best = max(score, key=lambda s: s[0] ** 3 * np.sqrt(s[1]["mask"].sum()))
        if best[0] < keep:
            continue
        g["best"], g["complete"] = best[1], best[0]
        out.append(g)
    return out


def part_of(g, groups):
    """Pillows on a bed and the like: already inside a larger object's cutout, so its shape includes them."""
    vol = np.prod(np.maximum(g["hi"] - g["lo"], 1e-3))
    for h in groups:
        hv = np.prod(np.maximum(h["hi"] - h["lo"], 1e-3))
        if h is g or hv <= vol * 2:
            continue
        inter = np.prod(np.clip(np.minimum(g["hi"], h["hi"]) - np.maximum(g["lo"], h["lo"]), 0, None))
        if inter < 0.6 * vol:
            continue
        o = g["best"]
        same = [k for k in h["obs"] if k["view"] == o["view"]]
        if same and (o["mask"] & same[0]["mask"]).sum() > 0.5 * o["mask"].sum():
            return True
    return False


def generate_shapes(cutouts: list[Path]):
    import cutout
    import torch

    todo = [c for c in cutouts if not c.with_suffix(".glb").exists()]
    if todo:
        cutout._models.clear()
        torch.mps.empty_cache()
        subprocess.run([str(SHAPE_PY), "-u", str(ROOT / "pipeline" / "shapegen.py"), *map(str, todo)])


def load_shape(path: Path):
    mesh = trimesh.load(path, force="mesh")
    parts = mesh.split(only_watertight=False)
    if len(parts) > 1:
        mesh = max(parts, key=lambda p: p.area)
    if len(mesh.faces) > FACES:
        import fast_simplification

        V, F = fast_simplification.simplify(mesh.vertices.astype(np.float32), mesh.faces.astype(np.int32),
                                            target_reduction=1 - FACES / len(mesh.faces))
        mesh = trimesh.Trimesh(V, F, process=True)
    mesh.vertices -= mesh.bounds.mean(0)
    return mesh


def yaw_R(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def place(V, p):
    t, sx, sy, sz, tx, ty, tz = p
    return (V * np.exp([sx, sy, sz])) @ yaw_R(t).T + [tx, ty, tz]


def view_masks(g):
    """Per photo, everything merged into this object: the bed, its pillows and sheets together."""
    out = {}
    for o in g["obs"]:
        out[o["view"]] = out[o["view"]] | o["mask"] if o["view"] in out else o["mask"].copy()
    return out


VOXEL = 0.03
FREE_TOL = 0.08


def carve(g, views: dict, room_views, room_box, on_floor):
    """The object's occupied volume: cells inside its cutout in every photo that detected it, and behind the measured
    surface in every photo of the room."""
    (bx0, bx1), (bz0, bz1), bh = room_box
    lo = np.maximum(g["lo"] - 0.6, [bx0, 0.0, bz0])
    hi = np.minimum(g["hi"] + 0.6, [bx1, bh, bz1])
    if on_floor:
        lo[1] = 0.0
    axes = [np.arange(lo[i], hi[i], VOXEL) + VOXEL / 2 for i in range(3)]
    X = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    keep = np.ones(len(X), bool)
    support = np.zeros(len(X), np.int32)
    whole = {o["view"] for o in g["obs"] if canonical(o["label"]) == g["kind"]}
    masks = {k: m for k, m in view_masks(g).items() if k in whole}
    for vid in room_views:
        v = views[vid]
        u, w, z = v.project(X)
        iu, iw = u.astype(int), w.astype(int)
        inf = (z > 0.1) & (iu >= 0) & (iu < v.W) & (iw >= 0) & (iw < v.H)
        idx = np.flatnonzero(inf)
        d = v.depth[iw[idx], iu[idx]]
        free = (d > 0.05) & (z[idx] < d - np.maximum(FREE_TOL, 0.04 * d))
        keep[idx[free]] = False
        if vid in masks:
            m = cv2.dilate(masks[vid].astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
            inside = m[iw[idx], iu[idx]]
            keep[idx[~inside]] = False
            support[idx[inside]] += 1
    keep &= support > 0
    shape = [len(a) for a in axes]
    from scipy.ndimage import binary_opening, label

    occ = binary_opening(keep.reshape(shape), iterations=1)
    lab, n = label(occ)
    if n == 0:
        return None
    sizes = np.bincount(lab.ravel())[1:]
    occ = lab == (np.argmax(sizes) + 1)
    return X[occ.ravel()]


def fit_pose(mesh, g, views: dict, room_views, room_box, on_floor):
    """Yaw, per-axis size and position that put the generated shape into the object's carved volume and onto its
    measured surface.

    The generator returns the object as its cutout showed it, facing +z toward that photo's camera, so yaw starts
    from that camera's heading.
    """
    hull = carve(g, views, room_views, room_box, on_floor)
    P = g["points"]
    P = P[np.random.default_rng(1).choice(len(P), min(len(P), 3000), replace=False)]
    if hull is None or len(hull) < 50:
        hull = P
    H = hull[np.random.default_rng(2).choice(len(hull), min(len(hull), 4000), replace=False)]
    hlo, hhi = np.quantile(hull, 0.02, 0), np.quantile(hull, 0.98, 0)
    if on_floor:
        hlo[1] = 0.0
    hhi[1] = max(hhi[1], np.quantile(P[:, 1], 0.98))
    S = mesh.sample(SAMPLES)
    ext = np.ptp(mesh.vertices, 0)
    f = views[g["best"]["view"]].C - (hlo + hhi) / 2
    heading = np.arctan2(f[0], f[2])
    if g["kind"] in BOXY:
        heading = np.round(heading / (np.pi / 2)) * (np.pi / 2)

    hull_tree = cKDTree(H)
    plo, phi = np.quantile(P, 0.03, 0), np.quantile(P, 0.97, 0)

    def cost(p):
        X = place(S, p)
        tree = cKDTree(X)
        d_obs, _ = tree.query(P, k=1)
        d_fill, _ = tree.query(H, k=1)
        d_out, _ = hull_tree.query(X, k=1)
        floor = abs(X[:, 1].min()) * 4 if on_floor else 0.0
        shape = np.var(p[1:4])
        return (np.mean(np.minimum(d_obs, 0.25)) + 0.5 * np.mean(np.minimum(d_out, 0.3))
                + 0.1 * np.mean(np.minimum(d_fill, 0.3)) + 0.3 * shape + floor)

    best, best_p = None, None
    for yaw in heading + np.radians([0, 90, -90, 180]):
        R = yaw_R(yaw)
        seen = np.abs(R.T) @ np.maximum(phi - plo, 0.05)
        s = np.full(3, np.log(np.clip(np.max(seen / ext), 0.05, 20)))
        c = (plo + phi) / 2
        c[1] = ext[1] * np.exp(s[1]) / 2 if on_floor else c[1]
        p0 = np.array([yaw, s[0], s[1], s[2], c[0], c[1], c[2]])
        r = minimize(cost, p0, method="Powell", options={"maxiter": 3000, "xtol": 1e-3, "ftol": 1e-5})
        if best is None or r.fun < best:
            best, best_p = r.fun, r.x
    p = best_p.copy()
    if g["kind"] in BOXY:
        q = np.round(p[0] / (np.pi / 2)) * (np.pi / 2)
        if abs(q - p[0]) < np.radians(15):
            p[0] = q
    V = place(mesh.vertices, p)
    if on_floor:
        V[:, 1] -= V[:, 1].min()
    return trimesh.Trimesh(V, mesh.faces, process=False), float(best), p


def raster_atlas(V, F, UV, size):
    """Per texel: the 3D point and face it samples, for every face of the unwrapped mesh."""
    pos = np.zeros((size, size, 3), np.float32)
    nrm = np.zeros((size, size, 3), np.float32)
    filled = np.zeros((size, size), bool)
    FN = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    FN /= np.linalg.norm(FN, axis=1, keepdims=True) + 1e-12
    T = UV[F] * (size - 1)
    T[..., 1] = (size - 1) - T[..., 1]
    for f in range(len(F)):
        t = T[f]
        x0, y0 = np.floor(t.min(0)).astype(int)
        x1, y1 = np.ceil(t.max(0)).astype(int)
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        px, py = xs.ravel() + 0.0, ys.ravel() + 0.0
        d = (t[1, 1] - t[2, 1]) * (t[0, 0] - t[2, 0]) + (t[2, 0] - t[1, 0]) * (t[0, 1] - t[2, 1])
        if abs(d) < 1e-12:
            continue
        a = ((t[1, 1] - t[2, 1]) * (px - t[2, 0]) + (t[2, 0] - t[1, 0]) * (py - t[2, 1])) / d
        b = ((t[2, 1] - t[0, 1]) * (px - t[2, 0]) + (t[0, 0] - t[2, 0]) * (py - t[2, 1])) / d
        c = 1 - a - b
        inside = (a >= -0.02) & (b >= -0.02) & (c >= -0.02)
        if not inside.any():
            continue
        a, b, c = a[inside], b[inside], c[inside]
        ix, iy = xs.ravel()[inside], ys.ravel()[inside]
        ok = (ix >= 0) & (ix < size) & (iy >= 0) & (iy < size)
        ix, iy, a, b, c = ix[ok], iy[ok], a[ok], b[ok], c[ok]
        pos[iy, ix] = a[:, None] * V[F[f, 0]] + b[:, None] * V[F[f, 1]] + c[:, None] * V[F[f, 2]]
        nrm[iy, ix] = FN[f]
        filled[iy, ix] = True
    return pos, nrm, filled


def mesh_depth(view: View, mesh, scale=0.5):
    """Nearest mesh depth per pixel of a downsized view, for self-occlusion."""
    S = mesh.sample(200000)
    u, w, z = view.project(S)
    W, H = int(view.W * scale), int(view.H * scale)
    iu, iw = (u * scale).astype(int), (w * scale).astype(int)
    ok = (z > 0.05) & (iu >= 0) & (iu < W) & (iw >= 0) & (iw < H)
    zb = np.full((H, W), np.inf, np.float32)
    np.minimum.at(zb, (iw[ok], iu[ok]), z[ok])
    zb = cv2.erode(np.where(np.isinf(zb), 1e3, zb), np.ones((3, 3), np.uint8))
    return zb, scale


def palette(g, views, k=10):
    """The object's own colours, from the clean cutout its shape was generated from."""
    o = g["best"]
    m = cv2.erode(o["mask"].astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    px = views[o["view"]].image[m]
    px = px[np.random.default_rng(0).choice(len(px), min(len(px), 20000), replace=False)]
    lab = cv2.cvtColor(px[None].astype(np.uint8), cv2.COLOR_RGB2LAB)[0].astype(np.float32)
    _, _, centers = cv2.kmeans(lab, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5), 3,
                               cv2.KMEANS_PP_CENTERS)
    return centers


def foreign(rgb, pal, tol=28.0):
    if len(rgb) == 0:
        return np.zeros(0, bool)
    lab = cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8)[None], cv2.COLOR_RGB2LAB)[0].astype(np.float32)
    d = np.linalg.norm((lab[:, None] - pal[None]) * [0.5, 1.0, 1.0], axis=2).min(1)
    return d > tol


def sample_views(X, N, mesh, g, views, room_views, masks):
    """Colour from the photo that sees each point most squarely, only where that photo shows this object and only
    in colours the object actually has."""
    col = np.zeros((len(X), 3), np.float32)
    best = np.zeros(len(X), np.float32)
    pal = palette(g, views)
    for vid in room_views:
        v = views[vid]
        zb, sc = mesh_depth(v, mesh)
        u, w, z = v.project(X)
        iu, iw = np.round(u).astype(int), np.round(w).astype(int)
        ok = (z > 0.1) & (iu >= 1) & (iu < v.W - 1) & (iw >= 1) & (iw < v.H - 1)
        to_cam = v.C - X
        to_cam /= np.linalg.norm(to_cam, axis=1, keepdims=True) + 1e-9
        cosang = (N * to_cam).sum(1)
        ok &= cosang > 0.15
        idx = np.flatnonzero(ok)
        zs = zb[np.clip((iw[idx] * sc).astype(int), 0, zb.shape[0] - 1), np.clip((iu[idx] * sc).astype(int), 0, zb.shape[1] - 1)]
        vis = z[idx] <= zs + 0.03
        m = masks.get(vid)
        vis &= m[iw[idx], iu[idx]] if m is not None else False
        idx = idx[vis]
        rgb = bilinear(v.image, u[idx], w[idx])
        own = ~foreign(rgb, pal)
        idx, rgb = idx[own], rgb[own]
        score = cosang[idx] * v.fx / np.maximum(z[idx], 0.1)
        better = score > best[idx]
        idx, score, rgb = idx[better], score[better], rgb[better]
        best[idx] = score
        col[idx] = rgb
    return col, best > 0


def bilinear(img, u, v):
    x0, y0 = np.floor(u - 0.5).astype(int), np.floor(v - 0.5).astype(int)
    fx, fy = (u - 0.5 - x0)[:, None], (v - 0.5 - y0)[:, None]
    H, W = img.shape[:2]
    x0, y0 = np.clip(x0, 0, W - 2), np.clip(y0, 0, H - 2)
    I = img.astype(np.float32)
    return (I[y0, x0] * (1 - fx) * (1 - fy) + I[y0, x0 + 1] * fx * (1 - fy)
            + I[y0 + 1, x0] * (1 - fx) * fy + I[y0 + 1, x0 + 1] * fx * fy)


def mirror_fill(X, N, seen, col, p):
    """Unseen points take the real colour of their mirror image across the object's own symmetry planes."""
    if not seen.any():
        return col, seen.copy()
    R = yaw_R(p[0])
    centre = np.array([p[4], 0, p[6]])
    L = (X - centre) @ R
    Ln = N @ R
    tree = cKDTree(L[seen])
    sidx = np.flatnonzero(seen)
    got = np.zeros(len(X), bool)
    out = col.copy()
    for flip in ([-1, 1, 1], [1, 1, -1], [-1, 1, -1]):
        todo = ~seen & ~got
        if not todo.any():
            break
        q = L[todo] * flip
        d, j = tree.query(q, k=1, distance_upper_bound=0.02)
        ok = np.isfinite(d)
        j = np.clip(j, 0, len(sidx) - 1)
        ok &= (Ln[todo] * flip * Ln[sidx[j]]).sum(1) > 0.6
        t = np.flatnonzero(todo)[ok]
        out[t] = col[sidx[j[ok]]]
        got[t] = True
    return out, seen | got


def patch_fill(tex, known, filled, pos, nrm, block=24, overlap=8):
    """Fill unknown texels with whole patches of this object's own known texture.

    Each patch is chosen to match the known surface nearest in 3D that faces the same way, then shifted to that
    surface's tone, so the weave, grain and print come from the object itself and only the shading is matched.
    """
    size = tex.shape[0]
    rng = np.random.default_rng(0)
    step = block - overlap
    full = cv2.erode(known.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    srcs = []
    for y in range(0, size - block, step // 2):
        for x in range(0, size - block, step // 2):
            if full[y:y + block, x:x + block].all():
                srcs.append((y, x))
    if not srcs:
        return tex
    srcs = np.array(srcs)
    lab = cv2.cvtColor(np.clip(tex, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    smean = np.array([lab[y:y + block, x:x + block].reshape(-1, 3).mean(0) for y, x in srcs])
    ky, kx = np.nonzero(known & filled)
    tree = cKDTree(pos[ky, kx])
    acc = np.zeros(tex.shape[:2], np.float32)
    paint = np.zeros_like(lab)
    ramp = np.minimum(np.minimum(np.arange(block) + 1, block - np.arange(block)) / overlap, 1.0)
    wblk = np.outer(ramp, ramp).astype(np.float32)
    need = filled & ~known
    for y in range(0, size, step):
        for x in range(0, size, step):
            sl = (slice(y, y + block), slice(x, x + block))
            n = need[sl]
            if not n.any():
                continue
            cy, cx = np.argwhere(n)[len(np.argwhere(n)) // 2]
            p, nv = pos[y + cy, x + cx], nrm[y + cy, x + cx]
            _, j = tree.query(p, k=48)
            facing = (nrm[ky[j], kx[j]] @ nv) > 0.5
            j = j[facing] if facing.sum() >= 4 else j[:8]
            target = np.median(lab[ky[j], kx[j]], 0)
            dist = np.linalg.norm((smean - target) * [1.0, 2.0, 2.0], axis=1)
            sy, sx = srcs[rng.choice(np.argsort(dist)[:6])]
            h, w = n.shape
            patch = lab[sy:sy + h, sx:sx + w]
            patch = patch - patch.reshape(-1, 3).mean(0) + target
            paint[sl] += patch * wblk[:h, :w, None]
            acc[sl] += wblk[:h, :w]
    fill = paint / np.maximum(acc, 1e-6)[..., None]
    lab[need] = fill[need]
    rgb = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)
    out = tex.copy()
    out[need] = rgb[need]
    return out


def bleed(tex, filled, px=4):
    out = tex.copy()
    have = filled.copy()
    k = np.ones((3, 3), np.uint8)
    for _ in range(px):
        grown = cv2.dilate(have.astype(np.uint8), k) > 0
        ring = grown & ~have
        if not ring.any():
            break
        s = cv2.blur(out * have[..., None], (3, 3))
        c = cv2.blur(have.astype(np.float32), (3, 3))
        out[ring] = s[ring] / np.maximum(c[ring, None], 1e-6)
        have = grown
    return out


def texture(mesh, g, p, views, room_views, out_png: Path):
    import xatlas

    vmap, F, UV = xatlas.parametrize(mesh.vertices.astype(np.float32), mesh.faces.astype(np.uint32))
    V = mesh.vertices[vmap]
    side = np.sqrt(mesh.area) / TEXEL_M
    size = int(np.clip(2 ** np.ceil(np.log2(side * 1.4)), 512, 2048))
    pos, nrm, filled = raster_atlas(V, F, UV, size)
    X, N = pos[filled], nrm[filled]
    masks = {vid: cv2.erode(m.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0 for vid, m in view_masks(g).items()}
    col, seen = sample_views(X, N, mesh, g, views, room_views, masks)
    col, have = mirror_fill(X, N, seen, col, p)
    tex = np.zeros((size, size, 3), np.float32)
    known = np.zeros((size, size), bool)
    tex[filled] = col
    known[filled] = have
    tex = patch_fill(tex, known, filled, pos, nrm)
    tex = bleed(tex, filled)
    Image.fromarray(np.clip(tex, 0, 255).astype(np.uint8)).save(out_png, quality=92)
    return V, F, UV, {"photo": float(seen.mean()), "mirror": float((have & ~seen).mean()), "patch": float((~have).mean())}


def write_glb(V, F, UV, tex_png: Path, out: Path):
    img = Image.open(tex_png)
    mat = trimesh.visual.material.PBRMaterial(baseColorTexture=img, metallicFactor=0.0, roughnessFactor=1.0, doubleSided=True)
    m = trimesh.Trimesh(V, F, visual=trimesh.visual.TextureVisuals(uv=UV, material=mat), process=False)
    m.export(out)


def room_box(room):
    c, s = np.array(room["center"]), np.array(room["size"])
    return ([c[0] - s[0] / 2, c[0] + s[0] / 2], [c[2] - s[2] / 2, c[2] + s[2] / 2], s[1])


_worker_views = {}


def finish(job):
    """Fit, texture and export one object. Runs in a worker process with its own copy of the room's views."""
    lid, room, g = job
    scene_dir = WEB / lid
    key = (lid, room["id"])
    if key not in _worker_views:
        scene = json.loads((scene_dir / "scene.json").read_text())
        _worker_views[key] = {v["id"]: View(scene_dir, v) for v in scene["views"] if v["room"] == room["id"]}
    views = _worker_views[key]
    t = time.time()
    mesh = load_shape(g["cutout"].with_suffix(".glb"))
    posed, err, p = fit_pose(mesh, g, views, room["views"], room_box(room), g["kind"] in ON_FLOOR)
    name = g["cutout"].stem
    tex_png = g["cutout"].with_name(f"{name}-albedo.jpg")
    V, F, UV, cover = texture(posed, g, p, views, room["views"], tex_png)
    write_glb(V, F, UV, tex_png, scene_dir / "models" / f"{name}.glb")
    size = np.ptp(posed.vertices, 0)
    print(f"  {room['id']} {g['kind']}: {size.round(2).tolist()} m, fit {err:.3f}, photo {cover['photo']:.0%} "
          f"mirror {cover['mirror']:.0%} patch {cover['patch']:.0%}, {time.time() - t:.0f}s", flush=True)
    return room["id"], {"file": f"models/{name}.glb", "kind": g["kind"], "size": size.round(3).tolist(),
                        "center": posed.bounds.mean(0).round(3).tolist(), "views": sorted({o["view"] for o in g["obs"]}),
                        "fit": round(err, 4), "texture": {k: round(v, 3) for k, v in cover.items()}}


def furnish(lid: str, rooms: list[str] | None = None, plan_only=False, workers=8):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    t0 = time.time()
    scene_dir = WEB / lid
    scene = json.loads((scene_dir / "scene.json").read_text())
    work = WORK / lid / "furnish"
    work.mkdir(parents=True, exist_ok=True)
    (scene_dir / "models").mkdir(exist_ok=True)
    views = {v["id"]: View(scene_dir, v) for v in scene["views"] if not rooms or v["room"] in rooms}
    jobs = []
    for room in scene["rooms"]:
        if rooms and room["id"] not in rooms or not room.get("views") or room.get("type") in SKIP_TYPES:
            continue
        rid = room["id"]
        box = room_box(room)
        obs = []
        for vid in room["views"]:
            v = views[vid]
            cache = work / f"obs-{vid}.pkl"
            if cache.exists():
                got = pickle.loads(cache.read_bytes())
            else:
                got = observe(v, Image.fromarray(v.image), box)
                cache.write_bytes(pickle.dumps(got))
            obs += [o for o in got if o["label"] in FURNITURE]
            print(f"  {rid} {vid}: {', '.join(o['label'] for o in got)}", flush=True)
        groups = pick_views(merge(obs), views)
        for g in groups:
            print(f"  {rid} -> {g['kind']} cut from {g['best']['view']} ({g['complete']:.0%} in frame), "
                  f"seen in {sorted({o['view'] for o in g['obs']})}", flush=True)
        for i, g in enumerate(groups):
            o = g["best"]
            sq, _ = square_on_white(Image.fromarray(views[o["view"]].image), (o["mask"] * 255).astype(np.uint8))
            g["cutout"] = work / f"{rid}-{i:02d}-{g['kind'].replace(' ', '_')}.png"
            sq.save(g["cutout"])
            jobs.append((lid, {k: room[k] for k in ("id", "views", "center", "size")}, g))
    print(f"[detect] {len(jobs)} objects in {time.time() - t0:.0f}s", flush=True)
    if plan_only:
        return
    t = time.time()
    generate_shapes([g["cutout"] for _, _, g in jobs])
    print(f"[generate] {time.time() - t:.0f}s", flush=True)
    t = time.time()
    jobs = [j for j in jobs if j[2]["cutout"].with_suffix(".glb").exists()]
    models = {}
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as pool:
        for rid, m in pool.map(finish, jobs):
            models.setdefault(rid, []).append(m)
    print(f"[fit+texture] {time.time() - t:.0f}s", flush=True)
    for room in scene["rooms"]:
        if room["id"] in models:
            room["models"] = models[room["id"]]
    (scene_dir / "scene.json").write_text(json.dumps(scene, indent=1))
    print(f"[total] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("listing")
    ap.add_argument("--rooms", nargs="*")
    ap.add_argument("--plan", action="store_true", help="stop after the cutouts, before generating shapes")
    a = ap.parse_args()
    furnish(a.listing, a.rooms, a.plan)
