"""Seamless photo textures for every wall and floor.

Each room surface is projected straight from the room's photos at PX texels per metre. A texel takes its colour from
the photo that sees it most squarely, unless furniture or anything else stands in front of it there. Photos are
exposure-matched per room and blended softly at the seams. Texels no photo sees get the surface's own paint tone plus
real grain quilted from the seen part of the same surface, so floors keep their planks running the right way.

    python walls.py <listing> [--rooms bedroom-1 ...]
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image

from furnish import WEB, WORK, View

PX = 300
SURFACES = ("n", "s", "w", "e", "floor")


def surface(room, sid):
    """Texel grid of one surface: origin of texel (0, 0), metres per column and per row, inward normal, size in px."""
    (x0, x1), (z0, z1), h = room["shell"]["x"], room["shell"]["z"], room["shell"]["height"]
    down = np.array([0, -1.0, 0])
    o, u, v, n, wm, hm = {
        "n": ((x0, h, z0), (1, 0, 0), down, (0, 0, 1), x1 - x0, h),
        "s": ((x1, h, z1), (-1, 0, 0), down, (0, 0, -1), x1 - x0, h),
        "w": ((x0, h, z1), (0, 0, -1), down, (1, 0, 0), z1 - z0, h),
        "e": ((x1, h, z0), (0, 0, 1), down, (-1, 0, 0), z1 - z0, h),
        "floor": ((x0, 0, z0), (1, 0, 0), (0, 0, 1), (0, 1, 0), x1 - x0, z1 - z0),
    }[sid]
    W, H = max(8, round(wm * PX)), max(8, round(hm * PX))
    o, u, v, n = (np.array(a, float) for a in (o, u, v, n))
    cs, rs = (np.arange(W) + 0.5) / PX, (np.arange(H) + 0.5) / PX
    X = o + rs[:, None, None] * v + cs[None, :, None] * u
    return X, n, (H, W)


def furniture_mask(lid, view):
    cache = WORK / lid / "furnish" / f"obs-{view.id}.pkl"
    m = np.zeros((view.H, view.W), np.uint8)
    if cache.exists():
        for o in pickle.loads(cache.read_bytes()):
            m |= o["mask"].astype(np.uint8)
    return cv2.dilate(m, np.ones((21, 21), np.uint8)) > 0


def model_meshes(room, scene_dir):
    out = []
    for m in room.get("models", []):
        p = scene_dir / m["file"]
        if p.exists():
            out.append(trimesh.load(p, force="mesh"))
    return out


def model_depth(view, meshes, scale=0.25):
    """Nearest fitted-furniture depth per pixel, upsampled to the photo."""
    W, H = int(view.W * scale), int(view.H * scale)
    zb = np.full((H, W), 1e3, np.float32)
    for mesh in meshes:
        S = mesh.sample(300000)
        u, w, z = view.project(S)
        iu, iw = (u * scale).astype(int), (w * scale).astype(int)
        ok = (z > 0.05) & (iu >= 0) & (iu < W) & (iw >= 0) & (iw < H)
        np.minimum.at(zb, (iw[ok], iu[ok]), z[ok])
    zb = cv2.erode(zb, np.ones((5, 5), np.uint8))
    return cv2.resize(zb, (view.W, view.H), interpolation=cv2.INTER_NEAREST)


def furniture_boxes(room):
    """Axis-aligned boxes of furniture that has no fitted model, slightly inflated."""
    if room.get("models"):
        return []
    boxes = []
    for m in room.get("model", {}).get("furniture", []):
        c, s = np.array(m["center"], float), np.array(m["size"], float)
        boxes.append((c - s / 2 - 0.06, c + s / 2 + 0.06))
    return boxes


def behind_boxes(C, P, boxes):
    """Texels whose line of sight from the camera passes through any furniture box."""
    hit = np.zeros(len(P), bool)
    d = P - C
    inv = 1 / np.where(np.abs(d) < 1e-9, 1e-9, d)
    for lo, hi in boxes:
        if np.all(C > lo) and np.all(C < hi):
            continue
        t0, t1 = (lo - C) * inv, (hi - C) * inv
        tn = np.minimum(t0, t1).max(1)
        tf = np.maximum(t0, t1).min(1)
        hit |= (tn < tf) & (tf > 0) & (tn < 0.995)
    return hit


def inside_rooms(P, rooms):
    hit = np.zeros(len(P), bool)
    for (x0, x1), (z0, z1) in rooms:
        hit |= (P[:, 0] > x0) & (P[:, 0] < x1) & (P[:, 2] > z0) & (P[:, 2] < z1)
    return hit


def project(view, X, n, blocked, near_depth, tol, boxes, mdepth=None, others=()):
    """Colour, score and visibility of every texel of a surface in one photo."""
    H, W = X.shape[:2]
    P = X.reshape(-1, 3)
    px, py, z = view.project(P)
    ray = P - view.C
    cos = -(ray @ n) / np.linalg.norm(ray, axis=1)
    ok = (z > 0.2) & (cos > 0.15) & (px >= 1) & (px < view.W - 1) & (py >= 1) & (py < view.H - 1)
    xi, yi = np.clip(px.astype(int), 0, view.W - 1), np.clip(py.astype(int), 0, view.H - 1)
    d = near_depth[yi, xi]
    seen = view.C + ray * (d / np.maximum(z, 1e-6))[:, None]
    rise = (seen - P) @ n
    sel = ok & (d > 0.05) & ~blocked[yi, xi]
    if sel.sum() > 500:
        # the shell is a box fit, so let the photo's own depth say where this surface really sits
        shift = float(np.median(rise[sel]))
        if abs(shift) < 0.3:
            rise = rise - shift
    ok &= (d <= 0.05) | (rise < tol)
    # seeing far past the surface is a window to the outside, or a doorway into another room
    through = ok & (rise < -0.3)
    if through.any() and others:
        ok[through] &= ~inside_rooms(seen[through], others)
    ok &= ~blocked[yi, xi]
    ok[ok] &= ~behind_boxes(view.C, P[ok], boxes)
    if mdepth is not None:
        ok &= mdepth[yi, xi] > z - 0.04
    mx, my = px.reshape(H, W).astype(np.float32), py.reshape(H, W).astype(np.float32)
    col = cv2.remap(view.image, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float32)
    score = np.where(ok, cos * view.fx / np.maximum(z, 0.2), 0).reshape(H, W)
    return col, score.astype(np.float32), ok.reshape(H, W)


def exposure(samples, vids):
    """Per-photo colour gain from the texels two photos share, anchored so the mean gain is 1."""
    idx = {v: i for i, v in enumerate(vids)}
    rows, rhs, wts = [], [], []
    for sid, per in samples.items():
        keys = list(per)
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                both = per[a][2] & per[b][2]
                if both.sum() < 1500:
                    continue
                la = np.log(np.median(per[a][0][both], 0) + 1)
                lb = np.log(np.median(per[b][0][both], 0) + 1)
                r = np.zeros(len(vids))
                r[idx[a]], r[idx[b]] = 1, -1
                rows.append(r)
                rhs.append(lb - la)
                wts.append(np.sqrt(both.sum()))
    gains = {v: np.ones(3) for v in vids}
    if not rows:
        return gains
    A = np.array(rows) * np.array(wts)[:, None]
    B = np.array(rhs) * np.array(wts)[:, None]
    A = np.vstack([A, np.ones(len(vids)) * 50])
    B = np.vstack([B, np.zeros(3)])
    A = np.vstack([A, np.eye(len(vids)) * 5])
    B = np.vstack([B, np.zeros((len(vids), 3))])
    g = np.linalg.lstsq(A, B, rcond=None)[0]
    return {v: np.exp(np.clip(g[idx[v]], -0.5, 0.5)) for v in vids}


def blend(per, gains, shape):
    """Pick one photo per region from smoothed scores, then feather the seams between them."""
    keys = list(per)
    H, W = shape
    if not keys:
        return np.zeros((H, W, 3), np.float32), np.zeros((H, W), bool)
    smooth = np.stack([cv2.GaussianBlur(per[k][1], (0, 0), 20) * per[k][2] for k in keys])
    label = smooth.argmax(0)
    known = np.any([per[k][2] for k in keys], 0)
    acc, wsum = np.zeros((H, W, 3), np.float32), np.zeros((H, W), np.float32)
    for i, k in enumerate(keys):
        w = cv2.GaussianBlur((label == i).astype(np.float32), (0, 0), 4) * per[k][2]
        acc += w[..., None] * per[k][0] * gains[k].astype(np.float32)
        wsum += w
    return acc / np.maximum(wsum, 1e-6)[..., None], known & (wsum > 1e-3)


def masked_mean(img, m, sigma):
    num = cv2.GaussianBlur(img * m[..., None], (0, 0), sigma)
    den = cv2.GaussianBlur(m.astype(np.float32), (0, 0), sigma)
    return num / np.maximum(den, 1e-6)[..., None], den


def base_tone(img, m, fallback):
    """Smooth paint tone everywhere: the masked average, grown outward from the seen paint into the holes."""
    H, W = m.shape
    if m.sum() < 200:
        return np.broadcast_to(np.array(fallback, np.float32), (H, W, 3)).copy()
    s = 16
    h, w = max(2, H // s), max(2, W // s)
    num = cv2.resize(img * m[..., None], (w, h), interpolation=cv2.INTER_AREA)
    den = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    have = den > 0.3
    small = num / np.maximum(den, 1e-6)[..., None]
    small = np.clip(small, 0, 255).astype(np.uint8)
    small = cv2.inpaint(small, (~have).astype(np.uint8), 5, cv2.INPAINT_TELEA)
    small = cv2.GaussianBlur(small.astype(np.float32), (0, 0), 1.0)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC)


def quilt(res, fill, src_res, src_ok, block, rng, tries=80):
    """Fill `fill` texels of a residual (grain) image with real blocks cut from `src_res` where `src_ok`,
    picking each block to agree with what is already there across the overlap."""
    H, W = fill.shape
    sh, sw = src_ok.shape
    ov = block // 4
    step = block - ov
    cover = cv2.boxFilter(src_ok.astype(np.float32), -1, (block, block), anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)
    ys, xs = np.nonzero(cover[:sh - block + 1, :sw - block + 1] > 0.9)
    if len(ys) == 0:
        return res
    out = res.copy()
    have = ~fill
    ramp = np.minimum(1, (np.arange(block) + 0.5) / ov).astype(np.float32)
    feather = np.minimum.outer(ramp, ramp)
    for y in range(0, H, step):
        for x in range(0, W, step):
            bh, bw = min(block, H - y), min(block, W - x)
            need = fill[y:y + bh, x:x + bw]
            if not need.any():
                continue
            cur, curm = out[y:y + bh, x:x + bw], have[y:y + bh, x:x + bw]
            pick = rng.choice(len(ys), min(tries, len(ys)), replace=False)
            best, best_e = None, np.inf
            for j in pick:
                cand = src_res[ys[j]:ys[j] + bh, xs[j]:xs[j] + bw]
                e = (((cand - cur) ** 2).sum(-1) * curm).sum() / max(curm.sum(), 1) if curm.any() else rng.random()
                if e < best_e:
                    best, best_e = cand, e
            a = np.where(curm, feather[:bh, :bw], 1.0)[..., None]
            a = np.where(need[..., None], a, 0)
            out[y:y + bh, x:x + bw] = a * best + (1 - a) * cur
            have[y:y + bh, x:x + bw] |= need
    return out


def paint_mask(img, known):
    """Seen texels of the surface's dominant material (paint, or the main floor finish), in Lab."""
    lab = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    pts = lab[known]
    if len(pts) < 200:
        return known.copy()
    sub = pts[np.random.default_rng(0).choice(len(pts), min(len(pts), 20000), replace=False)]
    k = 4
    _, lab_k, cent = cv2.kmeans(sub, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5), 3,
                                cv2.KMEANS_PP_CENTERS)
    main = cent[np.bincount(lab_k.ravel(), minlength=k).argmax()]
    return known & (np.linalg.norm(lab - main, axis=-1) < 24)


GRAIN_SIGMA = {"floor": 18, "wall": 6}
RUG_M2 = 0.4  # smaller off-colour floor patches are clutter (chair legs, cables), larger ones rugs


def big_regions(m, m2):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=4)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] > m2 * PX * PX
    return keep[lab]
OWN_GRAIN = {"floor": 0.6, "wall": 0.3}  # m^2 of seen material before a surface quilts from itself


def gather(lid, room, views, scene_dir, others):
    """Blended photo texture, seen mask and dominant-material mask of every surface of a room."""
    rv = [views[v] for v in room["views"] if v in views]
    blocked = {v.id: furniture_mask(lid, v) for v in rv}
    near = {v.id: cv2.erode(v.depth, np.ones((15, 15), np.uint8)) for v in rv}
    boxes = furniture_boxes(room)
    meshes = model_meshes(room, scene_dir)
    mdepth = {v.id: model_depth(v, meshes) for v in rv} if meshes else {}
    samples, shapes = {}, {}
    for sid in SURFACES:
        X, n, shapes[sid] = surface(room, sid)
        per = {}
        for v in rv:
            col, score, ok = project(v, X, n, blocked[v.id], near[v.id], 0.1 if sid == "floor" else 0.18, boxes,
                                     mdepth.get(v.id), others)
            if ok.sum() > 200:
                per[v.id] = (col, score, ok)
        samples[sid] = per
    gains = exposure(samples, [v.id for v in rv])
    out = {}
    for sid in SURFACES:
        img, known = blend(samples[sid], gains, shapes[sid])
        known = cv2.erode(known.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        paint = paint_mask(img, known)
        kind = "floor" if sid == "floor" else "wall"
        if kind == "floor":
            known = paint | big_regions(known & ~paint, RUG_M2)
        res = img - masked_mean(img, paint, GRAIN_SIGMA[kind])[0]
        out[sid] = {"img": img, "known": known, "paint": paint, "res": res, "kind": kind}
    return out


def lab(rgb):
    return cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8).reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(
        np.shape(rgb)).astype(np.float32)


def finish_surface(s, fallback, donor, rng):
    """Seen texels as photographed; the rest the surface's own tone plus real grain quilted from the same material."""
    img, known, paint, kind = s["img"], s["known"], s["paint"], s["kind"]
    own = paint.sum() > OWN_GRAIN[kind] * PX * PX
    if kind == "floor" and not own and donor is not None:
        # unseen or wrong-colour floor: ignore the room's own pixels and quilt the donor floor end to end
        if paint.sum() < 0.05 * OWN_GRAIN[kind] * PX * PX:
            known = np.zeros_like(known)
            paint = known
            base = base_tone(donor["img"], donor["paint"], fallback)
            res = quilt(np.zeros_like(img), np.ones(known.shape, bool), donor["res"], donor["paint"],
                        int(0.4 * PX), rng)
            return np.clip(base + res, 0, 255).astype(np.uint8), 0.0
        ref = lab(np.median(donor["img"][donor["paint"]], 0))
        known = known & (np.linalg.norm(lab(img) - ref, axis=-1) < 30)
        paint = known
    base = base_tone(img, paint, fallback)
    res = np.where(known[..., None], s["res"], 0)
    if own:
        res = quilt(res, ~known, s["res"], paint, int((0.4 if kind == "floor" else 0.16) * PX), rng)
    elif kind == "floor" and donor is not None:
        res = quilt(res, ~known, donor["res"], donor["paint"], int(0.4 * PX), rng)
    a = cv2.GaussianBlur(known.astype(np.float32), (0, 0), 3) * known
    out = a[..., None] * img + (1 - a[..., None]) * (base + res)
    return np.clip(out, 0, 255).astype(np.uint8), float(known.mean())


def main(lid, rooms=None):
    t0 = time.time()
    scene_dir = WEB / lid
    scene = json.loads((scene_dir / "scene.json").read_text())
    todo = [r for r in scene["rooms"] if r.get("shell") and r.get("model") and r.get("views")
            and (not rooms or r["id"] in rooms)]
    need = {v for r in todo for v in r["views"]}
    views = {v["id"]: View(scene_dir, v) for v in scene["views"] if v["id"] in need}
    boxes = {r["id"]: (r["shell"]["x"], r["shell"]["z"]) for r in scene["rooms"] if r.get("shell")}
    got = {}
    for r in todo:
        t = time.time()
        others = [b for k, b in boxes.items() if k != r["id"]]
        got[r["id"]] = gather(lid, r, views, scene_dir, others)
        print(f"  {r['id']}: " + ", ".join(f"{k} {s['known'].mean():.0%}" for k, s in got[r["id"]].items())
              + f" seen ({time.time() - t:.0f}s)", flush=True)
    def donor_for(room):
        """Best-seen floor of the same material, preferring a room of the same type (bedrooms share flooring)."""
        mat = room["model"].get("floor", {}).get("material", "wood")
        pool = [(r.get("type") == room.get("type"), got[r["id"]]["floor"]["paint"].sum(), r["id"]) for r in todo
                if r["model"].get("floor", {}).get("material", "wood") == mat]
        pool = [p for p in pool if p[1] > OWN_GRAIN["floor"] * PX * PX] or pool
        return got[max(pool)[2]]["floor"] if pool else None

    paint_px = [s["img"][s["paint"]] for r in todo for k, s in got[r["id"]].items() if k != "floor"]
    house_wall = np.median(np.concatenate(paint_px), 0) if paint_px else np.array([225, 224, 220.0])
    rng = np.random.default_rng(7)
    result = {}
    enough = OWN_GRAIN["wall"] * PX * PX
    for r in todo:
        surf = got[r["id"]]
        mat = r["model"].get("floor", {}).get("material", "wood")
        walls = [s for k, s in surf.items() if k != "floor" and s["paint"].sum() > enough]
        room_wall = np.median(np.concatenate([s["img"][s["paint"]] for s in walls]), 0) if walls else house_wall
        files, cover = {}, {}
        for sid, s in surf.items():
            donor = donor_for(r) if sid == "floor" else None
            if s["paint"].sum() > enough:
                fb = np.median(s["img"][s["paint"]], 0)
            elif sid == "floor":
                fb = np.median(donor["img"][donor["paint"]], 0) if donor is not None else [150, 130, 110]
            else:
                fb = room_wall
            tex, cover[sid] = finish_surface(s, fb, donor, rng)
            name = f"shell/{r['id']}-surf-{sid}.jpg"
            Image.fromarray(tex).save(scene_dir / name, quality=90)
            files[sid] = name
        result[r["id"]] = files, {k: round(v, 3) for k, v in cover.items()}
    scene = json.loads((scene_dir / "scene.json").read_text())
    for r in scene["rooms"]:
        if r["id"] in result:
            r["model"]["surfaces"], r["model"]["surfaceCover"] = result[r["id"]]
    (scene_dir / "scene.json").write_text(json.dumps(scene, indent=1))
    print(f"[walls] {len(result)} rooms in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("listing")
    ap.add_argument("--rooms", nargs="*")
    a = ap.parse_args()
    main(a.listing, a.rooms)
