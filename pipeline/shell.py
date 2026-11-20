from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

PX_PER_M = 200
DEFAULT_WALL = np.array([226, 222, 214], np.float32)
DEFAULT_FLOOR = np.array([150, 130, 110], np.float32)


@dataclass
class Cam:
    name: str
    Rw: np.ndarray
    C: np.ndarray
    K: np.ndarray
    W: int
    H: int
    depth: np.ndarray
    conf: np.ndarray
    image: np.ndarray
    outside: bool = False


def ransac_plane(P, iters=300, tol=0.02, rng=np.random.default_rng(0)):
    best, best_n = None, 0
    for _ in range(iters):
        s = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n = n / np.linalg.norm(n)
        inl = np.abs((P - s[0]) @ n) < tol
        if inl.sum() > best_n:
            best, best_n = inl, inl.sum()
    Q = P[best]
    c = Q.mean(0)
    n = np.linalg.svd(Q - c)[2][-1]
    return n, c, best


def plane_peaks(v, cell=0.04, frac=0.12):
    if len(v) < 40:
        return np.array([]), np.array([])
    edges = np.arange(v.min(), v.max() + cell, cell)
    if len(edges) < 3:
        return np.array([v.mean()]), np.array([len(v)])
    hist, edges = np.histogram(v, bins=edges)
    hist = np.convolve(hist, [0.25, 0.5, 0.25], "same")
    idx = [i for i in range(len(hist)) if hist[i] >= frac * hist.max()
           and hist[i] >= hist[max(i - 1, 0)] and hist[i] >= hist[min(i + 1, len(hist) - 1)]]
    return (edges[idx] + cell / 2), hist[idx]


def wall_extents(P, N, axis, cams_c):
    sel = (np.abs(N[:, axis]) > 0.8) & (P[:, 1] > 0.15) & (P[:, 1] < 2.3)
    v = P[sel, axis]
    c_lo, c_hi = cams_c[:, axis].min(), cams_c[:, axis].max()
    peaks, w = plane_peaks(v)
    out, seen = [], []
    for side in (0, 1):
        cand = [(p, ww) for p, ww in zip(peaks, w) if (p < c_lo - 0.15 if side == 0 else p > c_hi + 0.15)]
        if cand:
            strong = max(ww for _, ww in cand)
            cand = [p for p, ww in cand if ww >= 0.3 * strong]
            out.append(max(cand) if side == 0 else min(cand))
            seen.append(True)
        else:
            near = P[(P[:, 1] > 0.1) & (np.linalg.norm(P[:, [0, 2]] - cams_c[:, [0, 2]].mean(0), axis=1) < 6), axis]
            q = np.quantile(near, 0.06 if side == 0 else 0.94) if len(near) else (c_lo - 1.5 if side == 0 else c_hi + 1.5)
            out.append(min(q, c_lo - 0.6) if side == 0 else max(q, c_hi + 0.6))
            seen.append(False)
    return np.array(out), tuple(seen)


def ceiling_height(P, N, cam_h):
    sel = (np.abs(N[:, 1]) > 0.85) & (P[:, 1] > max(2.0, cam_h + 0.4)) & (P[:, 1] < 4.0)
    peaks, w = plane_peaks(P[sel, 1], 0.03, 0.2)
    if len(peaks):
        return float(np.clip(peaks.min(), 2.3, 3.4))
    return 2.7


def bake_plane(origin, du, dv, w, h, normal, cams: list[Cam], tol=0.035):
    tw, th = max(8, int(w * PX_PER_M)), max(8, int(h * PX_PER_M))
    uu, vv = np.meshgrid((np.arange(tw) + 0.5) / tw, (np.arange(th) + 0.5) / th)
    X = origin + uu[..., None] * du * w + vv[..., None] * dv * h
    X = X.reshape(-1, 3)
    best = np.zeros(len(X))
    out = np.zeros((len(X), 3), np.uint8)
    for c in cams:
        Xc = (X - c.C) @ c.Rw
        z = -Xc[:, 2]
        ok = z > 0.1
        px = c.K[0, 0] * Xc[:, 0] / np.maximum(z, 1e-6) + c.K[0, 2]
        py = -c.K[1, 1] * Xc[:, 1] / np.maximum(z, 1e-6) + c.K[1, 2]
        ok &= (px >= 0) & (px < c.W - 1) & (py >= 0) & (py < c.H - 1)
        ix = np.clip(px.astype(int), 0, c.W - 1)
        iy = np.clip(py.astype(int), 0, c.H - 1)
        d = c.depth[iy, ix]
        ok &= np.abs(d - z) < tol * z + 0.025
        ray = X - c.C
        ray /= np.linalg.norm(ray, axis=1, keepdims=True)
        cosang = np.abs(ray @ normal)
        score = c.conf[iy, ix] * cosang / (1 + z) * ok
        take = score > best
        if not take.any():
            continue
        best[take] = score[take]
        ih, iw = c.image.shape[:2]
        sx = np.clip((px[take] / c.W * iw).astype(int), 0, iw - 1)
        sy = np.clip((py[take] / c.H * ih).astype(int), 0, ih - 1)
        out[take] = c.image[sy, sx]
    seen = best > 0
    img = out.reshape(th, tw, 3)
    mask = seen.reshape(th, tw)
    return img, mask


def clean_mask(mask, min_area=0.002):
    import cv2

    m = mask.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area * m.size:
            keep[lab == i] = 1
    return keep.astype(bool)


def push_pull(img, mask, fallback):
    img = img.astype(np.float32)
    w = mask.astype(np.float32)
    if w.sum() < 1:
        return np.broadcast_to(fallback, img.shape).copy()
    levels = [(img * w[..., None], w)]
    while min(levels[-1][1].shape) > 2:
        c, ww = levels[-1]
        h, wd = ww.shape
        h2, w2 = h // 2 * 2, wd // 2 * 2
        c = c[:h2, :w2].reshape(h2 // 2, 2, w2 // 2, 2, 3).sum((1, 3))
        ww = ww[:h2, :w2].reshape(h2 // 2, 2, w2 // 2, 2).sum((1, 3))
        levels.append((c, ww))
    c, ww = levels[-1]
    col = np.where(ww[..., None] > 0, c / np.maximum(ww, 1e-6)[..., None], fallback)
    if (ww == 0).all():
        col[:] = (levels[0][0].sum((0, 1)) / max(w.sum(), 1))
    for c, ww in reversed(levels[:-1]):
        import cv2

        up = cv2.resize(col, (ww.shape[1], ww.shape[0]), interpolation=cv2.INTER_LINEAR)
        known = c / np.maximum(ww, 1e-6)[..., None]
        a = np.clip(ww, 0, 1)[..., None]
        col = known * a + up * (1 - a)
    return col


def feather(mask, px):
    import cv2

    d = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    return np.clip(d / max(px, 1), 0, 1)


def best_patch(mask, size):
    if size >= min(mask.shape):
        return None
    ii = np.pad(mask.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    s = ii[size:, size:] - ii[:-size, size:] - ii[size:, :-size] + ii[:-size, :-size]
    y, x = np.unravel_index(int(np.argmax(s)), s.shape)
    if s[y, x] < 0.92 * size * size:
        return None
    return y, x


def fill_surface(img, mask, kind, ref_color):
    import cv2

    mask = clean_mask(mask)
    lum = lambda x: x @ np.array([0.299, 0.587, 0.114], np.float32)
    shade = cv2.GaussianBlur(push_pull(img, mask, ref_color), (0, 0), 28)
    ratio = np.clip(lum(shade) / max(float(lum(ref_color)), 1.0), 0.7, 1.3)[..., None]
    base = ref_color[None, None, :] * ratio
    out = base
    if kind == "floor":
        size = int(0.5 * PX_PER_M)
        pos = best_patch(mask, size)
        if pos is not None:
            y, x = pos
            patch = img[y:y + size, x:x + size].astype(np.float32)
            patch = np.concatenate([patch, patch[:, ::-1]], 1)
            patch = np.concatenate([patch, patch[::-1]], 0)
            reps = (img.shape[0] // patch.shape[0] + 1, img.shape[1] // patch.shape[1] + 1, 1)
            tile = np.tile(patch, reps)[: img.shape[0], : img.shape[1]]
            low = cv2.GaussianBlur(tile, (0, 0), 12)
            out = tile - low + base
    a = feather(mask, 16)[..., None]
    res = img.astype(np.float32) * a + out * (1 - a)
    return np.clip(res, 0, 255).astype(np.uint8), mask


def robust_color(px, fallback):
    if len(px) < 50:
        return fallback
    px = px.astype(np.float32)
    l = px @ np.array([0.299, 0.587, 0.114], np.float32)
    lo, hi = np.quantile(l, [0.35, 0.9])
    sel = (l >= lo) & (l <= hi)
    return px[sel].mean(0) if sel.sum() > 10 else px.mean(0)


def build_shell(room_id, P, N, cams: list[Cam], out: Path, ceiling=None):
    cc = np.array([c.C for c in cams if not c.outside] or [c.C for c in cams])
    (xr, xs), (zr, zs) = wall_extents(P, N, 0, cc), wall_extents(P, N, 2, cc)
    inside = (P[:, 0] > xr[0]) & (P[:, 0] < xr[1]) & (P[:, 2] > zr[0]) & (P[:, 2] < zr[1])
    ceiling = ceiling_height(P[inside], N[inside], float(np.median(cc[:, 1])))
    x0, x1 = xr
    z0, z1 = zr
    h = ceiling
    out.mkdir(parents=True, exist_ok=True)
    surfaces = []
    specs = [
        ("floor", np.array([x0, 0, z1]), np.array([1, 0, 0]), np.array([0, 0, -1]), x1 - x0, z1 - z0, np.array([0, 1, 0]), True),
        ("wall-n", np.array([x0, h, z0]), np.array([1, 0, 0]), np.array([0, -1, 0]), x1 - x0, h, np.array([0, 0, 1]), zs[0]),
        ("wall-s", np.array([x1, h, z1]), np.array([-1, 0, 0]), np.array([0, -1, 0]), x1 - x0, h, np.array([0, 0, -1]), zs[1]),
        ("wall-w", np.array([x0, h, z1]), np.array([0, 0, -1]), np.array([0, -1, 0]), z1 - z0, h, np.array([1, 0, 0]), xs[0]),
        ("wall-e", np.array([x1, h, z0]), np.array([0, 0, 1]), np.array([0, -1, 0]), z1 - z0, h, np.array([-1, 0, 0]), xs[1]),
    ]
    raw = {}
    for name, o, du, dv, w, hh, n, seen_wall in specs:
        raw[name] = bake_plane(o, du, dv, w, hh, n, cams)
    wall_px = np.concatenate([raw[k][0][raw[k][1]] for k in raw if k != "floor"] or [np.zeros((0, 3))])
    wall_ref = robust_color(wall_px, DEFAULT_WALL)
    fl = raw["floor"]
    floor_ref = robust_color(fl[0][fl[1]], DEFAULT_FLOOR)
    for name, o, du, dv, w, hh, n, seen_wall in specs:
        img, mask = raw[name]
        filled, clean = fill_surface(img, mask, "floor" if name == "floor" else "wall", floor_ref if name == "floor" else wall_ref)
        f = f"{room_id}-{name}.jpg"
        Image.fromarray(filled).save(out / f, "JPEG", quality=84, optimize=True)
        Image.fromarray((clean * 255).astype(np.uint8)).resize((max(8, clean.shape[1] // 4), max(8, clean.shape[0] // 4))).save(out / f"{room_id}-{name}-seen.png", optimize=True)
        surfaces.append({
            "name": name, "origin": o.round(4).tolist(), "u": (du * w).round(4).tolist(), "v": (dv * hh).round(4).tolist(),
            "texture": f"shell/{f}", "seen": f"shell/{room_id}-{name}-seen.png",
            "coverage": round(float(clean.mean()), 3), "detected": bool(seen_wall),
        })
    ceil_o = np.array([x0, h, z0])
    surfaces.append({
        "name": "ceiling", "origin": ceil_o.round(4).tolist(), "u": [round(x1 - x0, 4), 0, 0], "v": [0, 0, round(z1 - z0, 4)],
        "color": "#%02x%02x%02x" % tuple(int(c) for c in np.clip(wall_ref * 1.04, 0, 255)), "coverage": 0, "detected": False,
    })
    return {"x": [float(x0), float(x1)], "z": [float(z0), float(z1)], "height": h, "surfaces": surfaces,
            "wallColor": surfaces[-1]["color"], "_raw": raw}
