from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

PX_PER_M = 140
HATCH = np.array([226, 224, 218], np.uint8)
HATCH_LINE = np.array([206, 203, 196], np.uint8)


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


def wall_extents(P, N, axis, lo_q=0.03, hi_q=0.97):
    other = 2 - axis
    sel = np.abs(N[:, axis]) > 0.85
    v = P[sel, axis] if sel.sum() > 60 else P[:, axis]
    fallback = np.quantile(P[:, axis], [lo_q, hi_q])
    if len(v) < 60:
        return fallback, (False, False)
    hist, edges = np.histogram(v, bins=max(10, int((v.max() - v.min()) / 0.04)))
    strong = np.where(hist > hist.max() * 0.22)[0]
    lo = (edges[strong[0]] + edges[strong[0] + 1]) / 2
    hi = (edges[strong[-1]] + edges[strong[-1] + 1]) / 2
    spread = np.quantile(P[:, axis], [0.01, 0.99])
    seen_lo = lo <= spread[0] + 0.35
    seen_hi = hi >= spread[1] - 0.35
    lo = lo if seen_lo else min(lo, fallback[0])
    hi = hi if seen_hi else max(hi, fallback[1])
    _ = other
    return np.array([lo, hi]), (bool(seen_lo), bool(seen_hi))


def bake_plane(origin, du, dv, w, h, normal, cams: list[Cam], tol=0.06):
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
        ok &= np.abs(d - z) < tol * z + 0.04
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
    ys, xs = np.mgrid[0:th, 0:tw]
    hatch = np.where(((xs + ys) // 6) % 4 == 0, 1, 0)[..., None]
    fill = np.where(hatch, HATCH_LINE, HATCH)
    img = np.where(mask[..., None], img, fill).astype(np.uint8)
    return img, float(mask.mean())


def build_shell(room_id, P, N, cams: list[Cam], out: Path, ceiling=2.5):
    (xr, xs), (zr, zs) = wall_extents(P, N, 0), wall_extents(P, N, 2)
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
    for name, o, du, dv, w, hh, n, seen_wall in specs:
        img, cov = bake_plane(o, du, dv, w, hh, n, cams)
        f = f"{room_id}-{name}.jpg"
        Image.fromarray(img).save(out / f, "JPEG", quality=82, optimize=True)
        surfaces.append({
            "name": name, "origin": o.round(4).tolist(), "u": (du * w).round(4).tolist(), "v": (dv * hh).round(4).tolist(),
            "texture": f"shell/{f}", "coverage": round(cov, 3), "detected": bool(seen_wall),
        })
    return {"x": [float(x0), float(x1)], "z": [float(z0), float(z1)], "height": h, "surfaces": surfaces}
