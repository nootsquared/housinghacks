"""Per-photo 3D object scenes, merged into one clean object mesh per room.

Stage 1 (`image_scene`): one photo + its metric depth -> structure / object split, object instances,
a photo-textured surface per instance and a solid body completed behind it (ray fill + gravity,
bounded by the space this photo sees as empty).

Stage 2 (`build_objects`): bodies from all photos are fused into one solid volume and carved by every
photo's free space; each piece of visible surface is owned by exactly one photo (best first, later photos
only add what is not covered yet), floaters and flat wall decor are dropped.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree

from shell import Cam

VOX = 0.04
BEHIND = 0.05
DEPTH_FILL = 0.6
MAX_TRIS = 240000
WALL_GAP = 0.1
OBJECT, STRUCTURE, OUTSIDE, INVALID = 1, 2, 3, 0
PLANES = ("floor", "ceil", "x0", "x1", "z0", "z1")
AXIS = {"floor": 1, "ceil": 1, "x0": 0, "x1": 0, "z0": 2, "z1": 2}
SIGN = {"floor": 1, "ceil": -1, "x0": 1, "x1": -1, "z0": 1, "z1": -1}


@dataclass
class Grid:
    lo: np.ndarray
    dims: tuple

    @classmethod
    def of(cls, box):
        (x0, x1), (z0, z1), h = box
        lo = np.array([x0, 0.0, z0])
        dims = tuple(int(v) for v in np.ceil(np.array([x1 - x0, h, z1 - z0]) / VOX).astype(int) + 1)
        return cls(lo, dims)

    def index(self, P):
        i = np.floor((P - self.lo) / VOX).astype(np.int64)
        ok = (i >= 0).all(-1) & (i < np.array(self.dims)).all(-1)
        return i, ok

    def flat(self, P):
        i, ok = self.index(P)
        f = np.ravel_multi_index(tuple(np.clip(i, 0, np.array(self.dims) - 1).T), self.dims)
        return np.where(ok, f, -1)

    def centers(self):
        g = [self.lo[a] + (np.arange(self.dims[a]) + 0.5) * VOX for a in range(3)]
        return np.stack(np.meshgrid(*g, indexing="ij"), -1).reshape(-1, 3)


@dataclass
class Instance:
    view: int
    mask: np.ndarray
    score: float
    points: np.ndarray
    body: np.ndarray = field(default=None)
    supported: bool = True


@dataclass
class ImageScene:
    cam: Cam
    planes: dict
    X: np.ndarray
    labels: np.ndarray
    instances: list
    slack: np.ndarray
    vdepth: np.ndarray
    rgb: np.ndarray
    cos: np.ndarray


def pixel_rays(c: Cam):
    ys, xs = np.mgrid[0:c.H, 0:c.W].astype(np.float64)
    ray = np.stack([(xs + 0.5 - c.K[0, 2]) / c.K[0, 0], -(ys + 0.5 - c.K[1, 2]) / c.K[1, 1], -np.ones_like(xs)], -1)
    return ray @ c.Rw.T


def project(c: Cam, X):
    Xc = (X - c.C) @ c.Rw
    z = -Xc[..., 2]
    zs = np.maximum(z, 1e-6)
    return c.K[0, 0] * Xc[..., 0] / zs + c.K[0, 2], -c.K[1, 1] * Xc[..., 1] / zs + c.K[1, 2], z


def depth_normals(X):
    dx = np.zeros_like(X)
    dy = np.zeros_like(X)
    dx[:, 1:-1] = X[:, 2:] - X[:, :-2]
    dy[1:-1] = X[2:] - X[:-2]
    n = np.cross(dx, dy)
    return n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9)


def signed(planes, X):
    return {p: SIGN[p] * (X[..., AXIS[p]] - planes[p]) for p in PLANES}


def nominal_planes(box):
    (x0, x1), (z0, z1), h = box
    return {"floor": 0.0, "ceil": float(h), "x0": x0, "x1": x1, "z0": z0, "z1": z1}


def refine_planes(planes, X, N, ok, search):
    """Snap each box plane to this photo's own evidence: the outermost strong axis-aligned plane near it."""
    out = dict(planes)
    for p in PLANES:
        a, v = AXIS[p], planes[p]
        reach = search if a != 1 else min(search, 0.2)
        sel = ok & (np.abs(N[..., a]) > 0.85) & (np.abs(X[..., a] - v) < reach)
        vals = X[..., a][sel]
        if len(vals) < 300:
            continue
        bins = np.arange(v - reach, v + reach + 0.01, 0.01)
        hist = np.convolve(np.histogram(vals, bins)[0], np.ones(5) / 5, "same")
        strong = np.flatnonzero((hist >= 0.2 * hist.max()) & (hist >= np.roll(hist, 1)) & (hist >= np.roll(hist, -1)))
        peak = bins[strong[np.argmin(SIGN[p] * bins[strong])]] + 0.005
        near = vals[np.abs(vals - peak) < 0.025]
        out[p] = float(near.mean())
    return out


def classify(c: Cam, box, structure):
    d = c.depth.astype(np.float64)
    X = c.C + pixel_rays(c) * d[..., None]
    N = depth_normals(X)
    gy, gx = np.gradient(d)
    edge = np.hypot(gx, gy) > 0.05 * d
    ok = (d > 0.1) & ~edge
    planes = refine_planes(nominal_planes(box), X, N, ok, 0.5 if structure else 0.8)
    s = signed(planes, X)
    tol = 0.035 + 0.012 * d
    near = np.zeros(d.shape, bool)
    for p in PLANES:
        aligned = np.abs(N[..., AXIS[p]]) > 0.6
        near |= (np.abs(s[p]) < tol) | ((np.abs(s[p]) < 2 * tol) & aligned)
    out = np.zeros(d.shape, bool)
    for p in PLANES:
        out |= s[p] < -tol
    labels = np.full(d.shape, INVALID, np.uint8)
    labels[ok & out] = OUTSIDE
    labels[ok & ~out & near] = STRUCTURE
    obj = ok & ~out & ~near
    obj &= ~inner_walls(X, N, obj)
    obj = cv2.morphologyEx(obj.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    labels[obj] = OBJECT
    return X, N, planes, labels


def inner_walls(X, N, obj):
    """Floor-to-ceiling planes the room box missed. Cabinets are tall but they stop at a counter, so they stay objects."""
    wall = np.zeros(obj.shape, bool)
    for a, b in ((0, 2), (2, 0)):
        facing = obj & (np.abs(N[..., a]) > 0.9)
        for v in plane_peaks(X[..., a][facing], 2000):
            band = obj & (np.abs(X[..., a] - v) < 0.04)
            if band.sum() < 3000:
                continue
            y = X[..., 1][band]
            w = X[..., b][band]
            y0, y1 = np.quantile(y, [0.03, 0.97])
            if y1 - y0 > 1.9 and y0 < 0.25 and y1 > 2.1 and np.quantile(w, 0.97) - np.quantile(w, 0.03) > 1.2:
                wall |= band & (np.abs(N[..., a]) > 0.5)
    return cv2.morphologyEx(wall.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0


def smooth_within(d, mask, sigma=1.2):
    m = mask.astype(np.float64)
    num = cv2.GaussianBlur(d * m, (0, 0), sigma)
    den = cv2.GaussianBlur(m, (0, 0), sigma)
    return np.where(mask, num / np.maximum(den, 1e-6), d)


def plane_peaks(vals, min_count):
    if len(vals) < min_count:
        return []
    lo, hi = vals.min(), vals.max()
    bins = np.arange(lo - 0.02, hi + 0.03, 0.01)
    hist = np.convolve(np.histogram(vals, bins)[0], np.ones(3), "same")
    peaks = []
    for i in np.argsort(-hist):
        if hist[i] < min_count:
            break
        v = bins[i] + 0.005
        if all(abs(v - q) > 0.06 for q in peaks):
            peaks.append(v)
    return [float(vals[np.abs(vals - v) < 0.02].mean()) for v in peaks]


def snap_planes(c: Cam, ray, X, N, mask):
    """Flatten axis-aligned planar parts of an instance; through-the-top glossy pixels land on the top."""
    X = X.copy()
    for a in (1, 0, 2):
        facing = mask & (np.abs(N[..., a]) > 0.9)
        for v in plane_peaks(X[..., a][facing], 250):
            band = mask & (np.abs(X[..., a] - v) < 0.035) & (np.abs(N[..., a]) > 0.6)
            if a == 1 and c.C[1] > v + 0.1:
                band |= through_top(ray, X, mask, v, band)
            t = (v - c.C[a]) / np.where(np.abs(ray[..., a]) > 1e-6, ray[..., a], 1e-6)
            band &= t > 0
            X[band] = c.C + ray[band] * t[band][:, None]
    return X


def through_top(ray, X, mask, h, top):
    pts = X[top]
    if len(pts) < 250:
        return np.zeros(mask.shape, bool)
    lo = pts[:, [0, 2]].min(0) - 0.1
    size = np.ceil((pts[:, [0, 2]].max(0) + 0.1 - lo) / VOX).astype(int) + 1
    occ = np.zeros(size, bool)
    ij = ((pts[:, [0, 2]] - lo) / VOX).astype(int)
    occ[ij[:, 0], ij[:, 1]] = True
    occ = ndimage.binary_fill_holes(ndimage.binary_closing(occ, np.ones((5, 5)), border_value=0))
    core = ndimage.binary_erosion(occ, np.ones((5, 5)))
    occ = ndimage.binary_erosion(occ, np.ones((3, 3)))
    own = ((X[..., [0, 2]] - lo) / VOX).astype(int)
    own_in = (own >= 0).all(-1) & (own < size).all(-1)
    own = np.clip(own, 0, size - 1)
    below = mask & (X[..., 1] < h - 0.04) & own_in & core[own[..., 0], own[..., 1]]
    t = (h - X[..., 1]) / np.where(np.abs(ray[..., 1]) > 1e-6, ray[..., 1], 1e-6)
    C = X - ray * t[..., None]
    hit = ((C[..., [0, 2]] - lo) / VOX).astype(int)
    inside = (hit >= 0).all(-1) & (hit < size).all(-1)
    hit = np.clip(hit, 0, size - 1)
    return below & inside & occ[hit[..., 0], hit[..., 1]]


def hidden_top(c: Cam, ray, X, labels, m, planes, depth=0.9):
    """A flat top the depth missed (glossy counters, vanities): the photo shows it above the object's top edge
    but the depth put it on the wall behind. Extend the top back to that wall over those pixels."""
    cols = np.flatnonzero(m.any(0))
    if len(cols) < 30:
        return None
    r0 = m[:, cols].argmax(0)
    edge = X[r0, cols]
    hist, bins = np.histogram(edge[:, 1], np.arange(0.3, 1.3, 0.02))
    if hist.max() < 30:
        return None
    h = float(np.median(edge[np.abs(edge[:, 1] - bins[hist.argmax()] - 0.01) < 0.03, 1]))
    flat = np.abs(edge[:, 1] - h) < 0.03
    if np.ptp(edge[flat][:, [0, 2]], 0).max() < 0.4 or c.C[1] < h + 0.12:
        return None
    under = np.clip(r0[flat] + 8, 0, m.shape[0] - 1)
    if np.median(np.abs(depth_normals(X)[under, cols[flat], 1])) > 0.5:
        return None
    lid = np.zeros(m.shape, bool)
    P = np.zeros(X.shape)
    dist = np.linalg.norm(X - c.C, axis=-1)
    walls = ("x0", "x1", "z0", "z1")
    for x, top in zip(cols[flat], r0[flat]):
        run = []
        for r in range(top - 1, -1, -1):
            if labels[r, x] == OBJECT or ray[r, x, 1] >= -1e-6:
                break
            t = (h - c.C[1]) / ray[r, x, 1]
            q = c.C + ray[r, x] * t
            gap = min(v[0] for p, v in signed(planes, q[None]).items() if p in walls)
            if gap < 0.03:
                break
            if dist[r, x] < t * np.linalg.norm(ray[r, x]) - 0.03 or np.hypot(*(q[[0, 2]] - X[top, x, [0, 2]])) > depth:
                run = []
                break
            run.append((r, q))
        else:
            run = []
        if len(run) >= 2:
            for r, q in run:
                lid[r, x] = True
                P[r, x] = q
    if lid.sum() < 150:
        return None
    return lid, P[lid]


def exit_distance(planes, X, r):
    s = signed(planes, X)
    t = np.full(X.shape[:-1], np.inf)
    for p in PLANES:
        rate = SIGN[p] * r[..., AXIS[p]]
        hit = np.where(rate < -1e-6, s[p] / np.maximum(-rate, 1e-6), np.inf)
        t = np.minimum(t, hit)
    return t


def view_slack(c: Cam, labels, grid: Grid, centers):
    """Per voxel: observed depth minus voxel depth in this view (positive = seen in front of the surface)."""
    dmin = np.where(c.depth > 0.1, c.depth, 0).astype(np.float32)
    px, py, z = project(c, centers)
    ix, iy = np.floor(px).astype(int), np.floor(py).astype(int)
    inb = (z > 0.1) & (ix >= 0) & (ix < c.W) & (iy >= 0) & (iy < c.H)
    slack = np.full(len(centers), np.nan, np.float32)
    dd = dmin[iy[inb], ix[inb]]
    slack[inb] = np.where(dd > 0, dd - z[inb], np.nan)
    depth = np.where(inb, z, 0).astype(np.float32)
    return slack.reshape(grid.dims), depth.reshape(grid.dims)


def free_mask(slack, depth, base, rel):
    with np.errstate(invalid="ignore"):
        return slack > base + rel * depth


def core_bounds(P):
    """Drop the flying pixels that turn one table into a cloud of shards."""
    if len(P) < 40:
        lo, hi = (P.min(0), P.max(0)) if len(P) else (np.zeros(3), np.zeros(3))
        return P, lo, hi
    tight_lo, tight_hi = np.quantile(P, [0.1, 0.9], 0)
    lo, hi = np.quantile(P, [0.04, 0.96], 0)
    if np.any((hi - lo) > (tight_hi - tight_lo) * 1.7 + 0.12):
        lo, hi = tight_lo, tight_hi
    keep = np.all((P >= lo - 0.05) & (P <= hi + 0.05), 1)
    if keep.mean() < 0.35:
        lo, hi = np.quantile(P, [0.04, 0.96], 0)
        return P, lo, hi
    return P[keep], lo, hi


def spatial_parts(X, mask, cell=0.06):
    """Pixels that touch in the photo can be different objects once depth puts space between them."""
    idx = np.flatnonzero(mask)
    P = X.reshape(-1, 3)[idx]
    if len(P) < 160:
        return [mask]
    ijk = np.floor((P - P.min(0)) / cell).astype(np.int32)
    dims = tuple(int(v) for v in ijk.max(0) + 1)
    if int(np.prod(dims)) > 2_000_000:
        return [mask]
    occ = np.zeros(dims, np.uint8)
    lin = np.ravel_multi_index((ijk[:, 0], ijk[:, 1], ijk[:, 2]), dims)
    occ.ravel()[np.unique(lin)] = 1
    lab, n = ndimage.label(occ)
    ids = lab.ravel()[lin]
    out = []
    for k, count in enumerate(np.bincount(ids)):
        if k == 0 or count < 80:
            continue
        part = np.zeros(mask.size, bool)
        part[idx[ids == k]] = True
        out.append(part.reshape(mask.shape))
    return out or [mask]


def extrude(P, direction, distance):
    if distance <= VOX:
        return np.concatenate([P, P + direction * max(distance, 0.0)])
    steps = np.arange(0.0, distance + 1e-9, VOX * 0.75)
    return np.concatenate([P + direction * s for s in steps])


def away_vector(P, cam: Cam):
    v = np.median(P, 0) - cam.C
    v[1] = 0.0
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])


def footprint_solid(P, y0, y1, grid: Grid, dilate=1):
    """A column under the pixels we actually saw, so the object has a body instead of a paper silhouette."""
    if len(P) == 0 or y1 < y0:
        return np.array([], np.int64)
    i, ok = grid.index(P)
    if not ok.any():
        return np.array([], np.int64)
    occ = np.zeros((grid.dims[0], grid.dims[2]), bool)
    occ[i[ok, 0], i[ok, 2]] = True
    occ = ndimage.binary_closing(occ, np.ones((3, 3)))
    if dilate:
        occ = ndimage.binary_dilation(occ, iterations=dilate)
    y0i = int(np.clip(np.floor((y0 - grid.lo[1]) / VOX), 0, grid.dims[1] - 1))
    y1i = int(np.clip(np.floor((y1 - grid.lo[1]) / VOX), 0, grid.dims[1] - 1))
    if y1i < y0i:
        return np.array([], np.int64)
    solid = np.zeros(grid.dims, bool)
    solid[:, y0i:y1i + 1, :] = occ[:, None, :]
    return np.flatnonzero(solid)


def object_body(P, cam: Cam, planes, grid: Grid):
    """Solid that matches the photo: a thin panel for a tv, a slab for a tabletop, a prism with real depth for a sofa."""
    P, lo, hi = core_bounds(P)
    if len(P) < 40:
        return np.array([], np.int64)
    span = np.maximum(hi - lo, 0.02)
    if float(span[0] * span[2]) > 3.2:
        return np.array([], np.int64)
    bottom, top = float(lo[1]), float(min(hi[1], planes["ceil"] - 0.02))
    tall = top - bottom
    wide = float(max(span[0], span[2]))
    thin = float(min(span[0], span[2]))
    if thin < 0.16 and 0.4 < wide < 2.4 and 0.25 < tall < 1.15 and bottom > 0.7:
        return footprint_solid(extrude(P, away_vector(P, cam), 0.1), bottom - 0.02, top + 0.02, grid, dilate=0)
    if tall < 0.28 and bottom > 0.32 and wide > 0.35:
        h = float(np.median(P[:, 1]))
        return footprint_solid(P, h - 0.12, h + 0.015, grid)
    # A flat top with stuff under it. A wide front is a cabinet; a few columns are table legs.
    if 0.3 < tall < 1.15 and wide > 0.4:
        tops = P[P[:, 1] > bottom + 0.62 * tall]
        if len(tops) > 80 and len(tops) > 0.4 * len(P) and float(np.quantile(tops[:, 1], 0.9) - np.quantile(tops[:, 1], 0.1)) < 0.14:
            h = float(np.median(tops[:, 1]))
            slab = footprint_solid(tops, h - 0.1, h + 0.02, grid)
            legs = P[P[:, 1] < h - 0.14]
            leg = np.array([], np.int64)
            if len(legs) > 30:
                span = np.quantile(legs[:, [0, 2]], 0.95, 0) - np.quantile(legs[:, [0, 2]], 0.05, 0)
                if float(max(span)) > 0.65 * wide:
                    legs = extrude(legs, away_vector(legs, cam), 0.5)
                    leg = footprint_solid(legs, 0.0, h - 0.04, grid, dilate=1)
                else:
                    leg = footprint_solid(legs, 0.0, h - 0.08, grid, dilate=0)
            return np.unique(np.concatenate([slab, leg])) if len(leg) else slab
    away = away_vector(P, cam)
    axis = 0 if abs(away[0]) >= abs(away[2]) else 2
    target = 0.82 if top < 1.25 and wide > 1.05 else (0.55 if wide > 0.65 else 0.36)
    extra = target - float(span[axis])
    Q = extrude(P, away, extra) if extra > 0.05 else P
    y0 = 0.0 if bottom < 0.55 else bottom
    return footprint_solid(Q, y0, top, grid)


def fill_body(inst_pts, inst_rays, inst_exit, grid: Grid):
    r = inst_rays / np.linalg.norm(inst_rays, axis=-1, keepdims=True)
    T = np.minimum(inst_exit - 0.02, DEPTH_FILL)
    steps = np.arange(BEHIND, DEPTH_FILL + 1e-6, VOX * 0.5)
    P = inst_pts[:, None] + r[:, None] * steps[None, :, None]
    keep = steps[None] < T[:, None]
    f = grid.flat(P[keep])
    return np.unique(f[f >= 0])


def architecture(planes, P):
    """Wall trim and frames. A television is flat too, but it is a compact plaque up on the wall, so it stays."""
    s = signed(planes, P)
    wall = np.min([s[p] for p in ("x0", "x1", "z0", "z1")], 0)
    reach = float(np.quantile(wall, 0.95))
    tall = float(np.quantile(P[:, 1], 0.98) - np.quantile(P[:, 1], 0.02))
    wide = float(max(np.ptp(np.quantile(P[:, 0], [0.05, 0.95])), np.ptp(np.quantile(P[:, 2], [0.05, 0.95]))))
    bottom = float(np.quantile(P[:, 1], 0.05))
    mounted = 0.4 < wide < 2.3 and 0.28 < tall < 1.2 and bottom > 0.65
    if reach < 0.1 and not mounted:
        return True
    if (reach < 0.22 and tall > 1.75 and wide > 1.8) or (reach < 0.14 and np.quantile(s["ceil"], 0.05) < 0.08 and tall > 1.1):
        return True
    Q = P[:: max(1, len(P) // 4000)]
    c = Q.mean(0)
    _, sv, vt = np.linalg.svd(Q - c, full_matrices=False)
    n = vt[-1]
    rms = sv[-1] / np.sqrt(len(Q))
    span = np.ptp((Q - c) @ vt[0]) if abs(vt[0][1]) < 0.5 else np.ptp((Q - c) @ vt[1])
    return abs(n[1]) < 0.15 and rms < 0.035 and tall > 1.6 and span > 1.3 and not mounted


def gravity(L, free, max_drop=2):
    """Close a crack under an object. A long drop is how a tabletop becomes a solid column."""
    orig = L.copy()
    ny = L.shape[1]
    for dy in range(1, max_drop + 1):
        above = np.zeros_like(L)
        above[:, : ny - dy] = orig[:, dy:]
        hole = (L == 0) & (above > 0) & ~free
        L[hole] = above[hole]
    return L


def image_scene(c: Cam, box, structure=True, view=0, grid: Grid | None = None, centers=None):
    grid = grid or Grid.of(box)
    centers = centers if centers is not None else grid.centers()
    X, N, planes, labels = classify(c, box, structure)
    ray = pixel_rays(c)
    obj = labels == OBJECT
    count, comp, stats, _ = cv2.connectedComponentsWithStats(obj.astype(np.uint8), connectivity=8)
    slack, vdepth = view_slack(c, labels, grid, centers)
    free = free_mask(slack, vdepth, 0.04, 0.02)
    rgb = np.asarray(Image.fromarray(c.image).resize((c.W, c.H), Image.BILINEAR))
    L = np.zeros(grid.dims, np.int32)
    cosimg = np.zeros(labels.shape)
    ys, xs = np.mgrid[0:c.H, 0:c.W]
    edge_dist = np.minimum(np.minimum(xs, c.W - 1 - xs) / c.W, np.minimum(ys, c.H - 1 - ys) / c.H)
    border = np.clip(edge_dist / 0.06, 0.3, 1.0)
    instances = []
    for k in range(1, count):
        if stats[k, cv2.CC_STAT_AREA] < 150:
            labels[comp == k] = INVALID
            continue
        m = comp == k
        if architecture(planes, X[m]):
            labels[m] = STRUCTURE
            continue
        d = smooth_within(c.depth.astype(np.float64), m)
        Xk = c.C + ray * d[..., None]
        Nk = depth_normals(np.where(m[..., None], Xk, X))
        Xk = snap_planes(c, ray, Xk, Nk, m)
        X[m] = Xk[m]
        parts = spatial_parts(X, m)
        if len(parts) != 1:
            labels[m] = INVALID
        for m in parts:
            labels[m] = OBJECT
            Pm = X[m]
            _, lo, hi = core_bounds(Pm)
            inside = np.all((Pm >= lo - 0.08) & (Pm <= hi + 0.08), 1)
            if 0.45 < inside.mean() < 0.995:
                trimmed = np.zeros_like(m)
                trimmed[m] = inside
                if trimmed.sum() > 120:
                    labels[m] = INVALID
                    m = trimmed
                    labels[m] = OBJECT
            lid = hidden_top(c, ray, X, labels, m, planes)
            if lid is not None:
                m = m | lid[0]
                X[lid[0]] = lid[1]
                labels[lid[0]] = OBJECT
            rn = ray[m] / np.linalg.norm(ray[m], axis=-1, keepdims=True)
            cos = np.abs((depth_normals(X)[m] * rn).sum(-1))
            cosimg[m] = cos
            score = float((cos * border[m] / np.maximum(d[m], 1.0) ** 2).sum() / np.sqrt(m.sum())) * (0.25 if c.outside else 1.0)
            inst = Instance(view, m, score, X[m])
            vox = object_body(X[m], c, planes, grid)
            flat = L.reshape(-1)
            vox = vox[flat[vox] == 0]
            flat[vox] = len(instances) + 1
            instances.append(inst)
    L = gravity(L, free)
    L[free_mask(slack, vdepth, -0.03, 0.0)] = 0
    for i, inst in enumerate(instances):
        inst.body = np.flatnonzero(L.reshape(-1) == i + 1)
        inst.supported = supported(planes, np.concatenate([inst.points, centers[inst.body]]))
    cosimg = smooth_within(cosimg, labels == OBJECT, 3.0)
    return ImageScene(c, planes, X, labels, instances, slack, vdepth, rgb, cosimg)


def surface_mesh(c: Cam, X, mask, stride=1):
    m = mask[::stride, ::stride]
    P = X[::stride, ::stride]
    q = m[:-1, :-1] & m[1:, :-1] & m[:-1, 1:] & m[1:, 1:]
    y, x = np.nonzero(q)
    corners = [(y, x), (y, x + 1), (y + 1, x), (y + 1, x + 1)]
    tris = [(0, 2, 1), (1, 2, 3)]
    dist = np.linalg.norm(P - c.C, axis=-1)
    pos, uv = [], []
    for t in tris:
        vy = np.stack([corners[i][0] for i in t], 1)
        vx = np.stack([corners[i][1] for i in t], 1)
        dd = dist[vy, vx]
        tri = P[vy, vx]
        edges = np.maximum(
            np.linalg.norm(tri[:, 0] - tri[:, 1], axis=-1),
            np.maximum(np.linalg.norm(tri[:, 1] - tri[:, 2], axis=-1), np.linalg.norm(tri[:, 2] - tri[:, 0], axis=-1)),
        )
        ok = (dd.max(1) - dd.min(1) < 0.02 + 0.03 * dd.min(1) * stride) & (edges < 0.07 * max(stride, 1))
        vy, vx = vy[ok], vx[ok]
        pos.append(P[vy, vx])
        uv.append(np.stack([(vx * stride + 0.5 * stride) / c.W, 1 - (vy * stride + 0.5 * stride) / c.H], -1))
    return np.concatenate(pos).astype(np.float32), np.concatenate(uv).astype(np.float32)


def supported(planes, P):
    """Stands on the floor, projects from a wall, or is a flat screen hung on one. Anything else floats."""
    if len(P) == 0:
        return False
    s = signed(planes, P)
    if np.quantile(s["floor"], 0.02) < 0.16:
        return True
    tall = float(np.quantile(P[:, 1], 0.95) - np.quantile(P[:, 1], 0.05))
    wide = float(max(np.ptp(np.quantile(P[:, 0], [0.05, 0.95])), np.ptp(np.quantile(P[:, 2], [0.05, 0.95]))))
    for p in ("x0", "x1", "z0", "z1"):
        if s[p].min() < WALL_GAP and np.quantile(s[p], 0.9) > 0.12:
            return True
        if s[p].min() < 0.08 and np.quantile(s[p], 0.85) < 0.18 and tall > 0.28 and wide > 0.4:
            return True
    return False


def body_meshes(solid, grid: Grid, max_tris=4000):
    import open3d as o3d
    from skimage.measure import marching_cubes

    comp, _ = ndimage.label(solid)
    out = []
    for k, sl in enumerate(ndimage.find_objects(comp), 1):
        if sl is None:
            continue
        sub = comp[sl] == k
        if sub.sum() < 20:
            continue
        vol = np.pad(sub.astype(np.float32), 2)
        vol = ndimage.gaussian_filter(vol, 0.55)
        if vol.max() < 0.5:
            continue
        V, F, _, _ = marching_cubes(vol, 0.5)
        V = grid.lo + (V - 2 + np.array([s.start for s in sl]) + 0.5) * VOX
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F[:, ::-1].copy()))
        mesh = mesh.filter_smooth_taubin(2)
        if len(mesh.triangles) > max_tris:
            mesh = mesh.simplify_quadric_decimation(max_tris)
        V, F = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        V[:, 1] = np.maximum(V[:, 1], 0.0)
        out.append((V, F))
    return out


def coloured_bodies(solid, grid: Grid, ref_pts, ref_col):
    """One colour per object, taken from its own photo, so a sofa side stays fabric instead of a blur."""
    tree = cKDTree(ref_pts)
    for V, F in body_meshes(solid, grid):
        dist, idx = tree.query(V, k=1, workers=-1)
        near = dist < 0.08
        src = ref_col[idx[near]] if near.sum() > 12 else ref_col[idx]
        C = np.broadcast_to(np.median(src, 0), V.shape).copy()
        yield V[F], shade(V, F, C)


def shade(V, F, colors):
    tri = V[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    f = 0.9 + 0.1 * np.abs(n[:, 1])
    return np.clip(colors[F] * f[:, None, None], 0, 255)


def merge(scenes: list[ImageScene], box, structure=True):
    grid = Grid.of(box)
    centers = grid.centers()
    solid = np.zeros(grid.dims, bool)
    free = np.zeros(grid.dims, bool)
    for s in scenes:
        free |= free_mask(s.slack, s.vdepth, 0.03, 0.01)
        for inst in s.instances:
            if inst.supported:
                solid.reshape(-1)[inst.body] = True
    solid &= ~free
    solid = ndimage.binary_opening(solid, np.ones((2, 2, 2)))
    comp, n = ndimage.label(solid)
    sizes = np.bincount(comp.ravel(), minlength=n + 1)
    solid = (sizes >= 12)[comp] & (comp > 0)
    order = sorted(((inst.score, -i, i, j) for i, s in enumerate(scenes) for j, inst in enumerate(s.instances)
                    if inst.supported), reverse=True)
    claimed = []
    pieces = []
    emitted = [np.zeros(sc.labels.shape, bool) for sc in scenes]
    for weak in (False, True):
        for _, _, i, j in order:
            s, inst = scenes[i], scenes[i].instances[j]
            base = inst.mask & ~emitted[i]
            m = base & ((s.cos < 0.2) if weak else (s.cos >= 0.2))
            if not weak:
                grow = cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
                m |= base & grow
            m = uncovered(m, s.X, [p for v, p in claimed if v != i])
            m &= ~occludes(s.X, m, scenes, emitted, i)
            m = prune(m, base & ~m, 400 if weak else 200)
            if not m.any():
                continue
            pieces.append((i, m))
            emitted[i] |= m
            claimed.append((i, s.X[m][::3]))
    if structure is False:
        for i, s in enumerate(scenes):
            m = clean_mask(uncovered(s.labels == STRUCTURE, s.X, [p for v, p in claimed if v != i]), 400)
            if m.any():
                pieces.append((i, m))
    for i in sorted({i for i, _ in pieces}):
        m = np.any([pm for j, pm in pieces if j == i], 0)
        solid &= ~pull_back(scenes[i].cam, m, grid, centers)
    return grid, solid, pieces


def pull_back(c: Cam, mask, grid: Grid, centers, gap=0.04):
    """Voxels just behind an emitted surface: the body stays clear of it so the photo texture always wins."""
    m = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    d = cv2.dilate(np.where(mask, c.depth, 0).astype(np.float32), np.ones((5, 5), np.uint8))
    px, py, z = project(c, centers)
    ix, iy = np.floor(px).astype(int), np.floor(py).astype(int)
    inb = (z > 0.1) & (ix >= 0) & (ix < c.W) & (iy >= 0) & (iy < c.H)
    out = np.zeros(len(centers), bool)
    ii = np.flatnonzero(inb)
    hit = m[iy[ii], ix[ii]]
    ii = ii[hit]
    out[ii] = z[ii] < d[iy[ii], ix[ii]] + gap
    return out.reshape(grid.dims)


def occludes(X, mask, scenes, emitted, own):
    """Pixels that would sit in front of a surface another photo already owns, as seen from that photo."""
    out = np.zeros(mask.shape, bool)
    if not mask.any():
        return out
    P = X[mask]
    bad = np.zeros(len(P), bool)
    for i, s in enumerate(scenes):
        if i == own or not emitted[i].any():
            continue
        c = s.cam
        E = emitted[i]
        px, py, z = project(c, P)
        ix, iy = np.floor(px).astype(int), np.floor(py).astype(int)
        inb = (z > 0.1) & (ix >= 0) & (ix < c.W) & (iy >= 0) & (iy < c.H)
        k = np.flatnonzero(inb)
        k = k[E[iy[k], ix[k]]]
        bad[k] |= z[k] < c.depth[iy[k], ix[k]] - (0.04 + 0.02 * z[k])
    out[mask] = bad
    return out


def uncovered(mask, X, claimed, radius=0.2):
    m = mask.copy()
    if claimed and m.any():
        dist, _ = cKDTree(np.concatenate(claimed)).query(X[m], k=1, distance_upper_bound=radius, workers=-1)
        m[m] = ~np.isfinite(dist)
    return m


def prune(m, removed, min_area):
    """Drop slivers left along cuts, keep thin structures that the photo itself shows."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    opened = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, k) > 0
    near_cut = cv2.dilate(removed.astype(np.uint8), k) > 0
    m = opened | (m & ~near_cut)
    count, comp, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    good = np.r_[False, stats[1:, cv2.CC_STAT_AREA] >= min_area]
    return good[comp]


def clean_mask(m, min_area):
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    count, comp, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    good = np.r_[False, stats[1:, cv2.CC_STAT_AREA] >= min_area]
    return good[comp]


def drop_chips(pos, uv, min_count=80):
    """Flying fragments of a table or cabinet are their own little meshes. Keep the real object."""
    if len(pos) < min_count:
        return pos, uv
    cen = pos.mean(1)
    ijk = np.floor((cen - cen.min(0)) / 0.07).astype(np.int32)
    dims = tuple(int(v) for v in ijk.max(0) + 1)
    if int(np.prod(dims)) > 4_000_000:
        return pos, uv
    occ = np.zeros(dims, bool)
    occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    lab, n = ndimage.label(occ)
    tri = lab[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    counts = np.bincount(tri, minlength=n + 1)
    keep = counts[tri] >= min_count
    if not keep.any():
        keep = tri == np.argmax(counts)
    return pos[keep], uv[keep]


def keep_dense(pos, uv, cell=0.06, min_per_cell=4):
    """A real surface packs many triangles into each patch. Scattered chips do not."""
    if len(pos) < 40:
        return pos, uv
    cen = pos.mean(1)
    ijk = np.floor((cen - cen.min(0)) / cell).astype(np.int32)
    dims = tuple(int(v) for v in ijk.max(0) + 1)
    if int(np.prod(dims)) > 4_000_000:
        return pos, uv
    counts = np.zeros(dims, np.int32)
    np.add.at(counts, (ijk[:, 0], ijk[:, 1], ijk[:, 2]), 1)
    dense = ndimage.binary_dilation(counts >= min_per_cell, iterations=1)
    keep = dense[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    if keep.sum() < 40:
        return pos[:0], uv[:0]
    return pos[keep], uv[keep]


def tidy_component(pos, uv):
    """A real cabinet face or tabletop is almost flat. A crumpled cloud is the broken table, so it goes."""
    ext = pos.reshape(-1, 3)
    c = np.median(ext, 0)
    sample = ext[:: max(1, len(ext) // 1500)]
    _, _, vt = np.linalg.svd(sample - c, full_matrices=False)
    nrm = vt[-1]
    dist = (pos - c) @ nrm
    p90 = float(np.quantile(np.abs(dist), 0.9))
    span = np.quantile(ext, 0.95, 0) - np.quantile(ext, 0.05, 0)
    # A small crumpled cloud is a broken chair or table fragment. A whole cabinet run stays.
    if p90 > 0.1 and float(max(span)) < 1.25:
        return None
    if p90 < 0.08 and float(np.sqrt(np.mean(dist ** 2))) > 0.015:
        pos = pos - dist[..., None] * nrm
    return pos, uv


def clean_surface(pos, uv):
    if len(pos) == 0:
        return pos, uv
    pos, uv = keep_dense(pos, uv)
    if len(pos) < 30:
        return pos[:0], uv[:0]
    cen = pos.mean(1)
    ijk = np.floor((cen - cen.min(0)) / 0.05).astype(np.int32)
    dims = tuple(int(v) for v in ijk.max(0) + 1)
    if int(np.prod(dims)) > 4_000_000:
        return drop_chips(pos, uv)
    occ = np.zeros(dims, bool)
    occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    lab, n = ndimage.label(occ)
    tri = lab[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    kept_p, kept_u = [], []
    for k in range(1, n + 1):
        m = tri == k
        if m.sum() < 60:
            continue
        piece = tidy_component(pos[m], uv[m])
        if piece is not None:
            kept_p.append(piece[0])
            kept_u.append(piece[1])
    if not kept_p:
        return pos[:0], uv[:0]
    return np.concatenate(kept_p), np.concatenate(kept_u)


def uv_color(rgb, uv):
    h, w = rgb.shape[:2]
    x = np.clip((uv[..., 0] * (w - 1)).astype(int), 0, w - 1)
    y = np.clip(((1 - uv[..., 1]) * (h - 1)).astype(int), 0, h - 1)
    return rgb[y, x].reshape(-1, 3).astype(np.float32)


def assemble(scenes, grid, solid, pieces, max_tris=MAX_TRIS):
    total = sum(int(m.sum()) * 2 for _, m in pieces)
    stride = 1 if total < max_tris * 0.8 else 2
    pos, uv, col, views = [], [], [], []
    ref_pts, ref_col = [], []
    for i, m in pieces:
        s = scenes[i]
        raw_p, raw_u = surface_mesh(s.cam, s.X, m, stride)
        p, u = clean_surface(raw_p, raw_u)
        if len(p):
            pos.append(p)
            uv.append(u)
            col.append(np.zeros((len(p), 3, 3), np.uint8))
            views.append(np.full(len(p), i))
            ref_pts.append(p.reshape(-1, 3))
            ref_col.append(uv_color(s.rgb, u))
        elif len(raw_p):
            # The photo surface was too broken to show, but its colour still belongs on the solid.
            ref_pts.append(np.median(raw_p.reshape(-1, 3), 0, keepdims=True))
            ref_col.append(np.median(uv_color(s.rgb, raw_u), 0, keepdims=True))
    if ref_pts:
        for tri, colors in coloured_bodies(solid, grid, np.concatenate(ref_pts), np.concatenate(ref_col)):
            pos.append(tri.astype(np.float32))
            uv.append(np.zeros((len(tri), 3, 2), np.float32))
            col.append(colors.astype(np.uint8))
            views.append(np.full(len(tri), -1))
    if not pos:
        return None
    pos, uv, col, views = map(np.concatenate, (pos, uv, col, views))
    if len(pos) == 0:
        return None
    order = np.argsort(np.where(views < 0, len(scenes), views), kind="stable")
    pos, uv, col, views = pos[order], uv[order], col[order], views[order]
    segments = []
    for i in list(range(len(scenes))) + [-1]:
        idx = np.flatnonzero(views == i)
        if len(idx):
            segments.append({"view": scenes[i].cam.name if i >= 0 else None, "start": int(idx[0]), "count": int(len(idx))})
    return {"pos": pos, "uv": uv, "col": col, "segments": segments}


def build_objects(cams: list[Cam], box, structure=True, inspect: Path | None = None):
    grid = Grid.of(box)
    centers = grid.centers()
    scenes = [image_scene(c, box, structure, i, grid, centers) for i, c in enumerate(cams)]
    inspect = inspect or (Path(os.environ["OBJECTS_INSPECT"]) if os.environ.get("OBJECTS_INSPECT") else None)
    if inspect:
        for s in scenes:
            write_inspection(s, box, inspect / f"{s.cam.name}.jpg")
    grid, solid, pieces = merge(scenes, box, structure)
    return assemble(scenes, grid, solid, pieces)


def bary(m):
    i, j = np.mgrid[0:m + 1, 0:m + 1]
    keep = i + j <= m
    a, b = i[keep] / m, j[keep] / m
    return np.stack([1 - a - b, a, b], 1)


def render(meshes, Rw, C, K, W, H, bg=(28, 28, 32)):
    zbuf_idx, zs, cols = [], [], []
    for pos, uv, col, tex in meshes:
        if len(pos) == 0:
            continue
        Xc = (pos - C) @ Rw
        z = -Xc[..., 2]
        ok = (z > 0.05).all(1)
        Xc, z = Xc[ok], z[ok]
        px = K[0, 0] * Xc[..., 0] / z + K[0, 2]
        py = -K[1, 1] * Xc[..., 1] / z + K[1, 2]
        inb = ((px > -50) & (px < W + 50) & (py > -50) & (py < H + 50)).all(1)
        e1 = np.stack([px[:, 1] - px[:, 0], py[:, 1] - py[:, 0]], 1)
        e2 = np.stack([px[:, 2] - px[:, 0], py[:, 2] - py[:, 0]], 1)
        edge = np.maximum(np.linalg.norm(e1, axis=1), np.linalg.norm(e2, axis=1))
        level = np.clip(np.ceil(edge * 2.2), 1, 40).astype(int)
        idx_all = np.flatnonzero(ok)
        for m in np.unique(level[inb]):
            sel = np.flatnonzero(inb & (level == m))
            B = bary(int(m))
            sx = (px[sel] @ B.T).ravel()
            sy = (py[sel] @ B.T).ravel()
            sz = (z[sel] @ B.T).ravel()
            if tex is not None:
                u = (uv[idx_all[sel], :, 0] @ B.T).ravel()
                v = (uv[idx_all[sel], :, 1] @ B.T).ravel()
                th, tw = tex.shape[:2]
                c = tex[np.clip(((1 - v) * th).astype(int), 0, th - 1), np.clip((u * tw).astype(int), 0, tw - 1)]
            else:
                c = np.einsum("tvc,sv->tsc", col[idx_all[sel]].astype(np.float32), B).reshape(-1, 3)
            ix, iy = sx.astype(int), sy.astype(int)
            keep = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
            zbuf_idx.append(iy[keep] * W + ix[keep])
            zs.append(sz[keep])
            cols.append(np.asarray(c, np.float32)[keep])
    img = np.tile(np.array(bg, np.uint8), (H * W, 1))
    if zbuf_idx:
        idx, z, c = np.concatenate(zbuf_idx), np.concatenate(zs), np.concatenate(cols)
        order = np.argsort(-z, kind="stable")
        img[idx[order]] = c[order].clip(0, 255).astype(np.uint8)
    return img.reshape(H, W, 3)


def look_at(eye, target):
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, [0.0, 1.0, 0.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return np.stack([r, u, -f], 1)


def scene_meshes(s: ImageScene):
    meshes = []
    obj = np.zeros(s.labels.shape, bool)
    for inst in s.instances:
        obj |= inst.mask
    for m in (s.labels == STRUCTURE, obj):
        p, u = surface_mesh(s.cam, s.X, m)
        meshes.append((p, u, None, s.cam.image))
    return meshes


def body_meshes_for(s: ImageScene, grid: Grid):
    solid = np.zeros(grid.dims, bool)
    for inst in s.instances:
        solid.reshape(-1)[inst.body] = True
    obj = np.zeros(s.labels.shape, bool)
    for inst in s.instances:
        obj |= inst.mask
    if not obj.any():
        return []
    rc = s.rgb[obj].astype(np.float32)
    return [(tri, None, colors, None) for tri, colors in coloured_bodies(solid, grid, s.X[obj], rc)]


def write_inspection(s: ImageScene, box, path: Path, width=480):
    """photo | structure (grey) / outside (navy) / object instances | its 3D scene from the photo pose | novel pose"""
    c = s.cam
    H = int(round(width * c.H / c.W))
    k = width / c.W
    K = c.K.copy()
    K[:2] *= k
    photo = np.asarray(Image.fromarray(c.image).resize((width, H), Image.BILINEAR))
    palette = np.array(
        [[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48], [145, 30, 180], [70, 240, 240],
         [240, 50, 230], [210, 245, 60], [250, 190, 212], [0, 128, 128], [170, 110, 40]], np.uint8)
    mask = np.zeros((c.H, c.W, 3), np.uint8)
    mask[s.labels == STRUCTURE] = (120, 120, 120)
    mask[s.labels == OUTSIDE] = (20, 30, 70)
    for i, inst in enumerate(s.instances):
        mask[inst.mask] = palette[i % len(palette)] // (1 if inst.supported else 3)
    mask = (0.55 * np.asarray(Image.fromarray(mask).resize((width, H), Image.NEAREST)) + 0.45 * photo).astype(np.uint8)
    grid = Grid.of(box)
    meshes = scene_meshes(s) + body_meshes_for(s, grid)
    same = render(meshes, c.Rw, c.C, K, width, H)
    obj = np.zeros(s.labels.shape, bool)
    for inst in s.instances:
        obj |= inst.mask
    target = np.median(s.X[obj], 0) if obj.sum() > 100 else c.C + c.Rw @ np.array([0, -0.3, -2.0])
    off = c.C - target
    a = np.radians(40)
    rot = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    eye = target + rot @ off * 1.15 + np.array([0, 0.9, 0])
    novel = render(meshes, look_at(eye, target), eye, K, width, H)
    sheet = np.concatenate([photo, mask, same, novel], 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet).save(path, quality=85)
