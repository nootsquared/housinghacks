import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage import color as skcolor

PX_PER_M = 200
TILE_PX = 1024
MIN_FLOOR_M2 = 0.12
MIN_WALL_M2 = 0.08
RELIABLE_EVIDENCE = 0.6
WALL_PLAUSIBLE_L = 62.0


def _rgb(img):
    if isinstance(img, (str, Path)):
        img = Image.open(img)
    if isinstance(img, Image.Image):
        img = np.asarray(img.convert("RGB"))
    img = np.asarray(img)
    return img.astype(np.float32) / 255.0 if img.dtype == np.uint8 else img.astype(np.float32)


def _mask(mask, shape, erode=2):
    if isinstance(mask, (str, Path)):
        mask = Image.open(mask)
    if isinstance(mask, Image.Image):
        mask = np.asarray(mask.convert("L"))
    m = np.asarray(mask)
    m = (m > 127) if m.dtype == np.uint8 else m.astype(bool)
    if m.shape != shape:
        m = cv2.resize(m.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR) > 0
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    if erode:
        m = cv2.erode(m, np.ones((2 * erode + 1, 2 * erode + 1), np.uint8))
    return m > 0


def _lab(rgb):
    return skcolor.rgb2lab(np.clip(rgb, 0, 1)).astype(np.float32)


def _to_rgb(lab):
    return np.clip(skcolor.lab2rgb(lab.astype(np.float64)), 0, 1).astype(np.float32)


def _hex(rgb):
    return "#%02x%02x%02x" % tuple(int(round(float(c) * 255)) for c in np.clip(rgb, 0, 1))


def _lab_hex(lab):
    return _hex(_to_rgb(np.asarray(lab, np.float32).reshape(1, 1, 3))[0, 0])


def _norm_blur(x, w, sigma):
    w = w.astype(np.float32)
    if x.ndim == 3:
        num = cv2.GaussianBlur(x * w[..., None], (0, 0), sigma)
        return num / np.maximum(cv2.GaussianBlur(w, (0, 0), sigma), 1e-4)[..., None]
    return cv2.GaussianBlur(x * w, (0, 0), sigma) / np.maximum(cv2.GaussianBlur(w, (0, 0), sigma), 1e-4)


def _save(rgb, path):
    Image.fromarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)).save(path, "JPEG", quality=88, optimize=True)




def _weighted_median(x, w):
    o = np.argsort(x, kind="stable")
    c = np.cumsum(w[o])
    return float(x[o][np.searchsorted(c, 0.5 * c[-1])])


def _region_mask(lab, seen):
    """Keep the dominant floor material: robust centre in chroma/contrast space (wall-adjacent pixels weigh more), drop outliers like rugs."""
    h, w = seen.shape
    f = 4
    small = cv2.resize(lab, (w // f, h // f), interpolation=cv2.INTER_AREA)
    sm = cv2.resize(seen.astype(np.uint8), (w // f, h // f), interpolation=cv2.INTER_NEAREST) > 0
    if sm.sum() < 30:
        return seen
    L = small[..., 0]
    mu = cv2.blur(L, (5, 5))
    contrast = np.sqrt(np.maximum(cv2.blur(L * L, (5, 5)) - mu * mu, 0))
    feats = np.stack([small[..., 1], small[..., 2], contrast, L], -1)[sm]
    yy, xx = np.nonzero(sm)
    edge = np.minimum.reduce([yy, xx, sm.shape[0] - 1 - yy, sm.shape[1] - 1 - xx]) * f / PX_PER_M
    wts = 1.0 + 3.0 * np.clip(1.0 - edge / 0.6, 0, 1)
    scale = np.array([2.5, 3.5, 3.0, 14.0], np.float32)
    keep = np.ones(len(feats), bool)
    for _ in range(3):
        centre = np.array([_weighted_median(feats[keep, i], wts[keep]) for i in range(4)])
        spread = np.array([_weighted_median(np.abs(feats[keep, i] - centre[i]), wts[keep]) for i in range(4)])
        d = np.abs(feats - centre) / np.maximum(1.8 * spread, scale)
        keep = (d[:, :3].max(1) < 2.5) & (d[:, 3] < 2.5)
    small_m = np.zeros(sm.shape, np.uint8)
    small_m[yy[keep], xx[keep]] = 1
    small_m = cv2.morphologyEx(small_m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    m = cv2.resize(small_m, (w, h), interpolation=cv2.INTER_NEAREST) > 0
    return m & seen


def _orientation_stats(L, m):
    g = cv2.GaussianBlur(L, (0, 0), 1.0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), 4)[m]
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), 4)[m]
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), 4)[m]
    e = jxx + jyy + 1e-6
    z2 = ((jxx - jyy) + 2j * jxy) / e
    local_coh = np.abs(z2)
    ang = np.angle(z2)
    z4 = np.exp(2j * ang) * local_coh
    return {
        "aniso": float(np.abs(z2.mean())),
        "quad": float(np.abs(z4.mean())),
        "coh": float(local_coh.mean()),
        "gx_dominant": float(jxx.mean()) > float(jyy.mean()),
        "energy": float(np.sqrt(e.mean())),
    }


def _classify(lab, m):
    L = lab[..., 0]
    st = _orientation_stats(L, m)
    hf = L - cv2.GaussianBlur(L, (0, 0), 2.0)
    st["fine"] = float(hf[m].std())
    st["chroma"] = float(np.median(np.hypot(lab[..., 1][m], lab[..., 2][m])))
    st["light"] = float(np.median(L[m]))
    st["grid"] = _tile_grid(L, m)
    if st["aniso"] > 0.22 and st["coh"] > 0.3 and st["chroma"] >= 4.0:
        return "wood", float(np.clip((st["aniso"] - 0.12) / 0.25, 0.3, 1.0)), st
    if st["aniso"] < 0.2 and st["quad"] > 0.15 and st["fine"] < 2.0:
        return "tile", float(np.clip(st["quad"] * 3, 0.3, 1.0)), st
    if st["light"] > 65 and st["fine"] < 1.6 and st["chroma"] < 10:
        return "tile", 0.5, st
    if st["fine"] > 2.2 and st["coh"] < 0.45:
        return "carpet", float(np.clip((st["fine"] - 1.5) / 3.0, 0.3, 1.0)), st
    return "other", 0.4, st


TONE_BAND = (1, 12)


def _tone_band(lab, m):
    """Pixels in the lower-middle luminance band: lit floor without window glare or deep shadow."""
    lo, hi = np.percentile(lab[..., 0][m], TONE_BAND)
    return m & (lab[..., 0] >= lo) & (lab[..., 0] <= hi)


def _target_lab(lab, m):
    """Lightness from the unglared band; hue and chroma from the most saturated pixels, since glare only washes colour out."""
    px = lab[m]
    chroma = np.hypot(px[:, 1], px[:, 2])
    band = lab[_tone_band(lab, m)]
    ab = np.median(px[chroma >= np.percentile(chroma, 85)][:, 1:], 0)
    base_c = float(np.hypot(*np.median(band[:, 1:], 0)))
    ab *= min(1.0, 1.35 * base_c / max(float(np.hypot(*ab)), 1e-6))
    return np.array([np.median(band[:, 0]), ab[0], ab[1]], np.float32)


def _flatten(lab, m, target, sigma=55):
    """Remove illumination gradients and glare: per-pixel normalise local mean and contrast to the target."""
    L = lab[..., 0]
    mu = _norm_blur(L, m, sigma)
    dev = L - mu
    sd = np.sqrt(np.maximum(_norm_blur(dev * dev, m, sigma), 1e-3))
    lt = _tone_band(lab, m)
    ref_sd = float(np.median(sd[lt])) if lt.sum() > 100 else float(np.median(sd[m]))
    gain = np.clip(ref_sd / sd, 0.6, 2.2)
    out = np.empty_like(lab)
    out[..., 0] = target[0] + dev * gain
    for c in (1, 2):
        cm = _norm_blur(lab[..., c], m, sigma)
        out[..., c] = target[c] + (lab[..., c] - cm) * 0.8
    return out




def _plank_width(L, m):
    hp = L - cv2.GaussianBlur(L, (0, 0), 6)
    line = cv2.GaussianBlur(-hp, (0, 0), sigmaX=0.8, sigmaY=12)
    line = np.where(m, line, 0)
    mm = m.astype(np.float32)
    lags = np.arange(int(0.09 * PX_PER_M), int(0.31 * PX_PER_M))
    ac = []
    for lag in lags:
        num = (line[:, :-lag] * line[:, lag:]).sum()
        den = (mm[:, :-lag] * mm[:, lag:]).sum() + 1.0
        ac.append(num / den)
    ac = np.array(ac)
    zero = (line * line).sum() / (mm.sum() + 1.0)
    ac = ac / max(zero, 1e-6)
    i = int(np.argmax(ac))
    if ac[i] < 0.08:
        return 0.18, float(ac[i])
    return float(lags[i]) / PX_PER_M, float(ac[i])


def _joint_profile(L):
    hp = L - cv2.GaussianBlur(L, (0, 0), 6)
    return cv2.GaussianBlur(-hp, (0, 0), sigmaX=0.8, sigmaY=10)


def _plank_crops(flat, lab, m, wpx, min_len, count, target):
    """Rank crops of single planks: fully seen, joints at the edges and none inside, sharp, evenly lit, near the target tone."""
    h, w = m.shape
    joint = _joint_profile(flat[..., 0])
    hf = np.abs(flat[..., 0] - cv2.GaussianBlur(flat[..., 0], (0, 0), 1.5))
    low = cv2.GaussianBlur(lab[..., 0], (0, 0), 4)
    ii = np.pad(m.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    cands = []
    for ln in (int(1.1 * PX_PER_M), int(0.7 * PX_PER_M), min_len):
        if ln > h:
            continue
        for y in range(0, h - ln, 10):
            for x in range(0, w - wpx, 3):
                cov = ii[y + ln, x + wpx] - ii[y, x + wpx] - ii[y + ln, x] + ii[y, x]
                if cov < 0.995 * ln * wpx:
                    continue
                prof = joint[y:y + ln, x:x + wpx].mean(0)
                edge = max(prof[:3].max(), prof[-3:].max())
                inner = prof[4:-4].max()
                sharp = hf[y:y + ln, x:x + wpx].mean()
                lit = low[y:y + ln, x + 3:x + wpx - 3]
                even = lit.std()
                tone = abs(float(lit.mean()) - target[0])
                score = (edge - inner) + 0.6 * sharp + 0.003 * ln - 1.5 * even - 0.15 * tone
                cands.append((score, y, x, ln))
        if len(cands) > 6 * count:
            break
    cands.sort(key=lambda c: -c[0])
    taken, used = [], np.zeros_like(m)
    for s, y, x, ln in cands:
        if used[y:y + ln, x + 2:x + wpx - 2].mean() > 0.25:
            continue
        used[y:y + ln, x:x + wpx] = True
        taken.append((y, x, ln))
        if len(taken) >= count:
            break
    return taken


GRAIN_GAIN = 0.9


def _noise(shape, sx, sy, rng):
    n = cv2.GaussianBlur(rng.normal(0, 1, shape).astype(np.float32), (0, 0), sigmaX=sx, sigmaY=sy)
    return n / (n.std() + 1e-6)


def _plank(prof, dev, w, h, rng):
    """One plank: the real cross-plank colour profile extruded along its length, plus flowing grain lines and pores."""
    xs = np.linspace(0, len(prof) - 1, w, dtype=np.float32)
    drift = _noise((h, 1), 0.1, 60, rng) * 0.04 * len(prof)
    coords = np.clip(xs[None, :] + drift, 0, len(prof) - 1)
    i0 = np.floor(coords).astype(int)
    i1 = np.minimum(i0 + 1, len(prof) - 1)
    t = (coords - i0)[..., None]
    out = (prof[i0] * (1 - t) + prof[i1] * t) * GRAIN_GAIN
    xx = np.arange(w, dtype=np.float32)[None, :] / w
    phi = xx + 0.07 * _noise((h, w), 0.6 * w, 0.2 * h, rng) + 0.015 * _noise((h, w), 0.15 * w, 0.05 * h, rng)
    grain = np.zeros((h, w), np.float32)
    for period, weight in ((1 / rng.uniform(9, 13), 1.0), (1 / rng.uniform(20, 28), 0.6), (1 / rng.uniform(4, 6), 0.5)):
        ridge = 1 - np.abs(np.sin(np.pi * (phi + rng.uniform()) / period))
        grain += weight * ridge**4 * np.clip(0.5 + 0.6 * _noise((h, w), 0.05 * w, 0.08 * h, rng), 0, 1.3)
    bands = _noise((h, w), 0.04 * w, 0.25 * h, rng)
    pores = np.clip(_noise((h, w), 0.6, 5, rng) - 2.2, 0, None)
    streak = _noise((h, w), 0.7, 30, rng)
    along = _noise((h, 1), 0.1, 90, rng)
    out[..., 0] += 0.9 * streak + 1.0 * along + 1.6 * bands + rng.normal(0, 0.45, (h, w)) - 4.5 * grain - 4.0 * pores
    out[..., 1] += 0.4 * grain
    out[..., 2] += 0.9 * grain
    tone = np.array([np.clip(1.1 * dev[0], -8.0, 8.0), np.clip(0.9 * dev[1], -2.5, 2.5), np.clip(0.9 * dev[2], -5.0, 5.0)], np.float32)
    return out + tone


def _wood_tile(flat, lab, m, target, rng):
    gx_dom = _orientation_stats(flat[..., 0], m)["gx_dominant"]
    transpose = not gx_dom
    if transpose:
        flat, lab, m = (np.ascontiguousarray(a.swapaxes(0, 1)) for a in (flat, lab, m))
    width_m, strength = _plank_width(flat[..., 0], m)
    n_cols = max(6, int(round(2.4 / width_m)))
    size_m = n_cols * width_m
    plank_m = size_m / 2
    out_ppm = TILE_PX / size_m
    wpx = int(round(width_m * PX_PER_M))
    crops = _plank_crops(flat, lab, m, wpx, int(0.35 * PX_PER_M), n_cols * 2, target)
    if not crops:
        return None
    col_w = TILE_PX / n_cols
    plank_px = int(round(plank_m * out_ppm))
    profiles = []
    for y, x, ln in crops:
        src = flat[y:y + ln, x + 2:x + wpx - 2]
        prof = src.mean(0)
        raw = lab[y:y + ln, x + 2:x + wpx - 2].reshape(-1, 3).mean(0)
        profiles.append((prof - prof.mean(0), raw - target))
    tile = np.zeros((TILE_PX, TILE_PX, 3), np.float32)
    seam = np.ones((TILE_PX, TILE_PX), np.float32)
    k = 0
    for c in range(n_cols):
        x0 = int(round(c * col_w))
        x1 = int(round((c + 1) * col_w))
        offset = int(rng.integers(0, plank_px))
        for p in range(2):
            prof, dev = profiles[k % len(profiles)]
            if (k // len(profiles)) % 2:
                prof = prof[::-1]
            k += 1
            start = (offset + p * plank_px) % TILE_PX
            rows = (start + np.arange(plank_px)) % TILE_PX
            tile[rows, x0:x1] = _plank(prof, dev, x1 - x0, plank_px, rng)
            seam[rows[:2], x0:x1] *= 0.72
            seam[rows[-1], x0:x1] *= 0.85
        seam[:, x0] *= 0.7
        seam[:, min(x0 + 1, TILE_PX - 1)] *= 0.88
    tile += target
    tile[..., 0] -= float(np.median(tile[..., 0])) - target[0]
    tile[..., 1:] += target[1:] - np.median(tile[..., 1:].reshape(-1, 2), 0)
    seam = _wrap_blur(seam, 0.6)
    rgb = _to_rgb(tile)
    rgb = _sharpen(rgb, 1.2, 0.5) * seam[..., None]
    if transpose:
        rgb = np.ascontiguousarray(rgb.swapaxes(0, 1))
    return rgb, size_m


def _wrap_blur(x, sigma):
    p = int(3 * sigma) + 1
    pad = ((p, p), (p, p)) + ((0, 0),) * (x.ndim - 2)
    return cv2.GaussianBlur(np.pad(x, pad, mode="wrap"), (0, 0), sigma)[p:-p, p:-p]


def _sharpen(rgb, sigma, amount):
    return np.clip(rgb + amount * (rgb - _wrap_blur(rgb, sigma)), 0, 1)


def _stochastic_tile(lab, m, target, size_m, rng):
    """Variance-preserving blend of random clean blocks on a torus (seamless by construction)."""
    out_ppm = TILE_PX / size_m
    ii = np.pad(m.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    for block_m in (0.2, 0.14, 0.1, 0.07):
        b_src = int(block_m * PX_PER_M)
        if b_src >= min(m.shape):
            continue
        cov = ii[b_src:, b_src:] - ii[:-b_src, b_src:] - ii[b_src:, :-b_src] + ii[:-b_src, :-b_src]
        ys, xs = np.nonzero(cov >= 0.97 * b_src * b_src)
        if len(ys) >= 20:
            break
    else:
        return None
    b_out = int(round(block_m * out_ppm))
    step = b_out // 2
    win = np.outer(np.hanning(b_out + 2)[1:-1], np.hanning(b_out + 2)[1:-1]).astype(np.float32)
    acc = np.zeros((TILE_PX, TILE_PX, 3), np.float32)
    wsum = np.zeros((TILE_PX, TILE_PX), np.float32)
    w2 = np.zeros((TILE_PX, TILE_PX), np.float32)
    mean = lab[m].mean(0)
    for gy in range(0, TILE_PX, step):
        for gx in range(0, TILE_PX, step):
            i = int(rng.integers(0, len(ys)))
            blk = lab[ys[i]:ys[i] + b_src, xs[i]:xs[i] + b_src]
            r = int(rng.integers(0, 4))
            blk = np.rot90(blk, r) if r else blk
            blk = cv2.resize(np.ascontiguousarray(blk), (b_out, b_out), interpolation=cv2.INTER_CUBIC) - mean
            rows = (gy + np.arange(b_out)) % TILE_PX
            cols = (gx + np.arange(b_out)) % TILE_PX
            acc[np.ix_(rows, cols)] += blk * win[..., None]
            wsum[np.ix_(rows, cols)] += win
            w2[np.ix_(rows, cols)] += win * win
    tile = acc / np.sqrt(np.maximum(w2, 1e-6))[..., None] + target
    return _to_rgb(tile)


def _tile_grid(L, m):
    """Detect a grout grid period (metres) from line autocorrelation along both axes."""
    hp = L - cv2.GaussianBlur(L, (0, 0), 5)
    best = []
    for axis in (0, 1):
        line = cv2.GaussianBlur(-hp, (0, 0), sigmaX=0.8 if axis == 1 else 14, sigmaY=14 if axis == 1 else 0.8)
        line = np.where(m, line, 0)
        if axis == 0:
            line = line.T
        mm = (m.T if axis == 0 else m).astype(np.float32)
        lags = np.arange(int(0.2 * PX_PER_M), min(int(0.9 * PX_PER_M), line.shape[1] - 4))
        if len(lags) == 0:
            best.append((0.0, 0.0))
            continue
        zero = (line * line).sum() / (mm.sum() + 1)
        ac = np.array([(line[:, :-g] * line[:, g:]).sum() / ((mm[:, :-g] * mm[:, g:]).sum() + 1) for g in lags]) / max(zero, 1e-6)
        i = int(np.argmax(ac))
        best.append((lags[i] / PX_PER_M, ac[i]))
    return best


def _crop_faces(face_rgb, rng):
    def face(ow, oh):
        fh, fw = face_rgb.shape[:2]
        if fh < oh or fw < ow:
            return cv2.resize(face_rgb, (ow, oh))
        sy = int(rng.integers(0, fh - oh + 1))
        sx = int(rng.integers(0, fw - ow + 1))
        return face_rgb[sy:sy + oh, sx:sx + ow]
    return face


def _tiled_floor(face, tile_w, tile_h, size_m, grout_rgb, rng, vary=0.015):
    """Lay rectangular tiles (running bond for long tiles) with thin grout; face(w, h) supplies each tile's surface."""
    ppm = TILE_PX / size_m
    out = np.zeros((TILE_PX, TILE_PX, 3), np.float32)
    nx = max(1, int(round(size_m / tile_w)))
    ny = max(1, int(round(size_m / tile_h)))
    tw, th = TILE_PX / nx, TILE_PX / ny
    bond = tile_w / tile_h > 1.5
    grout = np.ones((TILE_PX, TILE_PX), np.float32)
    gw = max(1.5, 0.003 * ppm)
    ow, oh = int(np.ceil(tw)), int(np.ceil(th))
    for j in range(ny):
        shift = tw / 2 if (bond and j % 2) else 0.0
        for i in range(nx + (1 if shift else 0)):
            x0 = int(round(i * tw + shift))
            y0 = int(round(j * th))
            rows = (y0 + np.arange(oh)) % TILE_PX
            cols = (x0 + np.arange(ow)) % TILE_PX
            out[np.ix_(rows, cols)] = face(ow, oh) * (1 + rng.normal(0, vary))
            for g in range(int(np.ceil(gw))):
                grout[(y0 + g) % TILE_PX, cols] = 0
                grout[rows, (x0 + g) % TILE_PX] = 0
    grout = _wrap_blur(grout, 0.7)[..., None]
    return out * grout + grout_rgb * (1 - grout)


def _marble_face(base_rgb, size_px, rng):
    """Procedural light marble: faint clouding plus long veins along the zero set of stretched smooth noise."""
    big = int(size_px * 1.5)
    ang = rng.uniform(20, 70)

    def stretched(sigma_across, ratio):
        n = cv2.GaussianBlur(rng.normal(0, 1, (big, big)).astype(np.float32), (0, 0), sigmaX=sigma_across * ratio, sigmaY=sigma_across)
        rot = cv2.getRotationMatrix2D((big / 2, big / 2), ang, 1.0)
        n = cv2.warpAffine(n, rot, (big, big), borderMode=cv2.BORDER_REFLECT)
        o = (big - size_px) // 2
        n = n[o:o + size_px, o:o + size_px]
        return n / (n.std() + 1e-6)

    warp = stretched(size_px * 0.05, 2.0) + 0.12 * stretched(size_px * 0.015, 1.5)
    main = stretched(size_px * 0.08, 4.5) + 0.35 * warp
    fine = stretched(size_px * 0.025, 3.0) + 0.3 * warp
    vis_main = np.clip(stretched(size_px * 0.1, 1.5) * 0.7 + 0.2, 0, 1)
    vis_fine = np.clip(stretched(size_px * 0.06, 1.5) * 0.6 - 0.1, 0, 1)
    veins = np.exp(-((main / 0.025) ** 2)) * vis_main
    veins = cv2.GaussianBlur(veins, (0, 0), 0.7) + np.exp(-((main / 0.12) ** 2)) * vis_main * 0.12
    hair = np.exp(-((fine / 0.04) ** 2)) * vis_fine
    cloud = cv2.GaussianBlur(rng.normal(0, 1, (size_px, size_px)).astype(np.float32), (0, 0), size_px * 0.06)
    cloud /= cloud.std() + 1e-6
    v = 1 - 0.42 * veins - 0.12 * hair + 0.005 * cloud
    tint = np.array([0.0, 0.01, 0.03], np.float32)
    return np.clip(base_rgb[None, None, :] * v[..., None] + tint * (veins[..., None] * 0.3), 0, 1)




def floor_tile(img, mask, out_path):
    rgb = _rgb(img)
    seen = _mask(mask, rgb.shape[:2])
    if seen.sum() / PX_PER_M**2 < MIN_FLOOR_M2:
        return None
    lab = _lab(rgb)
    m = _region_mask(lab, seen)
    area = m.sum() / PX_PER_M**2
    if area < MIN_FLOOR_M2:
        return None
    material, conf, st = _classify(lab, m)
    target = _target_lab(lab, m)
    conf *= float(np.clip((target[0] - 12.0) / 18.0, 0.1, 1.0))
    flat = _flatten(lab, m, target)
    rng = np.random.default_rng(7)
    out = None
    if material == "wood":
        res = _wood_tile(flat, lab, m, target, rng)
        if res is not None:
            out, size = res
        else:
            material = "other"
    if material == "tile":
        (gx, sx), (gy, sy) = st["grid"]
        tw, th = (gx, gy) if min(sx, sy) > 0.3 else (0.6, 0.3)
        size = float(np.clip(round(1.2 / tw) * tw, 0.6, 2.0))
        size = max(size, round(size / th) * th)
        face = _stochastic_tile(flat, m, target, size, rng)
        if face is not None:
            base = _to_rgb(target[None, None])[0, 0]
            out = _tiled_floor(_crop_faces(face, rng), tw, th, size, base * 0.85, rng)
    if out is None:
        size = 1.0
        out = _stochastic_tile(flat, m, target, size, rng)
    if out is None:
        return None
    out_path = Path(out_path)
    _save(out, out_path)
    return {
        "tile": out_path.name,
        "size": round(float(size), 3),
        "material": material,
        "color": _hex(out.reshape(-1, 3).mean(0)),
        "evidence": round(float(area * conf), 3),
    }


def default_tile(out_path, base="#e6e5e2", size=2.4):
    """Light marble-look porcelain, 60 cm squares: the usual bathroom floor when the photos never show it clearly."""
    rng = np.random.default_rng(11)
    base_rgb = np.array([int(base[i:i + 2], 16) for i in (1, 3, 5)], np.float32) / 255
    out = _tiled_floor(lambda w, h: _marble_face(base_rgb, max(w, h), rng)[:h, :w], 0.6, 0.6, size, base_rgb * 0.86, rng, vary=0.004)
    out_path = Path(out_path)
    _save(out, out_path)
    return {"tile": out_path.name, "size": size, "material": "tile", "color": _hex(out.reshape(-1, 3).mean(0)), "evidence": 0.0}



DOOR_TOP_M = 2.15
MIN_ACCENT_M = 0.6


def _smooth_pixels(lab, seen):
    """Paint candidates: seen pixels away from edges and texture (frames, art, furniture, window mullions)."""
    L = cv2.GaussianBlur(lab[..., 0], (0, 0), 1.0)
    grad = np.hypot(cv2.Sobel(L, cv2.CV_32F, 1, 0), cv2.Sobel(L, cv2.CV_32F, 0, 1)) / 8
    chroma_grad = sum(np.hypot(cv2.Sobel(lab[..., c], cv2.CV_32F, 1, 0), cv2.Sobel(lab[..., c], cv2.CV_32F, 0, 1)) / 8 for c in (1, 2))
    busy = cv2.dilate(((grad > 1.6) | (chroma_grad > 2.0)).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    return seen & ~busy


def _paint_modes(lab, sel):
    """Colour modes of smooth pixels in the a/b plane; each mode keeps its member pixel mask."""
    a, b = lab[..., 1][sel], lab[..., 2][sel]
    bins = np.arange(-80, 81, 2.0)
    hist, _, _ = np.histogram2d(a, b, bins=[bins, bins])
    hist = cv2.GaussianBlur(hist.astype(np.float32), (0, 0), 1.5)
    peaks = (hist == cv2.dilate(hist, np.ones((7, 7), np.uint8))) & (hist > 0.02 * hist.max())
    centres = [(bins[i] + 1, bins[j] + 1) for i, j in zip(*np.nonzero(peaks))]
    if not centres:
        return []
    centres = np.array(centres, np.float32)
    d = np.hypot(a[:, None] - centres[None, :, 0], b[:, None] - centres[None, :, 1])
    radius = 6 + 0.25 * np.hypot(centres[:, 0], centres[:, 1])
    lbl = np.argmin(d, 1)
    ok = d[np.arange(len(lbl)), lbl] < radius[lbl]
    ys, xs = np.nonzero(sel)
    modes = []
    for k in range(len(centres)):
        mk = ok & (lbl == k)
        if mk.sum() < 50:
            continue
        mask = np.zeros(sel.shape, bool)
        mask[ys[mk], xs[mk]] = True
        modes.append(mask)
    return modes


def _lit_colour(lab, mask):
    """The paint as it looks under even light: whites from the lit band, saturated paints from the middle band (glare washes them out)."""
    px = lab[mask]
    chroma = float(np.median(np.hypot(px[:, 1], px[:, 2])))
    lo, hi = np.percentile(px[:, 0], [55, 92] if chroma < 12 else [25, 65])
    band = px[(px[:, 0] >= lo) & (px[:, 0] <= hi)]
    return np.median(band if len(band) > 20 else px, 0)


def _plausibility(c):
    L, chroma = float(c[0]), float(np.hypot(c[1], c[2]))
    hue = np.degrees(np.arctan2(c[2], c[1])) % 360
    if chroma < 12:
        return 1.0 if L >= WALL_PLAUSIBLE_L else 0.12
    if L > 66 and 200 <= hue <= 275:
        return 0.25
    if chroma < 22 and L < 55:
        return 0.15
    return 0.9


def wall_paint(img, mask):
    rgb = _rgb(img)
    seen = _mask(mask, rgb.shape[:2], erode=3)
    if seen.sum() / PX_PER_M**2 < MIN_WALL_M2:
        return None
    lab = _lab(rgb)
    sel = _smooth_pixels(lab, seen)
    if sel.sum() / PX_PER_M**2 < MIN_WALL_M2 * 0.5:
        sel = seen
    modes = []
    for mk in _paint_modes(lab, sel):
        c = _lit_colour(lab, mk)
        modes.append({"mask": mk, "lab": c, "score": mk.sum() * _plausibility(c)})
    if not modes:
        return None
    modes.sort(key=lambda m: -m["score"])
    main = modes[0]
    h, w = seen.shape
    accent = []
    door_rows = slice(0, max(0, int((h / PX_PER_M - DOOR_TOP_M) * PX_PER_M)))
    cols = np.stack([m["mask"].sum(0) for m in modes], 0).astype(np.float32)
    for k, other in enumerate(modes[1:], 1):
        if other["score"] < 0.08 * main["score"] or _plausibility(other["lab"]) < 0.5:
            continue
        if np.linalg.norm(other["lab"][1:] - main["lab"][1:]) < 12:
            continue
        share = cv2.blur(cols[k][None], (41, 1))[0] / np.maximum(cv2.blur(cols.sum(0)[None], (41, 1))[0], 1e-3)
        on = share > 0.6
        on &= cv2.blur(cols.sum(0)[None], (41, 1))[0] > 2
        x = 0
        while x < w:
            if not on[x]:
                x += 1
                continue
            x1 = x
            while x1 < w and on[x1]:
                x1 += 1
            if (x1 - x) / PX_PER_M >= MIN_ACCENT_M:
                top_other = other["mask"][door_rows, x:x1].sum()
                top_main = main["mask"][door_rows, x:x1].sum()
                if not (top_main > 200 and top_main > 3 * top_other):
                    accent.append({"from": round(x / PX_PER_M, 2), "to": round(x1 / PX_PER_M, 2), "lab": other["lab"]})
            x = x1
    span = seen.any(0)
    covered = sum((a["to"] - a["from"]) * PX_PER_M for a in accent)
    colour = main["lab"]
    if accent and covered >= 0.8 * span.sum():
        colour = accent[0]["lab"]
        accent = []
    return {
        "color": _lab_hex(colour),
        "accent": [{"from": a["from"], "to": a["to"], "color": _lab_hex(a["lab"])} for a in accent],
        "lab": [round(float(v), 2) for v in colour],
        "evidence": round(float(sel.sum() / PX_PER_M**2), 3),
    }




WALLS = ("n", "s", "e", "w")
NEUTRAL_C = 9.0


def _hex_lab(h):
    rgb = np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], np.float32) / 255
    return _lab(rgb.reshape(1, 1, 3))[0, 0]


def _is_neutral(lab):
    return float(np.hypot(lab[1], lab[2])) < NEUTRAL_C


def _harmonise_walls(walls):
    """One white for the listing, one colour per accent paint (its deepest sighting: glare only washes paint out), photo cast removed."""
    entries = [(rid, w, d) for rid, ws in walls.items() for w, d in ws.items() if d is not None]
    whites = [(np.array(d["lab"]), d["evidence"]) for _, _, d in entries if _is_neutral(d["lab"]) and d["lab"][0] > WALL_PLAUSIBLE_L]
    if whites:
        labs = np.array([l for l, _ in whites])
        wts = np.array([e for _, e in whites]) + 0.05
        white = np.array([np.percentile(labs[:, 0], 90), np.average(labs[:, 1], weights=wts), np.average(labs[:, 2], weights=wts)])
    else:
        white = np.array([92.0, 0.0, 1.5])
    cast = white[1:] * 0.7
    white[1:] -= cast
    paints = []
    for rid, w, d in entries:
        lab = np.array(d["lab"])
        if _is_neutral(lab):
            continue
        for p in paints:
            if np.linalg.norm(p["ref"] - lab) < 20:
                if np.hypot(lab[1], lab[2]) > np.hypot(p["best"][1], p["best"][2]):
                    p["best"] = lab
                break
        else:
            paints.append({"ref": lab, "best": lab})

    def fix(lab):
        lab = np.array(lab, np.float32)
        if _is_neutral(lab):
            return white
        for p in paints:
            if np.linalg.norm(p["ref"] - lab) < 20:
                lab = p["best"]
                break
        return np.array([lab[0], lab[1] - cast[0], lab[2] - cast[1]])

    for rid, w, d in entries:
        d["color"] = _lab_hex(fix(d.pop("lab")))
        for a in d["accent"]:
            a["color"] = _lab_hex(fix(_hex_lab(a["color"])))
    return _lab_hex(white)


def wall_grain(out_path, size_m=1.0):
    """Faint eggshell/roller texture, a multiplier centred just under 1 so it never shifts the paint colour noticeably."""
    rng = np.random.default_rng(3)
    n = TILE_PX

    def periodic(sigma):
        g = rng.normal(0, 1, (n, n)).astype(np.float32)
        g = cv2.GaussianBlur(np.tile(g, (3, 3)), (0, 0), sigma)[n:2 * n, n:2 * n]
        return g / (g.std() + 1e-6)

    v = 0.985 + 0.004 * periodic(1.2) + 0.003 * periodic(6) + 0.002 * periodic(40)
    _save(np.repeat(np.clip(v, 0, 1)[..., None], 3, -1), out_path)
    return {"grain": Path(out_path).name, "grain_size": size_m}


def _share(src, rid):
    return {**src, "source": src.get("source", rid)}


def _pick_floors(floors, room_types, out_dir):
    reliable = {r: f for r, f in floors.items() if f and f["evidence"] >= RELIABLE_EVIDENCE and f["material"] != "other"}
    dry = {r: f for r, f in reliable.items() if room_types.get(r) != "bathroom"}
    main_id = max(dry, key=lambda r: dry[r]["evidence"]) if dry else None
    best = {}
    for r, f in reliable.items():
        if f["material"] not in best or f["evidence"] > reliable[best[f["material"]]]["evidence"]:
            best[f["material"]] = r
    out = {}
    for rid, f in floors.items():
        if room_types.get(rid) == "bathroom":
            if f and f["material"] == "tile" and f["evidence"] >= 0.5 * RELIABLE_EVIDENCE:
                out[rid] = {**f, "source": rid}
            elif "tile" in best:
                out[rid] = _share(floors[best["tile"]], best["tile"])
            else:
                out[rid] = {**default_tile(Path(out_dir) / f"{rid}-floor-tile.jpg"), "source": "default"}
            continue
        main = floors[main_id] if main_id else None
        if rid in reliable and (main is None or f["material"] == main["material"] or f["evidence"] >= 0.35 * main["evidence"]):
            out[rid] = {**f, "source": rid}
        elif f and f["material"] in best and f["material"] != "other" and f["evidence"] >= 0.5 * RELIABLE_EVIDENCE:
            out[rid] = _share(floors[best[f["material"]]], best[f["material"]])
        elif main_id:
            out[rid] = _share(main, main_id)
        elif f:
            out[rid] = {**f, "source": rid}
    return out


def listing_surfaces(shell_dir, room_ids, room_types, out_dir=None):
    shell_dir = Path(shell_dir)
    out_dir = Path(out_dir or shell_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    floors, walls = {}, {}
    for rid in room_ids:
        img, seen = shell_dir / f"{rid}-floor.jpg", shell_dir / f"{rid}-floor-seen.png"
        floors[rid] = floor_tile(img, seen, out_dir / f"{rid}-floor-tile.jpg") if img.exists() and seen.exists() else None
        walls[rid] = {}
        for w in WALLS:
            img, seen = shell_dir / f"{rid}-wall-{w}.jpg", shell_dir / f"{rid}-wall-{w}-seen.png"
            walls[rid][w] = wall_paint(img, seen) if img.exists() and seen.exists() else None
    picked = _pick_floors(floors, room_types, out_dir)
    for rid, ws in walls.items():
        known = [d for d in ws.values() if d is not None]
        room_main = max(known, key=lambda d: d["evidence"] * (1.0 if _is_neutral(d["lab"]) else 0.5), default=None)
        for w in WALLS:
            if ws[w] is None and room_main is not None and _is_neutral(room_main["lab"]):
                ws[w] = {"color": room_main["color"], "accent": [], "lab": list(room_main["lab"]), "evidence": 0.0}
    white = _harmonise_walls(walls)
    grain = wall_grain(out_dir / "wall-grain.jpg")
    result = {}
    for rid in room_ids:
        ws = {}
        for w in WALLS:
            d = walls[rid][w] or {"color": white, "accent": [], "evidence": 0.0}
            ws[w] = {**d, **grain}
        result[rid] = {"floor": picked.get(rid), "walls": ws}
    return result


if __name__ == "__main__":
    import sys

    data = Path(sys.argv[1])
    scene = json.loads((data / "scene.json").read_text())
    rooms = [r for r in scene["rooms"] if (data / "shell" / f"{r['id']}-floor.jpg").exists()]
    res = listing_surfaces(data / "shell", [r["id"] for r in rooms], {r["id"]: r["type"] for r in rooms})
    print(json.dumps(res, indent=1))
