import numpy as np
import torch
from PIL import Image

from shell import Cam, robust_color

CELL = 0.05
DEBUG = False
FURNITURE = {
    "bed": ["a bed with pillows and a mattress", "a made bed with white sheets"],
    "sofa": ["a sofa", "a couch", "a daybed sofa"],
    "table": ["a dining table", "a coffee table", "a round table"],
    "desk": ["a desk with a chair"],
    "chair": ["a chair", "a bar stool", "an armchair"],
    "counter": ["a kitchen island", "a kitchen counter with cabinets", "kitchen cabinets"],
    "cabinet": ["a dresser", "a cabinet", "a nightstand", "a bathroom vanity"],
    "appliance": ["a refrigerator", "a stove and oven", "a washing machine and dryer", "a dishwasher"],
    "toilet": ["a toilet"],
    "shower": ["a glass shower", "a bathtub"],
    "shelf": ["a bookshelf", "a closet with clothes", "a wardrobe"],
    "tv": ["a television on a wall", "a tv stand"],
    "rug": ["a rug on the floor"],
    "plant": ["a potted plant"],
    "lamp": ["a floor lamp", "a table lamp"],
}
REJECT = {
    "wall": ["a plain painted wall", "a wall corner"],
    "floor": ["a wooden floor", "a carpeted floor", "a tiled floor"],
    "window": ["a window with a city view", "a window with sky"],
    "door": ["a door", "a doorway"],
    "ceiling": ["a ceiling with a light"],
}
FLOORS = {
    "wood": ["a hardwood floor", "wood plank laminate flooring", "vinyl wood look floor"],
    "carpet": ["a carpeted floor", "beige carpet", "grey carpet"],
    "tile": ["a tiled floor", "ceramic floor tiles", "a bathroom tile floor"],
}

_clip = None


def clip():
    global _clip
    if _clip is None:
        from classify import load_model, text_bank

        model, pre, tok = load_model()
        _clip = (model, pre, tok, text_bank(model, tok, {**FURNITURE, **REJECT}), text_bank(model, tok, FLOORS))
    return _clip


@torch.no_grad()
def zero_shot(images, bank):
    model, pre, _, _, _ = clip()
    names, feats = bank
    dev = feats.device
    x = torch.stack([pre(im) for im in images]).to(dev)
    e = model.encode_image(x).float()
    e = e / e.norm(dim=-1, keepdim=True)
    p = (100 * e @ feats.T).softmax(-1).mean(0).cpu().numpy()
    i = int(p.argmax())
    return names[i], float(p[i])


def hexcol(c):
    c = np.clip(np.asarray(c, float), 0, 255).astype(int)
    return "#%02x%02x%02x" % tuple(c)


def project(c: Cam, X):
    Xc = (X - c.C) @ c.Rw
    z = -Xc[:, 2]
    px = c.K[0, 0] * Xc[:, 0] / np.maximum(z, 1e-6) + c.K[0, 2]
    py = -c.K[1, 1] * Xc[:, 1] / np.maximum(z, 1e-6) + c.K[1, 2]
    return px, py, z


def grid_points(c: Cam, stride=3):
    H, W = c.depth.shape
    v, u = np.mgrid[0:H:stride, 0:W:stride]
    d = c.depth[::stride, ::stride]
    x = (u + 0.5 - c.K[0, 2]) / c.K[0, 0] * d
    y = -(v + 0.5 - c.K[1, 2]) / c.K[1, 1] * d
    pc = np.stack([x, y, -d], -1).reshape(-1, 3)
    X = pc @ c.Rw.T + c.C
    ih, iw = c.image.shape[:2]
    cols = c.image[np.clip((v * ih / H).astype(int), 0, ih - 1), np.clip((u * iw / W).astype(int), 0, iw - 1)].reshape(-1, 3)
    conf = c.conf[::stride, ::stride].reshape(-1)
    return X, cols, conf


WALLS = {
    "n": (2, 0, +1),
    "s": (2, 1, -1),
    "w": (0, 0, +1),
    "e": (0, 1, -1),
}


def openings(cams, box):
    (x0, x1), (z0, z1), h = box
    ext = {"x": (x0, x1), "z": (z0, z1)}
    found = []
    for wall, (axis, side, inward) in WALLS.items():
        plane = (ext["z"] if axis == 2 else ext["x"])[side]
        along = 0 if axis == 2 else 2
        lo, hi = (x0, x1) if along == 0 else (z0, z1)
        nu, nv = max(1, int((hi - lo) / CELL)), max(1, int(h / CELL))
        through = np.zeros((nv, nu), np.int32)
        solid = np.zeros((nv, nu), np.int32)
        for c in cams:
            X, _, conf = grid_points(c, 3)
            d0 = (c.C[axis] - plane) * inward
            d1 = (X[:, axis] - plane) * inward
            if d0 <= 0.2:
                continue
            crosses = d1 < -0.25
            near = np.abs(d1) < 0.08
            t = d0 / np.maximum(d0 - d1, 1e-6)
            hit = c.C + (X - c.C) * t[:, None]
            for mask, acc in ((crosses, through), (near, solid)):
                hp = hit[mask] if acc is through else X[mask]
                iu = ((hp[:, along] - lo) / CELL).astype(int)
                iv = (hp[:, 1] / CELL).astype(int)
                ok = (iu >= 0) & (iu < nu) & (iv >= 0) & (iv < nv)
                np.add.at(acc, (iv[ok], iu[ok]), 1)
        import cv2

        m = ((through >= 2) & (through > 0.6 * solid)).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        for i in range(1, n):
            x, y, w, hh, area = stats[i]
            fw, fh = w * CELL, hh * CELL
            if fw < 0.45 or fh < 0.5 or area < 0.45 * w * hh:
                continue
            bottom, top = y * CELL, (y + hh) * CELL
            kind = "door" if bottom < 0.2 and fh > 1.6 and top < h - 0.3 and fw < 1.4 else "window"
            if kind == "door":
                bottom, top = 0.0, min(top, 2.1)
            else:
                bottom = max(bottom, 0.3)
            found.append({"wall": wall, "kind": kind, "from": round(lo + x * CELL, 3), "to": round(lo + (x + w) * CELL, 3),
                          "bottom": round(bottom, 3), "top": round(min(top, h - 0.05), 3)})
    return found


def visual_openings(raw, walls_ref, h, box):
    import cv2
    from shell import PX_PER_M

    (x0, x1), (z0, z1), _ = box
    found = []
    for wall in ("n", "s", "w", "e"):
        img, mask = raw[f"wall-{wall}"]
        ref = np.array([int(walls_ref[wall][i:i + 2], 16) for i in (1, 3, 5)], np.float32)
        diff = np.linalg.norm(img.astype(np.float32) - ref, axis=-1)
        lum = img.astype(np.float32) @ np.array([0.299, 0.587, 0.114])
        ref_l = float(ref @ np.array([0.299, 0.587, 0.114]))
        bright = mask & (lum > max(ref_l + 12, 215))
        dark = mask & (lum < 70)
        f = img.astype(np.int16)
        sky = mask & (f[..., 2] > f[..., 0] + 25) & (lum > 150) & (diff > 45)
        cand = mask & (diff > 45) & ~dark | bright
        m = cand.astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        span = (x0, x1) if wall in ("n", "s") else (z0, z1)
        reverse = wall in ("s", "w")
        for i in range(1, n):
            x, y, w, hh, area = stats[i]
            fw, fh = w / PX_PER_M, hh / PX_PER_M
            if fw < 0.6 or fh < 1.0 or area < 0.55 * w * hh:
                continue
            comp = lab[y:y + hh, x:x + w] == i
            if (bright | sky)[y:y + hh, x:x + w][comp].mean() < 0.25:
                continue
            rows = ((bright | sky)[y:y + hh, x:x + w] & comp).mean(1) > 0.2
            if rows.any():
                ys = np.flatnonzero(rows)
                y, hh = y + ys[0], ys[-1] - ys[0] + 1
            top = h - y / PX_PER_M
            bottom = max(h - (y + hh) / PX_PER_M, 0.0)
            a = x / PX_PER_M
            b = (x + w) / PX_PER_M
            if reverse:
                a, b = (span[1] - span[0]) - b, (span[1] - span[0]) - a
            kind = "door" if bottom < 0.15 and fw < 1.2 and top < h - 0.3 else "window"
            region = img[(lab == i) & mask]
            rl = region.astype(np.float32) @ np.array([0.299, 0.587, 0.114])
            frame = hexcol(np.median(region[rl < 70], 0)) if (rl < 70).mean() > 0.04 else None
            found.append({"wall": wall, "kind": kind, "from": round(span[0] + a, 3), "to": round(span[0] + b, 3),
                          "bottom": round(max(bottom, 0.0 if kind == "door" else 0.05), 3), "top": round(min(top, h - 0.05), 3),
                          "source": "appearance", **({"frame": frame} if frame else {})})
    return found


def ray_openings(cams, box, kind):
    found = openings(cams, box)
    if kind == "bathroom":
        found = [o for o in found if o["kind"] == "door"]
    return found


def merge_openings(a, b):
    out = list(a)
    for o in b:
        clash = any(p["wall"] == o["wall"] and min(p["to"], o["to"]) - max(p["from"], o["from"]) > 0.1 for p in out)
        if not clash:
            out.append(o)
    return out


def floor_material(img, mask):
    from shell import best_patch, PX_PER_M

    crops = []
    for size in (int(0.5 * PX_PER_M), int(0.3 * PX_PER_M)):
        for y in range(0, mask.shape[0] - size + 1, size // 2):
            for x in range(0, mask.shape[1] - size + 1, size // 2):
                if mask[y:y + size, x:x + size].mean() > 0.9:
                    crops.append(Image.fromarray(img[y:y + size, x:x + size]))
        if len(crops) >= 3:
            break
    if not crops:
        pos = best_patch(mask, int(0.3 * PX_PER_M))
        if pos:
            y, x = pos
            n = int(0.3 * PX_PER_M)
            crops.append(Image.fromarray(img[y:y + n, x:x + n]))
    crops = crops[:: max(1, len(crops) // 16)]
    seen = img[mask]
    base = robust_color(seen, np.array([150, 125, 100], np.float32))
    if not crops:
        return {"material": "wood", "color": hexcol(base), "evidence": 0.0}
    name, p = zero_shot(crops, clip()[4])
    l = seen.astype(np.float32) @ np.array([0.299, 0.587, 0.114])
    dark = seen[l < np.quantile(l, 0.3)].mean(0) if len(seen) > 50 else base * 0.8
    return {"material": name, "color": hexcol(base), "accent": hexcol(dark), "evidence": round(p, 3)}


def paint_fill(img, mask, fallback, scale=4):
    import cv2

    small = cv2.resize(img, (max(1, img.shape[1] // scale), max(1, img.shape[0] // scale)), interpolation=cv2.INTER_AREA)
    m = cv2.resize(mask.astype(np.uint8), small.shape[1::-1], interpolation=cv2.INTER_NEAREST) > 0
    out = np.empty_like(small)
    out[:] = np.asarray(fallback, np.uint8)
    px = small[m].astype(np.float32)
    if len(px) < 60:
        return out
    k = min(5, len(px) // 20)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
    cv2.setRNGSeed(0)
    _, lab, centers = cv2.kmeans(px, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    lab = lab.ravel()
    share = np.bincount(lab, minlength=k) / len(lab)
    l = centers @ np.array([0.299, 0.587, 0.114])
    chroma = centers - centers.mean(1, keepdims=True)
    sat = np.linalg.norm(chroma, axis=1)
    usable = ((l > 90) | ((l > 45) & (sat > 35))) & (l < 250) & (share > 0.05)
    if not usable.any():
        usable[np.argmax(share)] = True
    group = np.arange(k)
    for i in sorted(np.flatnonzero(usable), key=lambda i: -share[i]):
        for j in np.flatnonzero(usable):
            if share[j] > share[i] and group[j] == j and np.linalg.norm(chroma[i] - chroma[j]) < 20:
                group[i] = j
                break
    for i in range(k):
        group[i] = group[group[i]]
    paint = np.zeros(k, bool)
    colors = np.zeros((k, 3), np.uint8)
    for g in set(group[usable]):
        members = px[np.isin(lab, np.flatnonzero(usable & (group == g)))]
        if len(members) / len(px) < 0.12:
            continue
        ml = members @ np.array([0.299, 0.587, 0.114])
        colors[g] = np.median(members[ml >= np.quantile(ml, 0.3)], 0)
        paint[g] = True
    if not paint.any():
        g = int(np.argmax(share))
        paint[g], colors[g] = True, np.median(px[lab == g], 0)
    lab = np.where(usable[lab], group[lab], -1)
    labels = np.full(m.shape, -1)
    labels[m] = lab
    seed = (labels >= 0) & paint[np.maximum(labels, 0)]
    src = (~seed).astype(np.uint8)
    _, near = cv2.distanceTransformWithLabels(src, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(seed)
    ids = near[ys, xs]
    lut = np.zeros((ids.max() + 1, 3), np.uint8)
    lut[ids] = colors[labels[ys, xs]]
    return cv2.medianBlur(lut[near], 9)


def seen_alpha(mask):
    import cv2

    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.5)


def wall_fills(raw, walls, out_dir, room_id):
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for wall in ("n", "s", "w", "e"):
        img, mask = raw[f"wall-{wall}"]
        ref = [int(walls[wall][i:i + 2], 16) for i in (1, 3, 5)]
        fill = cv2.resize(paint_fill(img, mask, ref), img.shape[1::-1], interpolation=cv2.INTER_LINEAR)
        a = seen_alpha(mask)[..., None]
        paint = (img.astype(np.float32) * a + fill.astype(np.float32) * (1 - a)).clip(0, 255).astype(np.uint8)
        name = f"{room_id}-wall-{wall}-paint.jpg"
        Image.fromarray(paint).save(out_dir / name, quality=88)
        files[wall] = f"shell/{name}"
    img, mask = raw["floor"]
    rgba = np.dstack([img, (seen_alpha(mask) * 255).astype(np.uint8)])
    Image.fromarray(rgba, "RGBA").save(out_dir / f"{room_id}-floor-seen.png", optimize=True)
    files["floor"] = f"shell/{room_id}-floor-seen.png"
    return files


def wall_decals(raw, walls, out_dir, room_id, min_area=0.04):
    import cv2
    from shell import PX_PER_M

    files = {}
    for wall in ("n", "s", "w", "e"):
        img, mask = raw[f"wall-{wall}"]
        if mask.sum() < 400:
            continue
        paint = np.array([int(walls[wall][i:i + 2], 16) for i in (1, 3, 5)], np.float32)
        f = img.astype(np.float32)
        lum = f @ np.array([0.299, 0.587, 0.114], np.float32)
        near = mask & (np.linalg.norm(f - paint, axis=-1) < 40)
        num = cv2.GaussianBlur(np.where(near, lum, 0), (0, 0), 25)
        den = cv2.GaussianBlur(near.astype(np.float32), (0, 0), 25)
        shading = np.where(den > 0.02, num / np.maximum(den, 1e-3), float(paint @ np.array([0.299, 0.587, 0.114])))
        gain = float(paint @ np.array([0.299, 0.587, 0.114])) / np.maximum(shading, 20)
        flat = np.clip(f * gain[..., None], 0, 255)
        feature = mask & (np.linalg.norm(flat - paint, axis=-1) > 38)
        m = cv2.morphologyEx(feature.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        keep = np.zeros_like(m)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_area * PX_PER_M ** 2:
                keep[lab == i] = 1
        if not keep.any():
            continue
        seen = cv2.erode(mask.astype(np.uint8), np.ones((7, 7), np.uint8))
        keep = cv2.dilate(keep, np.ones((5, 5), np.uint8)) & seen
        keep = (cv2.GaussianBlur(keep.astype(np.float32), (0, 0), 6) > 0.5).astype(np.float32)
        alpha = cv2.GaussianBlur(keep, (0, 0), 3)
        rgba = np.dstack([flat.astype(np.uint8), (alpha * 255).astype(np.uint8)])
        name = f"{room_id}-wall-{wall}-decal.png"
        Image.fromarray(rgba, "RGBA").save(out_dir / name, optimize=True)
        files[wall] = f"shell/{name}"
    return files


def dominant_color(px):
    import cv2

    px = px.astype(np.float32)
    l = px @ np.array([0.299, 0.587, 0.114], np.float32)
    px = px[(l > 60) & (l < 245)]
    if len(px) < 400:
        return None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
    cv2.setRNGSeed(0)
    _, lab, centers = cv2.kmeans(px, 3, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    share = np.bincount(lab.ravel(), minlength=3) / len(lab)
    k = int(np.argmax(share))
    c = np.median(px[lab.ravel() == k], 0)
    return c if share[k] > 0.4 and c.max() - c.min() > 40 else None


def wall_colors(raw, room_ref):
    out = {}
    for wall in ("n", "s", "w", "e"):
        img, mask = raw[f"wall-{wall}"]
        px = img[mask]
        c = dominant_color(px) if len(px) > 800 else None
        if c is None and len(px) > 400:
            l = px.astype(np.float32) @ np.array([0.299, 0.587, 0.114])
            c = robust_color(px[(l > np.quantile(l, 0.2)) & (l < np.quantile(l, 0.95))], room_ref)
        out[wall] = hexcol(c if c is not None else room_ref)
    return out


def furniture(cams, box, max_items=12):
    import cv2

    (x0, x1), (z0, z1), h = box
    g = 0.05
    nx, nz = max(1, int((x1 - x0) / g)), max(1, int((z1 - z0) / g))
    pts, cols = [], []
    for c in cams:
        X, C, conf = grid_points(c, 2)
        m = (conf > 0.2) & (X[:, 0] > x0 + 0.1) & (X[:, 0] < x1 - 0.1) & (X[:, 2] > z0 + 0.1) & (X[:, 2] < z1 - 0.1)
        m &= (X[:, 1] > 0.08) & (X[:, 1] < 2.0)
        pts.append(X[m])
        cols.append(C[m])
    if sum(len(p) for p in pts) < 500:
        return []
    P, Cc = np.concatenate(pts), np.concatenate(cols).astype(np.float32)
    ix = np.clip(((P[:, 0] - x0) / g).astype(int), 0, nx - 1)
    iz = np.clip(((P[:, 2] - z0) / g).astype(int), 0, nz - 1)
    count = np.zeros((nz, nx), np.int32)
    np.add.at(count, (iz, ix), 1)
    top = np.zeros((nz, nx), np.float32)
    order = np.argsort(P[:, 1])
    top[iz[order], ix[order]] = P[order, 1]
    occ = (count >= 6).astype(np.uint8)
    occ = cv2.morphologyEx(occ, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(occ, 4)
    if DEBUG: print("   components", n - 1, "points", len(P), "occ cells", int(occ.sum()))
    items = []
    for i in range(1, n):
        x, y, w, hh, area = stats[i]
        if area * g * g < 0.08 or min(w, hh) * g < 0.2:
            continue
        cell = lab == i
        sel = cell[iz, ix]
        heights = P[sel, 1]
        if len(heights) < 80:
            continue
        H = float(np.quantile(heights, 0.92))
        lo = np.array([x0 + x * g, 0.0, z0 + y * g])
        hi = np.array([x0 + (x + w) * g, H, z0 + (y + hh) * g])
        fill = area / max(w * hh, 1)
        items.append({"lo": lo, "hi": hi, "n": int(sel.sum()), "colors": Cc[sel], "fill": fill, "Y": heights})
    items.sort(key=lambda it: -it["n"])
    out = []
    for it in items[:max_items]:
        lo, hi = it["lo"], it["hi"]
        crops = crops_for(cams, lo, hi)
        if not crops:
            if DEBUG: print("   no crop", (hi - lo).round(2))
            continue
        kind, p = zero_shot(crops, clip()[3])
        if DEBUG: print("   ", kind, round(p, 2), (hi - lo).round(2), it["n"])
        if kind in REJECT:
            continue
        colors = it["colors"]
        upper = colors[it["Y"] > np.quantile(it["Y"], 0.6)]
        src = upper if len(upper) > 30 else colors
        l = src @ np.array([0.299, 0.587, 0.114])
        body = src[(l > np.quantile(l, 0.2)) & (l < np.quantile(l, 0.9))].mean(0)
        if kind == "bed":
            lo, hi = snap_bed(lo, hi, box)
        item = {"kind": kind, "center": ((lo + hi) / 2).round(3).tolist(), "size": (hi - lo).round(3).tolist(),
                "color": hexcol(body), "evidence": round(p, 3), "points": it["n"]}
        if kind in ("bed", "sofa"):
            raised = colors[it["Y"] > (0.75 if kind == "bed" else 0.6)]
            accent = dominant_color(raised) if len(raised) > 60 else None
            if accent is not None and np.abs(accent - body).sum() > 60:
                item["accent"] = hexcol(accent)
        out.append(item)
    return out


BEDS = [(0.99, 1.91), (1.37, 1.91), (1.52, 2.03), (1.93, 2.03)]


def snap_bed(lo, hi, box):
    (x0, x1), (z0, z1), _ = box
    size = hi - lo
    along_x = size[0] >= size[2]
    length, width = (size[0], size[2]) if along_x else (size[2], size[0])
    bw, bl = min(BEDS, key=lambda b: abs(b[0] - width) + 0.5 * abs(b[1] - length))
    c = (lo + hi) / 2
    ax = 0 if along_x else 2
    walls = (x0, x1) if along_x else (z0, z1)
    gap_lo, gap_hi = lo[ax] - walls[0], walls[1] - hi[ax]
    c[ax] = lo[ax] + bl / 2 if gap_lo <= gap_hi else hi[ax] - bl / 2
    half = np.array([bl, 0, bw] if along_x else [bw, 0, bl]) / 2
    nlo, nhi = c - half, c + half
    nlo[1], nhi[1] = 0.0, min(hi[1], 0.65)
    return nlo, nhi


def crops_for(cams, lo, hi, n=2):
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    scored = []
    for c in cams:
        px, py, z = project(c, corners)
        if (z <= 0.15).sum() > 4:
            continue
        front = z > 0.15
        px, py = px[front], py[front]
        x0, x1 = np.clip([px.min(), px.max()], 0, c.W)
        y0, y1 = np.clip([py.min(), py.max()], 0, c.H)
        area = (x1 - x0) * (y1 - y0)
        full = (px.max() - px.min()) * (py.max() - py.min())
        if area < 400 or area / max(full, 1) < 0.3:
            continue
        scored.append((area, c, (x0, y0, x1, y1)))
    scored.sort(key=lambda s: -s[0])
    crops = []
    for _, c, (x0, y0, x1, y1) in scored[:n]:
        ih, iw = c.image.shape[:2]
        sx, sy = iw / c.W, ih / c.H
        pad = 0.08
        bw, bh = (x1 - x0) * pad, (y1 - y0) * pad
        box = (int(max(0, (x0 - bw) * sx)), int(max(0, (y0 - bh) * sy)), int(min(iw, (x1 + bw) * sx)), int(min(ih, (y1 + bh) * sy)))
        if box[2] - box[0] > 16 and box[3] - box[1] > 16:
            crops.append(Image.fromarray(c.image).crop(box))
    return crops


def recreate(raw, cams, box, kind=""):
    wall_px = np.concatenate([raw[k][0][raw[k][1]] for k in raw if k.startswith("wall")] or [np.zeros((0, 3))])
    ref = robust_color(wall_px, np.array([228, 226, 220], np.float32))
    fimg, fmask = raw["floor"]
    walls = wall_colors(raw, ref)
    return {
        "walls": walls,
        "wallColor": hexcol(ref),
        "floor": floor_material(fimg, fmask),
        "openings": merge_openings(ray_openings(cams, box, kind), visual_openings(raw, walls, box[2], box)),
        "furniture": furniture(cams, box),
    }
