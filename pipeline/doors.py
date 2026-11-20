import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from build_scene import align, load_run, smooth_depth

CELL = 0.05
WALLS = {"n": (2, 0, 1), "s": (2, 1, -1), "w": (0, 0, 1), "e": (0, 1, -1)}


def load_views(run: Path, room: dict):
    cams = load_run(run)
    with contextlib.redirect_stdout(io.StringIO()):
        A, origin, _, _, _ = align(cams, 1.0)
    raw = np.array([A @ (-c["R"].T @ c["t"]) - origin for c in cams])
    target = np.array([v["C"] for v in room["views"]])
    scale = float((raw * target).sum() / max((raw * raw).sum(), 1e-9))
    views = []
    for c, v in zip(cams, room["views"]):
        H, W = c["depth"].shape
        d = smooth_depth(c["depth"])
        vv, uu = np.mgrid[0:H, 0:W]
        K = c["K"]
        pc = np.stack([(uu + 0.5 - K[0, 2]) / K[0, 0] * d, (vv + 0.5 - K[1, 2]) / K[1, 1] * d, d], -1)
        X = ((pc - c["t"]) @ c["R"] @ A.T - origin) * scale
        conf = c["conf"]
        views.append({"photo": v["photo"], "file": c["file"], "crop": c.get("crop") or [0, 0, 1, 1], "C": np.array(v["C"]),
                      "X": X, "ok": conf >= np.quantile(conf, 0.05), "R": c["R"], "t": c["t"], "K": K, "A": A,
                      "origin": origin, "scale": scale, "W": W, "H": H})
    return views


def wall_frame(box, wall):
    (x0, x1), (z0, z1), h = box
    axis, side, inward = WALLS[wall]
    plane = (z0, z1)[side] if axis == 2 else (x0, x1)[side]
    along = 0 if axis == 2 else 2
    lo, hi = (x0, x1) if along == 0 else (z0, z1)
    return axis, plane, inward, along, lo, hi, h


def bin_wall(P, along, lo, nu, nv):
    iu = np.floor((P[:, along] - lo) / CELL).astype(int)
    iv = np.floor(P[:, 1] / CELL).astype(int)
    k = (iu >= 0) & (iu < nu) & (iv >= 0) & (iv < nv)
    return iu, iv, k


def through_map(views, box, wall, margin=0.3):
    axis, plane, inward, along, lo, hi, h = wall_frame(box, wall)
    nu, nv = int(np.ceil((hi - lo) / CELL)), int(np.ceil(h / CELL))
    through, solid, beyond = np.zeros((nv, nu)), np.zeros((nv, nu)), np.zeros((nv, nu))
    hits = {}
    for i, v in enumerate(views):
        C = v["C"]
        d0 = (C[axis] - plane) * inward
        if d0 <= 0.3:
            continue
        X = v["X"][v["ok"]]
        d1 = (X[:, axis] - plane) * inward
        near = np.abs(d1) < 0.12
        iu, iv, k = bin_wall(X[near], along, lo, nu, nv)
        np.add.at(solid, (iv[k], iu[k]), 1)
        cross = d1 < -margin
        t = d0 / (d0 - d1[cross])
        hit = C + (X[cross] - C) * t[:, None]
        iu, iv, k = bin_wall(hit, along, lo, nu, nv)
        np.add.at(through, (iv[k], iu[k]), 1)
        np.add.at(beyond, (iv[k], iu[k]), -d1[cross][k])
        low = k & (iv < int(1.8 / CELL))
        hits[i] = np.bincount(iu[low], minlength=nu).astype(float)
    return through, solid, beyond / np.maximum(through, 1), hits


def components(through, solid, beyond):
    import cv2

    tot = through + solid
    frac = np.where(tot >= 3, through / np.maximum(tot, 1), 0)
    m = cv2.morphologyEx((frac > 0.5).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 4)
    out = []
    for i in range(1, n):
        x, y, w, hh, area = stats[i]
        if area < 20:
            continue
        comp = lab == i
        b = beyond[comp & (through > 0)]
        out.append({"u0": x * CELL, "u1": (x + w) * CELL, "bottom": y * CELL, "top": (y + hh) * CELL,
                    "beyond": float(np.median(b)) if len(b) else 0.0, "rays": float(through[comp].sum())})
    return out


def project(v, X):
    P = X / v["scale"] + v["origin"]
    pc = P @ v["A"] @ v["R"].T + v["t"]
    z = pc[:, 2]
    K = v["K"]
    px = K[0, 0] * pc[:, 0] / np.maximum(z, 1e-6) + K[0, 2]
    py = K[1, 1] * pc[:, 1] / np.maximum(z, 1e-6) + K[1, 2]
    return px, py, z


def wall_point(box, wall, u, y):
    axis, plane, inward, along, lo, hi, h = wall_frame(box, wall)
    p = np.zeros(3)
    p[axis], p[along], p[1] = plane, lo + u, y
    return p


def photo_crop(v, box, wall, u0, u1, top, pad=0.25):
    corners = np.array([wall_point(box, wall, u, y) for u in (u0, u1) for y in (0.0, top)])
    px, py, z = project(v, corners)
    if (z <= 0.05).any():
        return None
    im = ImageOps.exif_transpose(Image.open(v["file"])).convert("RGB")
    W, H = im.size
    c = v["crop"]
    sx = lambda p: (c[0] + p / v["W"] * (c[2] - c[0])) * W
    sy = lambda p: (c[1] + p / v["H"] * (c[3] - c[1])) * H
    a, b, t, d = sx(px.min()), sx(px.max()), sy(py.min()), sy(py.max())
    pw, ph = max((b - a) * pad, ((d - t) * 0.6 - (b - a)) / 2), (d - t) * 0.08
    a, b, t, d = max(0, a - pw), min(W, b + pw), max(0, t - ph), min(H, d + ph)
    if b - a < 0.04 * W or d - t < 0.15 * H:
        return None
    return im.crop((int(a), int(t), int(b), int(d))), (b - a) / W


PROMPTS = {
    "door": ["an open doorway leading to another room", "a passage opening into a hallway", "an open doorway looking into a bathroom",
             "an open door to a hallway", "a door frame of an interior door", "a hallway seen through a doorway"],
    "closet": ["an open closet with shelves and hanging clothes", "a sliding closet door", "a wardrobe closet"],
    "mirror": ["a bathroom mirror above a vanity", "a mirror on a wall"],
    "shower": ["a glass shower enclosure", "a bathtub with a glass shower door"],
    "window": ["a window with a view outside"],
    "wall": ["a plain white wall", "a corner of a room"],
    "furniture": ["kitchen cabinets and appliances", "a bed", "a sofa"],
}

_clip = None


def clip_probs(images):
    global _clip
    import torch

    from classify import DEVICE, load_model, text_bank

    if _clip is None:
        model, pre, tok = load_model()
        _clip = (model, pre, text_bank(model, tok, PROMPTS))
    model, pre, (names, feats) = _clip
    with torch.no_grad():
        e = model.encode_image(torch.stack([pre(im) for im in images]).to(DEVICE)).float()
        e = e / e.norm(dim=-1, keepdim=True)
        p = (100 * e @ feats.T).softmax(-1).cpu().numpy()
    return [dict(zip(names, map(float, row))) for row in p]


def photo_name(v):
    return Path(v["photo"]).stem


def depth_candidates(views, box, wall):
    axis, plane, inward, along, lo, hi, h = wall_frame(box, wall)
    length = hi - lo
    through, solid, beyond, hits = through_map(views, box, wall)
    found = []
    for c in components(through, solid, beyond):
        width = c["u1"] - c["u0"]
        at_end = c["u0"] <= 0.15 or c["u1"] >= length - 0.15
        if c["rays"] < 300 or c["bottom"] > 0.6 or c["top"] - c["bottom"] < 1.2:
            continue
        if width < (0.25 if at_end else 0.5) or width > 2.6:
            continue
        a, b = int(round(c["u0"] / CELL)), int(round(c["u1"] / CELL))
        support = sorted(((float(cnt[a:b].sum()), i) for i, cnt in hits.items() if cnt[a:b].sum() > 50), reverse=True)
        found.append({**c, "wall": wall, "at_end": at_end, "support": support})
    return found


def score_candidate(cand, views, box):
    crops, weights, used = [], [], []
    top = min(max(cand["top"], 2.0), box[2])
    for w, i in cand["support"][:3]:
        res = photo_crop(views[i], box, cand["wall"], cand["u0"], cand["u1"], top)
        if res is None:
            continue
        crops.append(res[0])
        weights.append(w)
        used.append(photo_name(views[i]))
    if not crops:
        return None
    probs = clip_probs(crops)
    weights = np.array(weights) / sum(weights)
    return {k: float(sum(wt * p[k] for wt, p in zip(weights, probs))) for k in PROMPTS}, used


def wall_texture_closets(shell_dir: Path, room: dict):
    import cv2

    from shell import PX_PER_M

    (x0, x1), (z0, z1), h = room["box"]
    found = []
    for wall in "nsew":
        f = shell_dir / f"{room['id']}-wall-{wall}.jpg"
        if not f.exists():
            continue
        img = np.asarray(Image.open(f).convert("RGB")).astype(np.float32)
        seen = np.asarray(Image.open(shell_dir / f"{room['id']}-wall-{wall}-seen.png").convert("L").resize(img.shape[1::-1])) > 127
        lum = img @ np.array([0.299, 0.587, 0.114])
        sat = img.max(-1) - img.min(-1)
        if seen.mean() < 0.05:
            continue
        cand = seen & (lum < np.quantile(lum[seen], 0.9) - 50) & (lum > 50) & (sat < 60)
        m = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        span = (x0, x1) if wall in "ns" else (z0, z1)
        L = span[1] - span[0]
        for i in range(1, n):
            x, y, w, hh, area = stats[i]
            fw, fh = w / PX_PER_M, hh / PX_PER_M
            top = h - y / PX_PER_M
            if fw < 1.2 or fw > 3.2 or fh < 1.5 or top > 2.5 or top < 1.7 or area < 0.6 * w * hh:
                continue
            a, b = x / PX_PER_M, (x + w) / PX_PER_M
            if wall in "sw":
                a, b = L - b, L - a
            crop = Image.fromarray(img[max(0, y - 40):, max(0, x - 40):x + w + 40].astype(np.uint8))
            p = clip_probs([crop])[0]
            panel = p["door"] + p["closet"]
            if panel < 0.5:
                continue
            found.append({"wall": wall, "kind": "closet", "u0": a, "u1": b, "conf": panel,
                          "evidence": f"wall texture: closed dark panel {fw:.2f} m wide, {fh:.2f} m tall, top {top:.2f} m; "
                                      f"CLIP door/closet {panel:.2f}; no depth opening, too wide for a single door"})
    return found


def room_doors(run: Path, room: dict, shell_dir: Path):
    (x0, x1), (z0, z1), h = room["box"]
    views = load_views(run, room)
    out = []
    for wall in "nsew":
        lo = x0 if wall in "ns" else z0
        for cand in depth_candidates(views, room["box"], wall):
            scored = score_candidate(cand, views, room["box"])
            if scored is None:
                continue
            p, photos = scored
            opening = p["door"] + p["closet"]
            other = p["mirror"] + p["shower"] + p["window"] + p["furniture"]
            floor_door = cand["bottom"] <= 0.1 and cand["beyond"] >= 1.0 and cand["top"] <= h - 0.2
            if opening < 0.1 or (other > 0.5 and not floor_door):
                continue
            kind = "closet" if p["closet"] > p["door"] else "door"
            width = cand["u1"] - cand["u0"]
            reach = 1.0 if cand["bottom"] <= 0.1 else 0.75 if cand["bottom"] <= 0.3 else 0.5
            depth = 1.0 if kind == "closet" else min(1.0, 0.4 + cand["beyond"] / 2)
            geo = reach * depth * min(1.0, width / 0.7) * min(1.0, cand["rays"] / 3000) * (0.7 if cand["at_end"] else 1.0)
            conf = round(float(0.5 * geo + 0.5 * min(1.0, opening / 0.6)), 2)
            ev = (f"depth: rays pass {cand['beyond']:.2f} m beyond wall over {width:.2f} m, "
                  f"{cand['bottom']:.2f}-{cand['top']:.2f} m high{', clipped at wall end' if cand['at_end'] else ''}; "
                  f"CLIP door {p['door']:.2f} closet {p['closet']:.2f}; photos {','.join(photos)}")
            out.append({"wall": wall, "kind": kind, "u0": cand["u0"] + lo, "u1": cand["u1"] + lo, "conf": conf, "evidence": ev})
    for c in wall_texture_closets(shell_dir, room):
        lo = x0 if c["wall"] in "ns" else z0
        a, b = c["u0"] + lo, c["u1"] + lo
        clash = [o for o in out if o["wall"] == c["wall"] and min(o["u1"], b) - max(o["u0"], a) > 0.1]
        for o in clash:
            if o["kind"] == "closet":
                o["u0"], o["u1"] = min(o["u0"], a), max(o["u1"], b)
                o["evidence"] += f"; extended by closed panel in wall texture to {o['u1'] - o['u0']:.2f} m"
        if not clash:
            out.append({**c, "u0": a, "u1": b, "conf": round(c["conf"] * 0.7, 2)})
    return [{"wall": o["wall"], "kind": o["kind"], "from": round(o["u0"], 3), "to": round(o["u1"], 3),
             "center": round((o["u0"] + o["u1"]) / 2, 3), "confidence": o["conf"], "evidence": o["evidence"]}
            for o in sorted(out, key=lambda o: ("nsew".index(o["wall"]), o["u0"]))]


def detect_doors(listing_dir: Path, run_dirs: list[Path], rooms_json: Path) -> dict:
    rooms = json.loads(Path(rooms_json).read_text())
    runs = {Path(r).name: Path(r) for r in run_dirs}
    lid = json.loads((Path(listing_dir) / "listing.json").read_text())["id"]
    shell_dir = Path(__file__).resolve().parent.parent / "web" / "public" / "data" / lid / "shell"
    result = {}
    for room in rooms:
        if not room.get("shell") or room["id"] not in runs:
            result[room["id"]] = []
            continue
        result[room["id"]] = room_doors(runs[room["id"]], room, shell_dir)
    return result


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("listing", type=Path, nargs="?", default=root.parent / "data" / "listings" / "exhibit-a2-2bed-2bath")
    a = ap.parse_args()
    lid = json.loads((a.listing / "listing.json").read_text())["id"]
    out = root / "out" / lid
    rooms_json = out / "rooms_prelayout.json"
    runs = [out / r["id"] for r in json.loads(rooms_json.read_text()) if (out / r["id"] / "cameras.json").exists()]
    doors = detect_doors(a.listing, runs, rooms_json)
    for rid, found in doors.items():
        print(rid)
        for d in found:
            print(f"  {d['wall']} {d['kind']:6s} {d['from']:7.3f} .. {d['to']:7.3f}  conf {d['confidence']:.2f}  {d['evidence']}")
    (out / "doors.json").write_text(json.dumps(doors, indent=1))
