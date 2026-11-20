import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).parent
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MATCH_SIDE = 768
MAX_KEYPOINTS = 2048
DINO_SIDE = 448
VOCAB_SIZE = 64

OPEN_PLAN = {"living", "kitchen", "dining"}
EXCLUDED_KINDS = {"exterior", "floorplan", "amenity"}
NO_SHELL_TYPES = {"hallway", "laundry"}
KIND_THRESHOLD = 0.5
LINK_INLIERS = 25
LINK_COVER = 0.03
MERGE_INLIERS = 100
MERGE_COVER = 0.08
COLOUR_DISTANCE = 40.0
SAME_TYPE_SIMILARITY = 0.55
SEQUENCE_WINDOW = 2
TYPE_SMOOTHING = 0.5
RUN_SIMILARITY = 0.45
CLOSEUP_COVER = 0.4
CLOSEUP_CONTEXT = 0.12
LINK_SPREAD_MIN = 0.015
LINK_SPREAD_MAX = 0.05
FLOOR_LINE = 0.6
LINK_CONFIDENT = 40

KIND_PROMPTS = {
    "interior": [
        "a photo of a room inside a house", "an empty room in an apartment", "a photo of an apartment interior",
        "a modern apartment living room with large windows", "a furnished apartment with a kitchen and sofa",
        "a hallway inside a house", "a staircase inside a house", "a basement room", "a bathroom", "a kitchen", "a bedroom",
    ],
    "exterior": [
        "the outside of a house", "an apartment building exterior", "a street view of a house",
        "a front porch and yard", "a backyard of a house", "a driveway or parking lot",
    ],
    "floorplan": ["a floor plan drawing", "an architectural blueprint of an apartment", "a black and white floor plan diagram"],
    "amenity": [
        "a fitness center with exercise machines", "an apartment building lobby with a reception desk",
        "an outdoor swimming pool", "a rooftop terrace with outdoor furniture", "a building mailroom with mailboxes",
    ],
    "detail": ["a close-up photo of an appliance", "a close-up of a thermostat or light switch", "a close-up of a faucet", "a map or logo graphic"],
}

REGION_PROMPTS = {
    "doorway": [
        "an open doorway into another room", "a view through a door into a bedroom", "a view through a doorway into a kitchen",
        "a hallway seen through an open door", "a room seen through a wide opening",
    ],
    "window": [
        "a window with a view of the outside", "buildings seen through a window", "sky and trees through a window",
        "a glass block window", "a city skyline through a window", "a window with white blinds", "window blinds and a window frame",
    ],
    "surface": ["a blank wall", "a ceiling with a light fixture", "a framed picture on a wall", "a piece of furniture"],
}

ROOM_PROMPTS = {
    "living": ["a living room", "a living room with a couch", "an empty living room with a fireplace", "a lounge area with a sofa and tv"],
    "kitchen": ["a kitchen", "a kitchen with cabinets and a stove", "a kitchen with countertops and a refrigerator", "a small kitchenette"],
    "dining": ["a dining room", "a dining area with a table and chairs"],
    "bedroom": ["a bedroom", "a bedroom with a bed", "an empty bedroom with carpet and a closet", "a small bedroom with a window"],
    "bathroom": ["a bathroom", "a bathroom with a toilet and sink", "a bathroom with a shower or bathtub", "a bathroom vanity and mirror"],
    "hallway": ["a hallway", "a narrow hallway with doors", "a staircase", "an entryway with a front door", "a landing at the top of the stairs"],
    "laundry": ["a laundry room with a washer and dryer", "a utility room with a water heater"],
}


class Models:
    def __init__(self, device=DEVICE):
        self.device = device
        self._clip = self._dino = self._disk = self._glue = None

    @property
    def clip(self):
        if self._clip is None:
            import open_clip
            model, _, pre = open_clip.create_model_and_transforms("ViT-B-16", pretrained="laion2b_s34b_b88k", device=self.device)
            self._clip = (model.eval(), pre, open_clip.get_tokenizer("ViT-B-16"))
        return self._clip

    @property
    def dino(self):
        if self._dino is None:
            from transformers import AutoImageProcessor, AutoModel
            name = "facebook/dinov2-base"
            self._dino = (AutoModel.from_pretrained(name).to(self.device).eval(), AutoImageProcessor.from_pretrained(name))
        return self._dino

    @property
    def disk(self):
        if self._disk is None:
            import kornia.feature as KF
            self._disk = KF.DISK.from_pretrained("depth").to(self.device).eval()
        return self._disk

    @property
    def glue(self):
        if self._glue is None:
            import kornia.feature as KF
            self._glue = KF.LightGlueMatcher("disk").to(self.device).eval()
        return self._glue


def list_photos(listing_dir: Path):
    listing = json.loads((listing_dir / "listing.json").read_text())
    names, paths = [], []
    for p in listing["photos"]:
        path = listing_dir / p["file"]
        if path.exists():
            names.append(path.stem)
            paths.append(path)
    return names, paths


def load_rgb(path, side=None):
    im = Image.open(path).convert("RGB")
    if side:
        s = side / max(im.size)
        im = im.resize((max(16, round(im.width * s)), max(16, round(im.height * s))), Image.BILINEAR)
    return im


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


@torch.inference_mode()
def text_bank(models, groups):
    clip, _, tok = models.clip
    names, feats = [], []
    for name, prompts in groups.items():
        f = clip.encode_text(tok(prompts).to(models.device)).float()
        f = f / f.norm(dim=-1, keepdim=True)
        f = f.mean(0)
        names.append(name)
        feats.append((f / f.norm()).cpu().numpy())
    return names, np.stack(feats)


@torch.inference_mode()
def embed(models, paths):
    clip, pre, _ = models.clip
    dino, proc = models.dino
    size = {"shortest_edge": DINO_SIDE}
    crop = {"height": DINO_SIDE, "width": DINO_SIDE}
    clip_emb, dino_cls, tokens = [], [], []
    for path in paths:
        im = load_rgb(path)
        clip_emb.append(clip.encode_image(pre(im)[None].to(models.device)).float().cpu().numpy()[0])
        x = proc(images=im, return_tensors="pt", size=size, crop_size=crop).to(models.device)
        out = dino(**x).last_hidden_state[0].float().cpu().numpy()
        dino_cls.append(out[0])
        tokens.append(unit(out[1:]))
    return unit(np.stack(clip_emb)), unit(np.stack(dino_cls)), tokens


def visual_words(tokens):
    from scipy.cluster.vq import kmeans2, vq
    data = np.concatenate(tokens)[::2].astype(np.float64)
    centroids, _ = kmeans2(data, VOCAB_SIZE, seed=0, minit="++")
    hist = np.stack([np.bincount(vq(t.astype(np.float64), centroids)[0], minlength=VOCAB_SIZE) for t in tokens]).astype(float)
    idf = np.log(len(tokens) / (1 + (hist > 0).sum(0))) + 0.01
    vec = unit(np.sqrt(hist) * idf)
    return vec @ vec.T


def zero_shot(models, clip_emb, groups, temperature=100.0):
    names, bank = text_bank(models, groups)
    logits = temperature * clip_emb @ bank.T
    probs = np.exp(logits - logits.max(1, keepdims=True))
    return names, probs / probs.sum(1, keepdims=True)


@torch.inference_mode()
def local_features(models, paths):
    import kornia as K
    feats = []
    for path in paths:
        im = load_rgb(path, MATCH_SIDE)
        x = K.image_to_tensor(np.asarray(im).copy(), False).float().div(255).to(models.device)
        f = models.disk(x, n=MAX_KEYPOINTS, pad_if_not_divisible=True)[0]
        feats.append((f.keypoints, f.descriptors, (im.height, im.width)))
    return feats


@torch.inference_mode()
def match_pair(models, fa, fb):
    import kornia.feature as KF
    (ka, da, sa), (kb, db, sb) = fa, fb
    lafa = KF.laf_from_center_scale_ori(ka[None], torch.ones(1, len(ka), 1, 1, device=models.device))
    lafb = KF.laf_from_center_scale_ori(kb[None], torch.ones(1, len(kb), 1, 1, device=models.device))
    _, idx = models.glue(da, db, lafa, lafb, hw1=torch.tensor(sa), hw2=torch.tensor(sb))
    idx = idx.cpu().numpy()
    pa = ka.cpu().numpy()[idx[:, 0]]
    pb = kb.cpu().numpy()[idx[:, 1]]
    if len(pa) < 12:
        return np.zeros((0, 4), np.float32)
    _, mask = cv2.findFundamentalMat(pa, pb, cv2.USAC_MAGSAC, 1.0, 0.999, 10000)
    if mask is None:
        return np.zeros((0, 4), np.float32)
    keep = mask.ravel().astype(bool)
    norm_a = np.array([sa[1], sa[0]], np.float32)
    norm_b = np.array([sb[1], sb[0]], np.float32)
    return np.concatenate([pa[keep] / norm_a, pb[keep] / norm_b], 1).astype(np.float32)


def match_all(models, paths):
    feats = local_features(models, paths)
    pairs = {}
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            pairs[(i, j)] = match_pair(models, feats[i], feats[j])
    return pairs


def compute_signals(listing_dir: Path, cache_dir: Path, models=None, fresh=False):
    names, paths = list_photos(listing_dir)
    cache = cache_dir / "signals.npz"
    if cache.exists() and not fresh:
        z = np.load(cache, allow_pickle=False)
        if list(z["names"]) == names and "words" in z.files:
            pairs = {}
            for key in z.files:
                if key.startswith("m_"):
                    i, j = map(int, key[2:].split("_"))
                    pairs[(i, j)] = z[key]
            return {"names": names, "paths": paths, "clip": z["clip"], "dino_cls": z["dino_cls"], "words": z["words"], "pairs": pairs}
    models = models or Models()
    clip_emb, dino_cls, tokens = embed(models, paths)
    words = visual_words(tokens)
    pairs = match_all(models, paths)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache, names=np.array(names), clip=clip_emb, dino_cls=dino_cls, words=words,
        **{f"m_{i}_{j}": m for (i, j), m in pairs.items()},
    )
    return {"names": names, "paths": paths, "clip": clip_emb, "dino_cls": dino_cls, "words": words, "pairs": pairs}


def hull_area(pts, keep_pct=95):
    if len(pts) < 4:
        return 0.0
    d = np.linalg.norm(pts - np.median(pts, 0), axis=1)
    keep = pts[d <= np.percentile(d, keep_pct)]
    if len(keep) < 3:
        return 0.0
    return float(cv2.contourArea(cv2.convexHull(keep.astype(np.float32))))


def pair_geometry(pairs, n):
    inliers = np.zeros((n, n))
    cover = np.zeros((n, n))
    for (i, j), m in pairs.items():
        inliers[i, j] = inliers[j, i] = len(m)
        cover[i, j] = hull_area(m[:, :2])
        cover[j, i] = hull_area(m[:, 2:])
    return inliers, cover


def classify(models, sig):
    kind_names, kind_probs = zero_shot(models, sig["clip"], KIND_PROMPTS)
    room_names, room_probs = zero_shot(models, sig["clip"], ROOM_PROMPTS)
    kinds = []
    for p in kind_probs:
        top = int(p.argmax())
        kinds.append(kind_names[top] if kind_names[top] != "interior" and p[top] >= KIND_THRESHOLD else "interior")
    return kinds, room_names, room_probs


def colour_signature(path):
    lab = cv2.cvtColor(np.asarray(load_rgb(path, 256)), cv2.COLOR_RGB2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    floor = lab[int(h * 0.8):, int(w * 0.2):int(w * 0.8)].reshape(-1, 3)
    wall = lab[int(h * 0.25):int(h * 0.55)].reshape(-1, 3)
    return np.median(floor, 0), np.median(wall, 0)


def colour_distance(paths):
    sigs = [colour_signature(p) for p in paths]
    floor = np.stack([f for f, _ in sigs])
    wall = np.stack([w for _, w in sigs])
    dist = lambda x: np.linalg.norm(x[:, None] - x[None], axis=-1)
    return np.maximum(dist(floor), dist(wall))


def evidence(sig):
    n = len(sig["names"])
    inliers, cover = pair_geometry(sig["pairs"], n)
    overlap = np.minimum(cover, cover.T)
    linked = (inliers >= LINK_INLIERS) & (overlap >= LINK_COVER)
    return {
        "inliers": inliers, "cover": cover, "overlap": overlap, "linked": linked,
        "similarity": (sig["dino_cls"] @ sig["dino_cls"].T + sig["words"]) / 2,
        "colour": colour_distance(sig["paths"]),
    }


def closeups(ev, keep):
    parent = {}
    for i in keep:
        best = None
        for j in keep:
            if i == j or ev["inliers"][i, j] < LINK_INLIERS:
                continue
            if ev["cover"][i, j] >= CLOSEUP_COVER and ev["cover"][j, i] <= CLOSEUP_CONTEXT:
                if best is None or ev["inliers"][i, j] > ev["inliers"][i, best]:
                    best = j
        if best is not None:
            parent[i] = best
    return parent


def smoothed_types(room_names, room_probs, ev):
    weights = np.eye(len(room_probs)) + TYPE_SMOOTHING * ev["linked"]
    probs = weights @ room_probs
    probs /= probs.sum(1, keepdims=True)
    return [room_names[i] for i in probs.argmax(1)], probs.max(1)


def compatible(a, b):
    return a == b or (a in OPEN_PLAN and b in OPEN_PLAN)


def dominant(types, members):
    votes = [types[i] for i in members]
    return max(sorted(set(votes)), key=votes.count)


def segment(keep, types, kinds, ev):
    runs = [[keep[0]]]
    for b in keep[1:]:
        current = runs[-1]
        recent = current[-SEQUENCE_WINDOW:]
        geometric = any(ev["linked"][a, b] and compatible(types[a], types[b]) for a in recent)
        looks_same = any(
            types[a] == types[b] and ev["similarity"][a, b] >= SAME_TYPE_SIMILARITY and ev["colour"][a, b] <= COLOUR_DISTANCE
            for a in recent
        )
        fits_run = ev["similarity"][current, b].mean() >= RUN_SIMILARITY
        detail = types[current[-1]] == types[b] and "detail" in (kinds[current[-1]], kinds[b])
        if geometric or detail or (looks_same and fits_run):
            current.append(b)
        else:
            runs.append([b])
    return runs


def merge_runs(runs, types, ev, parent):
    groups = [list(r) for r in runs]
    changed = True
    while changed:
        changed = False
        for x in range(len(groups)):
            for y in range(x + 1, len(groups)):
                a, b = groups[x], groups[y]
                detail = any(parent.get(i) in b for i in a) or any(parent.get(i) in a for i in b)
                strong = any(
                    ev["inliers"][i, j] >= MERGE_INLIERS and ev["overlap"][i, j] >= MERGE_COVER for i in a for j in b
                )
                if detail or (strong and compatible(dominant(types, a), dominant(types, b))):
                    groups[x] = sorted(a + b)
                    del groups[y]
                    changed = True
                    break
            if changed:
                break
    return groups


def reconstruction_order(members, ev):
    if len(members) == 1:
        return list(members)
    sub = ev["inliers"][np.ix_(members, members)]
    order = [int(np.argmax(sub.sum(1)))]
    rest = [k for k in range(len(members)) if k != order[0]]
    while rest:
        nxt = max(rest, key=lambda k: (sub[order, k].max(), -k))
        order.append(nxt)
        rest.remove(nxt)
    return [members[k] for k in order]


def room_type(types, members):
    present = {types[i] for i in members}
    if len(present) > 1 and present <= OPEN_PLAN and "living" in present:
        return "living"
    return dominant(types, members)


def name_rooms(groups, types):
    typed = [(room_type(types, g), g) for g in groups]
    totals = {}
    for t, _ in typed:
        totals[t] = totals.get(t, 0) + 1
    seen, named = {}, []
    for t, g in typed:
        seen[t] = seen.get(t, 0) + 1
        base = "hall" if t == "hallway" else t
        named.append((base if totals[t] == 1 else f"{base}-{seen[t]}", t, g))
    return named


def box(points):
    lo = np.percentile(points, 3, axis=0)
    hi = np.percentile(points, 97, axis=0)
    return [round(float(v), 4) for v in (*lo, *hi)]


def box_area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def matched_regions(sig, i, j):
    m = sig["pairs"][(min(i, j), max(i, j))]
    pi, pj = (m[:, :2], m[:, 2:]) if i < j else (m[:, 2:], m[:, :2])
    return box(pi), box(pj)


def same_pairs(sig, room_of):
    out = []
    for (i, j), m in sig["pairs"].items():
        if room_of.get(i) is None or room_of.get(i) != room_of.get(j) or len(m) < LINK_INLIERS:
            continue
        bi, bj = matched_regions(sig, i, j)
        ai, aj = box_area(bi), box_area(bj)
        relation = "closeup" if min(ai, aj) < 0.3 * max(ai, aj) else "overlap"
        entry = {
            "photos": [sig["names"][i], sig["names"][j]], "room": room_of[i], "relation": relation,
            "inliers": int(len(m)), "regions": [bi, bj],
        }
        if relation == "closeup":
            entry["closeup"] = sig["names"][i] if ai > aj else sig["names"][j]
        out.append(entry)
    return sorted(out, key=lambda e: (e["room"], -e["inliers"]))


@torch.inference_mode()
def region_kinds(models, sig, regions):
    if not regions:
        return []
    clip, pre, _ = models.clip
    crops = []
    for i, b in regions:
        im = load_rgb(sig["paths"][i])
        w, h = im.size
        cx, cy = (b[0] + b[2]) / 2 * w, (b[1] + b[3]) / 2 * h
        half = max((b[2] - b[0]) * w, (b[3] - b[1]) * h, 0.2 * min(w, h)) * 0.65
        crops.append(pre(im.crop((int(cx - half), int(cy - half), int(cx + half), int(cy + half))))[None])
    emb = torch.cat([clip.encode_image(c.to(models.device)).float() for c in crops])
    emb = unit(emb.cpu().numpy())
    names, probs = zero_shot(models, emb, REGION_PROMPTS)
    return [names[int(p.argmax())] for p in probs]


def floor_only(points):
    return float(np.median(points[:, 1])) > FLOOR_LINE and float(np.percentile(points[:, 1], 10)) > FLOOR_LINE - 0.1


def cross_links(models, sig, room_of):
    links, viewers = [], []
    for (i, j), m in sig["pairs"].items():
        ri, rj = room_of.get(i), room_of.get(j)
        if ri is None or rj is None or ri == rj or len(m) < LINK_INLIERS:
            continue
        spread = sorted((hull_area(m[:, :2], 80), hull_area(m[:, 2:], 80)))
        if spread[0] < LINK_SPREAD_MIN or spread[1] < LINK_SPREAD_MAX:
            continue
        bi, bj = matched_regions(sig, i, j)
        viewer, target = (i, j) if box_area(bi) <= box_area(bj) else (j, i)
        bv, bt = (bi, bj) if viewer == i else (bj, bi)
        confident = len(m) >= LINK_CONFIDENT and not (floor_only(m[:, :2]) and floor_only(m[:, 2:]))
        links.append({
            "photo": sig["names"][viewer], "room": room_of[viewer],
            "sees_room": room_of[target], "match_photo": sig["names"][target],
            "kind": "doorway" if box_area(bv) < 0.5 else "overlap",
            "inliers": int(len(m)), "confidence": "high" if confident else "low",
            "region": bv, "match_region": bt,
        })
        viewers.append(viewer)
    seen = region_kinds(models, sig, [(v, link["region"]) for v, link in zip(viewers, links)])
    links = [dict(link, region_kind=k) for link, k in zip(links, seen)]
    return sorted([l for l in links if l["region_kind"] != "window"], key=lambda e: -e["inliers"])


def adjacency(links):
    pairs = {}
    for link in links:
        if link["confidence"] != "high":
            continue
        key = tuple(sorted((link["room"], link["sees_room"])))
        entry = pairs.setdefault(key, {"rooms": list(key), "links": 0, "inliers": 0, "photos": set()})
        entry["links"] += 1
        entry["inliers"] = max(entry["inliers"], link["inliers"])
        entry["photos"].add(link["photo"])
    out = []
    for entry in pairs.values():
        entry["photos"] = sorted(entry["photos"])
        out.append(entry)
    return sorted(out, key=lambda e: -e["inliers"])


def associate(listing_dir, models=None, fresh=False):
    listing_dir = Path(listing_dir).resolve()
    stage_dir = ROOT / "out" / listing_dir.name / "stages" / "2-associate"
    models = models or Models()
    sig = compute_signals(listing_dir, stage_dir, models, fresh)
    names = sig["names"]
    kinds, room_names, room_probs = classify(models, sig)
    ev = evidence(sig)
    keep = [i for i, k in enumerate(kinds) if k not in EXCLUDED_KINDS]
    parent = closeups(ev, keep)
    for i in parent:
        kinds[i] = "detail"
    types, type_conf = smoothed_types(room_names, room_probs, ev)
    groups = merge_runs(segment(keep, types, kinds, ev), types, ev, parent) if keep else []
    groups.sort(key=lambda g: g[0])
    rooms, room_of = [], {}
    for rid, rtype, members in name_rooms(groups, types):
        order = reconstruction_order(members, ev)
        all_detail = all(kinds[i] == "detail" for i in members)
        mixed = sorted({types[i] for i in members})
        room = {
            "id": rid, "type": rtype, "photos": [names[i] for i in order],
            "shell": not (rtype in NO_SHELL_TYPES or all_detail),
        }
        if len(mixed) > 1:
            room["label"] = " and ".join(mixed)
        rooms.append(room)
        for i in members:
            room_of[i] = rid
    photos = {}
    for i, n in enumerate(names):
        photos[n] = {
            "file": str(sig["paths"][i].relative_to(listing_dir)), "kind": kinds[i],
            "room_type": types[i] if i in room_of else None, "type_confidence": round(float(type_conf[i]), 3),
            "room": room_of.get(i),
        }
        if i in parent:
            photos[n]["detail_of"] = names[parent[i]]
    links = cross_links(models, sig, room_of)
    return {
        "listing": listing_dir.name, "photos": photos, "rooms": rooms,
        "same": same_pairs(sig, room_of), "links": links, "adjacency": adjacency(links),
    }


def rooms_json(result):
    out = {}
    for r in result["rooms"]:
        entry = {"type": r["type"], "photos": r["photos"]}
        if "label" in r:
            entry["label"] = r["label"]
        if not r["shell"]:
            entry["shell"] = False
        out[r["id"]] = entry
    return out


PALETTE = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
           (128, 128, 0), (0, 128, 128), (170, 110, 40), (128, 0, 0), (0, 0, 128), (255, 200, 0), (100, 100, 100)]


def font(size):
    from PIL import ImageFont
    for path in ("/System/Library/Fonts/Helvetica.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def thumb(path, height):
    im = load_rgb(path)
    return im.resize((max(1, round(im.width * height / im.height)), height), Image.BILINEAR)


def tag(draw, xy, text, size=18, fill=(255, 255, 0)):
    f = font(size)
    x, y = xy
    l, t, r, b = draw.textbbox((x, y), text, font=f)
    draw.rectangle([l - 3, t - 2, r + 3, b + 2], fill=(0, 0, 0))
    draw.text((x, y), text, font=f, fill=fill)


def room_block(title, colour, photos, result, listing_dir, height, width):
    from PIL import ImageDraw
    thumbs = [(n, thumb(listing_dir / result["photos"][n]["file"], height)) for n in photos]
    lines, line, x = [], [], 0
    for n, t in thumbs:
        if line and x + t.width > width:
            lines.append(line)
            line, x = [], 0
        line.append((n, t))
        x += t.width + 6
    lines.append(line)
    w = max(sum(t.width + 6 for _, t in ln) for ln in lines) + 20
    w = max(w, int(font(22).getlength(title)) + 30)
    block = Image.new("RGB", (w, 40 + len(lines) * (height + 6)), (250, 250, 250))
    draw = ImageDraw.Draw(block)
    draw.rectangle([0, 0, 8, block.height], fill=colour)
    draw.text((16, 8), title, font=font(22), fill=(20, 20, 20))
    y = 40
    for ln in lines:
        x = 16
        for n, t in ln:
            block.paste(t, (x, y))
            p = result["photos"][n]
            label = n if p["room"] is None else f"{n} {p['room_type']}"
            if p["kind"] != "interior":
                label += f" [{p['kind']}{' of ' + p['detail_of'] if p.get('detail_of') else ''}]"
            tag(draw, (x + 4, y + 4), label, size=16)
            x += t.width + 6
        y += height + 6
    return block


def rooms_sheet(result, listing_dir, height=170, width=1800):
    blocks = [room_block(f"{r['id']}  ({r['type']}{', no shell' if not r['shell'] else ''})", PALETTE[k % len(PALETTE)],
                         r["photos"], result, listing_dir, height, width)
              for k, r in enumerate(result["rooms"])]
    excluded = [n for n, p in result["photos"].items() if p["room"] is None]
    if excluded:
        blocks.append(room_block("excluded", (90, 90, 90), excluded, result, listing_dir, height, width))
    return stack(blocks, width, gap=12)


def pair_panel(listing_dir, result, sig, a, b, regions, caption, colour, height=260, lines=40):
    from PIL import ImageDraw
    names = sig["names"]
    ia, ib = names.index(a), names.index(b)
    ta, tb = thumb(listing_dir / result["photos"][a]["file"], height), thumb(listing_dir / result["photos"][b]["file"], height)
    panel = Image.new("RGB", (ta.width + tb.width + 30, height + 34), (255, 255, 255))
    panel.paste(ta, (0, 34))
    panel.paste(tb, (ta.width + 30, 34))
    draw = ImageDraw.Draw(panel)
    draw.text((4, 6), caption, font=font(20), fill=(20, 20, 20))
    m = sig["pairs"][(min(ia, ib), max(ia, ib))]
    pa, pb = (m[:, :2], m[:, 2:]) if ia < ib else (m[:, 2:], m[:, :2])
    step = max(1, len(pa) // lines)
    for (xa, ya), (xb, yb) in zip(pa[::step], pb[::step]):
        draw.line([(xa * ta.width, 34 + ya * height), (ta.width + 30 + xb * tb.width, 34 + yb * height)], fill=colour, width=1)
    for (bx, off, w) in ((regions[0], 0, ta.width), (regions[1], ta.width + 30, tb.width)):
        draw.rectangle([off + bx[0] * w, 34 + bx[1] * height, off + bx[2] * w, 34 + bx[3] * height], outline=colour, width=4)
    tag(draw, (4, 38), a)
    tag(draw, (ta.width + 34, 38), b)
    return panel


def stack(panels, width=1800, gap=10):
    if not panels:
        return None
    rows, row, x = [], [], 0
    for p in panels:
        if row and x + p.width > width:
            rows.append(row)
            row, x = [], 0
        row.append(p)
        x += p.width + gap
    rows.append(row)
    height = sum(max(p.height for p in r) + gap for r in rows)
    sheet = Image.new("RGB", (width, height), (235, 235, 235))
    y = 0
    for r in rows:
        x = 0
        for p in r:
            sheet.paste(p, (x, y))
            x += p.width + gap
        y += max(p.height for p in r) + gap
    return sheet


def write_sheets(result, listing_dir, sig, stage_dir, limit=40):
    listing_dir = Path(listing_dir)
    rooms_sheet(result, listing_dir).save(stage_dir / "rooms.jpg", quality=85)
    colours = {r["id"]: PALETTE[k % len(PALETTE)] for k, r in enumerate(result["rooms"])}
    same = [
        pair_panel(listing_dir, result, sig, *s["photos"], s["regions"],
                   f"{s['room']}: {s['relation']}{' (close-up ' + s['closeup'] + ')' if 'closeup' in s else ''}, {s['inliers']} matches",
                   colours[s["room"]])
        for s in result["same"][:limit]
    ]
    links = [
        pair_panel(listing_dir, result, sig, l["photo"], l["match_photo"], [l["region"], l["match_region"]],
                   f"{l['room']} sees {l['sees_room']} ({l['region_kind']}, {l['confidence']}), {l['inliers']}", colours[l["sees_room"]])
        for l in result["links"][:limit]
    ]
    for name, panels in (("same.jpg", same), ("links.jpg", links)):
        sheet = stack(panels)
        if sheet is not None:
            sheet.save(stage_dir / name, quality=85)
        elif (stage_dir / name).exists():
            (stage_dir / name).unlink()


def run(listing_dir, models=None, fresh=False):
    listing_dir = Path(listing_dir).resolve()
    stage_dir = ROOT / "out" / listing_dir.name / "stages" / "2-associate"
    models = models or Models()
    result = associate(listing_dir, models, fresh)
    out_dir = ROOT / "out" / listing_dir.name
    (out_dir / "associate.json").write_text(json.dumps(result, indent=1))
    (stage_dir / "rooms.json").write_text(json.dumps(rooms_json(result), indent=1))
    write_sheets(result, listing_dir, compute_signals(listing_dir, stage_dir, models), stage_dir)
    return result


def main():
    ap = argparse.ArgumentParser(description="group listing photos into rooms and find cross-photo links")
    ap.add_argument("listings", nargs="+", type=Path)
    ap.add_argument("--fresh", action="store_true", help="recompute embeddings and matches")
    args = ap.parse_args()
    models = Models()
    for listing_dir in args.listings:
        result = run(listing_dir, models, args.fresh)
        print(listing_dir.name)
        for r in result["rooms"]:
            print(f"  {r['id']:12s} {r['type']:9s} {'shell' if r['shell'] else '-':5s} {' '.join(r['photos'])}")
        skipped = [n for n, p in result["photos"].items() if p["room"] is None]
        print(f"  excluded: {' '.join(skipped) or '-'}  same: {len(result['same'])}  links: {len(result['links'])}")


if __name__ == "__main__":
    main()
