import argparse
import json
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image

ROOMS = {
    "bedroom": ["a bedroom with a bed", "an empty bedroom in an apartment", "a small bedroom with a window"],
    "living": ["a living room with a couch", "an empty apartment living room", "a lounge area in an apartment unit"],
    "kitchen": ["an apartment kitchen with cabinets and a stove", "a kitchen with countertops and a refrigerator"],
    "bathroom": ["a bathroom with a toilet and sink", "a bathroom with a shower or bathtub", "a bathroom vanity and mirror"],
    "dining": ["a dining area with a table and chairs"],
    "hallway": ["a hallway inside an apartment", "an entryway with a door", "a staircase inside a house"],
    "closet": ["a closet with shelves", "a laundry room with a washer and dryer"],
    "study": ["a desk area in an apartment", "a home office"],
}
OTHER = {
    "exterior": ["the outside of a house", "an apartment building exterior", "a street view of a building", "a porch or yard"],
    "amenity": ["an apartment building gym", "a building lobby", "a swimming pool", "a rooftop terrace", "a shared study lounge"],
    "floorplan": ["a floor plan drawing", "an architectural blueprint of an apartment"],
    "graphic": ["a map", "a logo or text graphic", "a promotional banner"],
}
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_model():
    model, _, pre = open_clip.create_model_and_transforms("ViT-B-16", pretrained="laion2b_s34b_b88k", device=DEVICE)
    tok = open_clip.get_tokenizer("ViT-B-16")
    return model.eval(), pre, tok


@torch.no_grad()
def text_bank(model, tok, groups):
    names, feats = [], []
    for name, prompts in groups.items():
        f = model.encode_text(tok(prompts).to(DEVICE)).float()
        f = f / f.norm(dim=-1, keepdim=True)
        f = f.mean(0)
        feats.append(f / f.norm())
        names.append(name)
    return names, torch.stack(feats)


@torch.no_grad()
def classify(listing_dir: Path, model=None):
    if model is None:
        model = load_model()
    clip, pre, tok = model
    listing = json.loads((listing_dir / "listing.json").read_text())
    names, bank = text_bank(clip, tok, {**ROOMS, **OTHER})
    files = [p["file"] for p in listing["photos"]]
    imgs = torch.stack([pre(Image.open(listing_dir / f).convert("RGB")) for f in files]).to(DEVICE)
    emb = clip.encode_image(imgs).float()
    emb = emb / emb.norm(dim=-1, keepdim=True)
    probs = (100 * emb @ bank.T).softmax(-1).cpu().numpy()
    out = {}
    for f, p, caption in zip(files, probs, [p.get("caption") for p in listing["photos"]]):
        top = names[int(p.argmax())]
        hint = caption_hint(caption)
        label = hint or top
        kind = "interior" if label in ROOMS else label
        out[f] = {"room": label if kind == "interior" else None, "kind": kind, "score": round(float(p.max()), 3)}
    groups = group_rooms(files, out, emb.cpu().numpy())
    for f, g in groups.items():
        out[f]["group"] = g
    (listing_dir / "labels.json").write_text(json.dumps(out, indent=1))
    np.save(listing_dir / "clip.npy", emb.cpu().numpy().astype(np.float16))
    return out


def caption_hint(caption):
    if not caption:
        return None
    c = caption.lower()
    for key, words in {
        "bedroom": ["bedroom", "bed room"],
        "bathroom": ["bath", "restroom"],
        "kitchen": ["kitchen"],
        "living": ["living", "family room"],
        "floorplan": ["floor plan", "floorplan"],
        "exterior": ["exterior", "outside", "front of"],
    }.items():
        if any(w in c for w in words):
            return key
    return None


def group_rooms(files, labels, emb, threshold=0.82):
    groups, count = {}, {}
    by_type = {}
    for i, f in enumerate(files):
        if labels[f]["kind"] == "interior":
            by_type.setdefault(labels[f]["room"], []).append(i)
    for room, idx in by_type.items():
        clusters = []
        for i in idx:
            best, best_sim = None, threshold
            for c in clusters:
                sim = float(np.max(emb[c] @ emb[i]))
                if sim > best_sim:
                    best, best_sim = c, sim
            if best is None:
                clusters.append([i])
            else:
                best.append(i)
        if room in ("kitchen", "living", "dining") and len(clusters) > 1:
            clusters = [sum(clusters, [])]
        for c in clusters:
            count[room] = count.get(room, 0) + 1
            for i in c:
                groups[files[i]] = f"{room}-{count[room]}"
    return groups


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", type=Path)
    a = ap.parse_args()
    m = load_model()
    for d in a.dirs:
        res = classify(d, m)
        print(d.name)
        for f, r in res.items():
            print(f"  {f}  {r['kind']:9s} {str(r['room']):9s} {r.get('group', '')}  {r['score']}")
