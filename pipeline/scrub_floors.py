"""Erase furniture footprints from photo floor textures so 3D models aren't doubled by flat ghosts.

Replaces each footprint with real wood patches sampled from the same floor, not blurry inpaint.
"""

import argparse
import json

import cv2
import numpy as np
from PIL import Image

from furnish import WEB
from walls import PX, quilt, surface


def scrub(lid):
    scene_dir = WEB / lid
    scene = json.loads((scene_dir / "scene.json").read_text())
    rng = np.random.default_rng(3)
    for room in scene["rooms"]:
        models = room.get("models") or []
        surf = (room.get("model") or {}).get("surfaces") or {}
        if not models or "floor" not in surf:
            continue
        path = scene_dir / surf["floor"]
        tex = np.asarray(Image.open(path).convert("RGB")).astype(np.float32)
        X, _, (H, W) = surface(room, "floor")
        if tex.shape[:2] != (H, W):
            tex = cv2.resize(tex, (W, H), interpolation=cv2.INTER_LINEAR)
        mask = np.zeros((H, W), np.uint8)
        for m in models:
            c, s = np.array(m["center"], float), np.array(m["size"], float)
            lo, hi = c - s / 2 - 0.05, c + s / 2 + 0.05
            inside = (X[..., 0] >= lo[0]) & (X[..., 0] <= hi[0]) & (X[..., 2] >= lo[2]) & (X[..., 2] <= hi[2])
            mask |= inside.astype(np.uint8)
        if not mask.any():
            continue
        mask = cv2.dilate(mask, np.ones((int(0.08 * PX) | 1,) * 2, np.uint8))
        keep = mask == 0
        if keep.mean() < 0.2:
            continue
        mean = tex[keep].mean(0)
        res = tex - mean
        filled = quilt(np.zeros_like(tex), mask.astype(bool), res, keep, int(0.35 * PX), rng)
        out = np.where(mask[..., None], mean + filled, tex)
        Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(path, quality=92)
        print(f"  {room['id']}: scrubbed {mask.mean():.0%} of floor under {len(models)} models")
    for room in scene["rooms"]:
        if room.get("model") and "furniture" in room["model"]:
            room["model"]["furniture"] = []
    (scene_dir / "scene.json").write_text(json.dumps(scene, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("listing")
    scrub(ap.parse_args().listing)
