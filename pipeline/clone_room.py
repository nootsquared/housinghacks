"""Copy a photographed room's surfaces and furniture models onto its unphotographed clone."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import trimesh

from furnish import WEB


def clone(lid, src_id, dst_id):
    scene_dir = WEB / lid
    scene = json.loads((scene_dir / "scene.json").read_text())
    src = next(r for r in scene["rooms"] if r["id"] == src_id)
    dst = next(r for r in scene["rooms"] if r["id"] == dst_id)
    off = np.array(dst["offset"], float) - np.array(src["offset"], float)
    (scene_dir / "models").mkdir(exist_ok=True)
    (scene_dir / "shell").mkdir(exist_ok=True)
    if src.get("model", {}).get("surfaces"):
        files = {}
        for sid, rel in src["model"]["surfaces"].items():
            src_p = scene_dir / rel
            dst_rel = f"shell/{dst_id}-surf-{sid}.jpg"
            shutil.copy2(src_p, scene_dir / dst_rel)
            files[sid] = dst_rel
        dst.setdefault("model", src["model"]).update({"surfaces": files, "surfaceCover": src["model"].get("surfaceCover")})
        for k in ("walls", "wallColor", "floor"):
            if k in src["model"]:
                dst["model"][k] = src["model"][k]
    models = []
    for m in src.get("models", []):
        mesh = trimesh.load(scene_dir / m["file"], force="mesh")
        mesh.apply_translation(off)
        name = Path(m["file"]).name.replace(src_id, dst_id, 1)
        out = f"models/{name}"
        mesh.export(scene_dir / out)
        models.append({**m, "file": out, "center": (np.array(m["center"]) + off).round(3).tolist()})
    dst["models"] = models
    dst["cloneOf"] = src_id
    (scene_dir / "scene.json").write_text(json.dumps(scene, indent=1))
    print(f"{dst_id}: {len(models)} models, surfaces={bool(dst.get('model', {}).get('surfaces'))}, off={off.round(3).tolist()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("listing")
    ap.add_argument("src")
    ap.add_argument("dst")
    a = ap.parse_args()
    clone(a.listing, a.src, a.dst)
