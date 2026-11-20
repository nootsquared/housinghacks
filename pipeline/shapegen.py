"""Image cutout -> solid mesh with Hunyuan3D-2mini-Turbo (step-distilled, 5 steps) and FlashVDM decoding on MPS.

Run in the third_party Hunyuan venv; the model loads once for every cutout given.
"""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
_cache = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))
if (_cache / "models--tencent--Hunyuan3D-2mini" / "snapshots").exists():
    # the loader re-lists the hub repo on every start, which can stall for many minutes
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline  # noqa: E402

MODEL, SUBFOLDER = "tencent/Hunyuan3D-2mini", "hunyuan3d-dit-v2-mini-turbo"


def generate(images: list[Path], steps=5, octree=256, seed=1234):
    t = time.time()
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(MODEL, subfolder=SUBFOLDER, use_safetensors=False,
                                                            device="mps", dtype=torch.float16)
    pipe.enable_flashvdm(topk_mode="merge")
    print(f"shape model ready in {time.time() - t:.0f}s", flush=True)
    for img in images:
        t = time.time()
        for s in (seed, seed + 1):
            try:
                mesh = pipe(image=Image.open(img).convert("RGBA"), num_inference_steps=steps, octree_resolution=octree,
                            num_chunks=20000, generator=torch.manual_seed(s), output_type="trimesh")[0]
            except (IndexError, ValueError, RuntimeError) as e:
                # an empty occupancy field (nothing above the iso level) surfaces as an empty reduction
                print(f"{img.name}: seed {s} gave no surface ({e})", flush=True)
                continue
            mesh.export(img.with_suffix(".glb"))
            print(f"{img.name}: {len(mesh.faces)} faces, {time.time() - t:.0f}s", flush=True)
            break


if __name__ == "__main__":
    generate([Path(p) for p in sys.argv[1:]])
