import argparse
import json
import resource
import sys
import threading
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def collect_images(inputs):
    files = []
    for p in map(Path, inputs):
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.suffix.lower() in IMG_EXT)
        elif p.suffix.lower() == ".txt":
            files += [Path(l.strip()) for l in p.read_text().splitlines() if l.strip()]
        else:
            files.append(p)
    if not files:
        sys.exit("no images found")
    return files


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class MemPeak:
    def __init__(self, device):
        self.device, self.peak, self._stop = device, 0, threading.Event()

    def _sample(self):
        if self.device.type == "mps":
            return torch.mps.driver_allocated_memory()
        return 0

    def __enter__(self):
        def loop():
            while not self._stop.is_set():
                self.peak = max(self.peak, self._sample())
                time.sleep(0.05)
        self.t = threading.Thread(target=loop, daemon=True)
        self.t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self.t.join()
        self.peak = max(self.peak, self._sample())


_MODEL = {}


def load_model(device, model_id="facebook/VGGT-1B"):
    key = (str(device), model_id)
    if key not in _MODEL:
        m = VGGT.from_pretrained(model_id)
        m.track_head = None
        m.point_head = None
        _MODEL[key] = m.to(device).eval()
    return _MODEL[key]


@torch.no_grad()
def run_vggt(model, images, device, dtype):
    images = images.to(device)[None]
    if dtype == torch.float32:
        tokens, ps = model.aggregator(images)
    else:
        with torch.autocast(device_type=device.type, dtype=dtype):
            tokens, ps = model.aggregator(images)
        tokens = [t.float() if t is not None else t for t in tokens]
    pose_enc = model.camera_head(tokens)[-1]
    depth, conf = model.depth_head(tokens, images=images, patch_start_idx=ps)
    extr, intr = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
    if device.type == "mps":
        torch.mps.synchronize()
    return {
        "extrinsic": extr[0].cpu().numpy(),
        "intrinsic": intr[0].cpu().numpy(),
        "depth": depth[0, ..., 0].cpu().numpy(),
        "conf": conf[0].cpu().numpy(),
    }


def build_cloud(pred, images, keep, max_points):
    pts = unproject_depth_map_to_point_map(pred["depth"][..., None], pred["extrinsic"], pred["intrinsic"])
    cols = images.permute(0, 2, 3, 1).cpu().numpy()
    conf = pred["conf"]
    thr = np.quantile(conf, 1 - keep)
    m = (conf >= thr) & np.isfinite(pts).all(-1) & (pred["depth"] > 1e-4)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts[m].astype(np.float64))
    pc.colors = o3d.utility.Vector3dVector(cols[m].astype(np.float64))
    if len(pc.points) > max_points:
        ext = np.linalg.norm(np.percentile(pts[m], 98, 0) - np.percentile(pts[m], 2, 0))
        vs = ext / 2000
        down = pc.voxel_down_sample(vs)
        while len(down.points) > max_points:
            vs *= 1.25
            down = pc.voxel_down_sample(vs)
        pc = down
    return pc


def load_keep_aspect(files, long_side=518):
    from PIL import Image, ImageOps
    import torchvision.transforms.functional as TF

    ims = [ImageOps.exif_transpose(Image.open(f)).convert("RGB") for f in files]
    portrait = sum(im.height > im.width for im in ims) * 2 >= len(ims)
    ar = np.median([max(im.size) / min(im.size) for im in ims])
    short = int(round(long_side / ar / 14)) * 14
    W, H = (short, long_side) if portrait else (long_side, short)
    out, boxes = [], []
    for im in ims:
        s = max(W / im.width, H / im.height)
        cw, ch = W / s, H / s
        x0, y0 = (im.width - cw) / 2, (im.height - ch) / 2
        boxes.append([x0 / im.width, y0 / im.height, (x0 + cw) / im.width, (y0 + ch) / im.height])
        out.append(TF.to_tensor(im.crop((round(x0), round(y0), round(x0 + cw), round(y0 + ch))).resize((W, H), Image.BICUBIC)))
    return torch.stack(out), boxes


def reconstruct(files, out, device="auto", dtype="auto", keep=0.6, max_points=400_000, mode="keep"):
    out = Path(out)
    (out / "depth").mkdir(parents=True, exist_ok=True)
    (out / "conf").mkdir(parents=True, exist_ok=True)
    device = pick_device(device)
    if dtype == "auto":
        dtype = "fp16" if device.type == "mps" else "fp32"
    t0 = time.time()
    model = load_model(device)
    t_load = time.time() - t0

    if mode == "keep":
        images, boxes = load_keep_aspect(files)
    else:
        images, boxes = load_and_preprocess_images([str(f) for f in files], mode=mode), [None] * len(files)
    H, W = images.shape[-2:]
    with MemPeak(device) as mem:
        t1 = time.time()
        pred = run_vggt(model, images, device, DTYPES[dtype])
        t_inf = time.time() - t1

    t2 = time.time()
    pc = build_cloud(pred, images, keep, max_points)
    o3d.io.write_point_cloud(str(out / "points.ply"), pc)
    cams = []
    for i, f in enumerate(files):
        stem = f"{i:03d}_{f.stem}"
        np.save(out / "depth" / f"{stem}.npy", pred["depth"][i].astype(np.float16))
        np.save(out / "conf" / f"{stem}.npy", pred["conf"][i].astype(np.float16))
        cams.append({
            "file": str(Path(f).absolute()),
            "name": stem,
            "extrinsic": pred["extrinsic"][i].tolist(),
            "intrinsic": pred["intrinsic"][i].tolist(),
            "width": int(W),
            "height": int(H),
            "crop": boxes[i],
        })
    meta = {"model": "facebook/VGGT-1B", "device": device.type, "dtype": dtype, "preprocess": mode,
            "convention": "extrinsic is world-to-camera OpenCV (x right, y down, z fwd); world = camera 0; scale is relative, not metric",
            "cameras": cams}
    (out / "cameras.json").write_text(json.dumps(meta, indent=1))
    stats = {
        "n_images": len(files), "res": [int(H), int(W)], "device": device.type, "dtype": dtype,
        "model_load_s": round(t_load, 2), "inference_s": round(t_inf, 2), "post_s": round(time.time() - t2, 2),
        "peak_mps_gb": round(mem.peak / 1e9, 2), "max_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2),
        "n_points": len(pc.points),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="image dir(s), image files, or a .txt list")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto", choices=["auto", *DTYPES])
    ap.add_argument("--keep", type=float, default=0.6, help="fraction of pixels kept by depth confidence")
    ap.add_argument("--max-points", type=int, default=400_000)
    ap.add_argument("--mode", default="keep", choices=["keep", "crop", "pad"])
    a = ap.parse_args()
    stats = reconstruct(collect_images(a.inputs), a.out, a.device, a.dtype, a.keep, a.max_points, a.mode)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
