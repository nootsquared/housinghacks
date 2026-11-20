import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
_models = {}


def _dino():
    if "dino" not in _models:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        rid = "IDEA-Research/grounding-dino-base"
        _models["dino"] = (AutoProcessor.from_pretrained(rid),
                           AutoModelForZeroShotObjectDetection.from_pretrained(rid).to(DEVICE).eval())
    return _models["dino"]


def _matte():
    if "matte" not in _models:
        from transformers import AutoModelForImageSegmentation

        m = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True)
        _models["matte"] = m.to(DEVICE).eval().float()
    return _models["matte"]


@torch.no_grad()
def detect(img: Image.Image, labels: list[str], box_thr=0.3, text_thr=0.25):
    proc, model = _dino()
    inputs = proc(images=img, text=[labels], return_tensors="pt").to(DEVICE)
    out = model(**inputs)
    res = proc.post_process_grounded_object_detection(
        out, inputs.input_ids, threshold=box_thr, text_threshold=text_thr, target_sizes=[img.size[::-1]])[0]
    names = res.get("text_labels", res.get("labels"))
    return [(str(n), float(s), [float(v) for v in b]) for n, s, b in zip(names, res["scores"], res["boxes"])]


@torch.no_grad()
def matte(img: Image.Image, box, pad=0.08):
    W, H = img.size
    x0, y0, x1, y1 = box
    px, py = (x1 - x0) * pad, (y1 - y0) * pad
    c = [int(max(0, x0 - px)), int(max(0, y0 - py)), int(min(W, x1 + px)), int(min(H, y1 + py))]
    crop = img.crop(c)
    x = torch.from_numpy(np.asarray(crop.resize((1024, 1024), Image.BILINEAR), np.float32) / 255)
    x = ((x - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor([0.229, 0.224, 0.225])).permute(2, 0, 1)[None]
    a = _matte()(x.to(DEVICE))[-1].sigmoid()[0, 0].cpu().numpy()
    a = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize(crop.size, Image.BILINEAR))
    full = np.zeros((H, W), np.uint8)
    full[c[1]:c[3], c[0]:c[2]] = a
    return full


def square_on_white(img: Image.Image, alpha: np.ndarray, size=1024, fill=0.85):
    ys, xs = np.nonzero(alpha > 127)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    rgba = np.dstack([np.asarray(img.convert("RGB")), alpha])[y0:y1, x0:x1]
    obj = Image.fromarray(rgba, "RGBA")
    s = fill * size / max(obj.size)
    obj = obj.resize((max(1, round(obj.width * s)), max(1, round(obj.height * s))), Image.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    canvas.paste(obj, ((size - obj.width) // 2, (size - obj.height) // 2), obj)
    return canvas, (int(x0), int(y0), int(x1), int(y1))


if __name__ == "__main__":
    photo, label, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    img = Image.open(photo).convert("RGB")
    dets = sorted(detect(img, [label]), key=lambda d: -d[1])
    print(dets[:3])
    a = matte(img, dets[0][2])
    sq, bbox = square_on_white(img, a)
    sq.save(out / f"{label}.png")
    Image.fromarray(a).save(out / f"{label}-alpha.png")
    print(bbox)
