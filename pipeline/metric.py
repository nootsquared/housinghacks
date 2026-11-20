import numpy as np
import torch
from PIL import Image

_pipe = None
MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf"


def model():
    global _pipe
    if _pipe is None:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        _pipe = (AutoImageProcessor.from_pretrained(MODEL), AutoModelForDepthEstimation.from_pretrained(MODEL).to(dev).eval(), dev)
    return _pipe


@torch.no_grad()
def metric_depth(img: Image.Image, size):
    proc, net, dev = model()
    x = proc(images=img, return_tensors="pt").to(dev)
    d = net(**x).predicted_depth
    d = torch.nn.functional.interpolate(d[:, None], size=size, mode="bilinear", align_corners=False)
    return d[0, 0].cpu().numpy()


ASSUMED_HFOV = 60.0


def scale_from_metric(images, depths, confs, hfovs=None):
    ratios = []
    for i, (img, d, c) in enumerate(zip(images, depths, confs)):
        m = metric_depth(img, d.shape)
        if hfovs:
            m = m * np.tan(np.radians(ASSUMED_HFOV / 2)) / np.tan(np.radians(hfovs[i] / 2))
        good = (c >= np.quantile(c, 0.5)) & (d > 1e-4) & (m > 0.3) & (m < 12)
        if good.sum() > 500:
            ratios.append(np.median(m[good] / d[good]))
    return float(np.median(ratios)) if ratios else None, [float(r) for r in ratios]
