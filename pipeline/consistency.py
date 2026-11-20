import json
import sys
from pathlib import Path

import numpy as np


def load(run: Path):
    meta = json.loads((run / "cameras.json").read_text())
    cams = []
    for c in meta["cameras"]:
        E = np.array(c["extrinsic"])
        cams.append({
            "name": c["name"], "R": E[:, :3], "t": E[:, 3], "K": np.array(c["intrinsic"]),
            "depth": np.load(run / "depth" / f"{c['name']}.npy").astype(np.float32),
            "conf": np.load(run / "conf" / f"{c['name']}.npy").astype(np.float32),
        })
    return cams


def world_points(c, stride=6):
    H, W = c["depth"].shape
    v, u = np.mgrid[0:H:stride, 0:W:stride]
    d = c["depth"][::stride, ::stride]
    keep = c["conf"][::stride, ::stride] >= np.median(c["conf"])
    K = c["K"]
    pc = np.stack([(u - K[0, 2]) / K[0, 0] * d, (v - K[1, 2]) / K[1, 1] * d, d], -1)[keep]
    return (pc - c["t"]) @ c["R"]


def pair_score(a, b, tol=0.08):
    P = world_points(a)
    Xc = P @ b["R"].T + b["t"]
    z = Xc[:, 2]
    H, W = b["depth"].shape
    u = b["K"][0, 0] * Xc[:, 0] / np.maximum(z, 1e-6) + b["K"][0, 2]
    v = b["K"][1, 1] * Xc[:, 1] / np.maximum(z, 1e-6) + b["K"][1, 2]
    inside = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if inside.sum() < 50:
        return {"overlap": float(inside.mean()), "agree": 0.0, "violate": 0.0}
    d = b["depth"][v[inside].astype(int), u[inside].astype(int)]
    zz = z[inside]
    agree = np.abs(zz - d) < tol * zz
    violate = zz < d * (1 - tol * 2)
    return {"overlap": float(inside.mean()), "agree": float(agree.mean()), "violate": float(violate.mean())}


def matrix(run: Path):
    cams = load(run)
    n = len(cams)
    out = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                out[(i, j)] = pair_score(cams[i], cams[j])
    return cams, out


if __name__ == "__main__":
    cams, m = matrix(Path(sys.argv[1]))
    for (i, j), s in m.items():
        print(f"{cams[i]['name']} -> {cams[j]['name']}  overlap {s['overlap']:.2f}  agree {s['agree']:.2f}  violate {s['violate']:.2f}")
