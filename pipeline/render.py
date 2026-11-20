import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def basis(cams):
    if not cams:
        return np.eye(3)
    R = np.array([np.array(c["extrinsic"])[:, :3] for c in cams])
    up = -R[:, 1, :].mean(0)
    up /= np.linalg.norm(up)
    fwd = R[0, 2, :] - up * (R[0, 2, :] @ up)
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    return np.stack([right, fwd, up])


def render(run_dir, png, n=120_000, title=None):
    run_dir = Path(run_dir)
    pc = o3d.io.read_point_cloud(str(run_dir / "points.ply"))
    P, C = np.asarray(pc.points), np.asarray(pc.colors)
    if len(P) > n:
        idx = np.random.default_rng(0).choice(len(P), n, replace=False)
        P, C = P[idx], C[idx]
    cj = run_dir / "cameras.json"
    cams = json.loads(cj.read_text())["cameras"] if cj.exists() else []
    B = basis(cams)
    Q = P @ B.T
    centers = np.array([-np.array(c["extrinsic"])[:, :3].T @ np.array(c["extrinsic"])[:, 3] for c in cams]).reshape(-1, 3) @ B.T
    lo, hi = np.percentile(Q, 1, 0), np.percentile(Q, 99, 0)
    fig = plt.figure(figsize=(18, 6), facecolor="#2b2b2b")
    views = [("top-down (ceiling cut)", 90, -90), ("oblique (ceiling cut)", 35, -60), ("front", 5, -90)]
    cut = centers[:, 2].mean() + 0.4 * (hi[2] - centers[:, 2].mean()) if len(centers) else hi[2]
    for k, (name, el, az) in enumerate(views):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d", facecolor="#2b2b2b")
        m = Q[:, 2] < cut if k < 2 else slice(None)
        ax.scatter(Q[m, 0], Q[m, 1], Q[m, 2], c=C[m], s=0.3, linewidths=0)
        if len(centers):
            ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c="red", s=25, marker="^")
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(hi - lo)
        ax.view_init(elev=el, azim=az)
        ax.set_title(name, color="w")
        ax.set_axis_off()
    fig.suptitle(title or run_dir.name, color="w")
    fig.tight_layout()
    fig.savefig(png, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--png")
    a = ap.parse_args()
    render(a.run_dir, a.png or str(Path(a.run_dir).with_suffix(".png")))
