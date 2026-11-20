import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web" / "public" / "data"
CAM_HEIGHT = 1.4
GRID_W = 192
FLIP = np.diag([1.0, -1.0, -1.0])


def quat(m):
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        return [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    i = int(np.argmax(np.diag(m)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k]) * 2
    q = [0.0] * 4
    q[i] = 0.25 * s
    q[j] = (m[j, i] + m[i, j]) / s
    q[k] = (m[k, i] + m[i, k]) / s
    q[3] = (m[k, j] - m[j, k]) / s
    return q


def rot_between(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-8:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def load_run(run: Path):
    meta = json.loads((run / "cameras.json").read_text())
    cams = []
    for c in meta["cameras"]:
        E = np.array(c["extrinsic"])
        K = np.array(c["intrinsic"])
        depth = np.load(run / "depth" / f"{c['name']}.npy").astype(np.float32)
        conf = np.load(run / "conf" / f"{c['name']}.npy").astype(np.float32)
        cams.append({**c, "R": E[:, :3], "t": E[:, 3], "K": K, "depth": depth, "conf": conf})
    return cams


def unproject(c, stride=4):
    H, W = c["depth"].shape
    v, u = np.mgrid[0:H:stride, 0:W:stride]
    d = c["depth"][::stride, ::stride]
    K = c["K"]
    x = (u - K[0, 2]) / K[0, 0] * d
    y = (v - K[1, 2]) / K[1, 1] * d
    pc = np.stack([x, y, d], -1).reshape(-1, 3)
    pw = (pc - c["t"]) @ c["R"]
    return pw, c["conf"][::stride, ::stride].reshape(-1)


def manhattan_yaw(points):
    import open3d as o3d

    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pc = pc.voxel_down_sample(np.linalg.norm(points.max(0) - points.min(0)) / 150)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(20))
    n = np.asarray(pc.normals)
    n = n[np.abs(n[:, 1]) < 0.2]
    if len(n) < 50:
        return 0.0
    ang = np.mod(np.arctan2(n[:, 2], n[:, 0]), np.pi / 2)
    hist, edges = np.histogram(ang, bins=90, range=(0, np.pi / 2))
    hist = np.convolve(np.r_[hist[-2:], hist, hist[:2]], np.ones(5) / 5, "valid")
    return float((edges[np.argmax(hist)] + edges[np.argmax(hist) + 1]) / 2)


def align(cams):
    down = np.mean([c["R"].T @ np.array([0, 1, 0]) for c in cams], 0)
    A = rot_between(down, np.array([0, -1, 0]))
    pts, conf = zip(*(unproject(c) for c in cams))
    pts, conf = np.concatenate(pts), np.concatenate(conf)
    good = conf >= np.quantile(conf, 0.4)
    P = pts[good] @ A.T
    centers = np.array([-c["R"].T @ c["t"] for c in cams]) @ A.T
    floor = np.quantile(P[:, 1], 0.02)
    cam_h = np.median(centers[:, 1] - floor)
    scale = CAM_HEIGHT / max(cam_h, 1e-6)
    origin = np.array([np.median(centers[:, 0]), floor, np.median(centers[:, 2])])
    yaw = manhattan_yaw((P - origin) * scale)
    Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    A = Ry @ A
    origin = Ry @ origin
    return A, origin, scale, float(np.quantile(conf, 0.3)), float(np.quantile(conf, 0.97))


def pool_depth(depth, conf, w):
    H, W = depth.shape
    h = max(2, round(H * w / W))
    di = Image.fromarray(depth).resize((w, h), Image.NEAREST)
    ci = Image.fromarray(conf).resize((w, h), Image.BILINEAR)
    return np.asarray(di), np.asarray(ci)


def crop_texture(src: Path, W: int, H: int, dst: Path):
    im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    ow, oh = im.size
    scaled_h = oh * (W / ow)
    if scaled_h > H + 1:
        keep = oh * H / scaled_h
        top = (oh - keep) / 2
        im = im.crop((0, round(top), ow, round(top + keep)))
    im.thumbnail((1600, 1600), Image.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, "JPEG", quality=85, optimize=True, progressive=True)


def room_box(points):
    lo = np.quantile(points, 0.04, 0)
    hi = np.quantile(points, 0.96, 0)
    size = np.maximum(hi - lo, 0.5)
    size[1] = float(np.clip(np.quantile(points[:, 1], 0.98), 2.2, 3.2))
    center = (lo + hi) / 2
    center[1] = size[1] / 2
    return center, size


def build(listing_dir: Path, runs: list[Path], out: Path):
    listing = json.loads((listing_dir / "listing.json").read_text())
    labels = json.loads((listing_dir / "labels.json").read_text()) if (listing_dir / "labels.json").exists() else {}
    out.mkdir(parents=True, exist_ok=True)
    views, room_pts, room_views = [], {}, {}
    offset_x = 0.0
    all_min, all_max = np.full(3, np.inf), np.full(3, -np.inf)
    for run in runs:
        cams = load_run(run)
        A, origin, scale, c_lo, c_hi = align(cams)
        run_pts = []
        for c in cams:
            src = Path(c["file"]).absolute()
            rel = src.relative_to(listing_dir).as_posix() if src.is_relative_to(listing_dir) else f"photos/{src.name}"
            lab = labels.get(rel, {})
            group = lab.get("group") or lab.get("room") or "room"
            Rc2w = A @ c["R"].T @ FLIP
            C = (A @ (-c["R"].T @ c["t"]) - origin) * scale + np.array([offset_x, 0, 0])
            K, H, W = c["K"], c["height"], c["width"]
            fov = float(np.degrees(2 * np.arctan(H / 2 / K[1, 1])))
            aspect = float((W / 2 / K[0, 0]) / (H / 2 / K[1, 1]))
            d, cf = pool_depth(c["depth"] * scale, c["conf"], GRID_W)
            mm = np.clip(d * 1000, 0, 65535).astype(np.uint16)
            cb = (np.clip((cf - c_lo) / max(c_hi - c_lo, 1e-6), 0, 1) * 255).astype(np.uint8)
            vid = f"v{len(views):02d}"
            (out / "depth").mkdir(exist_ok=True)
            with open(out / "depth" / f"{vid}.bin", "wb") as fh:
                fh.write(np.array([mm.shape[1], mm.shape[0]], np.uint16).tobytes())
                fh.write(mm.tobytes())
                fh.write(cb.tobytes())
            crop_texture(src, W, H, out / "views" / f"{vid}.jpg")
            views.append({
                "id": vid, "file": f"views/{vid}.jpg", "source": rel, "room": group,
                "position": C.round(4).tolist(), "quaternion": [round(x, 6) for x in quat(Rc2w)],
                "fov": round(fov, 3), "aspect": round(aspect, 4), "depth": f"depth/{vid}.bin",
            })
            p, conf = unproject(c, 6)
            p = ((p @ A.T) - origin) * scale + np.array([offset_x, 0, 0])
            p = p[conf >= c_lo]
            run_pts.append(p)
            room_pts.setdefault(group, []).append(p)
            room_views.setdefault(group, []).append(vid)
        rp = np.concatenate(run_pts)
        lo, hi = np.quantile(rp, 0.02, 0), np.quantile(rp, 0.98, 0)
        all_min, all_max = np.minimum(all_min, lo), np.maximum(all_max, hi)
        offset_x = all_max[0] + 1.0
    rooms = []
    for g, pts in room_pts.items():
        center, size = room_box(np.concatenate(pts))
        n = len(room_views[g])
        rtype = g.rsplit("-", 1)[0]
        rooms.append({
            "id": g, "type": rtype, "label": label(g, room_pts.keys()),
            "center": center.round(3).tolist(), "size": size.round(3).tolist(), "yaw": 0,
            "confidence": "measured" if n > 1 else "estimated", "views": room_views[g],
        })
    scene = {
        "id": listing["id"], "units": "m", "rooms": rooms, "views": views,
        "bounds": {"min": all_min.round(3).tolist(), "max": all_max.round(3).tolist()},
    }
    (out / "scene.json").write_text(json.dumps(scene, indent=1))
    return scene


NAMES = {"bedroom": "Bedroom", "living": "Living room", "kitchen": "Kitchen", "bathroom": "Bathroom",
         "dining": "Dining", "hallway": "Hall", "closet": "Closet", "study": "Study", "room": "Room"}


def label(g, all_groups):
    rtype, _, n = g.rpartition("-")
    if not rtype:
        return NAMES.get(g, g.title())
    same = [x for x in all_groups if x.rsplit("-", 1)[0] == rtype]
    return NAMES.get(rtype, rtype.title()) + (f" {n}" if len(same) > 1 else "")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("listing", type=Path)
    ap.add_argument("runs", nargs="+", type=Path)
    a = ap.parse_args()
    lid = json.loads((a.listing / "listing.json").read_text())["id"]
    s = build(a.listing.resolve(), [r.resolve() for r in a.runs], WEB / lid)
    print(f"{lid}: {len(s['views'])} views, {len(s['rooms'])} rooms, bounds {s['bounds']}")
