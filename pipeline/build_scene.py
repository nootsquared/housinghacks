import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from fuse import fuse_room, write_mesh
from layout import turn_room
from plan import solve
from recreate import recreate, wall_fills
from shell import Cam, build_shell, ransac_plane

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


def align(cams, metric_scale=None):
    down = np.mean([c["R"].T @ np.array([0, 1, 0]) for c in cams], 0)
    A = rot_between(down, np.array([0, -1, 0]))
    pts, conf = zip(*(unproject(c) for c in cams))
    pts, conf = np.concatenate(pts), np.concatenate(conf)
    good = conf >= np.quantile(conf, 0.4)
    P = pts[good] @ A.T
    centers = np.array([-c["R"].T @ c["t"] for c in cams]) @ A.T
    span = np.ptp(P[:, 1])
    low = P[P[:, 1] < np.quantile(P[:, 1], 0.02) + span * 0.15]
    if len(low) > 200:
        sample = low[np.random.default_rng(0).choice(len(low), min(len(low), 6000), replace=False)]
        n, _, _ = ransac_plane(sample, tol=span * 0.01)
        if n[1] < 0:
            n = -n
        if n[1] > 0.9:
            A = rot_between(n, np.array([0, 1, 0])) @ A
            P = pts[good] @ A.T
            centers = np.array([-c["R"].T @ c["t"] for c in cams]) @ A.T
    floor = np.quantile(P[:, 1], 0.02)
    cam_h = np.median(centers[:, 1] - floor)
    scale = metric_scale if metric_scale else CAM_HEIGHT / max(cam_h, 1e-6)
    origin = np.array([np.median(centers[:, 0]), floor, np.median(centers[:, 2])])
    yaw = manhattan_yaw((P - origin) * scale)
    Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    A = Ry @ A
    origin = Ry @ origin
    print(f"    scale {scale:.3f} camera height {cam_h * scale:.2f} m", flush=True)
    return A, origin, scale, float(np.quantile(conf, 0.3)), float(np.quantile(conf, 0.97))


def camera_height_units(cams):
    down = np.mean([c["R"].T @ np.array([0, 1, 0]) for c in cams], 0)
    A = rot_between(down, np.array([0, -1, 0]))
    pts, conf = zip(*(unproject(c) for c in cams))
    pts, conf = np.concatenate(pts), np.concatenate(conf)
    P = pts[conf >= np.quantile(conf, 0.4)] @ A.T
    centers = np.array([-c["R"].T @ c["t"] for c in cams]) @ A.T
    return float(np.median(centers[:, 1] - np.quantile(P[:, 1], 0.02)))


def resolve_scales(rooms):
    for r in rooms:
        cams = load_run(r["run"])
        r["scale"] = run_metric_scale(cams)
        r["cam_units"] = camera_height_units(cams)
    known = [r["scale"] * r["cam_units"] for r in rooms if r["scale"]]
    med = float(np.median(known)) if known else CAM_HEIGHT
    for r in rooms:
        h = r["scale"] * r["cam_units"] if r["scale"] else None
        if h is None or abs(h / med - 1) > 0.2:
            r["scale"] = med / r["cam_units"]
            r["scale_note"] = "camera height prior"
        else:
            r["scale_note"] = "metric depth"
        print(f"  {r['id']}: camera {r['scale'] * r['cam_units']:.2f} m via {r['scale_note']}", flush=True)
    return med


def smooth_depth(depth):
    import cv2

    d = cv2.medianBlur(depth.astype(np.float32), 5)
    sc = float(np.median(d)) * 0.04
    return cv2.bilateralFilter(d, 9, sc, 4)


def pool_depth(depth, conf, w):
    H, W = depth.shape
    h = max(2, round(H * w / W))
    di = Image.fromarray(depth).resize((w, h), Image.NEAREST)
    ci = Image.fromarray(conf).resize((w, h), Image.BILINEAR)
    return np.asarray(di), np.asarray(ci)


def crop_texture(src: Path, W: int, H: int, dst: Path, box=None):
    im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    ow, oh = im.size
    if box:
        im = im.crop((round(box[0] * ow), round(box[1] * oh), round(box[2] * ow), round(box[3] * oh)))
        im.thumbnail((1600, 1600), Image.LANCZOS)
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst, "JPEG", quality=85, optimize=True, progressive=True)
        return
    scaled_h = oh * (W / ow)
    if scaled_h > H + 1:
        keep = oh * H / scaled_h
        top = (oh - keep) / 2
        im = im.crop((0, round(top), ow, round(top + keep)))
    im.thumbnail((1600, 1600), Image.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, "JPEG", quality=85, optimize=True, progressive=True)




def normals(P):
    import open3d as o3d

    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(24))
    return np.asarray(pc.normals)


def object_mask(d, Rc2w, C, fov, aspect, box, margin=0.12):
    h, w = d.shape
    t = np.tan(np.radians(fov) / 2)
    uu, vv = np.meshgrid((np.arange(w) + 0.5) / w, (np.arange(h) + 0.5) / h)
    pc = np.stack([(uu * 2 - 1) * t * aspect * d, (1 - vv * 2) * t * d, -d], -1)
    X = pc.reshape(-1, 3) @ Rc2w.T + C
    (x0, x1), (z0, z1), hh = box
    inside = (X[:, 0] > x0 + margin) & (X[:, 0] < x1 - margin) & (X[:, 2] > z0 + margin) & (X[:, 2] < z1 - margin)
    inside &= (X[:, 1] > margin * 0.6) & (X[:, 1] < hh - margin)
    return inside.reshape(h, w).astype(np.uint8), X.reshape(h, w, 3)


def fuse_objects(samples, box, out_file: Path):
    import open3d as o3d

    pts = np.concatenate([s[0] for s in samples]) if samples else np.zeros((0, 3))
    if len(pts) < 2000:
        return None
    cols = np.concatenate([s[1] for s in samples])
    cams = np.concatenate([s[2] for s in samples])
    (x0, x1), (z0, z1), hh = box
    rng = np.random.default_rng(0)
    floor = np.stack([rng.uniform(x0, x1, 20000), np.zeros(20000), rng.uniform(z0, z1, 20000)], -1)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate([pts, floor])))
    pc.colors = o3d.utility.Vector3dVector(np.concatenate([cols, np.full((len(floor), 3), 0.5)]))
    pc = pc.voxel_down_sample(0.012)
    pc, _ = pc.remove_statistical_outlier(16, 2.0)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(0.06, 30))
    P = np.asarray(pc.points)
    N = np.asarray(pc.normals)
    tree = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)))
    view = np.zeros_like(P)
    for i, p in enumerate(P):
        _, idx, _ = tree.search_knn_vector_3d(p, 1)
        view[i] = cams[idx[0]] - p
    view[P[:, 1] < 0.02] = [0, 1, 0]
    flip = (N * view).sum(1) < 0
    N[flip] *= -1
    pc.normals = o3d.utility.Vector3dVector(N)
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=8, scale=1.05)
    dens = np.asarray(dens)
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, 0.12))
    V = np.asarray(mesh.vertices)
    keep = (V[:, 1] > 0.035) & (V[:, 0] > x0) & (V[:, 0] < x1) & (V[:, 2] > z0) & (V[:, 2] < z1) & (V[:, 1] < hh)
    d2, _ = nearest_dist(V, pts)
    keep &= d2 < 0.06
    mesh.remove_vertices_by_mask(~keep)
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    tri_clusters, counts, _ = mesh.cluster_connected_triangles()
    tri_clusters, counts = np.asarray(tri_clusters), np.asarray(counts)
    mesh.remove_triangles_by_mask(counts[tri_clusters] < 200)
    mesh.remove_unreferenced_vertices()
    mesh = mesh.filter_smooth_taubin(8)
    mesh.compute_vertex_normals()
    V = np.asarray(mesh.vertices).astype(np.float32)
    F = np.asarray(mesh.triangles).astype(np.uint32)
    if len(F) < 100:
        return None
    _, idx = nearest_dist(V, pts)
    Cc = (np.clip(cols[idx], 0, 1) * 255).astype(np.uint8)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "wb") as fh:
        fh.write(np.array([len(V), len(F)], np.uint32).tobytes())
        fh.write(V.tobytes())
        fh.write(Cc.tobytes())
        fh.write(F.tobytes())
    return {"vertices": int(len(V)), "faces": int(len(F))}


def nearest_dist(A, B):
    from scipy.spatial import cKDTree

    d, i = cKDTree(B).query(A, k=1, workers=-1)
    return d, i


def run_metric_scale(cams):
    from metric import scale_from_metric

    ims = []
    for c in cams:
        im = ImageOps.exif_transpose(Image.open(c["file"])).convert("RGB")
        b = c.get("crop") or [0, 0, 1, 1]
        W, H = im.size
        ims.append(im.crop((b[0] * W, b[1] * H, b[2] * W, b[3] * H)))
    hf = [float(np.degrees(2 * np.arctan(c["width"] / 2 / c["K"][0, 0]))) for c in cams]
    print(f"    hfov {np.round(hf, 0).tolist()}", flush=True)
    s, _ = scale_from_metric(ims, [c["depth"] for c in cams], [c["conf"] for c in cams], hf)
    return s


def room_from_run(listing_dir, room, out, views, labels):
    cams = load_run(room["run"])
    A, origin, scale, c_lo, c_hi = align(cams, room["scale"])
    pts_all, pending, rcams = [], [], []
    for c in cams:
        src = Path(c["file"]).absolute()
        rel = src.relative_to(listing_dir).as_posix() if src.is_relative_to(listing_dir) else f"photos/{src.name}"
        Rc2w = A @ c["R"].T @ FLIP
        C = (A @ (-c["R"].T @ c["t"]) - origin) * scale
        K, H, W = c["K"], c["height"], c["width"]
        fov = float(np.degrees(2 * np.arctan(H / 2 / K[1, 1])))
        aspect = float((W / 2 / K[0, 0]) / (H / 2 / K[1, 1]))
        sm = smooth_depth(c["depth"]) * scale
        d, cf = pool_depth(sm, c["conf"], GRID_W)
        vid = f"v{len(views) + len(pending):02d}"
        tex = out / "views" / f"{vid}.jpg"
        crop_texture(src, W, H, tex, c.get("crop"))
        cn = np.clip((c["conf"] - c_lo) / max(c_hi - c_lo, 1e-6), 0, 1)
        rcams.append(Cam(vid, Rc2w, C, K, W, H, sm, cn, np.asarray(Image.open(tex)), src.stem in room.get("outside", ())))
        pending.append({"vid": vid, "rel": rel, "Rc2w": Rc2w, "C": C, "fov": fov, "aspect": aspect, "d": d, "cf": cf, "tex": tex})
        p, conf = unproject(c, 4)
        p = ((p @ A.T) - origin) * scale
        pts_all.append(p[conf >= np.quantile(c["conf"], 0.45)])
    P = np.concatenate(pts_all)
    P = P[np.random.default_rng(0).choice(len(P), min(len(P), 60000), replace=False)]
    ceiling = float(np.clip(np.quantile(P[:, 1], 0.995), 2.3, 3.0))
    shell = build_shell(room["id"], P, normals(P), rcams, out / "shell") if room.get("shell", True) else None
    model = None
    if shell:
        box = (shell["x"], shell["z"], shell["height"])
        raw = shell.pop("_raw")
        model = recreate(raw, rcams, box, room["type"])
        model["fills"] = wall_fills(raw, model["walls"], out / "shell", room["id"])
    else:
        lo, hi = np.quantile(P, 0.03, 0), np.quantile(P, 0.97, 0)
        box = ([float(lo[0]), float(hi[0])], [float(lo[2]), float(hi[2])], ceiling)
    samples = []
    for pv in pending:
        obj, X = object_mask(pv["d"], pv["Rc2w"], pv["C"], pv["fov"], pv["aspect"], box)
        pv["obj"] = obj if shell else np.ones_like(obj)
        conf_ok = pv["cf"] >= np.quantile(pv["cf"], 0.4)
        sel = (obj > 0) & conf_ok
        img = np.asarray(Image.open(pv["tex"]).resize(obj.shape[::-1], Image.BILINEAR)).astype(np.float32) / 255
        samples.append((X[sel], img[sel], np.repeat(pv["C"][None], sel.sum(), 0)))
    objects = None
    fused = fuse_room(rcams, box, structure=shell is not None)
    return {"pending": pending, "shell": shell, "box": box, "ceiling": ceiling, "objects": objects, "model": model, "fused": fused,
            "c_lo": c_lo, "c_hi": c_hi}


def shift_shell(shell, off):
    shell = json.loads(json.dumps(shell))
    shell["x"] = [shell["x"][0] + off[0], shell["x"][1] + off[0]]
    shell["z"] = [shell["z"][0] + off[2], shell["z"][1] + off[2]]
    for s in shell["surfaces"]:
        s["origin"] = (np.array(s["origin"]) + off).round(4).tolist()
    return shell


def shift_model(model, off):
    m = json.loads(json.dumps(model))
    for o in m["openings"]:
        k = 0 if o["wall"] in ("n", "s") else 2
        o["from"] = round(o["from"] + off[k], 3)
        o["to"] = round(o["to"] + off[k], 3)
    for f in m["furniture"]:
        f["center"] = (np.array(f["center"]) + off).round(3).tolist()
    return m


def infer_floors(built, weak=0.6):
    models = [(r["room"], r["model"]) for r in built if r.get("model")]
    strong = [m["floor"] for room, m in models if room["type"] != "bathroom" and m["floor"]["evidence"] >= weak]
    best = max(strong, key=lambda f: f["evidence"], default=None)
    for room, m in models:
        if m["floor"]["evidence"] >= weak:
            continue
        if room["type"] == "bathroom":
            m["floor"] = {"material": "tile", "color": "#d9d6d0", "accent": "#bdb8b0", "evidence": 0.0, "inferred": "room type"}
        elif best:
            m["floor"] = {**best, "evidence": 0.0, "inferred": "matched to best seen floor"}


def dump_rooms(built, path):
    rooms = []
    for r in built:
        m = r.get("model") or {}
        rooms.append({
            "id": r["room"]["id"], "type": r["room"]["type"], "box": r["box"], "shell": bool(r.get("shell")),
            "openings": m.get("openings", []), "walls": m.get("walls"),
            "views": [{"id": pv["vid"], "photo": pv["rel"], "C": np.round(pv["C"], 3).tolist(),
                       "forward": np.round(-pv["Rc2w"][:, 2], 3).tolist(), "fov": round(pv["fov"], 2)} for pv in r["pending"]],
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rooms, indent=1))


def arrange(listing_dir, listing, built):
    from doors import detect_doors

    work = ROOT / "pipeline" / "out" / listing["id"]
    dump_rooms(built, work / "rooms_prelayout.json")
    doors = detect_doors(listing_dir, [r["room"]["run"] for r in built], work / "rooms_prelayout.json")
    (work / "doors.json").write_text(json.dumps(doors, indent=1))
    prior_path = listing_dir / "floorplan_prior.json"
    prior = json.loads(prior_path.read_text()) if prior_path.exists() else None
    rooms = json.loads((work / "rooms_prelayout.json").read_text())
    sol = solve(rooms, doors, listing.get("sqft"), prior)
    (work / "plan.json").write_text(json.dumps(sol, indent=1))
    offsets = []
    for r in built:
        p = sol[r["room"]["id"]]
        if "box" in p:
            r["box"] = p["box"]
        turn_room(r, p["turn"])
        offsets.append(np.array(p["offset"], float))
    return offsets


def build(listing_dir: Path, rooms: list[dict], out: Path):
    listing = json.loads((listing_dir / "listing.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    views, built = [], []
    cam_h = resolve_scales(rooms)
    for room in rooms:
        r = room_from_run(listing_dir, room, out, views, {})
        r["room"] = room
        built.append(r)
        for pv in r["pending"]:
            views.append(pv)
    infer_floors(built)
    offsets = arrange(listing_dir, listing, built)
    scene_views, scene_rooms = [], []
    all_min, all_max = np.full(3, np.inf), np.full(3, -np.inf)
    (out / "depth").mkdir(exist_ok=True)
    for r, off in zip(built, offsets):
        room = r["room"]
        vids = []
        for pv in r["pending"]:
            mm = np.clip(pv["d"] * 1000, 0, 65535).astype(np.uint16)
            cb = (np.clip((pv["cf"] - r["c_lo"]) / max(r["c_hi"] - r["c_lo"], 1e-6), 0, 1) * 255).astype(np.uint8)
            with open(out / "depth" / f"{pv['vid']}.bin", "wb") as fh:
                fh.write(np.array([mm.shape[1], mm.shape[0]], np.uint16).tobytes())
                fh.write(mm.tobytes())
                fh.write(cb.tobytes())
                fh.write(pv["obj"].tobytes())
            scene_views.append({
                "id": pv["vid"], "file": f"views/{pv['vid']}.jpg", "source": pv["rel"], "room": room["id"],
                "position": (pv["C"] + off).round(4).tolist(), "quaternion": [round(x, 6) for x in quat(pv["Rc2w"])],
                "fov": round(pv["fov"], 3), "aspect": round(pv["aspect"], 4), "depth": f"depth/{pv['vid']}.bin",
            })
            vids.append(pv["vid"])
        (x0, x1), (z0, z1), hh = r["box"]
        entry = {
            "id": room["id"], "type": room["type"], "label": room["label"],
            "center": [round((x0 + x1) / 2 + off[0], 3), round(hh / 2, 3), round((z0 + z1) / 2 + off[2], 3)],
            "size": [round(x1 - x0, 3), round(hh, 3), round(z1 - z0, 3)], "yaw": 0, "offset": off.round(4).tolist(),
            "confidence": "measured" if len(vids) > 1 else "estimated", "views": vids, "scaleSource": room["scale_note"],
        }
        if r["shell"]:
            entry["shell"] = {**shift_shell(r["shell"], off), "floorTurn": r.get("turn", 0)}
        if r["model"]:
            entry["model"] = shift_model(r["model"], off)
        if r["fused"]:
            entry["mesh"] = {"file": f"mesh/{room['id']}.bin", **write_mesh(out / "mesh" / f"{room['id']}.bin", r["fused"], off)}
        if r["objects"]:
            entry["objects"] = {"file": f"objects/{room['id']}.bin", **r["objects"]}
        scene_rooms.append(entry)
        all_min = np.minimum(all_min, [x0 + off[0], 0, z0 + off[2]])
        all_max = np.maximum(all_max, [x1 + off[0], hh, z1 + off[2]])
    scene = {
        "id": listing["id"], "units": "m", "rooms": scene_rooms, "views": scene_views,
        "bounds": {"min": all_min.round(3).tolist(), "max": all_max.round(3).tolist()},
        "cameraHeight": round(cam_h, 3),
        "floorArea": round(sum(r["size"][0] * r["size"][2] for r in scene_rooms if r.get("model")) * 10.764),
    }
    (out / "scene.json").write_text(json.dumps(scene, indent=1))
    return scene


NAMES = {"bedroom": "Bedroom", "living": "Living room", "kitchen": "Kitchen", "bathroom": "Bathroom",
         "dining": "Dining", "hallway": "Hall", "closet": "Closet", "study": "Study", "room": "Room", "other": "Room"}


def labels_for(rooms):
    counts = {}
    for r in rooms:
        counts[r["type"]] = counts.get(r["type"], 0) + 1
    seen = {}
    for r in rooms:
        t = r["type"]
        seen[t] = seen.get(t, 0) + 1
        base = NAMES.get(t, t.title())
        if r["id"] in ("attic", "basement", "bonus", "entry", "stairs"):
            base = r["id"].title() if r["id"] != "bonus" else "Bonus room"
            r["label"] = base
        else:
            r["label"] = base + (f" {seen[t]}" if counts[t] > 1 else "")
    return rooms


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("listing", type=Path)
    ap.add_argument("runs", nargs="+", type=Path)
    a = ap.parse_args()
    lid = json.loads((a.listing / "listing.json").read_text())["id"]
    spec = json.loads((a.listing / "rooms.json").read_text()) if (a.listing / "rooms.json").exists() else {}
    rooms = []
    for r in a.runs:
        meta = spec.get(r.name, {})
        rooms.append({"id": r.name, "type": meta.get("type", r.name.rsplit("-", 1)[0]), "run": r.resolve(),
                      "shell": meta.get("shell", True), "fixed_label": meta.get("label"),
                      "outside": set(meta.get("outside", [])),
                      "turn": meta.get("turn"), "at": meta.get("at")})
    rooms = labels_for(rooms)
    for r in rooms:
        if r.get("fixed_label"):
            r["label"] = r["fixed_label"]
    s = build(a.listing.resolve(), rooms, WEB / lid)
    print(f"{lid}: {len(s['views'])} views, {len(s['rooms'])} rooms")
