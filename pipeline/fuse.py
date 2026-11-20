import numpy as np
import open3d as o3d
from PIL import Image

from shell import Cam

VOXEL = 0.02
FLIP = np.diag([1.0, -1.0, -1.0])


def valid_depth(c: Cam, box, margin):
    (x0, x1), (z0, z1), h = box
    d = c.depth.astype(np.float32).copy()
    gy, gx = np.gradient(d)
    edge = np.hypot(gx, gy) > 0.04 * d
    ys, xs = np.mgrid[0:c.H, 0:c.W]
    ray = np.stack([(xs - c.K[0, 2]) / c.K[0, 0], -(ys - c.K[1, 2]) / c.K[1, 1], -np.ones_like(d)], -1)
    X = c.C + (ray * d[..., None]) @ c.Rw.T
    inside = (X[..., 0] > x0 - margin) & (X[..., 0] < x1 + margin) & (X[..., 2] > z0 - margin) & (X[..., 2] < z1 + margin)
    inside &= (X[..., 1] > -0.15) & (X[..., 1] < h + 0.25)
    d[edge | ~inside | (c.conf < 0.08)] = 0
    return d


def tsdf_mesh(cams: list[Cam], box, margin):
    from skimage.measure import marching_cubes

    (x0, x1), (z0, z1), h = box
    lo = np.array([x0 - margin, -0.15, z0 - margin])
    hi = np.array([x1 + margin, h + 0.25, z1 + margin])
    dims = np.ceil((hi - lo) / VOXEL).astype(int) + 1
    trunc = VOXEL * 4
    g = [lo[i] + np.arange(dims[i]) * VOXEL for i in range(3)]
    X = np.stack(np.meshgrid(*g, indexing="ij"), -1).reshape(-1, 3).astype(np.float32)
    tsdf = np.ones(len(X), np.float32)
    wsum = np.zeros(len(X), np.float32)
    col = np.zeros((len(X), 3), np.float32)
    for c in cams:
        d = valid_depth(c, box, margin)
        rgb = np.asarray(Image.fromarray(c.image).resize((c.W, c.H), Image.BILINEAR)).astype(np.float32)
        px, py, z = project(c, X)
        ix, iy = np.round(px).astype(int), np.round(py).astype(int)
        ok = (z > 0.1) & (ix >= 0) & (ix < c.W) & (iy >= 0) & (iy < c.H)
        idx = np.flatnonzero(ok)
        dd = d[iy[idx], ix[idx]]
        sdf = dd - z[idx]
        keep = (dd > 0) & (sdf > -trunc)
        idx, sdf = idx[keep], np.minimum(1.0, sdf[keep] / trunc)
        w = np.clip(c.conf[iy[idx], ix[idx]], 0.05, 1).astype(np.float32)
        tsdf[idx] = (tsdf[idx] * wsum[idx] + sdf * w) / (wsum[idx] + w)
        col[idx] = (col[idx] * wsum[idx, None] + rgb[iy[idx], ix[idx]] * w[:, None]) / (wsum[idx, None] + w[:, None])
        wsum[idx] += w
    vol = tsdf.reshape(dims)
    seen = (wsum > 0).reshape(dims)
    vol[~seen] = 1.0
    if vol.min() >= 0:
        return None
    V, F, _, _ = marching_cubes(vol, 0.0, spacing=(VOXEL,) * 3, mask=seen)
    V = V + lo
    vi = np.clip(np.round((V - lo) / VOXEL).astype(int), 0, dims - 1)
    C = col.reshape(*dims, 3)[vi[:, 0], vi[:, 1], vi[:, 2]]
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F[:, ::-1].copy()))
    mesh.vertex_colors = o3d.utility.Vector3dVector(C / 255)
    labels, counts, _ = mesh.cluster_connected_triangles()
    labels, counts = np.asarray(labels), np.asarray(counts)
    mesh.remove_triangles_by_mask(counts[labels] < 400)
    mesh.remove_unreferenced_vertices()
    return mesh


def project(c: Cam, X):
    Xc = (X - c.C) @ c.Rw
    z = -Xc[..., 2]
    zs = np.maximum(z, 1e-6)
    px = c.K[0, 0] * Xc[..., 0] / zs + c.K[0, 2]
    py = -c.K[1, 1] * Xc[..., 1] / zs + c.K[1, 2]
    return px, py, z


def assign_views(V, F, cams: list[Cam]):
    tri = V[F]
    cen = tri.mean(1)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    best = np.full(len(F), -1)
    score = np.zeros(len(F))
    uvs = np.zeros((len(F), 3, 2), np.float32)
    for i, c in enumerate(cams):
        px, py, z = project(c, tri)
        inb = (z > 0.1).all(1) & (px >= 0).all(1) & (px <= c.W - 1).all(1) & (py >= 0).all(1) & (py <= c.H - 1).all(1)
        cpx, cpy, cz = project(c, cen)
        ix = np.clip(cpx.astype(int), 0, c.W - 1)
        iy = np.clip(cpy.astype(int), 0, c.H - 1)
        vis = inb & (np.abs(c.depth[iy, ix] - cz) < 0.05 + 0.04 * cz)
        ray = cen - c.C
        ray /= np.linalg.norm(ray, axis=1, keepdims=True)
        cos = np.abs((n * ray).sum(1))
        s = np.where(vis & (cos > 0.12), cos / np.maximum(cz, 0.3) ** 1.5, 0)
        win = s > score
        best[win] = i
        score[win] = s[win]
        uvs[win, :, 0] = px[win] / c.W
        uvs[win, :, 1] = 1 - py[win] / c.H
    return best, uvs


def drop_structure(mesh, box, band=0.08):
    (x0, x1), (z0, z1), _ = box
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.triangles)
    t = V[F]
    on = lambda a, v: (np.abs(t[..., a] - v) < band).all(1)
    beyond = (t[..., 0] < x0 + band).all(1) | (t[..., 0] > x1 - band).all(1) | (t[..., 2] < z0 + band).all(1) | (t[..., 2] > z1 - band).all(1)
    floor = (t[..., 1] < band * 0.6).all(1)
    mesh.remove_triangles_by_mask(beyond | floor | on(0, x0) | on(0, x1) | on(2, z0) | on(2, z1))
    mesh.remove_unreferenced_vertices()
    labels, counts, _ = mesh.cluster_connected_triangles()
    labels, counts = np.asarray(labels), np.asarray(counts)
    mesh.remove_triangles_by_mask(counts[labels] < 150)
    mesh.remove_unreferenced_vertices()
    return mesh


def fuse_room(cams: list[Cam], box, margin=0.35, max_tris=90000, structure=False):
    mesh = tsdf_mesh(cams, box, margin)
    if mesh is None or len(mesh.triangles) == 0:
        return None
    if structure:
        mesh = drop_structure(mesh, box)
        if len(mesh.triangles) == 0:
            return None
    if len(mesh.triangles) > max_tris:
        colors = mesh.vertex_colors
        mesh = mesh.simplify_quadric_decimation(max_tris)
        mesh.remove_degenerate_triangles()
        if not mesh.has_vertex_colors():
            mesh.vertex_colors = colors
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.triangles)
    C = (np.asarray(mesh.vertex_colors) * 255).clip(0, 255).astype(np.uint8)
    best, uvs = assign_views(V, F, cams)
    order = np.argsort(np.where(best < 0, len(cams), best), kind="stable")
    best, uvs, F = best[order], uvs[order], F[order]
    segments = []
    for i, c in enumerate(cams):
        idx = np.flatnonzero(best == i)
        if len(idx):
            segments.append({"view": c.name, "start": int(idx[0]), "count": int(len(idx))})
    rest = np.flatnonzero(best < 0)
    if len(rest):
        segments.append({"view": None, "start": int(rest[0]), "count": int(len(rest))})
    return {"pos": V[F].astype(np.float32), "uv": uvs, "col": C[F], "segments": segments}


def write_mesh(path, fused, off):
    path.parent.mkdir(parents=True, exist_ok=True)
    pos = (fused["pos"] + np.asarray(off, np.float32)).astype(np.float32)
    with open(path, "wb") as fh:
        fh.write(pos.tobytes())
        fh.write(fused["uv"].astype(np.float32).tobytes())
        fh.write(fused["col"].astype(np.uint8).tobytes())
    return {"triangles": int(len(pos)), "segments": fused["segments"]}
