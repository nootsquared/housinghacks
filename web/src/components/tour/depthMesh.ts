import * as THREE from "three"
import type { View } from "@/lib/types"

export type DepthGrid = { w: number; h: number; depth: Float32Array; conf?: Uint8Array }

const gridCache = new Map<string, Promise<DepthGrid>>()

export function loadDepth(url: string): Promise<DepthGrid> {
  if (!gridCache.has(url)) {
    gridCache.set(
      url,
      fetch(url)
        .then((r) => r.arrayBuffer())
        .then((buf) => {
          const head = new Uint16Array(buf, 0, 2)
          const w = head[0]
          const h = head[1]
          const mm = new Uint16Array(buf, 4, w * h)
          const depth = new Float32Array(w * h)
          for (let i = 0; i < mm.length; i++) depth[i] = mm[i] / 1000
          const confOffset = 4 + w * h * 2
          const conf = buf.byteLength >= confOffset + w * h ? new Uint8Array(buf, confOffset, w * h) : undefined
          return { w, h, depth, conf }
        }),
    )
  }
  return gridCache.get(url)!
}

export function buildDepthGeometry(view: View, grid: DepthGrid, edgeRatio = 0.06) {
  const { w, h, depth, conf } = grid
  const t = Math.tan(THREE.MathUtils.degToRad(view.fov) / 2)
  const positions = new Float32Array(w * h * 3)
  const uvs = new Float32Array(w * h * 2)
  for (let j = 0; j < h; j++) {
    for (let i = 0; i < w; i++) {
      const k = j * w + i
      const u = (i + 0.5) / w
      const v = (j + 0.5) / h
      const d = depth[k]
      positions[k * 3] = (u * 2 - 1) * t * view.aspect * d
      positions[k * 3 + 1] = (1 - v * 2) * t * d
      positions[k * 3 + 2] = -d
      uvs[k * 2] = u
      uvs[k * 2 + 1] = 1 - v
    }
  }
  const index: number[] = []
  const ok = (k: number) => depth[k] > 0.05 && (!conf || conf[k] > 20)
  const close = (a: number, b: number) => Math.abs(depth[a] - depth[b]) < edgeRatio * Math.min(depth[a], depth[b]) * 4
  for (let j = 0; j < h - 1; j++) {
    for (let i = 0; i < w - 1; i++) {
      const a = j * w + i
      const b = a + 1
      const c = a + w
      const d = c + 1
      if (!ok(a) || !ok(b) || !ok(c) || !ok(d)) continue
      if (close(a, b) && close(a, c) && close(b, c)) index.push(a, c, b)
      if (close(b, d) && close(c, d) && close(b, c)) index.push(b, c, d)
    }
  }
  const g = new THREE.BufferGeometry()
  g.setAttribute("position", new THREE.BufferAttribute(positions, 3))
  g.setAttribute("uv", new THREE.BufferAttribute(uvs, 2))
  g.setIndex(index)
  g.computeBoundingSphere()
  return g
}
