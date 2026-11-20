import { useEffect, useMemo, useState } from "react"
import { useThree } from "@react-three/fiber"
import * as THREE from "three"
import type { Room, View } from "@/lib/types"
import type { Door } from "./plan"

type Part = { geometry: THREE.BufferGeometry; texture?: THREE.Texture }

const loader = new THREE.TextureLoader()
const cache = new Map<string, Promise<ArrayBuffer>>()

function fetchMesh(url: string) {
  if (!cache.has(url)) cache.set(url, fetch(url).then((r) => r.arrayBuffer()))
  return cache.get(url)!
}

function texture(url: string, anisotropy: number) {
  const t = loader.load(url)
  t.colorSpace = THREE.SRGBColorSpace
  t.anisotropy = anisotropy
  return t
}

function gather<T extends Float32Array | Uint8Array>(src: T, tris: number[], k: number): T {
  const out = new (src.constructor as new (n: number) => T)(tris.length * k)
  tris.forEach((t, i) => out.set(src.subarray(t * k, t * k + k), i * k))
  return out
}

function inDoorway(doors: Door[], x: number, z: number) {
  return doors.some((d) => {
    const [u, v] = d.axis === "x" ? [x, z] : [z, x]
    return Math.abs(v - d.at) < 0.45 && u > d.from - 0.3 && u < d.to + 0.3
  })
}

function stray(pos: Float32Array, n: number, room: Room, doors: Door[]) {
  const keep = new Uint8Array(n)
  const shell = room.shell
  const [x0, x1] = shell?.x ?? [-Infinity, Infinity]
  const [z0, z1] = shell?.z ?? [-Infinity, Infinity]
  const h = shell?.height ?? Infinity
  const m = 0.12
  const a = new THREE.Vector3()
  const b = new THREE.Vector3()
  const c = new THREE.Vector3()
  for (let t = 0; t < n; t++) {
    const o = t * 9
    a.set(pos[o], pos[o + 1], pos[o + 2])
    b.set(pos[o + 3], pos[o + 4], pos[o + 5])
    c.set(pos[o + 6], pos[o + 7], pos[o + 8])
    const cx = (a.x + b.x + c.x) / 3
    const cy = (a.y + b.y + c.y) / 3
    const cz = (a.z + b.z + c.z) / 3
    if (cx < x0 - m || cx > x1 + m || cz < z0 - m || cz > z1 + m || cy < -0.03 || cy > h + 0.05) continue
    if (inDoorway(doors, cx, cz)) continue
    b.sub(a)
    c.sub(a)
    const nrm = b.cross(c).normalize()
    const flat = Math.abs(nrm.y) > 0.85
    if (flat && (cy < 0.012 || cy > h - 0.2)) continue
    if (Math.abs(nrm.x) > 0.85 && Math.min(Math.abs(cx - x0), Math.abs(cx - x1)) < 0.025) continue
    if (Math.abs(nrm.z) > 0.85 && Math.min(Math.abs(cz - z0), Math.abs(cz - z1)) < 0.025) continue
    keep[t] = 1
  }
  return keep
}

function inside(room: Room, view: View | undefined) {
  if (room.cloneOf) return true
  if (!room.shell || !view) return true
  const [x, , z] = view.position
  const m = 0.3
  return x > room.shell.x[0] - m && x < room.shell.x[1] + m && z > room.shell.z[0] - m && z < room.shell.z[1] + m
}

type Props = { base: string; room: Room; views: View[]; doors: Door[]; clip: THREE.Plane[] }

export function PhotoMesh({ base, room, views, doors, clip }: Props) {
  const [buf, setBuf] = useState<ArrayBuffer | null>(null)
  const mesh = room.mesh

  useEffect(() => {
    if (!mesh) return
    let live = true
    fetchMesh(`${base}/${mesh.file}`).then((b) => live && setBuf(b))
    return () => {
      live = false
    }
  }, [base, mesh])

  const anisotropy = useThree((s) => s.gl.capabilities.getMaxAnisotropy())

  const parts = useMemo<Part[]>(() => {
    if (!buf || !mesh) return []
    const n = mesh.triangles
    const pos = new Float32Array(buf, 0, n * 9)
    const uv = new Float32Array(buf, n * 36, n * 6)
    const col = new Uint8Array(buf, n * 60, n * 9)
    const keep = stray(pos, n, room, doors)
    return mesh.segments.filter((s) => !s.view || inside(room, views.find((v) => v.id === s.view))).map((s) => {
      const tris: number[] = []
      for (let t = s.start; t < s.start + s.count; t++) if (keep[t]) tris.push(t)
      const g = new THREE.BufferGeometry()
      g.setAttribute("position", new THREE.BufferAttribute(gather(pos, tris, 9), 3))
      if (s.view) {
        g.setAttribute("uv", new THREE.BufferAttribute(gather(uv, tris, 6), 2))
        return { geometry: g, texture: texture(`${base}/views/${s.view}.jpg`, anisotropy) }
      }
      g.setAttribute("color", new THREE.BufferAttribute(gather(col, tris, 9), 3, true))
      return { geometry: g }
    })
  }, [buf, mesh, base, room, views, doors, anisotropy])

  useEffect(() => () => parts.forEach((p) => (p.geometry.dispose(), p.texture?.dispose())), [parts])

  return (
    <group>
      {parts.map((p, i) => (
        <mesh key={i} geometry={p.geometry}>
          {p.texture ? (
            <meshBasicMaterial map={p.texture} side={THREE.DoubleSide} toneMapped={false} clippingPlanes={clip} polygonOffset polygonOffsetFactor={-2} polygonOffsetUnits={-2} />
          ) : (
            <meshBasicMaterial vertexColors side={THREE.DoubleSide} toneMapped={false} clippingPlanes={clip} />
          )}
        </mesh>
      ))}
    </group>
  )
}
