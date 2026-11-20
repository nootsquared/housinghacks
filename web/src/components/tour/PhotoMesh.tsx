import { useEffect, useMemo, useState } from "react"
import * as THREE from "three"
import type { Room } from "@/lib/types"

type Part = { geometry: THREE.BufferGeometry; texture?: THREE.Texture }

const loader = new THREE.TextureLoader()
const cache = new Map<string, Promise<ArrayBuffer>>()

function fetchMesh(url: string) {
  if (!cache.has(url)) cache.set(url, fetch(url).then((r) => r.arrayBuffer()))
  return cache.get(url)!
}

function texture(url: string) {
  const t = loader.load(url)
  t.colorSpace = THREE.SRGBColorSpace
  t.anisotropy = 8
  return t
}

export function PhotoMesh({ base, room, clip }: { base: string; room: Room; clip: THREE.Plane[] }) {
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

  const parts = useMemo<Part[]>(() => {
    if (!buf || !mesh) return []
    const n = mesh.triangles
    const pos = new Float32Array(buf, 0, n * 9)
    const uv = new Float32Array(buf, n * 36, n * 6)
    const col = new Uint8Array(buf, n * 60, n * 9)
    return mesh.segments.map((s) => {
      const g = new THREE.BufferGeometry()
      g.setAttribute("position", new THREE.BufferAttribute(pos.subarray(s.start * 9, (s.start + s.count) * 9), 3))
      if (s.view) {
        g.setAttribute("uv", new THREE.BufferAttribute(uv.subarray(s.start * 6, (s.start + s.count) * 6), 2))
        return { geometry: g, texture: texture(`${base}/views/${s.view}.jpg`) }
      }
      g.setAttribute("color", new THREE.BufferAttribute(col.subarray(s.start * 9, (s.start + s.count) * 9), 3, true))
      return { geometry: g }
    })
  }, [buf, mesh, base])

  useEffect(() => () => parts.forEach((p) => (p.geometry.dispose(), p.texture?.dispose())), [parts])

  return (
    <group>
      {parts.map((p, i) => (
        <mesh key={i} geometry={p.geometry}>
          {p.texture ? (
            <meshBasicMaterial map={p.texture} side={THREE.DoubleSide} toneMapped={false} clippingPlanes={clip} />
          ) : (
            <meshBasicMaterial vertexColors side={THREE.DoubleSide} toneMapped={false} clippingPlanes={clip} />
          )}
        </mesh>
      ))}
    </group>
  )
}
