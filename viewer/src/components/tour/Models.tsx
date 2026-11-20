import { useEffect, useMemo } from "react"
import { useLoader } from "@react-three/fiber"
import * as THREE from "three"
import { GLTFLoader } from "three-stdlib"
import type { Room } from "@/lib/types"

type Props = { base: string; room: Room; clip: THREE.Plane[] }

export function Models({ base, room, clip }: Props) {
  const urls = useMemo(() => (room.models ?? []).map((m) => `${base}/${m.file}`), [base, room.models])
  const gltfs = useLoader(GLTFLoader, urls)

  const objects = useMemo(
    () =>
      gltfs.map((g) => {
        const root = g.scene.clone(true)
        root.traverse((o) => {
          const mesh = o as THREE.Mesh
          if (!mesh.isMesh) return
          const src = mesh.material as THREE.MeshStandardMaterial
          if (src.map) src.map.colorSpace = THREE.SRGBColorSpace
          mesh.material = new THREE.MeshBasicMaterial({ map: src.map, side: THREE.DoubleSide, toneMapped: false, clippingPlanes: clip })
        })
        return root
      }),
    [gltfs, clip],
  )

  useEffect(
    () => () =>
      objects.forEach((o) =>
        o.traverse((m) => {
          const mesh = m as THREE.Mesh
          if (mesh.isMesh) (mesh.material as THREE.Material).dispose()
        }),
      ),
    [objects],
  )

  return (
    <group position={room.place?.position} rotation={[0, room.place?.rotation ?? 0, 0]}>
      {objects.map((o, i) => (
        <primitive key={urls[i]} object={o} />
      ))}
    </group>
  )
}
