import { Suspense, useEffect, useMemo, useRef, useState } from "react"
import { Canvas, useFrame, useThree } from "@react-three/fiber"
import { Html, OrbitControls, useTexture } from "@react-three/drei"
import * as THREE from "three"
import type { OrbitControls as OrbitImpl } from "three-stdlib"
import { asset } from "@/lib/data"
import type { Room, Scene, View } from "@/lib/types"
import { buildDepthGeometry, loadDepth, type DepthGrid } from "./depthMesh"

export type TourMode = "walk" | "dollhouse"

type Props = {
  id: string
  scene: Scene
  mode: TourMode
  current: number
  onSelectView: (i: number) => void
}

const ease = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2)

export function Tour(props: Props) {
  return (
    <Canvas
      dpr={[1, 2]}
      gl={{ antialias: true, toneMapping: THREE.NoToneMapping }}
      camera={{ fov: 60, near: 0.02, far: 200 }}
      className="!absolute inset-0"
      onCreated={(s) => {
        const tick = () => {
          const out: unknown[] = []
          s.scene.traverse((o) => {
            const m = o as THREE.Mesh
            if (m.isMesh) out.push([m.visible, (m.material as THREE.Material).opacity, m.geometry.index?.count])
          })
          document.body.dataset.debug = JSON.stringify({ cam: s.camera.position.toArray(), n: out.length, out: out.slice(0, 9) })
        }
        setInterval(tick, 1000)
      }}
    >
      <color attach="background" args={[props.mode === "walk" ? "#16171b" : "#efede8"]} />
      <Suspense fallback={null}>
        <TourScene {...props} />
      </Suspense>
    </Canvas>
  )
}

function TourScene({ id, scene, mode, current, onSelectView }: Props) {
  const [opacity] = useState(() => new Map<string, { value: number }>())
  const views = scene.views
  const center = useMemo(() => {
    const { min, max } = scene.bounds
    return new THREE.Vector3((min[0] + max[0]) / 2, min[1], (min[2] + max[2]) / 2)
  }, [scene])

  return (
    <>
      {views.map((v) => (
        <ViewMesh key={v.id} id={id} view={v} opacity={getOpacity(opacity, v.id)} />
      ))}
      <Rig scene={scene} mode={mode} current={current} opacity={opacity} center={center} />
      {mode === "dollhouse" && (
        <>
          {scene.rooms.map((r) => (
            <RoomOutline key={r.id} room={r} />
          ))}
          {views.map((v, i) => (
            <ViewMarker key={v.id} view={v} onClick={() => onSelectView(i)} />
          ))}
        </>
      )}
    </>
  )
}

function getOpacity(map: Map<string, { value: number }>, key: string) {
  if (!map.has(key)) map.set(key, { value: 0 })
  return map.get(key)!
}

function ViewMesh({ id, view, opacity }: { id: string; view: View; opacity: { value: number } }) {
  const texture = useTexture(asset(id, view.file))
  const [grid, setGrid] = useState<DepthGrid | null>(null)
  const mat = useRef<THREE.MeshBasicMaterial>(null)
  const mesh = useRef<THREE.Mesh>(null)

  useEffect(() => {
    texture.colorSpace = THREE.SRGBColorSpace
    texture.anisotropy = 8
  }, [texture])

  useEffect(() => {
    let live = true
    loadDepth(asset(id, view.depth)).then((g) => live && setGrid(g))
    return () => {
      live = false
    }
  }, [id, view.depth])

  const geometry = useMemo(() => (grid ? buildDepthGeometry(view, grid) : null), [grid, view])

  useFrame(() => {
    if (!mat.current || !mesh.current) return
    mat.current.opacity = opacity.value
    mesh.current.visible = opacity.value > 0.002
    mesh.current.renderOrder = Math.round(opacity.value * 10)
  })

  if (!geometry) return null
  return (
    <mesh ref={mesh} geometry={geometry} position={view.position} quaternion={view.quaternion}>
      <meshBasicMaterial
        ref={mat}
        map={texture}
        transparent
        depthWrite
        side={THREE.DoubleSide}
        toneMapped={false}
      />
    </mesh>
  )
}

function Rig({
  scene,
  mode,
  current,
  opacity,
  center,
}: {
  scene: Scene
  mode: TourMode
  current: number
  opacity: Map<string, { value: number }>
  center: THREE.Vector3
}) {
  const { camera, gl } = useThree()
  const cam = camera as THREE.PerspectiveCamera
  const controls = useRef<OrbitImpl>(null)
  const anim = useRef<{
    t: number
    from: { p: THREE.Vector3; q: THREE.Quaternion; fov: number }
    to: { p: THREE.Vector3; q: THREE.Quaternion; fov: number }
    prev: string | null
  } | null>(null)
  const look = useRef({ yaw: 0, pitch: 0, dragging: false, x: 0, y: 0 })
  const base = useRef(new THREE.Quaternion())
  const shown = useRef<string | null>(null)

  const overview = useMemo(() => {
    const { min, max } = scene.bounds
    const span = Math.max(max[0] - min[0], max[2] - min[2], 4)
    const p = new THREE.Vector3(center.x + span * 0.55, center.y + span * 1.1, center.z + span * 0.9)
    const q = new THREE.Quaternion().setFromRotationMatrix(
      new THREE.Matrix4().lookAt(p, center, new THREE.Vector3(0, 1, 0)),
    )
    return { p, q, fov: 45 }
  }, [scene, center])

  useEffect(() => {
    const view = scene.views[current]
    const to =
      mode === "walk" && view
        ? {
            p: new THREE.Vector3(...view.position),
            q: new THREE.Quaternion(...view.quaternion),
            fov: Math.min(view.fov + 4, 90),
          }
        : overview
    anim.current = {
      t: 0,
      from: { p: cam.position.clone(), q: cam.quaternion.clone(), fov: cam.fov },
      to,
      prev: shown.current,
    }
    look.current.yaw = 0
    look.current.pitch = 0
    if (mode === "walk" && view) shown.current = view.id
  }, [mode, current, scene, overview, cam])

  useEffect(() => {
    if (mode !== "walk") return
    const el = gl.domElement
    const down = (e: PointerEvent) => {
      look.current.dragging = true
      look.current.x = e.clientX
      look.current.y = e.clientY
    }
    const move = (e: PointerEvent) => {
      if (!look.current.dragging) return
      const k = 0.0025
      look.current.yaw = THREE.MathUtils.clamp(look.current.yaw + (e.clientX - look.current.x) * k, -0.6, 0.6)
      look.current.pitch = THREE.MathUtils.clamp(look.current.pitch + (e.clientY - look.current.y) * k, -0.35, 0.35)
      look.current.x = e.clientX
      look.current.y = e.clientY
    }
    const up = () => (look.current.dragging = false)
    el.addEventListener("pointerdown", down)
    window.addEventListener("pointermove", move)
    window.addEventListener("pointerup", up)
    return () => {
      el.removeEventListener("pointerdown", down)
      window.removeEventListener("pointermove", move)
      window.removeEventListener("pointerup", up)
    }
  }, [mode, gl])

  useFrame((_, dt) => {
    const a = anim.current
    if (a && a.t < 1) {
      a.t = Math.min(1, a.t + dt / 1.1)
      const k = ease(a.t)
      cam.position.lerpVectors(a.from.p, a.to.p, k)
      base.current.slerpQuaternions(a.from.q, a.to.q, k)
      cam.fov = THREE.MathUtils.lerp(a.from.fov, a.to.fov, k)
      cam.updateProjectionMatrix()
      for (const v of scene.views) {
        const o = opacity.get(v.id)
        if (!o) continue
        if (mode === "dollhouse") o.value = THREE.MathUtils.lerp(o.value, 1, k)
        else if (v.id === shown.current) o.value = Math.max(o.value, k)
        else if (v.id === a.prev) o.value = 1 - k * k
        else o.value = Math.min(o.value, 1 - k)
      }
      if (a.t === 1 && controls.current) {
        controls.current.target.copy(center)
        controls.current.update()
      }
    }
    if (mode === "walk" && (!a || a.t >= 1 || a.t > 0)) {
      const yaw = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), look.current.yaw)
      const pitch = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), look.current.pitch)
      cam.quaternion.copy(yaw).multiply(base.current).multiply(pitch)
    } else if (a && a.t < 1) {
      cam.quaternion.copy(base.current)
    }
  })

  const settled = mode === "dollhouse" && (!anim.current || anim.current.t >= 1)
  return (
    <OrbitControls
      ref={controls}
      enabled={settled}
      enableDamping
      dampingFactor={0.08}
      maxPolarAngle={Math.PI * 0.45}
      minDistance={2}
      maxDistance={40}
    />
  )
}

function RoomOutline({ room }: { room: Room }) {
  const geo = useMemo(() => new THREE.EdgesGeometry(new THREE.BoxGeometry(...room.size)), [room])
  const color = room.confidence === "inferred" ? "#9a9890" : "#23252b"
  return (
    <group position={room.center} rotation={[0, room.yaw, 0]}>
      <lineSegments geometry={geo}>
        <lineBasicMaterial color={color} transparent opacity={0.5} />
      </lineSegments>
      <Html position={[0, room.size[1] / 2 + 0.25, 0]} center className="pointer-events-none">
        <span className="rounded-md bg-card/90 px-2 py-0.5 text-[11px] font-medium whitespace-nowrap shadow-sm ring-1 ring-border">
          {room.label}
        </span>
      </Html>
    </group>
  )
}

function ViewMarker({ view, onClick }: { view: View; onClick: () => void }) {
  const [hover, setHover] = useState(false)
  return (
    <group position={view.position}>
      <mesh
        onClick={(e) => {
          e.stopPropagation()
          onClick()
        }}
        onPointerOver={() => setHover(true)}
        onPointerOut={() => setHover(false)}
      >
        <sphereGeometry args={[hover ? 0.16 : 0.12, 24, 16]} />
        <meshBasicMaterial color={hover ? "#d9683a" : "#23252b"} />
      </mesh>
    </group>
  )
}
