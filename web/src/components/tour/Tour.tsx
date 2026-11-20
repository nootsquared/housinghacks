import { Suspense, useEffect, useMemo, useRef } from "react"
import { Canvas, useFrame, useThree } from "@react-three/fiber"
import { Html, OrbitControls } from "@react-three/drei"
import { EffectComposer, SMAA } from "@react-three/postprocessing"
import * as THREE from "three"
import type { OrbitControls as OrbitImpl } from "three-stdlib"
import type { Room, Scene, View } from "@/lib/types"
import { PhotoMesh } from "./PhotoMesh"
import { FilledShell, RoomModel } from "./RoomModel"

export type TourMode = "walk" | "dollhouse"

type Props = {
  scene: Scene
  views: View[]
  mode: TourMode
  current: number
  onSelectView: (i: number) => void
}

const ease = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2)

export function Tour(props: Props) {
  return (
    <Canvas
      shadows
      dpr={[1, 2]}
      gl={{ antialias: false, toneMapping: THREE.NeutralToneMapping, toneMappingExposure: 1.0, localClippingEnabled: true }}
      camera={{ fov: 50, near: 0.05, far: 200 }}
      className="absolute! inset-0"
    >
      <color attach="background" args={["#efede8"]} />
      <Suspense fallback={null}>
        <TourScene {...props} />
      </Suspense>
    </Canvas>
  )
}

function Lights({ scene }: { scene: Scene }) {
  const { min, max } = scene.bounds
  const cx = (min[0] + max[0]) / 2
  const cz = (min[2] + max[2]) / 2
  const span = Math.max(max[0] - min[0], max[2] - min[2]) / 2 + 2
  return (
    <>
      <hemisphereLight args={["#ffffff", "#e6dfd2", 2.1]} />
      <ambientLight intensity={0.55} />
      <directionalLight
        position={[cx + span * 0.15, 12, cz + span * 0.25]}
        intensity={1.1}
        castShadow
        shadow-mapSize={[2048, 2048]}
        shadow-bias={-0.0004}
        shadow-normalBias={0.02}
        shadow-camera-left={-span}
        shadow-camera-right={span}
        shadow-camera-top={span}
        shadow-camera-bottom={-span}
        shadow-camera-near={1}
        shadow-camera-far={30}
      >
        <object3D attach="target" position={[cx, 0, cz]} />
      </directionalLight>
    </>
  )
}

function TourScene({ scene, views, mode, current, onSelectView }: Props) {
  const center = useMemo(() => {
    const { min, max } = scene.bounds
    return new THREE.Vector3((min[0] + max[0]) / 2, 0, (min[2] + max[2]) / 2)
  }, [scene])

  const clip = useMemo(() => (mode === "dollhouse" ? [new THREE.Plane(new THREE.Vector3(0, -1, 0), 2.1)] : []), [mode])

  return (
    <>
      <Lights scene={scene} />
      {scene.rooms.map((r) =>
        r.mesh ? (
          <group key={r.id}>
            <PhotoMesh base={`/data/${scene.id}`} room={r} clip={clip} />
            <FilledShell room={r} base={`/data/${scene.id}`} mode={mode} />
          </group>
        ) : (
          <RoomModel key={r.id} room={r} mode={mode} />
        ),
      )}
      <Rig scene={scene} views={views} mode={mode} current={current} center={center} />
      {scene.rooms.filter((r) => r.model || r.mesh).map((r) => <RoomLabel key={r.id} room={r} hidden={mode !== "dollhouse"} />)}
      {views.map((v, i) => <ViewMarker key={v.id} view={v} hidden={mode !== "dollhouse"} onClick={() => onSelectView(i)} />)}
      <EffectComposer multisampling={0} enableNormalPass={false}>
        <SMAA />
      </EffectComposer>
    </>
  )
}

function coverFov(view: View, viewport: number) {
  const t = Math.tan(THREE.MathUtils.degToRad(view.fov) / 2)
  const fitWidth = (t * view.aspect) / viewport
  return THREE.MathUtils.radToDeg(2 * Math.atan(Math.min(t, fitWidth)))
}

function Rig({
  scene,
  views,
  mode,
  current,
  center,
}: {
  scene: Scene
  views: View[]
  mode: TourMode
  current: number
  center: THREE.Vector3
}) {
  const { camera, gl, size } = useThree()
  const cam = camera as THREE.PerspectiveCamera
  const controls = useRef<OrbitImpl>(null)
  const anim = useRef<{
    t: number
    from: { p: THREE.Vector3; q: THREE.Quaternion; fov: number }
    to: { p: THREE.Vector3; q: THREE.Quaternion; fov: number }
  } | null>(null)
  const look = useRef({ yaw: 0, pitch: 0, dragging: false, x: 0, y: 0 })
  const base = useRef(new THREE.Quaternion())

  const overview = useMemo(() => {
    const { min, max } = scene.bounds
    const span = Math.max(max[0] - min[0], max[2] - min[2], 4)
    const p = new THREE.Vector3(center.x + span * 0.35, span * 1.05, center.z + span * 0.95)
    const q = new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().lookAt(p, center, new THREE.Vector3(0, 1, 0)))
    return { p, q, fov: 40 }
  }, [scene, center])

  useEffect(() => {
    const view = views[current]
    const walk = mode === "walk" && !!view
    const to = walk
      ? { p: new THREE.Vector3(...view.position), q: new THREE.Quaternion(...view.quaternion), fov: coverFov(view, size.width / size.height) }
      : overview
    const from = anim.current
      ? { p: cam.position.clone(), q: cam.quaternion.clone(), fov: cam.fov }
      : { p: overview.p.clone().sub(center).multiplyScalar(1.3).add(center), q: overview.q.clone(), fov: overview.fov }
    anim.current = { t: 0, from, to }
    look.current.yaw = 0
    look.current.pitch = 0
  }, [mode, current, views, overview, cam, center, size])

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
      look.current.yaw += (e.clientX - look.current.x) * 0.004
      look.current.pitch = THREE.MathUtils.clamp(look.current.pitch + (e.clientY - look.current.y) * 0.004, -0.6, 0.6)
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
    if (!a) return
    if (a.t < 1) {
      a.t = Math.min(1, a.t + Math.min(dt, 0.05) / 1.3)
      const k = ease(a.t)
      cam.position.lerpVectors(a.from.p, a.to.p, k)
      base.current.slerpQuaternions(a.from.q, a.to.q, k)
      cam.fov = THREE.MathUtils.lerp(a.from.fov, a.to.fov, k)
      cam.updateProjectionMatrix()
      if (a.t === 1 && controls.current && mode === "dollhouse") {
        controls.current.target.copy(center)
        controls.current.update()
      }
    }
    if (mode === "walk") {
      const yaw = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), -look.current.yaw)
      const pitch = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), -look.current.pitch)
      cam.quaternion.copy(yaw).multiply(base.current).multiply(pitch)
    } else if (a.t < 1) {
      cam.quaternion.copy(base.current)
    }
  })

  return (
    <OrbitControls
      ref={controls}
      enabled={mode === "dollhouse"}
      enableDamping
      dampingFactor={0.08}
      maxPolarAngle={Math.PI * 0.42}
      minDistance={3}
      maxDistance={45}
    />
  )
}

function RoomLabel({ room, hidden }: { room: Room; hidden: boolean }) {
  return (
    <Html position={[room.center[0], room.size[1] + 0.3, room.center[2]]} center className="pointer-events-none" zIndexRange={[10, 0]}>
      <span style={{ opacity: hidden ? 0 : 1 }} className="transition-opacity duration-300 rounded-md bg-card/95 px-2 py-0.5 text-[11px] font-medium whitespace-nowrap shadow-sm ring-1 ring-border">
        {room.label}
        <span className="ml-1.5 text-muted-foreground tabular">
          {(room.size[0] * 3.281).toFixed(0)}′ × {(room.size[2] * 3.281).toFixed(0)}′
        </span>
      </span>
    </Html>
  )
}

function ViewMarker({ view, hidden, onClick }: { view: View; hidden: boolean; onClick: () => void }) {
  return (
    <Html position={[view.position[0], 0.02, view.position[2]]} center zIndexRange={[20, 10]}>
      <button
        onClick={onClick}
        style={{ opacity: hidden ? 0 : 1, pointerEvents: hidden ? "none" : "auto" }}
        className="group grid size-5 place-items-center rounded-full bg-card/90 shadow-sm ring-1 ring-border transition hover:scale-110"
        aria-label="Stand here"
      >
        <span className="size-1.5 rounded-full bg-foreground transition group-hover:bg-brand" />
      </button>
    </Html>
  )
}
