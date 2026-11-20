import { Suspense, useEffect, useMemo, useRef, useState } from "react"
import { Canvas, useFrame, useThree } from "@react-three/fiber"
import { Html, OrbitControls, useTexture } from "@react-three/drei"
import * as THREE from "three"
import type { OrbitControls as OrbitImpl } from "three-stdlib"
import { asset } from "@/lib/data"
import type { Room, Scene, Surface, View } from "@/lib/types"
import { buildDepthGeometry, loadDepth, type DepthGrid } from "./depthMesh"

export type TourMode = "walk" | "dollhouse"

type Props = {
  id: string
  scene: Scene
  mode: TourMode
  current: number
  onSelectView: (i: number) => void
}

type Fade = { value: number }

const ease = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2)

export function Tour(props: Props) {
  return (
    <Canvas
      dpr={[1, 2]}
      gl={{ antialias: true, toneMapping: THREE.NoToneMapping }}
      camera={{ fov: 50, near: 0.02, far: 200 }}
      className="absolute! inset-0"
    >
      <color attach="background" args={["#efede8"]} />
      <Suspense fallback={null}>
        <TourScene {...props} />
      </Suspense>
    </Canvas>
  )
}

function TourScene({ id, scene, mode, current, onSelectView }: Props) {
  const fades = useMemo(() => new Map<string, Fade>(), [])
  const fade = (k: string) => {
    if (!fades.has(k)) fades.set(k, { value: 0 })
    return fades.get(k)!
  }
  const shellFade = fade("__shell")
  const center = useMemo(() => {
    const { min, max } = scene.bounds
    return new THREE.Vector3((min[0] + max[0]) / 2, 0, (min[2] + max[2]) / 2)
  }, [scene])

  return (
    <>
      {scene.rooms.map((r) =>
        r.shell?.surfaces.map((s) => <ShellSurface key={r.id + s.name} id={id} surface={s} fade={shellFade} />),
      )}
      {scene.views.map((v) => (
        <ViewMesh key={v.id} id={id} view={v} fade={fade(v.id)} objects={fade(`${v.id}:obj`)} />
      ))}
      <Rig scene={scene} mode={mode} current={current} fades={fades} center={center} />
      {mode === "dollhouse" && scene.rooms.map((r) => <RoomLabel key={r.id} room={r} />)}
      {mode === "dollhouse" &&
        scene.views.map((v, i) => <ViewMarker key={v.id} view={v} onClick={() => onSelectView(i)} />)}
    </>
  )
}

function ShellSurface({ id, surface, fade }: { id: string; surface: Surface; fade: Fade }) {
  const tex = useTexture(asset(id, surface.texture))
  const mat = useRef<THREE.MeshBasicMaterial>(null)
  const mesh = useRef<THREE.Mesh>(null)
  const geometry = useMemo(() => {
    const o = new THREE.Vector3(...surface.origin)
    const u = new THREE.Vector3(...surface.u)
    const v = new THREE.Vector3(...surface.v)
    const p = [o, o.clone().add(u), o.clone().add(v), o.clone().add(u).add(v)]
    const g = new THREE.BufferGeometry()
    g.setAttribute("position", new THREE.Float32BufferAttribute(p.flatMap((x) => x.toArray()), 3))
    g.setAttribute("uv", new THREE.Float32BufferAttribute([0, 1, 1, 1, 0, 0, 1, 0], 2))
    g.setIndex([0, 2, 1, 2, 3, 1])
    return g
  }, [surface])

  useEffect(() => {
    tex.colorSpace = THREE.SRGBColorSpace
    tex.anisotropy = 8
  }, [tex])

  useFrame(() => {
    if (!mat.current || !mesh.current) return
    mat.current.opacity = fade.value
    mesh.current.visible = fade.value > 0.002
  })

  return (
    <mesh ref={mesh} geometry={geometry} renderOrder={-1}>
      <meshBasicMaterial
        ref={mat}
        map={tex}
        side={surface.name === "floor" ? THREE.DoubleSide : THREE.FrontSide}
        transparent
        polygonOffset
        polygonOffsetFactor={2}
        polygonOffsetUnits={2}
        toneMapped={false}
      />
    </mesh>
  )
}

function ViewMesh({ id, view, fade, objects }: { id: string; view: View; fade: Fade; objects: Fade }) {
  const texture = useTexture(asset(id, view.file))
  const [grid, setGrid] = useState<DepthGrid | null>(null)
  const full = useRef<THREE.Mesh>(null)
  const obj = useRef<THREE.Mesh>(null)

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

  const geometries = useMemo(
    () => (grid ? [buildDepthGeometry(view, grid), buildDepthGeometry(view, grid, true)] : null),
    [grid, view],
  )

  useFrame(() => {
    const pairs: [THREE.Mesh | null, Fade][] = [
      [full.current, fade],
      [obj.current, objects],
    ]
    for (const [m, f] of pairs) {
      if (!m) continue
      ;(m.material as THREE.MeshBasicMaterial).opacity = f.value
      m.visible = f.value > 0.002
    }
  })

  if (!geometries) return null
  return (
    <group position={view.position} quaternion={view.quaternion}>
      <mesh ref={full} geometry={geometries[0]} renderOrder={2}>
        <meshBasicMaterial map={texture} transparent side={THREE.DoubleSide} toneMapped={false} />
      </mesh>
      <mesh ref={obj} geometry={geometries[1]}>
        <meshBasicMaterial map={texture} transparent side={THREE.DoubleSide} toneMapped={false} />
      </mesh>
    </group>
  )
}

function Rig({
  scene,
  mode,
  current,
  fades,
  center,
}: {
  scene: Scene
  mode: TourMode
  current: number
  fades: Map<string, Fade>
  center: THREE.Vector3
}) {
  const { camera, gl, size } = useThree()
  const cam = camera as THREE.PerspectiveCamera
  const controls = useRef<OrbitImpl>(null)
  const anim = useRef<{
    t: number
    from: { p: THREE.Vector3; q: THREE.Quaternion; fov: number }
    to: { p: THREE.Vector3; q: THREE.Quaternion; fov: number }
    start: Map<string, number>
    target: Map<string, number>
  } | null>(null)
  const look = useRef({ yaw: 0, pitch: 0, dragging: false, x: 0, y: 0 })
  const base = useRef(new THREE.Quaternion())

  const overview = useMemo(() => {
    const { min, max } = scene.bounds
    const span = Math.max(max[0] - min[0], max[2] - min[2], 3)
    const p = new THREE.Vector3(center.x + span * 0.7, span * 1.35, center.z + span * 1.1)
    const q = new THREE.Quaternion().setFromRotationMatrix(
      new THREE.Matrix4().lookAt(p, center, new THREE.Vector3(0, 1, 0)),
    )
    return { p, q, fov: 40 }
  }, [scene, center])

  useEffect(() => {
    const view = scene.views[current]
    const walk = mode === "walk" && !!view
    const to = walk
      ? { p: new THREE.Vector3(...view.position), q: new THREE.Quaternion(...view.quaternion), fov: coverFov(view, size.width / size.height) }
      : overview
    const target = new Map<string, number>()
    target.set("__shell", walk ? 0 : 1)
    for (const v of scene.views) {
      target.set(v.id, walk && v.id === view.id ? 1 : 0)
      target.set(`${v.id}:obj`, walk ? 0 : 1)
    }
    const start = new Map<string, number>()
    fades.forEach((f, k) => start.set(k, f.value))
    const from = anim.current
      ? { p: cam.position.clone(), q: cam.quaternion.clone(), fov: cam.fov }
      : { p: overview.p.clone().sub(center).multiplyScalar(1.35).add(center), q: overview.q.clone(), fov: overview.fov }
    anim.current = { t: 0, from, to, start, target }
    look.current.yaw = 0
    look.current.pitch = 0
  }, [mode, current, scene, overview, cam, fades, center, size])

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
      const k = 0.003
      look.current.yaw = THREE.MathUtils.clamp(look.current.yaw + (e.clientX - look.current.x) * k, -0.5, 0.5)
      look.current.pitch = THREE.MathUtils.clamp(look.current.pitch + (e.clientY - look.current.y) * k, -0.3, 0.3)
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
      a.t = Math.min(1, a.t + dt / 1.2)
      const k = ease(a.t)
      cam.position.lerpVectors(a.from.p, a.to.p, k)
      base.current.slerpQuaternions(a.from.q, a.to.q, k)
      cam.fov = THREE.MathUtils.lerp(a.from.fov, a.to.fov, k)
      cam.updateProjectionMatrix()
      a.target.forEach((to, key) => {
        const f = fades.get(key)
        if (!f) return
        const s = a.start.get(key) ?? f.value
        const kk = to > s ? Math.min(1, k * 1.6) : Math.max(0, (k - 0.25) / 0.75)
        f.value = THREE.MathUtils.lerp(s, to, kk)
      })
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
      minDistance={2}
      maxDistance={40}
    />
  )
}

function coverFov(view: View, viewport: number) {
  const t = Math.tan(THREE.MathUtils.degToRad(view.fov) / 2)
  const fitWidth = (t * view.aspect) / viewport
  return THREE.MathUtils.radToDeg(2 * Math.atan(Math.min(t, fitWidth))) * 0.97
}

function RoomLabel({ room }: { room: Room }) {
  return (
    <Html position={[room.center[0], 0.05, room.center[2]]} center className="pointer-events-none" zIndexRange={[10, 0]}>
      <span className="rounded-md bg-card/95 px-2 py-0.5 text-[11px] font-medium whitespace-nowrap shadow-sm ring-1 ring-border">
        {room.label}
        <span className="ml-1.5 text-muted-foreground tabular">
          {(room.size[0] * 3.281).toFixed(0)}′ × {(room.size[2] * 3.281).toFixed(0)}′
        </span>
      </span>
    </Html>
  )
}

function ViewMarker({ view, onClick }: { view: View; onClick: () => void }) {
  return (
    <Html position={[view.position[0], 0.02, view.position[2]]} center zIndexRange={[20, 10]}>
      <button
        onClick={onClick}
        className="group grid size-6 place-items-center rounded-full bg-card/90 shadow-sm ring-1 ring-border transition hover:scale-110"
        aria-label="Stand here"
      >
        <span className="size-2 rounded-full bg-foreground transition group-hover:bg-brand" />
      </button>
    </Html>
  )
}
