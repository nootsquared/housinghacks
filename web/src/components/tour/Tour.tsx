import { Suspense, useEffect, useLayoutEffect, useMemo, useRef } from "react"
import { Canvas, useFrame, useThree } from "@react-three/fiber"
import { Html, OrbitControls } from "@react-three/drei"
import * as THREE from "three"
import type { OrbitControls as OrbitImpl } from "three-stdlib"
import type { Room, Scene, View } from "@/lib/types"
import { Models } from "./Models"
import { PhotoMesh } from "./PhotoMesh"
import { RoomModel } from "./RoomModel"
import { Staging } from "./Staging"
import { Shell } from "./Shell"
import { ringTexture } from "./materials"
import { Plan, type XZ } from "./plan"

export type TourMode = "walk" | "dollhouse"

type Props = {
  scene: Scene
  views: View[]
  mode: TourMode
  current: number
  atPhoto: boolean
  onSelectView: (i: number) => void
  onFreeMove: () => void
  onEnterWalk: () => void
}

const SPOT = 0.35
const ease = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2)

export function Tour(props: Props) {
  return (
    <Canvas
      flat
      dpr={[1, 2]}
      gl={{ antialias: true, localClippingEnabled: true, powerPreference: "high-performance" }}
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
  return (
    <>
      <ambientLight intensity={0.64 * Math.PI} />
      <hemisphereLight args={["#ffffff", "#ddd5c8", 0.3 * Math.PI]} />
      <directionalLight position={[cx + 4, 9, cz + 6]} intensity={0.1 * Math.PI}>
        <object3D attach="target" position={[cx, 0, cz]} />
      </directionalLight>
    </>
  )
}

function TourScene(props: Props) {
  const { scene, views, mode, atPhoto, current } = props
  const plan = useMemo(() => new Plan(scene), [scene])
  const shells = useRef<THREE.Group>(null)
  const photos = useRef<THREE.Group>(null)
  const clip = useMemo(() => (mode === "dollhouse" ? [new THREE.Plane(new THREE.Vector3(0, -1, 0), 2.1)] : []), [mode])
  const base = `/data/${scene.id}`

  return (
    <>
      <Lights scene={scene} />
      <group ref={shells}>
        {scene.rooms.map((r) =>
          r.shell && r.model ? <Shell key={r.id} room={r} doors={plan.doors} base={base} mode={mode} /> : null,
        )}
      </group>
      <group ref={photos}>
        {scene.rooms.map((r) =>
          r.staging?.length ? (
            <Staging key={r.id} room={r} clip={clip} mode={mode} />
          ) : r.models?.length ? (
            <Models key={r.id} base={base} room={r} clip={clip} />
          ) : r.model?.surfaces ? null : r.mesh ? (
            <PhotoMesh key={r.id} base={base} room={r} views={scene.views} doors={plan.doors} clip={clip} />
          ) : (
            <RoomModel key={r.id} room={r} mode={mode} />
          ),
        )}
      </group>
      <Rig {...props} plan={plan} shells={shells} photos={photos} />
      {scene.rooms.filter((r) => r.model || r.mesh).map((r) => <RoomLabel key={r.id} room={r} hidden={mode !== "dollhouse"} />)}
      <Spots views={views} mode={mode} skip={mode === "walk" && atPhoto ? current : -1} />
    </>
  )
}

function coverFov(view: View, viewport: number) {
  const t = Math.tan(THREE.MathUtils.degToRad(view.fov) / 2)
  const fitWidth = (t * view.aspect) / viewport
  return THREE.MathUtils.radToDeg(2 * Math.atan(Math.min(t, fitWidth)))
}

type Glide = {
  pts: THREE.Vector3[]
  lens: number[]
  total: number
  t: number
  dur: number
  q0: THREE.Quaternion
  q1: THREE.Quaternion
  f0: number
  f1: number
  done?: () => void
}

function roundCorners(pts: THREE.Vector3[], radius = 0.42, steps = 6) {
  if (pts.length < 3) return pts
  const out = [pts[0]]
  for (let i = 1; i < pts.length - 1; i++) {
    const a = pts[i - 1]
    const b = pts[i]
    const c = pts[i + 1]
    const ab = a.clone().sub(b)
    const cb = c.clone().sub(b)
    const lab = ab.length()
    const lcb = cb.length()
    const d = Math.min(radius, lab * 0.45, lcb * 0.45)
    if (d < 0.06 || lab < 1e-4 || lcb < 1e-4) {
      out.push(b)
      continue
    }
    const p0 = b.clone().add(ab.multiplyScalar(d / lab))
    const p1 = b.clone().add(cb.multiplyScalar(d / lcb))
    out.push(p0)
    for (let s = 1; s < steps; s++) {
      const t = s / steps
      const u = 1 - t
      out.push(new THREE.Vector3(
        u * u * p0.x + 2 * u * t * b.x + t * t * p1.x,
        u * u * p0.y + 2 * u * t * b.y + t * t * p1.y,
        u * u * p0.z + 2 * u * t * b.z + t * t * p1.z,
      ))
    }
    out.push(p1)
  }
  out.push(pts[pts.length - 1])
  return out
}

function glide(pts: THREE.Vector3[], q0: THREE.Quaternion, q1: THREE.Quaternion, f0: number, f1: number, dur?: number): Glide {
  pts = roundCorners(pts)
  const lens = [0]
  for (let i = 1; i < pts.length; i++) lens.push(lens[i - 1] + pts[i].distanceTo(pts[i - 1]))
  const total = lens[lens.length - 1]
  const turn = q0.angleTo(q1)
  return { pts, lens, total, t: 0, q0, q1, f0, f1, dur: dur ?? THREE.MathUtils.clamp(0.55 + total / 2.6 + turn * 0.25, 0.6, 2.4) }
}

function along(g: Glide, k: number, out: THREE.Vector3) {
  const s = k * g.total
  let i = 1
  while (i < g.lens.length - 1 && g.lens[i] < s) i++
  const seg = g.lens[i] - g.lens[i - 1]
  return out.lerpVectors(g.pts[i - 1], g.pts[i], seg > 0 ? (s - g.lens[i - 1]) / seg : 1)
}

const pick = { ray: new THREE.Raycaster(), ndc: new THREE.Vector2(), inside: false, hit: null as THREE.Vector3 | null, room: "" }

function Rig({
  scene,
  views,
  mode,
  current,
  atPhoto,
  onSelectView,
  onFreeMove,
  onEnterWalk,
  plan,
  shells,
  photos,
}: Props & { plan: Plan; shells: React.RefObject<THREE.Group | null>; photos: React.RefObject<THREE.Group | null> }) {
  const { camera, gl, size } = useThree()
  const cam = camera as THREE.PerspectiveCamera
  const controls = useRef<OrbitImpl>(null)
  const anim = useRef<Glide | null>(null)
  const look = useRef({ yaw: 0, pitch: 0, roll: 0, down: false, moved: 0, x: 0, y: 0 })
  const pending = useRef<XZ | null>(null)
  const last = useRef({ t: 0, p: new THREE.Vector3() })
  const prevMode = useRef<TourMode | null>(null)
  const cursor = useRef<THREE.Mesh>(null)
  const live = useRef({ mode, onSelectView, onFreeMove, onEnterWalk })
  useLayoutEffect(() => {
    live.current = { mode, onSelectView, onFreeMove, onEnterWalk }
  })

  const center = useMemo(() => {
    const { min, max } = scene.bounds
    return new THREE.Vector3((min[0] + max[0]) / 2, 0, (min[2] + max[2]) / 2)
  }, [scene])

  const eye = useMemo(() => {
    const ys = views.map((v) => v.position[1]).sort((a, b) => a - b)
    return ys[Math.floor(ys.length / 2)] ?? scene.cameraHeight ?? 1.2
  }, [views, scene])

  const overview = useMemo(() => {
    const { min, max } = scene.bounds
    const span = Math.max(max[0] - min[0], max[2] - min[2], 4)
    const p = new THREE.Vector3(center.x + span * 0.35, span * 1.05, center.z + span * 0.95)
    const q = new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().lookAt(p, center, new THREE.Vector3(0, 1, 0)))
    return { p, q, fov: 40 }
  }, [scene, center])

  const lift = (path: XZ[], y: number) => {
    const lens = [0]
    for (let i = 1; i < path.length; i++) lens.push(lens[i - 1] + Math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1]))
    const total = lens[lens.length - 1] || 1
    return path.map((p, i) => new THREE.Vector3(p[0], THREE.MathUtils.lerp(cam.position.y, y, lens[i] / total), p[1]))
  }
  const here = (): XZ => [cam.position.x, cam.position.z]

  useEffect(() => {
    const from = prevMode.current
    prevMode.current = mode
    if (!from) {
      cam.position.copy(overview.p).sub(center).multiplyScalar(1.3).add(center)
      cam.quaternion.copy(overview.q)
      cam.fov = overview.fov
      cam.updateProjectionMatrix()
    }
    const q0 = cam.quaternion.clone()
    if (mode === "dollhouse") {
      anim.current = glide([cam.position.clone(), overview.p.clone()], q0, overview.q.clone(), cam.fov, overview.fov, 1.3)
      anim.current.done = () => {
        controls.current?.target.copy(center)
        controls.current?.update()
      }
      return
    }
    const view = views[current]
    if (atPhoto && view) {
      const p = new THREE.Vector3(...view.position)
      const pts = from === "walk" ? lift(plan.route(here(), [p.x, p.z]) ?? [here(), [p.x, p.z]], p.y) : [cam.position.clone(), p]
      anim.current = glide(pts, q0, new THREE.Quaternion(...view.quaternion), cam.fov, coverFov(view, size.width / size.height), from === "walk" ? undefined : 1.4)
      return
    }
    const spot = pending.current
    pending.current = null
    if (spot) {
      const ahead = new THREE.Vector3(0, 0, -1).applyQuaternion(cam.quaternion)
      const yaw = Math.atan2(-ahead.x, -ahead.z)
      const q1 = new THREE.Quaternion().setFromEuler(new THREE.Euler(-0.08, yaw, 0, "YXZ"))
      anim.current = glide([cam.position.clone(), new THREE.Vector3(spot[0], eye, spot[1])], q0, q1, cam.fov, 62, 1.4)
    }
  }, [mode, current, atPhoto, views, overview, cam, center, size, eye])

  useEffect(() => {
    const el = gl.domElement
    const setNdc = (e: PointerEvent | MouseEvent) => {
      const r = el.getBoundingClientRect()
      pick.ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1)
    }
    const nearSpot = (p: THREE.Vector3) => {
      let best = -1
      let dist = SPOT
      views.forEach((v, i) => {
        const d = Math.hypot(v.position[0] - p.x, v.position[2] - p.z)
        if (d < dist) {
          dist = d
          best = i
        }
      })
      return best
    }
    const blocker = (hit: THREE.Vector3) => {
      if (!photos.current) return null
      const reach = cam.position.distanceTo(hit)
      const obj = pick.ray.intersectObject(photos.current, true)[0]
      if (!obj || obj.distance > reach - 0.3) return null
      const back = new THREE.Vector3(cam.position.x - obj.point.x, 0, cam.position.z - obj.point.z)
      const gap = Math.max(0, Math.min(0.9, back.length() - 0.2))
      return obj.point.clone().add(back.normalize().multiplyScalar(gap))
    }
    const click = () => {
      const hit = pick.hit
      if (!hit) return
      const spot = nearSpot(hit)
      if (spot >= 0) return live.current.onSelectView(spot)
      if (live.current.mode === "dollhouse") {
        const now = performance.now()
        if (now - last.current.t < 400 && last.current.p.distanceTo(hit) < 0.4) {
          pending.current = plan.clampInto(pick.room || plan.roomAt(hit.x, hit.z) || "", hit.x, hit.z)
          live.current.onEnterWalk()
        }
        last.current = { t: now, p: hit.clone() }
        return
      }
      if (anim.current) return
      const stop = blocker(hit)
      const room = pick.room || plan.roomAt(hit.x, hit.z) || ""
      const to = plan.clampInto(room, stop?.x ?? hit.x, stop?.z ?? hit.z)
      const path = plan.route(here(), to)
      if (!path) return
      const l = look.current
      const q1 = new THREE.Quaternion().setFromEuler(new THREE.Euler(l.pitch, l.yaw, 0, "YXZ"))
      anim.current = glide(lift(path, eye), cam.quaternion.clone(), q1, cam.fov, cam.fov)
      live.current.onFreeMove()
    }
    const down = (e: PointerEvent) => {
      setNdc(e)
      const l = look.current
      l.down = true
      l.moved = 0
      l.x = e.clientX
      l.y = e.clientY
    }
    const move = (e: PointerEvent) => {
      const l = look.current
      if (e.target === el) {
        setNdc(e)
        pick.inside = true
      }
      if (!l.down) return
      const dx = e.clientX - l.x
      const dy = e.clientY - l.y
      l.moved += Math.abs(dx) + Math.abs(dy)
      l.x = e.clientX
      l.y = e.clientY
      if (live.current.mode !== "walk" || anim.current) return
      const k = THREE.MathUtils.degToRad(cam.fov) / el.clientHeight
      l.yaw += dx * k
      l.pitch = THREE.MathUtils.clamp(l.pitch + dy * k, -1.2, 1.2)
    }
    const up = () => {
      const l = look.current
      if (l.down && l.moved < 6) click()
      l.down = false
    }
    const leave = () => (pick.inside = false)
    const wheel = (e: WheelEvent) => {
      if (live.current.mode !== "walk" || anim.current) return
      e.preventDefault()
      cam.fov = THREE.MathUtils.clamp(cam.fov + e.deltaY * 0.04, 30, 85)
      cam.updateProjectionMatrix()
    }
    el.addEventListener("pointerdown", down)
    el.addEventListener("pointerleave", leave)
    el.addEventListener("wheel", wheel, { passive: false })
    window.addEventListener("pointermove", move)
    window.addEventListener("pointerup", up)
    return () => {
      el.removeEventListener("pointerdown", down)
      el.removeEventListener("pointerleave", leave)
      el.removeEventListener("wheel", wheel)
      window.removeEventListener("pointermove", move)
      window.removeEventListener("pointerup", up)
    }
  }, [gl, cam, plan, views, eye])

  const euler = useMemo(() => new THREE.Euler(0, 0, 0, "YXZ"), [])
  const pos = useMemo(() => new THREE.Vector3(), [])

  useFrame((_, dt) => {
    const a = anim.current
    const l = look.current
    if (a) {
      a.t = Math.min(1, a.t + Math.min(dt, 0.1) / a.dur)
      const k = ease(a.t)
      along(a, k, cam.position)
      cam.quaternion.slerpQuaternions(a.q0, a.q1, k)
      cam.fov = THREE.MathUtils.lerp(a.f0, a.f1, k)
      cam.updateProjectionMatrix()
      if (a.t === 1) {
        anim.current = null
        euler.setFromQuaternion(a.q1, "YXZ")
        l.pitch = euler.x
        l.yaw = euler.y
        l.roll = euler.z
        a.done?.()
      }
    } else if (mode === "walk") {
      if (l.down && l.moved > 6) l.roll = THREE.MathUtils.damp(l.roll, 0, 6, dt)
      cam.quaternion.setFromEuler(euler.set(l.pitch, l.yaw, l.roll, "YXZ"))
    }
    if (controls.current) controls.current.enabled = mode === "dollhouse" && !anim.current

    pick.hit = null
    pick.room = ""
    if (pick.inside && shells.current) {
      pick.ray.setFromCamera(pick.ndc, cam)
      const hits = pick.ray.intersectObject(shells.current, true)
      const first = hits[0] && hits[0].point.y < 0.05 ? hits.find((h) => h.object.userData.floor) : hits[0]
      if (first?.object.userData.floor) {
        pick.hit = pos.copy(first.point)
        pick.room = first.object.userData.room ?? ""
      }
    }
    const c = cursor.current
    if (c) {
      const show = mode === "walk" && !!pick.hit && !anim.current && !(l.down && l.moved > 6)
      c.visible = show
      if (show) {
        const v = views.find((v) => Math.hypot(v.position[0] - pos.x, v.position[2] - pos.z) < SPOT)
        c.position.set(v ? v.position[0] : pos.x, 0.012, v ? v.position[2] : pos.z)
        c.scale.setScalar(THREE.MathUtils.damp(c.scale.x, v ? 0.75 : 1, 14, dt))
      }
    }
    gl.domElement.style.cursor = pick.hit && (mode === "walk" || nearAny(views, pos)) ? "pointer" : ""
  })

  return (
    <>
      <OrbitControls
        ref={controls}
        enabled={false}
        enableDamping
        dampingFactor={0.08}
        maxPolarAngle={Math.PI * 0.42}
        minDistance={3}
        maxDistance={45}
      />
      <mesh ref={cursor} rotation={[-Math.PI / 2, 0, 0]} renderOrder={10} visible={false}>
        <planeGeometry args={[0.42, 0.42]} />
        <meshBasicMaterial map={ringTexture()} transparent depthWrite={false} polygonOffset polygonOffsetFactor={-4} />
      </mesh>
    </>
  )
}

function nearAny(views: View[], p: THREE.Vector3) {
  return views.some((v) => Math.hypot(v.position[0] - p.x, v.position[2] - p.z) < SPOT)
}

function Spots({ views, mode, skip }: { views: View[]; mode: TourMode; skip: number }) {
  const refs = useRef<(THREE.Mesh | null)[]>([])
  useFrame(({ camera }, dt) => {
    views.forEach((v, i) => {
      const m = refs.current[i]
      if (!m) return
      const mat = m.material as THREE.MeshBasicMaterial
      let target = 0
      if (i !== skip) {
        if (mode === "walk") {
          const d = Math.hypot(v.position[0] - camera.position.x, v.position[2] - camera.position.z)
          target = THREE.MathUtils.clamp(1 - (d - 3.5) / 2.5, 0, 1) * 0.9
        } else if (pick.hit) {
          const d = Math.hypot(v.position[0] - pick.hit.x, v.position[2] - pick.hit.z)
          target = THREE.MathUtils.clamp(1 - (d - 0.4) / 1.4, 0, 1)
        }
      }
      mat.opacity = THREE.MathUtils.damp(mat.opacity, target, 10, dt)
      m.visible = mat.opacity > 0.01
    })
  })
  return (
    <group>
      {views.map((v, i) => (
        <mesh
          key={v.id}
          ref={(m) => {
            refs.current[i] = m
          }}
          position={[v.position[0], 0.01, v.position[2]]}
          rotation={[-Math.PI / 2, 0, 0]}
          renderOrder={5}
        >
          <planeGeometry args={[mode === "walk" ? 0.34 : 0.42, mode === "walk" ? 0.34 : 0.42]} />
          <meshBasicMaterial map={ringTexture()} transparent opacity={0} depthWrite={false} polygonOffset polygonOffsetFactor={-2} />
        </mesh>
      ))}
    </group>
  )
}

function RoomLabel({ room, hidden }: { room: Room; hidden: boolean }) {
  const x = room.shell ? (room.shell.x[0] + room.shell.x[1]) / 2 : room.center[0]
  const z = room.shell ? (room.shell.z[0] + room.shell.z[1]) / 2 : room.center[2]
  return (
    <Html position={[x, 0.05, z]} center className="pointer-events-none" zIndexRange={[10, 0]}>
      <span
        style={{ opacity: hidden ? 0 : 1 }}
        className="rounded-md bg-card/90 px-1.5 py-0.5 text-[11px] font-medium whitespace-nowrap ring-1 ring-border transition-opacity duration-300"
      >
        {room.label}
        <span className="ml-1.5 text-muted-foreground tabular">
          {(room.size[0] * 3.281).toFixed(0)}′ × {(room.size[2] * 3.281).toFixed(0)}′
        </span>
      </span>
    </Html>
  )
}
