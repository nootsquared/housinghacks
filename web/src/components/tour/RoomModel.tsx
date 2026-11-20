import { useMemo, useRef } from "react"
import { useFrame } from "@react-three/fiber"
import * as THREE from "three"
import type { Opening, Room, WallId } from "@/lib/types"
import { FurnitureModel } from "./Furniture"
import { floorTexture } from "./materials"

const T = 0.1
const CUT = 0.18

type Piece = { a: number; b: number; y0: number; y1: number }

function pieces(a: number, b: number, h: number, ops: Opening[]): Piece[] {
  const out: Piece[] = []
  let cur = a
  for (const o of [...ops].sort((p, q) => p.from - q.from)) {
    const f = Math.max(o.from, cur)
    const t = Math.min(o.to, b)
    if (t <= f) continue
    if (f > cur) out.push({ a: cur, b: f, y0: 0, y1: h })
    if (o.bottom > 0.01) out.push({ a: f, b: t, y0: 0, y1: o.bottom })
    if (o.top < h - 0.01) out.push({ a: f, b: t, y0: o.top, y1: h })
    cur = t
  }
  if (cur < b) out.push({ a: cur, b, y0: 0, y1: h })
  return out
}

function Wall({
  id,
  room,
  color,
  mode,
}: {
  id: WallId
  room: Room
  color: string
  mode: "walk" | "dollhouse"
}) {
  const shell = room.shell!
  const [x0, x1] = shell.x
  const [z0, z1] = shell.z
  const h = shell.height
  const alongX = id === "n" || id === "s"
  const fixed = id === "n" ? z0 - T / 2 : id === "s" ? z1 + T / 2 : id === "w" ? x0 - T / 2 : x1 + T / 2
  const inward = new THREE.Vector3(id === "w" ? 1 : id === "e" ? -1 : 0, 0, id === "n" ? 1 : id === "s" ? -1 : 0)
  const [a, b] = alongX ? [x0 - T, x1 + T] : [z0, z1]
  const ops = (room.model?.openings ?? []).filter((o) => o.wall === id)
  const parts = useMemo(() => pieces(a, b, h, ops), [a, b, h, ops])
  const group = useRef<THREE.Group>(null)
  const mid = new THREE.Vector3(alongX ? (x0 + x1) / 2 : fixed, h / 2, alongX ? fixed : (z0 + z1) / 2)

  useFrame(({ camera }, dt) => {
    if (!group.current) return
    const facing = camera.position.clone().sub(mid).dot(inward) < 0
    const target = mode === "dollhouse" && facing ? CUT / h : 1
    const s = group.current.scale
    s.y = THREE.MathUtils.damp(s.y, target, 8, dt)
  })

  return (
    <group ref={group}>
      {parts.map((p, i) => {
        const len = p.b - p.a
        const c = (p.a + p.b) / 2
        const ph = p.y1 - p.y0
        const pos: [number, number, number] = alongX ? [c, p.y0 + ph / 2, fixed] : [fixed, p.y0 + ph / 2, c]
        const size: [number, number, number] = alongX ? [len, ph, T] : [T, ph, len]
        return (
          <mesh key={i} position={pos} receiveShadow>
            <boxGeometry args={size} />
            <meshStandardMaterial color={color} roughness={0.92} />
          </mesh>
        )
      })}
      {ops
        .filter((o) => o.kind === "window")
        .map((o, i) => (
          <Window key={`w${i}`} opening={o} alongX={alongX} fixed={fixed} />
        ))}
    </group>
  )
}

const BAR = 0.05

function Window({ opening: o, alongX, fixed }: { opening: Opening; alongX: boolean; fixed: number }) {
  const len = o.to - o.from
  const ph = o.top - o.bottom
  const color = o.frame ?? "#ecebe8"
  const panes = Math.max(1, Math.round(len / 1.2))
  const bars: { u: number; v: number; w: number; h: number }[] = [
    { u: 0, v: ph / 2 - BAR / 2, w: len, h: BAR },
    { u: 0, v: -ph / 2 + BAR / 2, w: len, h: BAR },
    { u: -len / 2 + BAR / 2, v: 0, w: BAR, h: ph },
    { u: len / 2 - BAR / 2, v: 0, w: BAR, h: ph },
  ]
  for (let k = 1; k < panes; k++) bars.push({ u: -len / 2 + (len * k) / panes, v: 0, w: BAR * 0.8, h: ph })
  if (ph > 1.6) bars.push({ u: 0, v: -ph / 2 + Math.min(0.55, ph * 0.25), w: len, h: BAR * 0.8 })
  const at = (u: number, v: number): [number, number, number] => (alongX ? [u, v, 0] : [0, v, u])
  const box = (w: number, h: number, d: number): [number, number, number] => (alongX ? [w, h, d] : [d, h, w])
  return (
    <group position={alongX ? [(o.from + o.to) / 2, o.bottom + ph / 2, fixed] : [fixed, o.bottom + ph / 2, (o.from + o.to) / 2]}>
      <mesh>
        <boxGeometry args={box(len, ph, 0.015)} />
        <meshPhysicalMaterial color="#a9c4d6" transparent opacity={0.4} roughness={0.08} metalness={0.2} />
      </mesh>
      {bars.map((b, k) => (
        <mesh key={k} position={at(b.u, b.v)} castShadow>
          <boxGeometry args={box(b.w, b.h, T * 0.7)} />
          <meshStandardMaterial color={color} roughness={0.5} />
        </mesh>
      ))}
      {o.bottom > 0.2 && (
        <mesh position={at(0, -ph / 2 - 0.015)}>
          <boxGeometry args={box(len + 0.08, 0.03, T * 1.8)} />
          <meshStandardMaterial color="#f2f1ee" roughness={0.6} />
        </mesh>
      )}
    </group>
  )
}

function backSide(item: { center: number[]; size: number[] }, room: Room): "-x" | "+x" | "-z" | "+z" {
  const [x0, x1] = room.shell!.x
  const [z0, z1] = room.shell!.z
  const [cx, , cz] = item.center
  const [w, , d] = item.size
  const gaps: [number, "-x" | "+x" | "-z" | "+z"][] = [
    [cx - w / 2 - x0, "-x"],
    [x1 - (cx + w / 2), "+x"],
    [cz - d / 2 - z0, "-z"],
    [z1 - (cz + d / 2), "+z"],
  ]
  gaps.sort((p, q) => p[0] - q[0])
  return gaps[0][1]
}

const SINK = 0.03
const loader = new THREE.TextureLoader()

function FillWall({ id, room, base, mode }: { id: WallId; room: Room; base: string; mode: "walk" | "dollhouse" }) {
  const shell = room.shell!
  const [x0, x1] = shell.x
  const [z0, z1] = shell.z
  const h = shell.height
  const file = room.model?.fills?.[id]
  const map = useMemo(() => {
    if (!file) return null
    const t = loader.load(`${base}/${file}`)
    t.colorSpace = THREE.SRGBColorSpace
    return t
  }, [base, file])
  const group = useRef<THREE.Group>(null)
  const alongX = id === "n" || id === "s"
  const len = alongX ? x1 - x0 + 2 * SINK : z1 - z0 + 2 * SINK
  const pos: [number, number, number] =
    id === "n" ? [(x0 + x1) / 2, 0, z0 - SINK] : id === "s" ? [(x0 + x1) / 2, 0, z1 + SINK] : id === "w" ? [x0 - SINK, 0, (z0 + z1) / 2] : [x1 + SINK, 0, (z0 + z1) / 2]
  const rot = id === "n" ? 0 : id === "s" ? Math.PI : id === "w" ? Math.PI / 2 : -Math.PI / 2
  const inward = new THREE.Vector3(id === "w" ? 1 : id === "e" ? -1 : 0, 0, id === "n" ? 1 : id === "s" ? -1 : 0)
  const mid = new THREE.Vector3(pos[0], h / 2, pos[2])

  useFrame(({ camera }, dt) => {
    if (!group.current) return
    const facing = camera.position.clone().sub(mid).dot(inward) < 0
    const s = group.current.scale
    s.y = THREE.MathUtils.damp(s.y, mode === "dollhouse" && facing ? CUT / h : 1, 8, dt)
  })

  return (
    <group ref={group} position={pos} rotation={[0, rot, 0]}>
      <mesh position={[0, h / 2, 0]}>
        <planeGeometry args={[len, h]} />
        <meshBasicMaterial map={map} color={map ? "#ffffff" : room.model?.walls[id] ?? room.model?.wallColor} toneMapped={false} />
      </mesh>
    </group>
  )
}

export function FilledShell({ room, base, mode }: { room: Room; base: string; mode: "walk" | "dollhouse" }) {
  const shell = room.shell
  const model = room.model
  const floor = useMemo(() => {
    if (!model || !shell) return null
    const t = floorTexture(model.floor.material, model.floor.color, model.floor.accent).clone()
    t.needsUpdate = true
    t.repeat.set((shell.x[1] - shell.x[0]) / 4, (shell.z[1] - shell.z[0]) / 4)
    return t
  }, [model, shell])
  const seen = useMemo(() => {
    const file = model?.fills?.floor
    if (!file) return null
    const t = loader.load(`${base}/${file}`)
    t.colorSpace = THREE.SRGBColorSpace
    t.flipY = false
    t.center.set(0.5, 0.5)
    t.rotation = ((room.shell?.floorTurn ?? 0) * Math.PI) / 2
    return t
  }, [base, model, room.shell?.floorTurn])
  if (!shell || !model) return null
  const [x0, x1] = shell.x
  const [z0, z1] = shell.z
  const cx = (x0 + x1) / 2
  const cz = (z0 + z1) / 2
  const w = x1 - x0 + 2 * SINK
  const d = z1 - z0 + 2 * SINK
  return (
    <group>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[cx, -0.025, cz]}>
        <planeGeometry args={[w, d]} />
        <meshBasicMaterial map={floor} toneMapped={false} />
      </mesh>
      {seen && (
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[cx, -0.02, cz]}>
          <planeGeometry args={[x1 - x0, z1 - z0]} />
          <meshBasicMaterial map={seen} transparent toneMapped={false} depthWrite={false} />
        </mesh>
      )}
      {(["n", "s", "w", "e"] as WallId[]).map((id) => (
        <FillWall key={id} id={id} room={room} base={base} mode={mode} />
      ))}
      {mode === "walk" && (
        <mesh rotation={[Math.PI / 2, 0, 0]} position={[cx, shell.height + SINK, cz]}>
          <planeGeometry args={[w, d]} />
          <meshBasicMaterial color="#f1f0ed" toneMapped={false} />
        </mesh>
      )}
    </group>
  )
}

export function RoomModel({ room, mode }: { room: Room; mode: "walk" | "dollhouse" }) {
  const shell = room.shell
  const model = room.model
  const floor = useMemo(() => {
    if (!model) return null
    const t = floorTexture(model.floor.material, model.floor.color, model.floor.accent).clone()
    t.needsUpdate = true
    if (shell) t.repeat.set((shell.x[1] - shell.x[0]) / 4, (shell.z[1] - shell.z[0]) / 4)
    return t
  }, [model, shell])
  if (!shell || !model) return null
  const [x0, x1] = shell.x
  const [z0, z1] = shell.z
  const w = x1 - x0
  const d = z1 - z0
  return (
    <group>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[(x0 + x1) / 2, 0, (z0 + z1) / 2]} receiveShadow>
        <planeGeometry args={[w + 2 * T, d + 2 * T]} />
        <meshStandardMaterial map={floor} roughness={model.floor.material === "carpet" ? 1 : 0.55} />
      </mesh>
      <mesh position={[(x0 + x1) / 2, -0.056, (z0 + z1) / 2]}>
        <boxGeometry args={[w + 2 * T, 0.1, d + 2 * T]} />
        <meshStandardMaterial color="#d8d4cc" roughness={1} />
      </mesh>
      {(["n", "s", "w", "e"] as WallId[]).map((id) => (
        <Wall key={id} id={id} room={room} color={model.walls[id] ?? model.wallColor} mode={mode} />
      ))}
      {mode === "walk" && (
        <mesh rotation={[Math.PI / 2, 0, 0]} position={[(x0 + x1) / 2, shell.height, (z0 + z1) / 2]}>
          <planeGeometry args={[w + 2 * T, d + 2 * T]} />
          <meshStandardMaterial color="#f4f3f0" roughness={1} side={THREE.DoubleSide} />
        </mesh>
      )}
      {model.furniture.map((f, i) => (
        <group key={i} position={f.center as [number, number, number]}>
          <FurnitureModel item={f} back={backSide(f, room)} />
        </group>
      ))}
    </group>
  )
}
