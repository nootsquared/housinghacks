import { useEffect, useMemo, useRef } from "react"
import { useFrame, useThree } from "@react-three/fiber"
import * as THREE from "three"
import type { Room, WallId } from "@/lib/types"
import { floorTexture, shade } from "./materials"
import { WALL, type Door } from "./plan"

const CUT = 0.16
const SINK = 0.03
const TRIM = "#f3f1ec"
const CASING = 0.07
const PROUD = 0.016
const JAMB = 0.02
const SIDES: WallId[] = ["n", "s", "w", "e"]

type Mode = "walk" | "dollhouse"
type Gap = { from: number; to: number; top: number; cut: boolean; closed: boolean; bare?: boolean }
type Pane = { from: number; to: number; bottom: number; top: number; frame: string }

const lambert = new Map<string, THREE.MeshLambertMaterial>()
function paint(color: string) {
  if (!lambert.has(color)) lambert.set(color, new THREE.MeshLambertMaterial({ color }))
  return lambert.get(color)!
}

function frame(room: Room, id: WallId) {
  const { x, z } = room.shell!
  const alongX = id === "n" || id === "s"
  const plane = id === "n" ? z[0] : id === "s" ? z[1] : id === "w" ? x[0] : x[1]
  const sign = id === "n" || id === "w" ? 1 : -1
  const span: [number, number] = alongX ? x : z
  const at = (u: number, v: number, w: number): [number, number, number] =>
    alongX ? [u, v, plane + sign * w] : [plane + sign * w, v, u]
  const dims = (lu: number, lv: number, lw: number): [number, number, number] => (alongX ? [lu, lv, lw] : [lw, lv, lu])
  return { alongX, plane, sign, span, at, dims }
}

function solids(a: number, b: number, h: number, gaps: Gap[]) {
  const out: { a: number; b: number; y0: number; y1: number }[] = []
  let cur = a
  for (const g of [...gaps].sort((p, q) => p.from - q.from)) {
    if (g.from > cur) out.push({ a: cur, b: g.from, y0: 0, y1: h })
    if (g.top < h - 0.01) out.push({ a: g.from, b: g.to, y0: g.top, y1: h })
    cur = Math.max(cur, g.to)
  }
  if (cur < b) out.push({ a: cur, b, y0: 0, y1: h })
  return out
}

function Box({ at, size, color }: { at: [number, number, number]; size: [number, number, number]; color: string }) {
  return (
    <mesh position={at} material={paint(color)}>
      <boxGeometry args={size} />
    </mesh>
  )
}

const overlaps = (a: { from: number; to: number }, b: { from: number; to: number }) =>
  Math.min(a.to, b.to) - Math.max(a.from, b.from) > 0.3 * Math.min(a.to - a.from, b.to - b.from)

function ShellWall({
  room,
  id,
  doors,
  mode,
  floorTone,
  base,
}: {
  room: Room
  id: WallId
  doors: Door[]
  mode: Mode
  floorTone: string
  base: string
}) {
  const h = room.shell!.height
  const color = room.model?.walls[id] ?? room.model?.wallColor ?? "#e4e2de"
  const f = useMemo(() => frame(room, id), [room, id])
  const surface = room.model?.surfaces?.[id]
  const fill = surface ? undefined : room.model?.fills?.[id]
  const decal = surface ?? room.model?.decals?.[id]
  const { cuts, gaps, panes } = useMemo(() => {
    const cuts: Gap[] = doors
      .filter((d) => d.walls.some((w) => w.room === room.id && w.wall === id))
      .map((d) => ({ from: d.from, to: d.to, top: Math.min(d.top, d.bare ? h : h - 0.1), cut: true, closed: d.rooms.length === 1, bare: d.bare }))
    const inner = (o: { from: number; to: number }) => o.from > f.span[0] + 0.05 && o.to < f.span[1] - 0.05
    const own = surface
      ? []
      : (room.model?.openings ?? []).filter((o) => o.wall === id && inner(o) && !cuts.some((c) => overlaps(c, o)))
    const gaps: Gap[] = [
      ...cuts,
      ...own
        .filter((o) => o.kind === "door")
        .map((o) => ({ from: o.from, to: o.to, top: THREE.MathUtils.clamp(o.top, 2.0, h - 0.1), cut: false, closed: true })),
    ]
    const panes: Pane[] = fill
      ? own
          .filter((o) => o.kind === "window")
          .map((o) => ({ from: o.from, to: o.to, bottom: o.bottom, top: Math.min(o.top, h - 0.04), frame: o.frame ?? TRIM }))
      : []
    return { cuts, gaps, panes }
  }, [doors, room, id, h, f, fill, surface])
  const [a, b] = f.alongX ? [f.span[0] - WALL, f.span[1] + WALL] : f.span
  const mats = useMemo(() => {
    const side = paint(color)
    const cap = paint(shade(color, 0.78))
    return [side, side, cap, side, side, side]
  }, [color])
  const group = useRef<THREE.Group>(null)
  const mid = useMemo(() => new THREE.Vector3(...f.at((f.span[0] + f.span[1]) / 2, h / 2, 0)), [f, h])
  const inward = useMemo(() => new THREE.Vector3(...f.at(0, 0, 1)).sub(new THREE.Vector3(...f.at(0, 0, 0))), [f])
  const look = useMemo(() => new THREE.Vector3(), [])

  useFrame(({ camera }, dt) => {
    if (!group.current) return
    const facing = look.copy(camera.position).sub(mid).dot(inward) < 0
    const s = group.current.scale
    s.y = THREE.MathUtils.damp(s.y, mode === "dollhouse" && facing ? CUT / h : 1, 8, dt)
  })

  return (
    <group ref={group}>
      {solids(a, b, h, cuts).map((p, i) => (
        <mesh key={i} position={f.at((p.a + p.b) / 2, (p.y0 + p.y1) / 2, -WALL / 2)} material={mats} userData={{ solid: true }}>
          <boxGeometry args={f.dims(p.b - p.a, p.y1 - p.y0, WALL)} />
        </mesh>
      ))}
      {gaps.filter((g) => !g.bare).map((g, i) => (
        <Casing key={i} gap={g} f={f} span={f.span} floorTone={floorTone} wall={color} />
      ))}
      {fill &&
        panes.map((p, i) => <WindowPane key={`p${i}`} pane={p} f={f} url={`${base}/${fill}`} h={h} wallId={id} />)}
      {decal &&
        solids(f.span[0], f.span[1], h, cuts).map((p, i) => (
          <Decal key={`d${i}`} url={`${base}/${decal}`} piece={p} f={f} h={h} wallId={id} opaque={!!surface} />
        ))}
    </group>
  )
}

function Decal({
  url,
  piece: p,
  f,
  h,
  wallId,
  opaque,
}: {
  url: string
  piece: { a: number; b: number; y0: number; y1: number }
  f: ReturnType<typeof frame>
  h: number
  wallId: WallId
  opaque?: boolean
}) {
  const anisotropy = useThree((s) => s.gl.capabilities.getMaxAnisotropy())
  const map = useMemo(() => {
    const t = loader.load(url)
    t.colorSpace = THREE.SRGBColorSpace
    t.anisotropy = anisotropy
    const [s0, s1] = f.span
    const len = s1 - s0
    const flip = wallId === "s" || wallId === "w"
    t.repeat.set((p.b - p.a) / len, (p.y1 - p.y0) / h)
    t.offset.set(flip ? (s1 - p.b) / len : (p.a - s0) / len, p.y0 / h)
    return t
  }, [url, f, p, h, wallId, anisotropy])
  useEffect(() => () => map.dispose(), [map])
  const rot = wallId === "n" ? 0 : wallId === "s" ? Math.PI : wallId === "w" ? Math.PI / 2 : -Math.PI / 2
  return (
    <mesh position={f.at((p.a + p.b) / 2, (p.y0 + p.y1) / 2, 0.003)} rotation={[0, rot, 0]} renderOrder={1}>
      <planeGeometry args={[p.b - p.a, p.y1 - p.y0]} />
      <meshBasicMaterial
        map={map}
        transparent={!opaque}
        depthWrite={!!opaque}
        toneMapped={!opaque}
        polygonOffset
        polygonOffsetFactor={-2}
      />
    </mesh>
  )
}

function PhotoFloor({ url, x, z }: { url: string; x: [number, number]; z: [number, number] }) {
  const anisotropy = useThree((s) => s.gl.capabilities.getMaxAnisotropy())
  const map = useMemo(() => {
    const t = loader.load(url)
    t.colorSpace = THREE.SRGBColorSpace
    t.anisotropy = anisotropy
    return t
  }, [url, anisotropy])
  useEffect(() => () => map.dispose(), [map])
  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[(x[0] + x[1]) / 2, 0.002, (z[0] + z[1]) / 2]} renderOrder={1}>
      <planeGeometry args={[x[1] - x[0], z[1] - z[0]]} />
      <meshBasicMaterial map={map} toneMapped={false} polygonOffset polygonOffsetFactor={-2} />
    </mesh>
  )
}

function WindowPane({ pane: p, f, url, h, wallId }: { pane: Pane; f: ReturnType<typeof frame>; url: string; h: number; wallId: WallId }) {
  const anisotropy = useThree((s) => s.gl.capabilities.getMaxAnisotropy())
  const map = useMemo(() => {
    const t = loader.load(url)
    t.colorSpace = THREE.SRGBColorSpace
    t.anisotropy = anisotropy
    const [s0, s1] = f.span
    const len = s1 - s0 + 2 * SINK
    const flip = wallId === "s" || wallId === "w"
    const u0 = flip ? (s1 + SINK - p.to) / len : (p.from - (s0 - SINK)) / len
    t.repeat.set((p.to - p.from) / len, (p.top - p.bottom) / h)
    t.offset.set(u0, p.bottom / h)
    return t
  }, [url, f, p, h, wallId, anisotropy])
  useEffect(() => () => map.dispose(), [map])
  const w = p.to - p.from
  const ph = p.top - p.bottom
  const mid = (p.from + p.to) / 2
  const cy = (p.bottom + p.top) / 2
  const rot = wallId === "n" ? 0 : wallId === "s" ? Math.PI : wallId === "w" ? Math.PI / 2 : -Math.PI / 2
  const bar = 0.035
  return (
    <group>
      <mesh position={f.at(mid, cy, 0.002)} rotation={[0, rot, 0]}>
        <planeGeometry args={[w, ph]} />
        <meshBasicMaterial map={map} />
      </mesh>
      <Box at={f.at(mid, p.top + bar / 2, 0.012)} size={f.dims(w + 2 * bar, bar, 0.024)} color={p.frame} />
      <Box at={f.at(mid, p.bottom - bar / 2, 0.012)} size={f.dims(w + 2 * bar, bar, 0.024)} color={p.frame} />
      <Box at={f.at(p.from - bar / 2, cy, 0.012)} size={f.dims(bar, ph, 0.024)} color={p.frame} />
      <Box at={f.at(p.to + bar / 2, cy, 0.012)} size={f.dims(bar, ph, 0.024)} color={p.frame} />
    </group>
  )
}

function Casing({
  gap: g,
  f,
  span,
  floorTone,
  wall,
}: {
  gap: Gap
  f: ReturnType<typeof frame>
  span: [number, number]
  floorTone: string
  wall: string
}) {
  const width = g.to - g.from
  const mid = (g.from + g.to) / 2
  const left = Math.max(g.from - CASING, span[0])
  const right = Math.min(g.to + CASING, span[1])
  return (
    <group>
      {g.from - span[0] > CASING && (
        <Box at={f.at(g.from - CASING / 2, (g.top + CASING) / 2, PROUD / 2)} size={f.dims(CASING, g.top + CASING, PROUD)} color={TRIM} />
      )}
      {span[1] - g.to > CASING && (
        <Box at={f.at(g.to + CASING / 2, (g.top + CASING) / 2, PROUD / 2)} size={f.dims(CASING, g.top + CASING, PROUD)} color={TRIM} />
      )}
      <Box at={f.at((left + right) / 2, g.top + CASING / 2, PROUD / 2)} size={f.dims(right - left, CASING, PROUD)} color={TRIM} />
      {g.cut && (
        <group>
          <Box at={f.at(g.from + JAMB / 2, g.top / 2, -WALL / 2)} size={f.dims(JAMB, g.top, WALL)} color={TRIM} />
          <Box at={f.at(g.to - JAMB / 2, g.top / 2, -WALL / 2)} size={f.dims(JAMB, g.top, WALL)} color={TRIM} />
          <Box at={f.at(mid, g.top - JAMB / 2, -WALL / 2)} size={f.dims(width, JAMB, WALL)} color={TRIM} />
          <mesh position={f.at(mid, 0.003, -WALL / 2)} material={paint(floorTone)} userData={{ floor: true }}>
            <boxGeometry args={f.dims(width - 2 * JAMB, 0.006, WALL)} />
          </mesh>
        </group>
      )}
      {g.closed && (
        <group>
          <Box
            at={f.at(mid, (g.top - JAMB) / 2, g.cut ? -WALL + 0.025 : 0.006)}
            size={f.dims(width - 2 * JAMB, g.top - JAMB, g.cut ? 0.04 : 0.012)}
            color={shade(wall, 0.9)}
          />
          <Box at={f.at(g.to - JAMB - 0.1, 1.0, g.cut ? -WALL + 0.06 : 0.025)} size={f.dims(0.11, 0.022, 0.026)} color="#3b3936" />
        </group>
      )}
    </group>
  )
}

const loader = new THREE.TextureLoader()

function useFloorMap(room: Room, base: string) {
  const anisotropy = useThree((s) => s.gl.capabilities.getMaxAnisotropy())
  const model = room.model!
  const { x, z, floorTurn = 0 } = room.shell!
  const map = useMemo(() => {
    const w = x[1] - x[0] + 2 * WALL
    const d = z[1] - z[0] + 2 * WALL
    const tile = model.floor.tile
    const size = tile ? (model.floor.size ?? 1) : 4
    const t = tile ? loader.load(`${base}/${tile}`) : floorTexture(model.floor.material, model.floor.color, model.floor.accent).clone()
    t.wrapS = t.wrapT = THREE.RepeatWrapping
    t.colorSpace = THREE.SRGBColorSpace
    t.anisotropy = anisotropy
    t.center.set(0.5, 0.5)
    t.rotation = (floorTurn * Math.PI) / 2
    t.repeat.set((floorTurn % 2 ? d : w) / size, (floorTurn % 2 ? w : d) / size)
    if (!tile) t.needsUpdate = true
    return t
  }, [base, model, x, z, floorTurn, anisotropy])
  useEffect(() => () => map.dispose(), [map])
  return map
}

export function Shell({ room, doors, base, mode }: { room: Room; doors: Door[]; base: string; mode: Mode }) {
  const shell = room.shell!
  const model = room.model!
  const map = useFloorMap(room, base)
  const [x0, x1] = shell.x
  const [z0, z1] = shell.z
  const cx = (x0 + x1) / 2
  const cz = (z0 + z1) / 2
  const w = x1 - x0 + 2 * WALL
  const d = z1 - z0 + 2 * WALL
  const floorTone = model.floor.accent ? shade(model.floor.color, 0.85) : model.floor.color
  return (
    <group>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[cx, 0, cz]} userData={{ floor: true, room: room.id }}>
        <planeGeometry args={[w, d]} />
        <meshLambertMaterial map={map} />
      </mesh>
      {model.surfaces?.floor && <PhotoFloor url={`${base}/${model.surfaces.floor}`} x={shell.x} z={shell.z} />}
      {SIDES.map((id) => (
        <ShellWall key={id} room={room} id={id} doors={doors} mode={mode} floorTone={floorTone} base={base} />
      ))}
      {mode === "walk" && (
        <mesh rotation={[Math.PI / 2, 0, 0]} position={[cx, shell.height, cz]} userData={{ solid: true }}>
          <planeGeometry args={[w, d]} />
          <meshLambertMaterial color="#f3f2ef" />
        </mesh>
      )}
    </group>
  )
}
