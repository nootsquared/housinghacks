import type { Scene, WallId } from "@/lib/types"

export const WALL = 0.06
const PAD = 0.18
const EDGE = 0.1
const STAND = 0.6

export type XZ = [number, number]

type Rect = { id: string; x0: number; x1: number; z0: number; z1: number }

export type Door = {
  rooms: string[]
  axis: "x" | "z"
  at: number
  from: number
  to: number
  bottom: number
  top: number
  walls: { room: string; wall: WallId }[]
  bare?: boolean
}

export function wallPlane(r: Rect, wall: WallId) {
  return wall === "n" ? r.z0 : wall === "s" ? r.z1 : wall === "w" ? r.x0 : r.x1
}

function doors(scene: Scene, rects: Map<string, Rect>): Door[] {
  const out: Door[] = []
  for (const d of scene.doorways ?? []) {
    const [[ax, az], [bx, bz]] = [d.p0, d.p1]
    const axis = Math.abs(bx - ax) >= Math.abs(bz - az) ? "x" : "z"
    const at = axis === "x" ? (az + bz) / 2 : (ax + bx) / 2
    let from = axis === "x" ? Math.min(ax, bx) : Math.min(az, bz)
    let to = axis === "x" ? Math.max(ax, bx) : Math.max(az, bz)
    let lo = -Infinity
    let hi = Infinity
    const walls: Door["walls"] = []
    for (const id of d.rooms) {
      const r = rects.get(id)
      if (!r) continue
      const sides: WallId[] = axis === "x" ? ["n", "s"] : ["w", "e"]
      const wall = sides
        .map((w) => ({ w, dist: Math.abs(wallPlane(r, w) - at) }))
        .filter((c) => c.dist < 0.25)
        .sort((p, q) => p.dist - q.dist)[0]?.w
      if (!wall) continue
      const [s0, s1] = axis === "x" ? [r.x0, r.x1] : [r.z0, r.z1]
      if (Math.min(to, s1) - Math.max(from, s0) < 0.3) continue
      lo = Math.max(lo, s0 + EDGE)
      hi = Math.min(hi, s1 - EDGE)
      walls.push({ room: id, wall })
    }
    if (!walls.length || hi - lo < 0.5) continue
    const w = Math.min(to - from, hi - lo)
    if (from < lo) from = lo
    if (from + w > hi) from = hi - w
    to = from + w
    out.push({ rooms: walls.map((x) => x.room), axis, at, from, to, bottom: d.bottom, top: d.top, walls, bare: !!d.bare })
  }
  return out
}

export class Plan {
  rects: Rect[]
  doors: Door[]
  private byId: Map<string, Rect>

  constructor(scene: Scene) {
    this.rects = scene.rooms
      .filter((r) => r.shell && (r.model || r.mesh))
      .map((r) => ({ id: r.id, x0: r.shell!.x[0], x1: r.shell!.x[1], z0: r.shell!.z[0], z1: r.shell!.z[1] }))
    this.byId = new Map(this.rects.map((r) => [r.id, r]))
    this.doors = doors(scene, this.byId)
  }

  rect(id: string) {
    return this.byId.get(id)
  }

  roomAt(x: number, z: number): string | null {
    let best: string | null = null
    let score = -Infinity
    for (const r of this.rects) {
      const s = Math.min(x - r.x0, r.x1 - x, z - r.z0, r.z1 - z)
      if (s > score) {
        score = s
        best = r.id
      }
    }
    return best
  }

  clampInto(id: string, x: number, z: number): XZ {
    const r = this.byId.get(id)
    if (!r) return [x, z]
    const cx = (r.x0 + r.x1) / 2
    const cz = (r.z0 + r.z1) / 2
    const mx = Math.min(STAND, (r.x1 - r.x0) / 2)
    const mz = Math.min(STAND, (r.z1 - r.z0) / 2)
    return [
      x < cx ? Math.max(x, r.x0 + mx) : Math.min(x, r.x1 - mx),
      z < cz ? Math.max(z, r.z0 + mz) : Math.min(z, r.z1 - mz),
    ]
  }

  walkable(x: number, z: number) {
    for (const r of this.rects) {
      if (x > r.x0 + PAD && x < r.x1 - PAD && z > r.z0 + PAD && z < r.z1 - PAD) return true
    }
    for (const d of this.doors) {
      if (d.rooms.length < 2) continue
      const [u, v] = d.axis === "x" ? [x, z] : [z, x]
      if (u > d.from + 0.12 && u < d.to - 0.12 && Math.abs(v - d.at) < 0.45) return true
    }
    return false
  }

  clear(a: XZ, b: XZ) {
    const len = Math.hypot(b[0] - a[0], b[1] - a[1])
    const n = Math.ceil(len / 0.04)
    for (let k = 1; k < n; k++) {
      const s = (k / n) * len
      if (s < 0.3 || len - s < 0.3) continue
      const t = k / n
      if (!this.walkable(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)) return false
    }
    return true
  }

  route(a: XZ, b: XZ): XZ[] | null {
    if (this.clear(a, b)) return [a, b]
    const ra = this.roomAt(...a)
    const rb = this.roomAt(...b)
    if (!ra || !rb || ra === rb) return null
    const prev = new Map<string, { door: Door; room: string }>([[ra, { door: null!, room: "" }]])
    const queue = [ra]
    while (queue.length) {
      const u = queue.shift()!
      if (u === rb) break
      for (const d of this.doors) {
        if (d.rooms.length < 2 || !d.rooms.includes(u)) continue
        const v = d.rooms.find((r) => r !== u)!
        if (prev.has(v)) continue
        prev.set(v, { door: d, room: u })
        queue.push(v)
      }
    }
    if (!prev.has(rb)) return null
    const steps: { door: Door; from: string }[] = []
    for (let r = rb; r !== ra; r = prev.get(r)!.room) steps.unshift({ door: prev.get(r)!.door, from: prev.get(r)!.room })
    const pts: XZ[] = [a]
    for (const { door, from } of steps) {
      const mid = (door.from + door.to) / 2
      const r = this.byId.get(from)!
      const sign = door.axis === "x" ? Math.sign((r.z0 + r.z1) / 2 - door.at) : Math.sign((r.x0 + r.x1) / 2 - door.at)
      const side = (o: number): XZ => (door.axis === "x" ? [mid, door.at + o] : [door.at + o, mid])
      pts.push(side(sign * 0.5), side(-sign * 0.5))
    }
    pts.push(b)
    const out: XZ[] = [a]
    let i = 0
    while (i < pts.length - 1) {
      let j = pts.length - 1
      while (j > i + 1 && !this.clear(pts[i], pts[j])) j--
      out.push(pts[j])
      i = j
    }
    return out
  }
}
