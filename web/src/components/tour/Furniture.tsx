import { RoundedBox } from "@react-three/drei"
import type { Furniture as Item } from "@/lib/types"
import { fabricTint } from "./materials"

type Side = "-x" | "+x" | "-z" | "+z"

type Props = { item: Item; back: Side }

const WOOD = "#8a6f57"
const METAL = "#c4c7ca"
const DARK = "#2a2b2e"

function Box({ p, s, c, r = 0.02, rough = 0.8, metal = 0 }: { p: number[]; s: number[]; c: string; r?: number; rough?: number; metal?: number }) {
  return (
    <RoundedBox args={s as [number, number, number]} radius={Math.min(r, ...s.map((x) => x / 2.2))} smoothness={3} position={p as [number, number, number]} castShadow receiveShadow>
      <meshStandardMaterial color={c} roughness={rough} metalness={metal} />
    </RoundedBox>
  )
}

function backOffset(back: Side, w: number, d: number, t: number): { p: [number, number]; s: [number, number] } {
  if (back === "-x") return { p: [-w / 2 + t / 2, 0], s: [t, d] }
  if (back === "+x") return { p: [w / 2 - t / 2, 0], s: [t, d] }
  if (back === "-z") return { p: [0, -d / 2 + t / 2], s: [w, t] }
  return { p: [0, d / 2 - t / 2], s: [w, t] }
}

export function FurnitureModel({ item, back }: Props) {
  const [w, h, d] = item.size
  const c = item.color
  const y0 = -h / 2
  switch (item.kind) {
    case "bed": {
      const frameH = Math.min(0.3, h * 0.45)
      const top = Math.max(h, 0.5)
      const hb = backOffset(back, w, d, 0.08)
      return (
        <group position={[0, (top - h) / 2, 0]}>
          <Box p={[0, -top / 2 + frameH / 2, 0]} s={[w, frameH, d]} c={fabricTint(c, 0.7)} />
          <Box p={[0, -top / 2 + frameH + 0.11, 0]} s={[w * 0.97, 0.22, d * 0.97]} c={fabricTint(c, 1.05)} r={0.06} rough={0.95} />
          <Box p={[hb.p[0], -top / 2 + 0.55, hb.p[1]]} s={[hb.s[0], 1.1, hb.s[1]]} c={item.accent ?? fabricTint(c, 0.6)} r={0.03} />
        </group>
      )
    }
    case "sofa": {
      const seatH = Math.min(0.42, h * 0.7)
      const bk = backOffset(back, w, d, Math.min(0.22, Math.min(w, d) * 0.3))
      const top = Math.max(h, 0.75)
      return (
        <group position={[0, (top - h) / 2, 0]}>
          <Box p={[0, -top / 2 + seatH / 2, 0]} s={[w, seatH, d]} c={c} r={0.06} rough={0.95} />
          <Box p={[bk.p[0], -top / 2 + seatH + (top - seatH) / 2, bk.p[1]]} s={[bk.s[0], top - seatH, bk.s[1]]} c={c} r={0.06} rough={0.95} />
        </group>
      )
    }
    case "table":
    case "desk": {
      const t = 0.04
      const leg = 0.045
      const top = Math.max(h, 0.4)
      const legs = [
        [-w / 2 + leg, -d / 2 + leg],
        [w / 2 - leg, -d / 2 + leg],
        [-w / 2 + leg, d / 2 - leg],
        [w / 2 - leg, d / 2 - leg],
      ]
      return (
        <group position={[0, (top - h) / 2, 0]}>
          <Box p={[0, top / 2 - t / 2, 0]} s={[w, t, d]} c={c} rough={0.6} />
          {legs.map(([x, z], i) => (
            <Box key={i} p={[x, -t / 2, z]} s={[leg, top - t, leg]} c={fabricTint(c, 0.7)} r={0.01} />
          ))}
        </group>
      )
    }
    case "chair":
      return (
        <group>
          <Box p={[0, y0 + 0.45, 0]} s={[w * 0.9, 0.05, d * 0.9]} c={c} />
          <Box p={[0, y0 + 0.21, 0]} s={[0.05, 0.42, 0.05]} c={DARK} r={0.01} />
        </group>
      )
    case "counter":
    case "cabinet": {
      const slab = 0.035
      return (
        <group>
          <Box p={[0, -slab / 2, 0]} s={[w, h - slab, d]} c={c} r={0.01} rough={0.55} />
          <Box p={[0, h / 2 - slab / 2, 0]} s={[w + 0.02, slab, d + 0.02]} c={fabricTint(c, 1.08)} r={0.005} rough={0.3} />
        </group>
      )
    }
    case "appliance":
      return <Box p={[0, 0, 0]} s={[w, h, d]} c={METAL} r={0.02} rough={0.35} metal={0.6} />
    case "toilet":
      return (
        <group>
          <Box p={[0, y0 + 0.2, 0]} s={[Math.min(w, 0.4), 0.4, Math.min(d, 0.6)]} c="#f2f2f0" r={0.08} rough={0.2} />
        </group>
      )
    case "shower":
      return (
        <group>
          <Box p={[0, y0 + 0.03, 0]} s={[w, 0.06, d]} c="#efefed" r={0.01} rough={0.2} />
          <mesh position={[0, 0.03, 0]}>
            <boxGeometry args={[w, Math.max(h, 1.9), d]} />
            <meshPhysicalMaterial color="#dfe9ec" transparent opacity={0.18} roughness={0.05} />
          </mesh>
        </group>
      )
    case "tv":
      return <Box p={[0, 0, 0]} s={[w, h, Math.min(d, 0.06)]} c={DARK} r={0.005} rough={0.2} />
    case "rug":
      return <Box p={[0, y0 + 0.006, 0]} s={[w, 0.012, d]} c={c} r={0.004} rough={1} />
    case "lamp":
      return (
        <group>
          <Box p={[0, 0, 0]} s={[0.03, h, 0.03]} c={DARK} r={0.01} />
          <mesh position={[0, h / 2 - 0.12, 0]} castShadow>
            <cylinderGeometry args={[0.12, 0.17, 0.24, 24]} />
            <meshStandardMaterial color="#f3efe6" roughness={0.9} />
          </mesh>
        </group>
      )
    case "plant":
      return (
        <group>
          <Box p={[0, y0 + 0.15, 0]} s={[0.3, 0.3, 0.3]} c="#c9c2b6" r={0.04} />
          <mesh position={[0, y0 + 0.3 + h * 0.25, 0]} castShadow>
            <icosahedronGeometry args={[Math.min(w, d) * 0.5, 1]} />
            <meshStandardMaterial color="#5f7a52" roughness={0.9} flatShading />
          </mesh>
        </group>
      )
    default:
      return <Box p={[0, 0, 0]} s={[w, h, d]} c={item.kind === "shelf" ? fabricTint(c, 0.9) : c || WOOD} r={0.015} />
  }
}
