import { RoundedBox } from "@react-three/drei"
import * as THREE from "three"
import type { Room, StagedItem } from "@/lib/types"

type V3 = [number, number, number]

const CAB = "#ddd3c4"
const QUARTZ = "#f4f3f0"
const STEEL = "#b9bcbf"
const STEEL_DARK = "#8d9094"
const BLACK = "#1e1f22"
const PULL = "#9a9ca0"
const WHITE = "#f6f6f4"

function Box({ p, s, c, r = 0.015, rough = 0.7, metal = 0, clip }: { p: V3; s: V3; c: string; r?: number; rough?: number; metal?: number; clip: THREE.Plane[] }) {
  return (
    <RoundedBox args={s} radius={Math.min(r, ...s.map((x) => x / 2.2))} smoothness={3} position={p} castShadow receiveShadow>
      <meshStandardMaterial color={c} roughness={rough} metalness={metal} clippingPlanes={clip} />
    </RoundedBox>
  )
}

function Cyl({ p, r, h, c, rough = 0.6, metal = 0, r2, seg = 32, clip }: { p: V3; r: number; h: number; c: string; rough?: number; metal?: number; r2?: number; seg?: number; clip: THREE.Plane[] }) {
  return (
    <mesh position={p} castShadow receiveShadow>
      <cylinderGeometry args={[r, r2 ?? r, h, seg]} />
      <meshStandardMaterial color={c} roughness={rough} metalness={metal} clippingPlanes={clip} />
    </mesh>
  )
}

function Glass({ p, s, clip }: { p: V3; s: V3; clip: THREE.Plane[] }) {
  return (
    <mesh position={p}>
      <boxGeometry args={s} />
      <meshPhysicalMaterial color="#f1f6f7" transparent opacity={0.14} roughness={0.04} metalness={0.1} clippingPlanes={clip} depthWrite={false} />
    </mesh>
  )
}

type P = { w: number; h: number; d: number; c?: string; clip: THREE.Plane[] }

function Kitchen({ w, d, clip }: P) {
  const base = 0.9
  const y = -base / 2
  const upperY = 1.45
  const upperH = 0.8
  const range = w * 0.12
  const sink = -w * 0.2
  const dw = -w * 0.38
  const doors = Math.round(w / 0.5)
  return (
    <group>
      <Box p={[0, y + 0.05, -0.04]} s={[w, 0.1, d - 0.08]} c="#3a3631" clip={clip} />
      <Box p={[0, y + 0.1 + (base - 0.14) / 2, 0]} s={[w, base - 0.14, d]} c={CAB} r={0.006} rough={0.55} clip={clip} />
      <Box p={[0, y + base - 0.02, 0.015]} s={[w + 0.02, 0.04, d + 0.03]} c={QUARTZ} r={0.004} rough={0.25} clip={clip} />
      {Array.from({ length: doors - 1 }, (_, i) => (
        <Box key={i} p={[-w / 2 + ((i + 1) * w) / doors, y + 0.48, d / 2 + 0.002]} s={[0.006, base - 0.18, 0.004]} c="#c8bead" r={0.001} clip={clip} />
      ))}
      {Array.from({ length: doors }, (_, i) => (
        <Box key={`h${i}`} p={[-w / 2 + ((i + 0.5) * w) / doors, y + 0.72, d / 2 + 0.015]} s={[0.16, 0.012, 0.02]} c={PULL} r={0.004} metal={0.7} rough={0.3} clip={clip} />
      ))}
      <Box p={[dw, y + 0.5, d / 2 + 0.004]} s={[0.6, base - 0.2, 0.01]} c={STEEL} metal={0.75} rough={0.3} r={0.004} clip={clip} />
      <Box p={[range, y + 0.48, d / 2 - 0.27]} s={[0.76, base - 0.02, 0.54]} c={STEEL} metal={0.75} rough={0.3} r={0.006} clip={clip} />
      <Box p={[range, y + 0.42, d / 2 + 0.015]} s={[0.6, 0.42, 0.01]} c={BLACK} rough={0.1} r={0.004} clip={clip} />
      <Box p={[range, y + base - 0.005, d / 2 - 0.27]} s={[0.74, 0.012, 0.52]} c={BLACK} rough={0.15} r={0.003} clip={clip} />
      {[-0.18, 0.18].flatMap((dx) =>
        [-0.14, 0.12].map((dz) => <Cyl key={`${dx}${dz}`} p={[range + dx, y + base + 0.005, d / 2 - 0.27 + dz * 0.85]} r={0.09} h={0.004} c="#3b3c40" clip={clip} />),
      )}
      <Box p={[range, upperY - 0.18, -d / 2 + 0.19]} s={[0.76, 0.42, 0.38]} c={STEEL} metal={0.75} rough={0.3} r={0.01} clip={clip} />
      <Box p={[range - 0.06, upperY - 0.18, -d / 2 + 0.385]} s={[0.5, 0.3, 0.008]} c={BLACK} rough={0.1} r={0.003} clip={clip} />
      <Box p={[sink, y + base + 0.001, 0.02]} s={[0.7, 0.01, 0.42]} c="#9fa2a6" metal={0.8} rough={0.2} r={0.004} clip={clip} />
      <Cyl p={[sink, y + base + 0.18, -0.2]} r={0.016} h={0.36} c={STEEL_DARK} metal={0.9} rough={0.2} clip={clip} />
      <Box p={[sink, y + base + 0.34, -0.12]} s={[0.025, 0.025, 0.18]} c={STEEL_DARK} metal={0.9} rough={0.2} clip={clip} />
      <Box p={[0, y + base + 0.3, -d / 2 + 0.005]} s={[w, 0.56, 0.01]} c="#f1f0ec" rough={0.3} r={0.002} clip={clip} />
      {[[-w / 2, range - 0.4], [range + 0.4, w / 2]].map(([a, b], k) => (
        <group key={k}>
          <Box p={[(a + b) / 2, upperY + upperH / 2 - 0.05, -d / 2 + 0.18]} s={[b - a, upperH, 0.34]} c={CAB} r={0.006} rough={0.55} clip={clip} />
          {Array.from({ length: Math.max(1, Math.round((b - a) / 0.45)) }, (_, i, n = Math.max(1, Math.round((b - a) / 0.45))) => (
            <Box key={i} p={[a + ((i + 0.5) * (b - a)) / n, upperY + 0.05, -d / 2 + 0.36]} s={[0.012, 0.14, 0.02]} c={PULL} metal={0.7} rough={0.3} r={0.004} clip={clip} />
          ))}
        </group>
      ))}
      <Box p={[range, upperY + 0.47, -d / 2 + 0.18]} s={[0.8, 0.36, 0.34]} c={CAB} r={0.006} rough={0.55} clip={clip} />
      <Box p={[-w * 0.42, y + base + 0.12, -0.12]} s={[0.22, 0.22, 0.3]} c="#26272a" r={0.02} rough={0.4} clip={clip} />
      <Cyl p={[range + 0.62, y + base + 0.11, -0.15]} r={0.07} h={0.22} c="#6d5a45" clip={clip} />
    </group>
  )
}

function Fridge({ w, h, d, clip }: P) {
  return (
    <group>
      <Box p={[0, 0, 0]} s={[w, h, d]} c={STEEL} metal={0.75} rough={0.28} r={0.02} clip={clip} />
      <Box p={[0, h * 0.18, d / 2 + 0.003]} s={[w - 0.02, 0.006, 0.006]} c={STEEL_DARK} clip={clip} />
      <Box p={[0, h / 2 - (h * 0.32) / 2 - 0.01, d / 2 + 0.003]} s={[0.004, h * 0.62, 0.006]} c={STEEL_DARK} clip={clip} />
      {[-1, 1].map((s) => (
        <Box key={s} p={[s * 0.04, h * 0.32, d / 2 + 0.04]} s={[0.02, h * 0.3, 0.025]} c="#dcdee0" metal={0.9} rough={0.2} r={0.008} clip={clip} />
      ))}
      <Box p={[0, -h / 2 + h * 0.25, d / 2 + 0.04]} s={[w * 0.5, 0.02, 0.025]} c="#dcdee0" metal={0.9} rough={0.2} r={0.008} clip={clip} />
    </group>
  )
}

function Island({ w, h, d, clip }: P) {
  return (
    <group>
      <Box p={[0, -0.02, -0.1]} s={[w - 0.06, h - 0.04, d - 0.3]} c={CAB} r={0.006} rough={0.55} clip={clip} />
      <Box p={[0, h / 2 - 0.025, 0]} s={[w, 0.05, d]} c={QUARTZ} r={0.006} rough={0.25} clip={clip} />
      {[-1, 1].map((s) => (
        <Box key={s} p={[(s * (w - 0.05)) / 2, 0, 0]} s={[0.05, h, d]} c={QUARTZ} r={0.006} rough={0.25} clip={clip} />
      ))}
      {[-0.5, 0, 0.5].map((u) => (
        <Box key={u} p={[u * (w - 0.3), 0.2, -d / 2 - 0.01]} s={[0.14, 0.012, 0.02]} c={PULL} metal={0.7} rough={0.3} r={0.004} clip={clip} />
      ))}
    </group>
  )
}

function Stool({ h, c = "#e6e4df", clip }: P) {
  const seat = 0.66
  const y = -h / 2
  return (
    <group>
      <Box p={[0, y + seat, 0]} s={[0.42, 0.07, 0.4]} c={c} r={0.03} rough={0.85} clip={clip} />
      <Box p={[0, y + seat + 0.2, -0.17]} s={[0.4, 0.34, 0.05]} c={c} r={0.025} rough={0.85} clip={clip} />
      {[[-0.16, -0.15], [0.16, -0.15], [-0.16, 0.15], [0.16, 0.15]].map(([x, z], i) => (
        <Box key={i} p={[x, y + seat / 2, z]} s={[0.022, seat, 0.022]} c={BLACK} r={0.008} metal={0.5} clip={clip} />
      ))}
      <Box p={[0, y + 0.24, 0.15]} s={[0.32, 0.018, 0.018]} c={BLACK} r={0.006} clip={clip} />
    </group>
  )
}

function DiningTable({ w, h, d, c = "#c99a5f", clip }: P) {
  const t = 0.045
  return (
    <group>
      <Box p={[0, h / 2 - t / 2, 0]} s={[w, t, d]} c={c} r={0.008} rough={0.55} clip={clip} />
      {[[-1, -1], [1, -1], [-1, 1], [1, 1]].map(([sx, sz], i) => (
        <Box key={i} p={[(sx * (w - 0.12)) / 2, -t / 2, (sz * (d - 0.12)) / 2]} s={[0.055, h - t, 0.055]} c={c} r={0.01} rough={0.55} clip={clip} />
      ))}
      <Box p={[0, h / 2 - t - 0.04, 0]} s={[w - 0.16, 0.07, d - 0.16]} c="#b8894f" r={0.004} clip={clip} />
      <Cyl p={[0, h / 2 + 0.08, 0]} r={0.07} r2={0.05} h={0.16} c="#e9e6df" rough={0.3} clip={clip} />
    </group>
  )
}

function Chair({ h, c = "#e19a35", clip }: P) {
  const y = -h / 2
  const seat = 0.46
  return (
    <group>
      <Box p={[0, y + seat, 0.02]} s={[0.46, 0.035, 0.44]} c={c} r={0.015} rough={0.45} clip={clip} />
      <Box p={[0, y + seat + 0.2, -0.21]} s={[0.46, 0.36, 0.03]} c={c} r={0.015} rough={0.45} clip={clip} />
      {[[-0.19, -0.18], [0.19, -0.18], [-0.19, 0.2], [0.19, 0.2]].map(([x, z], i) => (
        <Box key={i} p={[x, y + seat / 2, z]} s={[0.03, seat, 0.03]} c={c} r={0.01} rough={0.45} clip={clip} />
      ))}
    </group>
  )
}

function Daybed({ w, h, d, c = "#5d6064", clip }: P) {
  const y = -h / 2
  const base = 0.14
  const seatTop = 0.42
  return (
    <group>
      <Box p={[0, y + 0.1, 0]} s={[w + 0.04, 0.05, d + 0.02]} c="#9a5f36" r={0.02} rough={0.45} clip={clip} />
      {[[-1, -1], [1, -1], [-1, 1], [1, 1]].map(([sx, sz], i) => (
        <Box key={i} p={[(sx * (w - 0.1)) / 2, y + 0.04, (sz * (d - 0.1)) / 2]} s={[0.04, 0.08, 0.04]} c={BLACK} r={0.01} clip={clip} />
      ))}
      <Box p={[0, y + base + (seatTop - base) / 2, 0.05]} s={[w, seatTop - base, d - 0.1]} c={c} r={0.07} rough={0.97} clip={clip} />
      <Box p={[0, y + seatTop + 0.17, -d / 2 + 0.12]} s={[w, 0.34, 0.22]} c={c} r={0.09} rough={0.97} clip={clip} />
      <Box p={[0, y + seatTop - 0.002, 0.05]} s={[0.006, 0.01, d - 0.16]} c="#4a4d51" clip={clip} />
      <mesh position={[w / 2 - 0.2, y + seatTop + 0.11, 0.05]} rotation={[Math.PI / 2, 0, 0]} castShadow>
        <cylinderGeometry args={[0.11, 0.11, d - 0.2, 24]} />
        <meshStandardMaterial color={c} roughness={0.97} clippingPlanes={clip} />
      </mesh>
      <Box p={[w / 2 - 0.2, y + seatTop + 0.11, 0.05]} s={[0.24, 0.03, 0.06]} c="#a0643a" r={0.01} clip={clip} />
    </group>
  )
}

function Rug({ w, d, c = "#3d3f44", clip }: P) {
  return (
    <group>
      <Box p={[0, 0, 0]} s={[w, 0.012, d]} c={c} r={0.004} rough={1} clip={clip} />
      <Box p={[0, 0.0065, 0]} s={[w - 0.3, 0.002, d - 0.3]} c="#55585e" r={0.001} rough={1} clip={clip} />
    </group>
  )
}

function CoffeeTable({ w, h, clip }: P) {
  return (
    <group>
      <Cyl p={[0, h / 2 - 0.02, 0]} r={w / 2} h={0.04} c="#ecebe8" rough={0.25} seg={48} clip={clip} />
      <Cyl p={[0, -0.02, 0]} r={w / 2 - 0.05} r2={w / 2 - 0.08} h={h - 0.04} c="#f2f1ee" rough={0.3} seg={48} clip={clip} />
    </group>
  )
}

function Tv({ w, h, clip }: P) {
  return (
    <group>
      <Box p={[0, 0, 0]} s={[w, h, 0.04]} c={BLACK} r={0.006} rough={0.25} clip={clip} />
      <Box p={[0, 0, 0.021]} s={[w - 0.03, h - 0.03, 0.002]} c="#0d0e10" r={0.001} rough={0.05} clip={clip} />
    </group>
  )
}

function Media({ w, h, d, clip }: P) {
  const y = -h / 2
  const top = 0.46
  return (
    <group>
      <Box p={[0, y + 0.04, 0]} s={[w - 0.1, 0.08, d - 0.08]} c={BLACK} r={0.01} clip={clip} />
      <Box p={[0, y + 0.08 + (top - 0.08) / 2, 0]} s={[w, top - 0.08, d]} c="#efede8" r={0.01} rough={0.45} clip={clip} />
      {[-1, 0, 1].map((u) => (
        <Box key={u} p={[(u * w) / 3, y + 0.27, d / 2 + 0.002]} s={[0.005, top - 0.14, 0.004]} c="#cfccc5" r={0.001} clip={clip} />
      ))}
      <Box p={[0, y + top + 0.04, -0.02]} s={[0.3, 0.012, 0.2]} c="#2a2b2e" r={0.004} clip={clip} />
      <Box p={[0, y + top + 0.12, -0.02]} s={[0.05, 0.16, 0.03]} c="#2a2b2e" r={0.004} clip={clip} />
      <group position={[0, y + top + 0.2 + 0.33, -0.02]}>
        <Tv w={1.15} h={0.66} d={0.04} clip={clip} />
      </group>
      <Box p={[w / 2 - 0.22, y + top + 0.1, 0]} s={[0.14, 0.2, 0.14]} c="#5f7a52" r={0.05} rough={0.9} clip={clip} />
    </group>
  )
}

function Pouf({ w, h, clip }: P) {
  return <Cyl p={[0, 0, 0]} r={w / 2} r2={w / 2 + 0.03} h={h} c="#e2dfd8" rough={1} clip={clip} />
}

function Vanity({ w, d, h, clip }: P) {
  const y = -h / 2
  const top = 0.86
  return (
    <group>
      <Box p={[0, y + 0.1 + (top - 0.13) / 2, 0]} s={[w, top - 0.13, d]} c="#d0cfcb" r={0.006} rough={0.45} clip={clip} />
      <Box p={[0, y + 0.06, -0.03]} s={[w - 0.04, 0.1, d - 0.06]} c="#6b6c6e" clip={clip} />
      <Box p={[0, y + top - 0.015, 0.01]} s={[w + 0.02, 0.03, d + 0.02]} c={QUARTZ} r={0.004} rough={0.2} clip={clip} />
      <Box p={[0, y + 0.45, d / 2 + 0.002]} s={[0.006, top - 0.25, 0.004]} c="#b5b4b0" clip={clip} />
      {[-1, 1].map((s) => (
        <Box key={s} p={[s * 0.05, y + 0.62, d / 2 + 0.015]} s={[0.014, 0.14, 0.02]} c={BLACK} r={0.005} clip={clip} />
      ))}
      <Box p={[0, y + top + 0.002, 0.03]} s={[0.42, 0.008, 0.3]} c={WHITE} r={0.003} rough={0.15} clip={clip} />
      <Cyl p={[0, y + top + 0.09, -d / 2 + 0.07]} r={0.014} h={0.18} c={STEEL} metal={0.9} rough={0.2} clip={clip} />
      <Box p={[0, y + top + 0.17, -d / 2 + 0.13]} s={[0.025, 0.02, 0.13]} c={STEEL} metal={0.9} rough={0.2} clip={clip} />
      <Box p={[0, y + 1.6, -d / 2 + 0.02]} s={[w * 0.95, 0.8, 0.03]} c="#f7f7f3" r={0.01} rough={0.2} clip={clip} />
      <mesh position={[0, y + 1.6, -d / 2 + 0.036]}>
        <boxGeometry args={[w * 0.95 - 0.08, 0.72, 0.002]} />
        <meshStandardMaterial color="#c9d4d8" roughness={0.02} metalness={0.9} clippingPlanes={clip} />
      </mesh>
      <Box p={[0, y + 2.05, -d / 2 + 0.06]} s={[0.5, 0.05, 0.08]} c="#ededea" r={0.015} clip={clip} />
      <Box p={[w / 2 - 0.08, y + top + 0.06, 0.05]} s={[0.07, 0.12, 0.07]} c="#d6a637" r={0.01} rough={0.3} clip={clip} />
    </group>
  )
}

function Toilet({ h, clip }: P) {
  const y = -h / 2
  return (
    <group>
      <Box p={[0, y + 0.2, 0.06]} s={[0.36, 0.4, 0.46]} c={WHITE} r={0.12} rough={0.15} clip={clip} />
      <Box p={[0, y + 0.41, 0.08]} s={[0.38, 0.03, 0.46]} c={WHITE} r={0.012} rough={0.15} clip={clip} />
      <Box p={[0, y + 0.6, -0.2]} s={[0.42, 0.38, 0.18]} c={WHITE} r={0.04} rough={0.15} clip={clip} />
      <Box p={[0, y + 0.8, -0.2]} s={[0.44, 0.03, 0.2]} c={WHITE} r={0.01} rough={0.15} clip={clip} />
    </group>
  )
}

function Shower({ w, h, d, clip }: P) {
  const y = -h / 2
  return (
    <group>
      <Box p={[0, y + 0.03, 0]} s={[w, 0.06, d]} c={WHITE} r={0.01} rough={0.2} clip={clip} />
      <Box p={[0, 0, -d / 2 + 0.01]} s={[w, h, 0.02]} c="#eceae6" rough={0.25} r={0.002} clip={clip} />
      <Box p={[-w / 2 + 0.01, 0, 0]} s={[0.02, h, d]} c="#eceae6" rough={0.25} r={0.002} clip={clip} />
      <Box p={[w / 2 - 0.01, 0, 0]} s={[0.02, h, d]} c="#eceae6" rough={0.25} r={0.002} clip={clip} />
      {[[-0.3, 0.2, 0.9], [0.1, -0.3, 0.6], [0.35, 0.4, 1.1]].map(([u, v, l], i) => (
        <Box key={i} p={[u * w, v * h, -d / 2 + 0.021]} s={[l * 0.5, 0.006, 0.002]} c="#b9bbbd" r={0.001} clip={clip} />
      ))}
      <Glass p={[0, 0.03, d / 2 - 0.02]} s={[w - 0.04, h - 0.06, 0.012]} clip={clip} />
      <Box p={[0, h / 2 - 0.08, d / 2 - 0.02]} s={[w, 0.03, 0.04]} c={STEEL} metal={0.9} rough={0.2} clip={clip} />
      <Cyl p={[0, h / 2 - 0.35, -d / 2 + 0.12]} r={0.1} h={0.015} c={STEEL} metal={0.9} rough={0.2} clip={clip} />
      <Box p={[0, h / 2 - 0.3, -d / 2 + 0.06]} s={[0.025, 0.025, 0.1]} c={STEEL} metal={0.9} rough={0.2} clip={clip} />
      <Box p={[0.25, 0.05, -d / 2 + 0.05]} s={[0.08, 0.14, 0.06]} c={STEEL} metal={0.9} rough={0.2} clip={clip} />
    </group>
  )
}

function Bed({ w, h, d, c = "#f4f4f2", clip }: P) {
  const y = -h / 2
  return (
    <group>
      <Box p={[0, y + 0.13, 0.03]} s={[w, 0.26, d - 0.06]} c="#e8e4dd" r={0.02} rough={0.9} clip={clip} />
      <Box p={[0, y + 0.42, 0.04]} s={[w - 0.04, 0.32, d - 0.1]} c={c} r={0.09} rough={0.98} clip={clip} />
      <Box p={[0, y + 0.5, 0.36]} s={[w + 0.06, 0.2, d * 0.62]} c="#fbfbfa" r={0.08} rough={0.98} clip={clip} />
      {[-1, 1].map((s) => (
        <Box key={s} p={[(s * w) / 4.2, y + 0.67, -d / 2 + 0.3]} s={[w / 2.4, 0.2, 0.42]} c={WHITE} r={0.09} rough={0.98} clip={clip} />
      ))}
      {[-1, 1].map((s) => (
        <Box key={`b${s}`} p={[(s * w) / 4.2, y + 0.82, -d / 2 + 0.16]} s={[w / 2.5, 0.38, 0.14]} c="#fdfdfc" r={0.07} rough={0.98} clip={clip} />
      ))}
    </group>
  )
}

function Nightstand({ w, h, d, c = "#2b2b2d", clip }: P) {
  return (
    <group>
      <Cyl p={[0, 0, 0]} r={w / 2} r2={w / 2.6} h={h} c={c} rough={0.5} clip={clip} />
      <Cyl p={[0, h / 2 + 0.12, 0]} r={0.025} h={0.24} c="#c79a3a" metal={0.6} rough={0.3} clip={clip} />
      <Cyl p={[0, h / 2 + 0.28, 0]} r={0.06} r2={0.11} h={0.12} c="#e6b440" rough={0.4} clip={clip} />
      <Box p={[0, h / 2 + 0.005, d / 4]} s={[0.12, 0.01, 0.08]} c="#ddd" r={0.002} clip={clip} />
    </group>
  )
}

function Closet({ w, h, clip }: P) {
  return (
    <group>
      <Box p={[0, 0, 0]} s={[w + 0.08, h + 0.04, 0.03]} c="#f1efea" r={0.004} clip={clip} />
      {[-1, 1].map((s) => (
        <group key={s}>
          <Box p={[(s * w) / 4, -0.01, 0.025 + (s > 0 ? 0.03 : 0)]} s={[w / 2, h - 0.02, 0.03]} c="#4b433e" r={0.004} rough={0.55} clip={clip} />
          <Box p={[(s * w) / 4, -0.01, 0.042 + (s > 0 ? 0.03 : 0)]} s={[w / 2 - 0.16, h - 0.2, 0.004]} c="#433b36" r={0.002} rough={0.55} clip={clip} />
          <Cyl p={[s * 0.06 + (s * w) / 4 - (s * w) / 4 + s * (w / 2 - 0.1), 0.05, 0.05 + (s > 0 ? 0.03 : 0)]} r={0.025} h={0.01} c={PULL} metal={0.7} clip={clip} />
        </group>
      ))}
    </group>
  )
}

function Laundry({ w, h, d, clip }: P) {
  const unit = h / 2
  return (
    <group>
      {[-1, 1].map((s) => (
        <group key={s} position={[0, (s * unit) / 2, 0]}>
          <Box p={[0, 0, 0]} s={[w, unit - 0.01, d]} c={WHITE} r={0.02} rough={0.3} clip={clip} />
          <mesh position={[0, -0.04, d / 2 + 0.005]} rotation={[Math.PI / 2, 0, 0]}>
            <cylinderGeometry args={[w * 0.3, w * 0.3, 0.02, 40]} />
            <meshStandardMaterial color="#c9ccd0" roughness={0.1} metalness={0.4} clippingPlanes={clip} />
          </mesh>
          <Box p={[0, unit / 2 - 0.08, d / 2 + 0.004]} s={[w - 0.04, 0.1, 0.006]} c="#e0e1e3" clip={clip} />
        </group>
      ))}
    </group>
  )
}

function Panel({ w, h, d, c = "#5fae3a", clip }: P) {
  return <Box p={[0, 0, 0]} s={[w, h, d]} c={c} r={0.03} rough={0.9} clip={clip} />
}

function Niche({ w, h, d, clip }: P) {
  const t = 0.04
  return (
    <group>
      <Box p={[0, 0, -d / 2 + t / 2]} s={[w, h, t]} c="#ecebe7" r={0.002} clip={clip} />
      <Box p={[-w / 2 + t / 2, 0, 0]} s={[t, h, d]} c="#ecebe7" r={0.002} clip={clip} />
      <Box p={[w / 2 - t / 2, 0, 0]} s={[t, h, d]} c="#ecebe7" r={0.002} clip={clip} />
      <Box p={[0, h / 2 - t / 2, 0]} s={[w, t, d]} c="#ecebe7" r={0.002} clip={clip} />
    </group>
  )
}

const KINDS: Record<string, (p: P) => React.ReactElement> = {
  kitchen: Kitchen,
  fridge: Fridge,
  island: Island,
  stool: Stool,
  diningTable: DiningTable,
  chair: Chair,
  daybed: Daybed,
  rug: Rug,
  coffeeTable: CoffeeTable,
  tv: Tv,
  pouf: Pouf,
  media: Media,
  vanity: Vanity,
  toilet: Toilet,
  shower: Shower,
  bed: Bed,
  nightstand: Nightstand,
  closet: Closet,
  laundry: Laundry,
  panel: Panel,
  niche: Niche,
}

export function Staging({ room, clip, mode }: { room: Room; clip: THREE.Plane[]; mode: "walk" | "dollhouse" }) {
  return (
    <group>
      {(room.staging ?? []).map((it: StagedItem, i) => {
        const Kind = KINDS[it.kind]
        if (!Kind || (it.wall && mode !== "walk")) return null
        const [w, h, d] = it.size
        return (
          <group key={i} position={it.center} rotation={[0, it.rot ?? 0, 0]}>
            <Kind w={w} h={h} d={d} c={it.color} clip={clip} />
          </group>
        )
      })}
    </group>
  )
}
