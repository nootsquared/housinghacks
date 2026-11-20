import * as THREE from "three"

const cache = new Map<string, THREE.CanvasTexture>()

function rng(seed: number) {
  let s = seed >>> 0 || 1
  return () => {
    s ^= s << 13
    s ^= s >>> 17
    s ^= s << 5
    return ((s >>> 0) % 100000) / 100000
  }
}

function hash(str: string) {
  let h = 2166136261
  for (let i = 0; i < str.length; i++) h = Math.imul(h ^ str.charCodeAt(i), 16777619)
  return h >>> 0
}

export function shade(hex: string, k: number) {
  const c = new THREE.Color(hex)
  const hsl = { h: 0, s: 0, l: 0 }
  c.getHSL(hsl)
  c.setHSL(hsl.h, hsl.s, THREE.MathUtils.clamp(hsl.l * k, 0, 1))
  return `#${c.getHexString()}`
}

export const PX_PER_M = 256

export function floorTexture(material: string, color: string, accent?: string) {
  const key = `${material}:${color}:${accent}`
  if (cache.has(key)) return cache.get(key)!
  const size = 1024
  const cv = document.createElement("canvas")
  cv.width = cv.height = size
  const g = cv.getContext("2d")!
  const r = rng(hash(key))
  if (material === "wood") {
    const plankW = 0.19 * PX_PER_M
    const plankL = 1.22 * PX_PER_M
    for (let x = 0, col = 0; x < size; x += plankW, col++) {
      let y = -r() * plankL
      while (y < size) {
        g.fillStyle = shade(color, 0.9 + r() * 0.2)
        g.fillRect(x, y, plankW, plankL)
        g.globalAlpha = 0.07
        g.strokeStyle = shade(accent ?? color, 0.75)
        for (let k = 0; k < 7; k++) {
          const gx = x + r() * plankW
          g.beginPath()
          g.moveTo(gx, y)
          g.bezierCurveTo(gx + (r() - 0.5) * 6, y + plankL * 0.33, gx + (r() - 0.5) * 6, y + plankL * 0.66, gx, y + plankL)
          g.lineWidth = 1 + r() * 1.5
          g.stroke()
        }
        g.globalAlpha = 0.35
        g.fillStyle = shade(color, 0.62)
        g.fillRect(x, y, plankW, 1.5)
        g.fillRect(x, y, 1.5, plankL)
        g.globalAlpha = 1
        y += plankL
      }
    }
  } else if (material === "tile") {
    const t = 0.6 * PX_PER_M
    g.fillStyle = shade(color, 0.86)
    g.fillRect(0, 0, size, size)
    for (let x = 0; x < size; x += t) {
      for (let y = 0; y < size; y += t) {
        g.fillStyle = shade(color, 0.97 + r() * 0.06)
        g.fillRect(x + 2, y + 2, t - 4, t - 4)
      }
    }
  } else {
    g.fillStyle = color
    g.fillRect(0, 0, size, size)
    const img = g.getImageData(0, 0, size, size)
    for (let i = 0; i < img.data.length; i += 4) {
      const n = (r() - 0.5) * 22
      img.data[i] += n
      img.data[i + 1] += n
      img.data[i + 2] += n
    }
    g.putImageData(img, 0, 0)
  }
  const tex = new THREE.CanvasTexture(cv)
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping
  tex.colorSpace = THREE.SRGBColorSpace
  tex.anisotropy = 8
  cache.set(key, tex)
  return tex
}

export function fabricTint(hex: string, k = 1) {
  return shade(hex, k)
}

let ring: THREE.CanvasTexture | null = null

export function ringTexture() {
  if (ring) return ring
  const size = 256
  const cv = document.createElement("canvas")
  cv.width = cv.height = size
  const g = cv.getContext("2d")!
  const c = size / 2
  const halo = g.createRadialGradient(c, c, c * 0.55, c, c, c)
  halo.addColorStop(0, "rgba(0,0,0,0.16)")
  halo.addColorStop(1, "rgba(0,0,0,0)")
  g.fillStyle = halo
  g.fillRect(0, 0, size, size)
  g.beginPath()
  g.arc(c, c, c * 0.62, 0, Math.PI * 2)
  g.fillStyle = "rgba(255,255,255,0.22)"
  g.fill()
  g.lineWidth = c * 0.09
  g.strokeStyle = "rgba(255,255,255,0.95)"
  g.stroke()
  ring = new THREE.CanvasTexture(cv)
  ring.colorSpace = THREE.SRGBColorSpace
  ring.anisotropy = 8
  return ring
}
