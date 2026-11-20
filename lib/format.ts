import type { Listing, Scene } from "./types"

export function usd(n: number) {
  return n.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 })
}

export function shortUsd(n: number) {
  if (n >= 10000) return `$${Math.round(n / 1000)}K`
  if (n >= 1000) return `$${(n / 1000).toFixed(1).replace(/\.0$/, "")}K`
  return `$${n}`
}

export function bedsLabel(beds: number) {
  return beds === 0 ? "Studio" : `${beds} bd`
}

export function bathsLabel(baths: number | null) {
  return baths == null ? "— ba" : `${baths} ba`
}

export function sqftLabel(sqft: number | null) {
  return sqft ? `${sqft.toLocaleString()} sqft` : "— sqft"
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

export function availability(l: Listing) {
  if (l.status === "now") return "Available now"
  if (l.status === "unavailable") return "Leased"
  if (l.availableOn) {
    const [y, m, d] = l.availableOn.split("-").map(Number)
    return `Available ${MONTHS[m - 1]} ${d}, ${y}`
  }
  return "Ask for dates"
}

export function shortAvailability(l: Listing) {
  if (l.status === "now") return "Now"
  if (l.status === "unavailable") return "Leased"
  if (l.availableOn) {
    const [y, m] = l.availableOn.split("-").map(Number)
    return `${MONTHS[m - 1]} ’${String(y).slice(2)}`
  }
  return "—"
}

export const SCENES: { id: Scene; label: string; hint: string }[] = [
  { id: "day", label: "Weekday", hint: "Traffic, trains, construction" },
  { id: "night", label: "Weekend night", hint: "Bars, Greek row, late traffic" },
  { id: "game", label: "Game day", hint: "Stadium crowd and game traffic" },
]

export function loudness(db: number) {
  if (db < 50) return { word: "Very quiet", tone: "#5fb3a1" }
  if (db < 56) return { word: "Quiet", tone: "#7fbf8e" }
  if (db < 62) return { word: "Moderate", tone: "#e3c565" }
  if (db < 68) return { word: "Busy", tone: "#ec9a4a" }
  return { word: "Loud", tone: "#e5625c" }
}

export function proxy(src: string, size: "medium" | "large" = "large") {
  return size === "medium" ? src.replace(/\/large\.(jpg|jpeg|png|webp)$/, "/medium.$1") : src
}

export function has(l: Listing, pattern: RegExp) {
  return l.amenities.some((a) => pattern.test(a)) || l.appliances.some((a) => pattern.test(a))
}

export function petsAllowed(l: Listing, animal: "cats" | "dogs") {
  const line = l.pets.find((p) => p.toLowerCase().startsWith(animal))
  return !!line && !/not allowed/i.test(line)
}
