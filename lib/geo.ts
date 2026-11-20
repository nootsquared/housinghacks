export function meters(lat1: number, lng1: number, lat2: number, lng2: number) {
  const R = 6371000
  const p1 = (lat1 * Math.PI) / 180
  const p2 = (lat2 * Math.PI) / 180
  const dp = ((lat2 - lat1) * Math.PI) / 180
  const dl = ((lng2 - lng1) * Math.PI) / 180
  const h = Math.sin(dp / 2) ** 2 + Math.cos(p1) * Math.cos(p2) * Math.sin(dl / 2) ** 2
  return 2 * R * Math.asin(Math.min(1, Math.sqrt(h)))
}

export function nearest<T extends { lat: number; lng: number }>(lat: number, lng: number, items: T[]) {
  let best = items[0]
  let bestD = Infinity
  for (const item of items) {
    const d = meters(lat, lng, item.lat, item.lng)
    if (d < bestD) {
      best = item
      bestD = d
    }
  }
  return { item: best, meters: bestD }
}

export function walkMinutes(meters: number) {
  return Math.max(1, Math.round((meters * 1.3) / 80))
}
