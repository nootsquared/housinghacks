import { useEffect, useRef } from "react"
import * as maplibregl from "maplibre-gl"
import type { Listing } from "@/lib/types"

type Props = {
  listings: Listing[]
  hovered: string | null
  onHover: (id: string | null) => void
  onSelect: (id: string) => void
}

const CAMPUS: [number, number] = [-83.7382, 42.2780]

function pillLabel(l: Listing) {
  if (l.rent == null) return `${l.beds} bd`
  const k = l.rent >= 1000 ? `$${(l.rent / 1000).toFixed(l.rent % 1000 === 0 ? 0 : 1)}k` : `$${l.rent}`
  return l.rentBasis === "per person" ? `${k}/pp` : k
}

export function ListingMap({ listings, hovered, onHover, onSelect }: Props) {
  const el = useRef<HTMLDivElement>(null)
  const map = useRef<maplibregl.Map | null>(null)
  const markers = useRef(new Map<string, HTMLElement>())

  useEffect(() => {
    if (!el.current) return
    const m = new maplibregl.Map({
      container: el.current,
      style: "https://tiles.openfreemap.org/styles/positron",
      center: CAMPUS,
      zoom: 13.6,
      pitch: 0,
      attributionControl: { compact: true },
    })
    m.addControl(new maplibregl.NavigationControl({ showCompass: false }), "bottom-right")
    map.current = m
    return () => m.remove()
  }, [])

  useEffect(() => {
    const m = map.current
    if (!m || !listings.length) return
    const added: maplibregl.Marker[] = []
    const bounds = new maplibregl.LngLatBounds()
    for (const l of listings) {
      const node = document.createElement("button")
      node.className = "map-pill"
      node.dataset.tour = l.tour
      node.textContent = pillLabel(l)
      node.onmouseenter = () => onHover(l.id)
      node.onmouseleave = () => onHover(null)
      node.onclick = () => onSelect(l.id)
      markers.current.set(l.id, node)
      added.push(new maplibregl.Marker({ element: node }).setLngLat([l.lng, l.lat]).addTo(m))
      bounds.extend([l.lng, l.lat])
    }
    m.fitBounds(bounds, { padding: 80, maxZoom: 15, duration: 0 })
    return () => {
      added.forEach((mk) => mk.remove())
      markers.current.clear()
    }
  }, [listings, onHover, onSelect])

  useEffect(() => {
    markers.current.forEach((node, id) => node.toggleAttribute("data-active", id === hovered))
  }, [hovered])

  return <div ref={el} className="size-full" />
}
