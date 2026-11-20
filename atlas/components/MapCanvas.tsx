"use client"

import { useEffect, useRef, useState } from "react"
import * as maplibregl from "maplibre-gl"
import "maplibre-gl/dist/maplibre-gl.css"
import { shortUsd } from "@/lib/format"
import type { Atlas, Bounds, Listing, Scene } from "@/lib/types"

type Props = {
  atlas: Atlas
  listings: Listing[]
  selectedId: string | null
  hoveredId: string | null
  noise: Scene | null
  tilt: boolean
  onSelect: (id: string | null) => void
  onHover: (id: string | null) => void
  onBounds: (bounds: Bounds) => void
}

const SATELLITE = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

const SOURCE_COLORS: Record<string, string> = {
  bars: "#f08a4b",
  greek: "#d36fd8",
  stadium: "#e5625c",
  helicopter: "#69b7ff",
  crossing: "#e8d36a",
  fire: "#ff7a6b",
}

export const groupKey = (l: Listing) => (l.building != null ? `b${l.building}` : `p${l.lat.toFixed(5)},${l.lng.toFixed(5)}`)

function style(): maplibregl.StyleSpecification {
  return {
    version: 8,
    glyphs: "https://tiles.openfreemap.org/fonts/{fontstack}/{range}.pbf",
    sources: {
      satellite: {
        type: "raster",
        tiles: [SATELLITE],
        tileSize: 256,
        maxzoom: 19,
        attribution: "Imagery © Esri, Maxar, Earthstar Geographics",
      },
      omt: { type: "vector", url: "https://tiles.openfreemap.org/planet" },
    },
    layers: [
      { id: "bg", type: "background", paint: { "background-color": "#0a0a0b" } },
      {
        id: "satellite",
        type: "raster",
        source: "satellite",
        paint: { "raster-saturation": -0.25, "raster-brightness-max": 0.78, "raster-contrast": 0.08 },
      },
      {
        id: "roads-major",
        type: "line",
        source: "omt",
        "source-layer": "transportation",
        filter: ["in", ["get", "class"], ["literal", ["motorway", "trunk", "primary", "secondary"]]],
        paint: {
          "line-color": "rgba(255,255,255,0.16)",
          "line-width": ["interpolate", ["linear"], ["zoom"], 12, 0.6, 17, 2],
        },
      },
      {
        id: "road-labels",
        type: "symbol",
        source: "omt",
        "source-layer": "transportation_name",
        minzoom: 14,
        layout: {
          "symbol-placement": "line",
          "text-field": ["coalesce", ["get", "name:en"], ["get", "name"]],
          "text-font": ["Noto Sans Regular"],
          "text-size": ["interpolate", ["linear"], ["zoom"], 14, 10, 18, 13],
          "text-letter-spacing": 0.02,
        },
        paint: { "text-color": "rgba(255,255,255,0.82)", "text-halo-color": "rgba(0,0,0,0.75)", "text-halo-width": 1.2 },
      },
      {
        id: "place-labels",
        type: "symbol",
        source: "omt",
        "source-layer": "place",
        filter: ["in", ["get", "class"], ["literal", ["suburb", "neighbourhood", "quarter"]]],
        maxzoom: 15,
        layout: {
          "text-field": ["coalesce", ["get", "name:en"], ["get", "name"]],
          "text-font": ["Noto Sans Bold"],
          "text-size": 11,
          "text-transform": "uppercase",
          "text-letter-spacing": 0.12,
        },
        paint: { "text-color": "rgba(255,255,255,0.6)", "text-halo-color": "rgba(0,0,0,0.7)", "text-halo-width": 1 },
      },
    ],
  }
}

export default function MapCanvas(props: Props) {
  const ref = useRef<HTMLDivElement>(null)
  const mapRef = useRef<maplibregl.Map | null>(null)
  const live = useRef(props)
  const pins = useRef(new Map<string, { node: HTMLButtonElement; marker: maplibregl.Marker; ids: string[] }>())
  const featureState = useRef<{ sel: number | null; hov: number | null }>({ sel: null, hov: null })
  const [loaded, setLoaded] = useState(0)
  live.current = props

  useEffect(() => {
    if (!ref.current) return
    maplibregl.setWorkerUrl("/maplibre-gl-worker.mjs")
    const map = new maplibregl.Map({
      container: ref.current,
      style: style(),
      center: [-83.7395, 42.2765],
      zoom: 14.6,
      pitch: 55,
      bearing: -18,
      maxPitch: 75,
      attributionControl: { compact: true },
    })
    map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), "bottom-right")
    const emitBounds = () => {
      const b = map.getBounds()
      live.current.onBounds({ west: b.getWest(), south: b.getSouth(), east: b.getEast(), north: b.getNorth() })
    }
    map.on("moveend", emitBounds)
    map.on("move", () => declutter(map))
    map.on("load", () => {
      const { atlas } = live.current
      map.addSource("noise", { type: "image", url: "/noise/day.png", coordinates: atlas.noise.bounds })
      map.addLayer(
        {
          id: "noise",
          type: "raster",
          source: "noise",
          layout: { visibility: "none" },
          paint: { "raster-opacity": 0.82, "raster-resampling": "linear", "raster-fade-duration": 200 },
        },
        "roads-major",
      )
      map.addSource("homes", { type: "geojson", data: empty(), promoteId: "id" })
      map.addLayer({
        id: "homes-base",
        type: "line",
        source: "homes",
        paint: {
          "line-color": [
            "case",
            ["boolean", ["feature-state", "selected"], false],
            "#f4b860",
            "rgba(255,255,255,0.9)",
          ],
          "line-width": 1.4,
        },
      })
      map.addLayer({
        id: "homes",
        type: "fill-extrusion",
        source: "homes",
        paint: {
          "fill-extrusion-color": [
            "case",
            ["boolean", ["feature-state", "selected"], false],
            "#f4b860",
            ["boolean", ["feature-state", "hover"], false],
            "#ffffff",
            "#e9e6df",
          ],
          "fill-extrusion-height": ["get", "height"],
          "fill-extrusion-opacity": 0.9,
          "fill-extrusion-vertical-gradient": true,
        },
      })
      map.addSource("sources", {
        type: "geojson",
        data: {
          type: "FeatureCollection",
          features: atlas.noise.sources.map((s) => ({
            type: "Feature",
            properties: { label: s.label, group: s.group },
            geometry: { type: "Point", coordinates: [s.lng, s.lat] },
          })),
        },
      })
      map.addLayer({
        id: "source-dots",
        type: "circle",
        source: "sources",
        layout: { visibility: "none" },
        paint: {
          "circle-radius": ["interpolate", ["linear"], ["zoom"], 12, 2, 16, 4.5],
          "circle-color": ["match", ["get", "group"], ...Object.entries(SOURCE_COLORS).flat(), "#ffffff"] as unknown as maplibregl.ExpressionSpecification,
          "circle-stroke-color": "#0a0a0b",
          "circle-stroke-width": 1.2,
        },
      })
      map.addLayer({
        id: "source-labels",
        type: "symbol",
        source: "sources",
        minzoom: 15.4,
        filter: ["!=", ["get", "group"], "crossing"],
        layout: {
          visibility: "none",
          "text-field": ["get", "label"],
          "text-font": ["Noto Sans Regular"],
          "text-size": 11,
          "text-offset": [0, 1.1],
          "text-anchor": "top",
          "text-optional": true,
        },
        paint: { "text-color": "#ffffff", "text-halo-color": "rgba(0,0,0,0.85)", "text-halo-width": 1.2 },
      })
      map.on("click", "homes", (event) => {
        const id = event.features?.[0]?.properties?.first
        if (typeof id === "string") live.current.onSelect(id)
      })
      map.on("mousemove", "homes", (event) => {
        map.getCanvas().style.cursor = "pointer"
        const id = event.features?.[0]?.properties?.first
        if (typeof id === "string" && live.current.hoveredId !== id) live.current.onHover(id)
      })
      map.on("mouseleave", "homes", () => {
        map.getCanvas().style.cursor = ""
        live.current.onHover(null)
      })
      setLoaded((n) => n + 1)
      emitBounds()
    })
    mapRef.current = map
    const pinMap = pins.current
    return () => {
      pinMap.forEach((p) => p.marker.remove())
      pinMap.clear()
      map.remove()
      mapRef.current = null
    }
  }, [])

  useEffect(() => {
    const map = mapRef.current
    if (!loaded || !map?.getSource("homes")) return
    const groups = new Map<string, Listing[]>()
    for (const l of props.listings) {
      const key = groupKey(l)
      const list = groups.get(key) ?? []
      list.push(l)
      groups.set(key, list)
    }
    const shapes = new Map(props.atlas.buildings.map((b) => [b.id, b]))
    const features: GeoJSON.Feature[] = []
    for (const [, units] of groups) {
      const first = [...units].sort((a, b) => a.rent - b.rent)[0]
      const shape = first.building != null ? shapes.get(first.building) : undefined
      if (!shape) continue
      const fallback = units.length > 6 ? 15 : units.length > 1 ? 10 : 8
      features.push({
        type: "Feature",
        id: shape.id,
        properties: { id: shape.id, first: first.id, height: shape.height ?? fallback },
        geometry: { type: "Polygon", coordinates: [shape.ring] },
      })
    }
    ;(map.getSource("homes") as maplibregl.GeoJSONSource).setData({ type: "FeatureCollection", features })

    const keep = new Set(groups.keys())
    pins.current.forEach((pin, key) => {
      if (!keep.has(key)) {
        pin.marker.remove()
        pins.current.delete(key)
      }
    })
    for (const [key, units] of groups) {
      const sorted = [...units].sort((a, b) => a.rent - b.rent)
      const label =
        units.length === 1
          ? shortUsd(sorted[0].rent)
          : `${shortUsd(sorted[0].rent)}+<small>${units.length}</small>`
      const existing = pins.current.get(key)
      if (existing) {
        existing.node.innerHTML = label
        existing.ids = sorted.map((u) => u.id)
        continue
      }
      const node = document.createElement("button")
      node.type = "button"
      node.className = "pin"
      node.innerHTML = label
      const entry = { node, ids: sorted.map((u) => u.id), marker: null as unknown as maplibregl.Marker }
      node.onmouseenter = () => live.current.onHover(entry.ids[0])
      node.onmouseleave = () => live.current.onHover(null)
      node.onclick = (event) => {
        event.stopPropagation()
        live.current.onSelect(entry.ids[0])
      }
      entry.marker = new maplibregl.Marker({ element: node, anchor: "bottom", offset: [0, -6] })
        .setLngLat([sorted[0].lng, sorted[0].lat])
        .addTo(map)
      pins.current.set(key, entry)
    }
    paintActive(map)
    declutter(map)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loaded, props.listings])

  useEffect(() => {
    const map = mapRef.current
    if (!loaded || !map?.getSource("homes")) return
    paintActive(map)
    declutter(map)
     
  }, [loaded, props.selectedId, props.hoveredId])

  useEffect(() => {
    const map = mapRef.current
    if (!loaded || !map?.getSource("homes")) return
    const on = props.noise != null
    for (const id of ["noise", "source-dots", "source-labels"]) map.setLayoutProperty(id, "visibility", on ? "visible" : "none")
    if (props.noise) (map.getSource("noise") as maplibregl.ImageSource).updateImage({ url: `/noise/${props.noise}.png` })
  }, [loaded, props.noise])

  useEffect(() => {
    const map = mapRef.current
    if (!loaded || !map?.getSource("homes")) return
    map.easeTo({ pitch: props.tilt ? 55 : 0, bearing: props.tilt ? map.getBearing() || -18 : 0, duration: 700 })
  }, [loaded, props.tilt])

  useEffect(() => {
    const map = mapRef.current
    if (!loaded || !map?.getSource("homes") || !props.selectedId) return
    const l = props.atlas.listings.find((item) => item.id === props.selectedId)
    if (!l) return
    map.easeTo({ center: [l.lng, l.lat], zoom: Math.max(map.getZoom(), 16.4), duration: 800, essential: true })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loaded, props.selectedId])

  function paintActive(map: maplibregl.Map) {
    const { selectedId, hoveredId, atlas } = live.current
    const buildingOf = (id: string | null) => atlas.listings.find((l) => l.id === id)?.building ?? null
    const sel = buildingOf(selectedId)
    const hov = buildingOf(hoveredId)
    const prev = featureState.current
    for (const id of new Set([prev.sel, prev.hov])) {
      if (id != null) map.removeFeatureState({ source: "homes", id })
    }
    if (sel != null) map.setFeatureState({ source: "homes", id: sel }, { selected: true })
    if (hov != null && hov !== sel) map.setFeatureState({ source: "homes", id: hov }, { hover: true })
    featureState.current = { sel, hov }
    pins.current.forEach((pin) => {
      const isSel = !!selectedId && pin.ids.includes(selectedId)
      const isHov = !!hoveredId && pin.ids.includes(hoveredId)
      pin.node.toggleAttribute("data-active", isSel)
      pin.node.toggleAttribute("data-hover", isHov)
      pin.marker.getElement().style.zIndex = isSel ? "5" : isHov ? "4" : "1"
    })
  }

  function declutter(map: maplibregl.Map) {
    const { selectedId, hoveredId } = live.current
    const entries = [...pins.current.values()]
    const priority = (ids: string[]) => (selectedId && ids.includes(selectedId) ? 2 : hoveredId && ids.includes(hoveredId) ? 1 : 0)
    entries.sort((a, b) => priority(b.ids) - priority(a.ids) || b.ids.length - a.ids.length)
    const placed: { x: number; y: number }[] = []
    for (const pin of entries) {
      const p = map.project(pin.marker.getLngLat())
      const forced = priority(pin.ids) > 0
      const clear = placed.every((o) => Math.abs(o.x - p.x) > 54 || Math.abs(o.y - p.y) > 24)
      const show = forced || clear
      if (show) placed.push({ x: p.x, y: p.y })
      pin.node.style.visibility = show ? "visible" : "hidden"
    }
  }

  return <div ref={ref} className="size-full bg-bg" data-loaded={loaded ? "yes" : "no"} />
}

function empty(): GeoJSON.FeatureCollection {
  return { type: "FeatureCollection", features: [] }
}
