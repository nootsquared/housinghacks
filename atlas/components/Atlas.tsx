"use client"

import dynamic from "next/dynamic"
import { useCallback, useEffect, useMemo, useState } from "react"
import { Detail } from "@/components/Detail"
import { FilterBar, DEFAULT_FILTERS } from "@/components/FilterBar"
import { DEMO_ID } from "@/components/MapCanvas"
import { MapControls } from "@/components/MapControls"
import { Results } from "@/components/Results"
import { has, petsAllowed } from "@/lib/format"
import type { Atlas as AtlasData, Bounds, Filters, Listing, Scene, Sort } from "@/lib/types"

const MapCanvas = dynamic(() => import("@/components/MapCanvas"), { ssr: false })

export default function Atlas() {
  const [atlas, setAtlas] = useState<AtlasData | null>(null)
  const [filters, setFilters] = useState<Filters>(DEFAULT_FILTERS)
  const [sort, setSort] = useState<Sort>("price-asc")
  const [bounds, setBounds] = useState<Bounds | null>(null)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [hoveredId, setHoveredId] = useState<string | null>(null)
  const [noise, setNoise] = useState<Scene | null>(null)
  const [lastScene, setLastScene] = useState<Scene>("day")
  const [tilt, setTilt] = useState(true)
  const [me, setMe] = useState<{ lat: number; lng: number } | null>(null)

  useEffect(() => {
    fetch("/atlas.json")
      .then((r) => r.json())
      .then(setAtlas)
  }, [])

  const scene = noise ?? lastScene

  const filtered = useMemo(() => {
    if (!atlas) return []
    const q = filters.query.trim().toLowerCase()
    return atlas.listings.filter((l) => {
      if (l.id === DEMO_ID) return true
      if (q && !`${l.address} ${l.area} ${l.manager} ${l.title ?? ""}`.toLowerCase().includes(q)) return false
      if (filters.minPrice != null && l.rent < filters.minPrice) return false
      if (filters.maxPrice != null && l.rent > filters.maxPrice) return false
      if (filters.beds >= 0) {
        if (filters.exactBeds ? l.beds !== filters.beds : l.beds < filters.beds) return false
      }
      if (filters.baths > 0 && (l.baths ?? 0) < filters.baths) return false
      if (filters.moveIn === "now" && l.status !== "now") return false
      if (filters.moveIn === "summer" && !(l.status === "now" || (l.availableOn && l.availableOn <= "2027-06-30"))) return false
      if (filters.moveIn === "fall" && !(l.availableOn && l.availableOn >= "2027-07-01" && l.availableOn <= "2027-09-30")) return false
      if (filters.pets !== "any" && !petsAllowed(l, filters.pets)) return false
      if (filters.laundry && !has(l, /in-unit laundry|onsite laundry|washer/i)) return false
      if (filters.furnished && !has(l, /^furnished/i)) return false
      if (filters.parking && !has(l, /onsite parking|covered parking|garage|driveway/i)) return false
      if (filters.tour && !l.tour) return false
      if (!filters.leased && l.status === "unavailable") return false
      if (filters.maxDb != null && l.noise[scene] > filters.maxDb) return false
      return true
    })
  }, [atlas, filters, scene])

  const inView = useMemo(() => {
    const list = bounds
      ? filtered.filter((l) => l.lng >= bounds.west && l.lng <= bounds.east && l.lat >= bounds.south && l.lat <= bounds.north)
      : filtered
    const order: Record<Sort, (a: Listing, b: Listing) => number> = {
      "price-asc": (a, b) => a.rent - b.rent,
      "price-desc": (a, b) => b.rent - a.rent,
      beds: (a, b) => b.beds - a.beds || a.rent - b.rent,
      sqft: (a, b) => (b.sqft ?? 0) - (a.sqft ?? 0),
      quiet: (a, b) => a.noise[scene] - b.noise[scene],
    }
    return [...list].sort((a, b) => Number(!!b.tour) - Number(!!a.tour) || order[sort](a, b))
  }, [filtered, bounds, sort, scene])

  const selected = atlas?.listings.find((l) => l.id === selectedId) ?? null
  const close = useCallback(() => setSelectedId(null), [])

  function toggleNoise(next: Scene | null) {
    setNoise(next)
    if (next) setLastScene(next)
  }

  return (
    <div className="flex h-dvh flex-col bg-bg text-text">
      <FilterBar filters={filters} onChange={setFilters} scene={scene} total={atlas?.listings.length ?? 0} />
      <div className="flex min-h-0 flex-1">
        <div className="relative min-w-0 flex-1">
          {atlas && (
            <MapCanvas
              atlas={atlas}
              listings={filtered}
              selectedId={selectedId}
              hoveredId={hoveredId}
              noise={noise}
              tilt={tilt}
              onSelect={setSelectedId}
              onHover={setHoveredId}
              onBounds={setBounds}
              onLocate={setMe}
            />
          )}
          <MapControls noise={noise} onNoise={toggleNoise} tilt={tilt} onTilt={setTilt} />
          {!atlas && (
            <div className="absolute inset-0 grid place-items-center">
              <p className="eyebrow">Loading the map</p>
            </div>
          )}
        </div>
        <Results
          listings={inView}
          total={filtered.length}
          sort={sort}
          scene={scene}
          noiseOn={noise != null}
          hoveredId={hoveredId}
          selectedId={selectedId}
          onSort={setSort}
          onSelect={setSelectedId}
          onHover={setHoveredId}
          onClear={() => setFilters(DEFAULT_FILTERS)}
        />
      </div>
      {selected && atlas && (
        <Detail
          key={selected.id}
          listing={selected}
          atlas={atlas}
          scene={scene}
          me={me}
          onScene={(s) => toggleNoise(s)}
          onClose={close}
          onSelect={setSelectedId}
          onShowNoise={() => {
            toggleNoise(scene)
            close()
          }}
        />
      )}
    </div>
  )
}
