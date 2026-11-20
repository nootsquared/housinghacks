"use client"

import { Check, Popover, Segmented } from "@/components/ui"
import { SCENES, shortUsd } from "@/lib/format"
import type { Filters, Scene } from "@/lib/types"

export const DEFAULT_FILTERS: Filters = {
  query: "",
  minPrice: null,
  maxPrice: null,
  beds: -1,
  exactBeds: false,
  baths: 0,
  moveIn: "any",
  pets: "any",
  laundry: false,
  furnished: false,
  parking: false,
  tour: false,
  leased: false,
  maxDb: null,
}

const PRICE_STEPS = [800, 1000, 1250, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 8000, 10000]

type Props = { filters: Filters; onChange: (f: Filters) => void; scene: Scene; total: number }

export function FilterBar({ filters, onChange, scene, total }: Props) {
  const set = <K extends keyof Filters>(key: K, value: Filters[K]) => onChange({ ...filters, [key]: value })
  const priceOn = filters.minPrice != null || filters.maxPrice != null
  const bedsOn = filters.beds >= 0 || filters.baths > 0
  const moreCount = [
    filters.moveIn !== "any",
    filters.pets !== "any",
    filters.laundry,
    filters.furnished,
    filters.parking,
    filters.tour,
    filters.leased,
    filters.maxDb != null,
  ].filter(Boolean).length

  return (
    <header className="relative z-40 flex h-14 shrink-0 items-center gap-3 border-b border-line bg-bg px-4">
      <div className="mr-2 flex items-center gap-2.5">
        <Logo />
        <span className="text-[15px] font-semibold tracking-[-0.02em]">Openhouse</span>
        <span className="hidden font-mono text-[11px] text-faint lg:inline">ANN ARBOR</span>
      </div>
      <div className="relative w-[280px] shrink-0">
        <svg className="absolute top-1/2 left-2.5 -translate-y-1/2 text-faint" width="14" height="14" viewBox="0 0 16 16">
          <circle cx="7" cy="7" r="5" fill="none" stroke="currentColor" strokeWidth="1.5" />
          <path d="m11 11 3.5 3.5" stroke="currentColor" strokeWidth="1.5" />
        </svg>
        <input
          value={filters.query}
          onChange={(e) => set("query", e.target.value)}
          placeholder="Street, neighborhood, or manager"
          aria-label="Search"
          className="h-8 w-full rounded-md border border-line-strong bg-surface pr-3 pl-8 text-[13px] placeholder:text-faint focus:border-text/30 focus:outline-none"
        />
      </div>

      <Popover
        active={priceOn}
        label={
          priceOn
            ? `${filters.minPrice ? shortUsd(filters.minPrice) : "$0"} – ${filters.maxPrice ? shortUsd(filters.maxPrice) : "Any"}`
            : "Price"
        }
      >
        <p className="eyebrow mb-3">Monthly rent</p>
        <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-2">
          <PriceSelect value={filters.minPrice} placeholder="No min" onChange={(v) => set("minPrice", v)} />
          <span className="text-faint">–</span>
          <PriceSelect value={filters.maxPrice} placeholder="No max" onChange={(v) => set("maxPrice", v)} />
        </div>
        <div className="mt-3 flex flex-wrap gap-1.5">
          {[
            ["Under $1.5K", null, 1500],
            ["$1.5K–3K", 1500, 3000],
            ["$3K–6K", 3000, 6000],
            ["$6K+", 6000, null],
          ].map(([label, min, max]) => (
            <button
              key={String(label)}
              type="button"
              onClick={() => onChange({ ...filters, minPrice: min as number | null, maxPrice: max as number | null })}
              className="h-7 rounded-md border border-line-strong px-2 text-[12px] text-muted hover:bg-hover hover:text-text"
            >
              {label}
            </button>
          ))}
        </div>
      </Popover>

      <Popover active={bedsOn} label={bedsOn ? bedsSummary(filters) : "Beds & baths"} width={340}>
        <p className="eyebrow mb-2">Bedrooms</p>
        <Segmented
          value={filters.beds}
          onChange={(v) => set("beds", v)}
          options={[
            { id: -1, label: "Any" },
            { id: 0, label: "Studio" },
            ...[1, 2, 3, 4, 5, 6].map((n) => ({ id: n, label: filters.exactBeds ? `${n}` : `${n}+` })),
          ]}
        />
        <Check checked={filters.exactBeds} onChange={(v) => set("exactBeds", v)}>
          <span className="text-muted">Use exact match</span>
        </Check>
        <p className="eyebrow mt-3 mb-2">Bathrooms</p>
        <Segmented
          value={filters.baths}
          onChange={(v) => set("baths", v)}
          options={[0, 1, 1.5, 2, 3].map((n) => ({ id: n, label: n === 0 ? "Any" : `${n}+` }))}
        />
      </Popover>

      <Popover active={filters.moveIn !== "any"} label={moveInLabel(filters.moveIn)}>
        <p className="eyebrow mb-2">Move-in</p>
        <div className="grid gap-1">
          {(
            [
              ["any", "Any time"],
              ["now", "Available now"],
              ["summer", "By summer 2027"],
              ["fall", "Fall 2027 lease"],
            ] as const
          ).map(([id, label]) => (
            <button
              key={id}
              type="button"
              onClick={() => set("moveIn", id)}
              className={`flex h-8 items-center justify-between rounded-md px-2 text-left text-[13px] ${
                filters.moveIn === id ? "bg-hover text-text" : "text-muted hover:bg-hover hover:text-text"
              }`}
            >
              {label}
              {filters.moveIn === id && <span className="size-1.5 rounded-full bg-accent" />}
            </button>
          ))}
        </div>
      </Popover>

      <Popover active={moreCount > 0} label={moreCount ? `More · ${moreCount}` : "More"} width={320}>
        <p className="eyebrow mb-1">Home</p>
        <Check checked={filters.laundry} onChange={(v) => set("laundry", v)}>Laundry in unit or on site</Check>
        <Check checked={filters.furnished} onChange={(v) => set("furnished", v)}>Furnished</Check>
        <Check checked={filters.parking} onChange={(v) => set("parking", v)}>Parking on site</Check>
        <Check checked={filters.tour} onChange={(v) => set("tour", v)}>Has a 3D walkthrough</Check>
        <Check checked={filters.leased} onChange={(v) => set("leased", v)}>Show leased units</Check>
        <p className="eyebrow mt-3 mb-2">Pets</p>
        <Segmented
          value={filters.pets}
          onChange={(v) => set("pets", v)}
          options={[
            { id: "any", label: "Any" },
            { id: "cats", label: "Cats OK" },
            { id: "dogs", label: "Dogs OK" },
          ]}
        />
        <div className="mt-4 flex items-baseline justify-between">
          <p className="eyebrow">Max noise</p>
          <span className="font-mono text-[12px]">{filters.maxDb ? `${filters.maxDb} dB` : "Any"}</span>
        </div>
        <input
          type="range"
          min={48}
          max={72}
          step={1}
          value={filters.maxDb ?? 72}
          onChange={(e) => set("maxDb", Number(e.target.value) >= 72 ? null : Number(e.target.value))}
          className="mt-2 w-full"
          aria-label="Maximum modeled noise"
        />
        <p className="mt-1.5 text-[11.5px] text-faint">Uses the {SCENES.find((s) => s.id === scene)?.label.toLowerCase()} model.</p>
      </Popover>

      <div className="ml-auto flex items-center gap-3">
        <span className="hidden font-mono text-[11px] text-faint xl:inline">{total} LISTINGS · 6 MANAGERS</span>
      </div>
    </header>
  )
}

function PriceSelect({ value, placeholder, onChange }: { value: number | null; placeholder: string; onChange: (v: number | null) => void }) {
  return (
    <select
      value={value ?? ""}
      onChange={(e) => onChange(e.target.value ? Number(e.target.value) : null)}
      className="h-8 rounded-md border border-line-strong bg-bg px-2 text-[13px]"
    >
      <option value="">{placeholder}</option>
      {PRICE_STEPS.map((p) => (
        <option key={p} value={p}>
          ${p.toLocaleString()}
        </option>
      ))}
    </select>
  )
}

function bedsSummary(f: Filters) {
  const beds = f.beds < 0 ? "" : f.beds === 0 ? "Studio" : `${f.beds}${f.exactBeds ? "" : "+"} bd`
  const baths = f.baths ? `${f.baths}+ ba` : ""
  return [beds, baths].filter(Boolean).join(", ")
}

function moveInLabel(m: Filters["moveIn"]) {
  return { any: "Move-in", now: "Available now", summer: "By summer", fall: "Fall 2027" }[m]
}

function Logo() {
  return (
    <svg width="20" height="20" viewBox="0 0 20 20" aria-hidden>
      <rect x="0.5" y="0.5" width="19" height="19" rx="5" fill="#ededed" />
      <path d="M5 14V9.2L10 5.5l5 3.7V14h-3.4v-3.2H8.4V14z" fill="#0a0a0b" />
    </svg>
  )
}
