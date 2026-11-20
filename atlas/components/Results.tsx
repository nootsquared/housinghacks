"use client"

import { useEffect, useState } from "react"
import { Tag } from "@/components/ui"
import { cn } from "@/lib/cn"
import { bathsLabel, bedsLabel, loudness, proxy, shortAvailability, sqftLabel, usd } from "@/lib/format"
import type { Listing, Scene, Sort } from "@/lib/types"

const PAGE = 40

type Props = {
  listings: Listing[]
  total: number
  sort: Sort
  scene: Scene
  noiseOn: boolean
  hoveredId: string | null
  selectedId: string | null
  onSort: (s: Sort) => void
  onSelect: (id: string) => void
  onHover: (id: string | null) => void
  onClear: () => void
}

export function Results(props: Props) {
  const [limit, setLimit] = useState(PAGE)
  useEffect(() => setLimit(PAGE), [props.listings])

  return (
    <aside className="flex w-[min(720px,48vw)] shrink-0 flex-col border-l border-line bg-bg">
      <div className="flex items-end justify-between gap-4 px-5 pt-5 pb-4">
        <div>
          <h1 className="text-[20px] font-semibold tracking-[-0.025em]">Ann Arbor rentals</h1>
          <p className="mt-1 text-[13px] text-muted">
            <span className="tabular text-text">{props.listings.length}</span> in this view
            <span className="text-faint"> · {props.total} match your filters</span>
          </p>
        </div>
        <label className="flex items-center gap-2 text-[12px] text-muted">
          Sort
          <select
            value={props.sort}
            onChange={(e) => props.onSort(e.target.value as Sort)}
            className="h-8 rounded-md border border-line-strong bg-surface px-2 text-[13px] text-text"
          >
            <option value="price-asc">Price, low to high</option>
            <option value="price-desc">Price, high to low</option>
            <option value="beds">Most bedrooms</option>
            <option value="sqft">Largest</option>
            <option value="quiet">Quietest</option>
          </select>
        </label>
      </div>
      <div className="scroll-thin min-h-0 flex-1 overflow-y-auto px-5 pb-6">
        {props.listings.length === 0 ? (
          <div className="rounded-xl border border-dashed border-line-strong px-6 py-12 text-center">
            <p className="text-[14px]">Nothing here matches.</p>
            <p className="mt-1 text-[13px] text-muted">Zoom out on the map or loosen a filter.</p>
            <button
              type="button"
              onClick={props.onClear}
              className="mt-4 h-8 rounded-md bg-text px-3 text-[13px] font-medium text-bg"
            >
              Clear filters
            </button>
          </div>
        ) : (
          <>
            <ul className="grid grid-cols-1 gap-4 lg:grid-cols-2">
              {props.listings.slice(0, limit).map((l) => (
                <li key={l.id}>
                  <Card
                    listing={l}
                    scene={props.scene}
                    noiseOn={props.noiseOn}
                    hovered={props.hoveredId === l.id}
                    onSelect={props.onSelect}
                    onHover={props.onHover}
                  />
                </li>
              ))}
            </ul>
            {limit < props.listings.length && (
              <button
                type="button"
                onClick={() => setLimit(limit + PAGE)}
                className="mt-5 h-9 w-full rounded-md border border-line-strong text-[13px] text-muted hover:bg-hover hover:text-text"
              >
                Show {Math.min(PAGE, props.listings.length - limit)} more
              </button>
            )}
          </>
        )}
        <p className="mt-6 text-[11.5px] leading-relaxed text-faint">
          Pulled from the public listing pages of Varsity Management, Campus Realty, Old Town Realty, Barnes &amp; Barnes, JMZ
          and Howard Hanna. Building outlines are from OpenStreetMap.
        </p>
      </div>
    </aside>
  )
}

function Card({
  listing: l,
  scene,
  noiseOn,
  hovered,
  onSelect,
  onHover,
}: {
  listing: Listing
  scene: Scene
  noiseOn: boolean
  hovered: boolean
  onSelect: (id: string) => void
  onHover: (id: string | null) => void
}) {
  const level = loudness(l.noise[scene])
  return (
    <button
      type="button"
      onClick={() => onSelect(l.id)}
      onMouseEnter={() => onHover(l.id)}
      onMouseLeave={() => onHover(null)}
      className={cn(
        "group block w-full overflow-hidden rounded-xl border bg-surface text-left transition-colors",
        hovered ? "border-text/30" : "border-line hover:border-line-strong",
      )}
    >
      <div className="relative aspect-[16/10] overflow-hidden bg-raised">
        {l.photos[0] && (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={proxy(l.photos[0])}
            alt=""
            loading="lazy"
            className="size-full object-cover transition-transform duration-500 group-hover:scale-[1.03]"
          />
        )}
        <div className="absolute inset-x-0 top-0 flex justify-between p-2.5">
          <div className="flex gap-1.5">
            {l.tour && (
              <span className="rounded-[5px] bg-accent px-1.5 py-0.5 font-mono text-[10.5px] font-medium text-bg">3D TOUR</span>
            )}
            {l.status === "now" && (
              <span className="rounded-[5px] bg-bg/80 px-1.5 py-0.5 font-mono text-[10.5px] text-emerald-300 backdrop-blur">
                AVAILABLE NOW
              </span>
            )}
            {l.status === "unavailable" && (
              <span className="rounded-[5px] bg-bg/80 px-1.5 py-0.5 font-mono text-[10.5px] text-muted backdrop-blur">LEASED</span>
            )}
          </div>
          <span className="rounded-[5px] bg-bg/70 px-1.5 py-0.5 font-mono text-[10.5px] text-text/80 backdrop-blur">
            {l.photos.length}
          </span>
        </div>
      </div>
      <div className="px-3.5 pt-3 pb-3.5">
        <div className="flex items-baseline justify-between gap-2">
          <p className="text-[19px] font-semibold tracking-[-0.03em] tabular">
            {usd(l.rent)}
            <span className="text-[13px] font-normal tracking-normal text-muted">/mo</span>
          </p>
          <span className="flex items-center gap-1.5 font-mono text-[11px] text-muted" title={`${level.word}, modeled`}>
            <span className="size-1.5 rounded-full" style={{ background: level.tone }} />
            {Math.round(l.noise[scene])} dB
          </span>
        </div>
        <p className="mt-1 flex items-center gap-2 text-[13px] text-text/90 tabular">
          <span>{bedsLabel(l.beds)}</span>
          <span className="h-3 w-px bg-line-strong" />
          <span>{bathsLabel(l.baths)}</span>
          <span className="h-3 w-px bg-line-strong" />
          <span className={l.sqft ? "" : "text-faint"}>{sqftLabel(l.sqft)}</span>
        </p>
        <p className="mt-1.5 truncate text-[13px] text-muted">
          {l.street}
          {l.unit ? `, Unit ${l.unit}` : ""}
        </p>
        <div className="mt-3 flex items-center justify-between gap-2 border-t border-line pt-2.5">
          <span className="truncate font-mono text-[10.5px] tracking-wide text-faint uppercase">{l.manager}</span>
          <div className="flex shrink-0 gap-1.5">
            {noiseOn && <Tag>{level.word}</Tag>}
            <Tag tone={l.status === "now" ? "good" : "muted"}>{shortAvailability(l)}</Tag>
          </div>
        </div>
      </div>
    </button>
  )
}
