"use client"

import { useEffect, useMemo, useState } from "react"
import { groupKey } from "@/components/MapCanvas"
import { Section, Segmented, Tag } from "@/components/ui"
import { cn } from "@/lib/cn"
import { availability, bathsLabel, bedsLabel, loudness, petsAllowed, proxy, SCENES, sqftLabel, usd } from "@/lib/format"
import { meters, walkMinutes } from "@/lib/geo"
import { costs, homeFacts, management, market, community } from "@/lib/estimates"
import { DESTINATIONS, STOPS } from "@/lib/places"
import type { Atlas, Listing, NoiseSource, Scene } from "@/lib/types"

type Props = {
  listing: Listing
  atlas: Atlas
  scene: Scene
  me: { lat: number; lng: number } | null
  onScene: (s: Scene) => void
  onClose: () => void
  onSelect: (id: string) => void
  onTour: () => void
  onShowNoise: () => void
}

export function Detail({ listing: l, atlas, scene, me, onScene, onClose, onSelect, onTour, onShowNoise }: Props) {
  const [photo, setPhoto] = useState<number | null>(null)

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return
      if (photo != null) setPhoto(null)
      else onClose()
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [photo, onClose])

  const siblings = useMemo(
    () => atlas.listings.filter((o) => o.id !== l.id && groupKey(o) === groupKey(l)).sort((a, b) => a.rent - b.rent),
    [atlas, l],
  )

  const shots = l.photos.slice(0, 5)

  return (
    <div className="fade-in fixed inset-0 z-50 flex justify-center bg-black/70 backdrop-blur-sm" onMouseDown={onClose}>
      <div
        className="rise-in scroll-thin relative my-4 w-full max-w-[1120px] overflow-y-auto rounded-lg border border-line-strong bg-bg shadow-[0_40px_120px_rgba(0,0,0,0.7)]"
        onMouseDown={(e) => e.stopPropagation()}
      >
        <header className="sticky top-0 z-20 flex h-12 items-center justify-between gap-4 border-b border-line bg-bg/90 px-5 backdrop-blur-md">
          <button type="button" onClick={onClose} className="flex items-center gap-1.5 text-[13px] text-muted hover:text-text">
            <svg width="14" height="14" viewBox="0 0 16 16" aria-hidden>
              <path d="M10 3 5 8l5 5" fill="none" stroke="currentColor" strokeWidth="1.5" />
            </svg>
            Back to map
          </button>
          <p className="truncate text-[13px] text-muted">{l.address}</p>
          <a
            href={l.source}
            target="_blank"
            rel="noreferrer"
            className="flex items-center gap-1 text-[13px] text-muted hover:text-text"
          >
            Original listing
            <svg width="12" height="12" viewBox="0 0 16 16" aria-hidden>
              <path d="M6 3h7v7M13 3 4 12" fill="none" stroke="currentColor" strokeWidth="1.5" />
            </svg>
          </a>
        </header>

        <div className="grid h-[400px] grid-cols-4 grid-rows-2 gap-1.5 p-1.5">
          {shots.map((src, i) => (
            <button
              key={src}
              type="button"
              onClick={() => setPhoto(i)}
              className={cn(
                "group relative overflow-hidden bg-raised",
                i === 0 ? "col-span-2 row-span-2 rounded-l-md" : "",
                i === 2 && "rounded-tr-md",
                i === 4 && "rounded-br-md",
                shots.length === 1 && "col-span-4 rounded-md",
              )}
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={proxy(src, i === 0 ? "large" : "medium")} alt="" className="size-full object-cover transition-opacity group-hover:opacity-85" />
              {i === shots.length - 1 && l.photos.length > shots.length && (
                <span className="absolute right-3 bottom-3 rounded-md bg-bg/85 px-2.5 py-1.5 text-[12px] font-medium backdrop-blur">
                  See all {l.photos.length} photos
                </span>
              )}
            </button>
          ))}
        </div>

        <div className="grid grid-cols-1 gap-6 px-6 pt-6 pb-10 lg:grid-cols-[minmax(0,1fr)_320px]">
          <div className="min-w-0 space-y-5">
            <div>
              <div className="flex flex-wrap items-center gap-1.5">
                {l.tour && <Tag tone="accent">3D walkthrough</Tag>}
                <Tag tone={l.status === "now" ? "good" : "muted"}>{availability(l)}</Tag>
                <Tag>{l.area}</Tag>
              </div>
              <div className="mt-3 flex flex-wrap items-end justify-between gap-x-8 gap-y-3">
                <div>
                  <p className="text-[36px] leading-none font-semibold tracking-[-0.04em] tabular">
                    {usd(l.rent)}
                    <span className="text-[16px] font-normal tracking-normal text-muted">/mo</span>
                  </p>
                  <p className="mt-2 text-[15px] text-text/90">
                    {l.street}
                    {l.unit ? `, Unit ${l.unit}` : ""}
                    <span className="text-muted">, Ann Arbor, MI {l.zip}</span>
                  </p>
                </div>
                <dl className="flex gap-6">
                  <Big value={l.beds === 0 ? "Studio" : String(l.beds)} label={l.beds === 0 ? "" : l.beds === 1 ? "bed" : "beds"} />
                  <Big value={l.baths != null ? String(l.baths) : "—"} label={l.baths === 1 ? "bath" : "baths"} />
                  <Big value={l.sqft ? l.sqft.toLocaleString() : "—"} label="sqft" />
                </dl>
              </div>
              {l.title && <p className="mt-4 text-[14px] text-muted">{l.title}</p>}
            </div>

            <Facts listing={l} />
            <NoiseCard listing={l} scene={scene} onScene={onScene} onShow={onShowNoise} />
            <Costs listing={l} />
            <Getting listing={l} me={me} />
            <Market listing={l} />
            <People listing={l} />

            {l.amenities.length > 0 && (
              <Section title="Amenities" aside={<span className="font-mono text-[11px] text-faint">{l.amenities.length}</span>}>
                <ul className="grid grid-cols-2 gap-x-6 gap-y-2 sm:grid-cols-3">
                  {[...l.amenities, ...l.appliances].filter((a) => a.length < 48).map((a) => (
                    <li key={a} className="flex items-center gap-2 text-[13px] text-text/90">
                      <span className="size-1 rounded-full bg-faint" />
                      {a}
                    </li>
                  ))}
                </ul>
              </Section>
            )}

            {l.description && (
              <Section title="From the listing">
                <p className="text-[14px] leading-relaxed whitespace-pre-line text-text/85">{tidy(l.description)}</p>
              </Section>
            )}

            {l.floorplan && (
              <Section title="Floor plan">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={l.floorplan} alt="Floor plan" className="w-full rounded-lg bg-white" />
              </Section>
            )}

            {siblings.length > 0 && (
              <Section title="Other units in this building" aside={<span className="font-mono text-[11px] text-faint">{siblings.length}</span>}>
                <ul className="-my-2 divide-y divide-line">
                  {siblings.slice(0, 12).map((s) => (
                    <li key={s.id}>
                      <button
                        type="button"
                        onClick={() => onSelect(s.id)}
                        className="grid w-full grid-cols-[1fr_auto_auto] items-center gap-4 py-2.5 text-left text-[13px] hover:text-text"
                      >
                        <span className="truncate text-muted">{s.unit ? `Unit ${s.unit}` : s.street}</span>
                        <span className="text-muted tabular">
                          {bedsLabel(s.beds)} · {bathsLabel(s.baths)} · {sqftLabel(s.sqft)}
                        </span>
                        <span className="w-20 text-right font-medium tabular">{usd(s.rent)}</span>
                      </button>
                    </li>
                  ))}
                </ul>
              </Section>
            )}
          </div>

          <aside className="lg:sticky lg:top-16 lg:self-start">
            <div className="rounded-md border border-line-strong bg-surface p-5">
              <p className="eyebrow">Monthly rent</p>
              <p className="mt-1 text-[26px] font-semibold tracking-[-0.03em] tabular">{usd(l.rent)}</p>
              <dl className="mt-4 space-y-2 border-t border-line pt-4 text-[13px]">
                <Row label="Move-in" value={availability(l).replace("Available ", "")} />
                <Row label="Deposit" value={l.deposit ? usd(l.deposit) : "Ask"} />
                <Row label="Application fee" value={l.applicationFee ? usd(l.applicationFee) : "Ask"} />
                {l.sqft && <Row label="Per sqft" value={`$${(l.rent / l.sqft).toFixed(2)}`} />}
              </dl>
              {l.tour && (
                <button
                  type="button"
                  onClick={onTour}
                  className="mt-5 h-10 w-full rounded-lg bg-accent text-[14px] font-medium text-bg transition-opacity hover:opacity-90"
                >
                  Walk through in 3D
                </button>
              )}
              <a
                href={l.source}
                target="_blank"
                rel="noreferrer"
                className={cn(
                  "mt-2 flex h-10 w-full items-center justify-center rounded-lg text-[14px] font-medium transition-colors",
                  l.tour ? "border border-line-strong hover:bg-hover" : "mt-5 bg-text text-bg hover:opacity-90",
                )}
              >
                Contact {l.manager}
              </a>
            </div>
            <p className="mt-3 px-1 text-[11.5px] leading-relaxed text-faint">
              Listed by {l.manager}. Pulled from their public listing page on {l.accessed}. Noise is a model built from
              OpenStreetMap, not a reading taken at the address.
            </p>
          </aside>
        </div>
      </div>

      {photo != null && <Lightbox photos={l.photos} index={photo} onIndex={setPhoto} onClose={() => setPhoto(null)} />}
    </div>
  )
}

function Big({ value, label }: { value: string; label: string }) {
  return (
    <div>
      <dt className="text-[22px] leading-none font-semibold tracking-[-0.03em] tabular">{value}</dt>
      <dd className="mt-1 text-[12px] text-muted">{label}</dd>
    </div>
  )
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <dt className="text-muted">{label}</dt>
      <dd className="tabular">{value}</dd>
    </div>
  )
}

function Facts({ listing: l }: { listing: Listing }) {
  const cats = petsAllowed(l, "cats")
  const dogs = petsAllowed(l, "dogs")
  const pets = l.pets.length === 0 ? "Cats only" : cats && dogs ? "Cats and dogs" : cats ? "Cats only" : dogs ? "Dogs only" : "No pets"
  const f = homeFacts(l)
  const facts = [
    ["Type", l.unit ? "Apartment" : l.beds >= 4 ? "House" : "Unit"],
    ["Laundry", f.laundry],
    ["Parking", f.parking],
    ["Air conditioning", f.ac],
    ["Pets", pets],
    ["Furnished", f.furnished],
    ["Kitchen", f.kitchen],
    ["Bathrooms", f.bath],
    ["Appliances", f.appliances],
  ]
  return (
    <div className="grid grid-cols-2 overflow-hidden rounded-md border border-line bg-line sm:grid-cols-3" style={{ gap: 1 }}>
      {facts.map(([label, value]) => (
        <div key={label} className="bg-surface px-4 py-3">
          <p className="eyebrow">{label}</p>
          <p className="mt-1 text-[13.5px] text-text">{value}</p>
        </div>
      ))}
    </div>
  )
}

const GROUP_LABEL: Record<NoiseSource["group"], string> = {
  traffic: "Traffic",
  rail: "Rail",
  crossing: "Train horn",
  bars: "Bar",
  greek: "Greek house",
  stadium: "Venue",
  helicopter: "Helicopter",
  fire: "Sirens",
  construction: "Construction",
}

function NoiseCard({ listing: l, scene, onScene, onShow }: { listing: Listing; scene: Scene; onScene: (s: Scene) => void; onShow: () => void }) {
  const db = l.noise[scene]
  const level = loudness(db)
  const pct = (v: number) => `${Math.min(100, Math.max(0, ((v - 42) / (78 - 42)) * 100))}%`
  const sources = l.noise.sources[scene]
  const max = Math.max(...sources.map((s) => s.db), 1)
  return (
    <Section
      title="Noise"
      aside={
        <button type="button" onClick={onShow} className="text-[12px] text-muted hover:text-text">
          See it on the map
        </button>
      }
    >
      <Segmented size="sm" value={scene} onChange={onScene} options={SCENES.map((s) => ({ id: s.id, label: s.label }))} />
      <div className="mt-5 flex items-end gap-4">
        <p className="text-[40px] leading-none font-semibold tracking-[-0.04em] tabular">
          {Math.round(db)}
          <span className="ml-1 text-[14px] font-normal tracking-normal text-muted">dB</span>
        </p>
        <div className="pb-1">
          <p className="text-[14px] font-medium" style={{ color: level.tone }}>
            {level.word}
          </p>
          <p className="text-[12px] text-muted">{SCENES.find((s) => s.id === scene)?.hint}</p>
        </div>
      </div>
      <div className="relative mt-4 h-1.5 rounded-full" style={{ background: "linear-gradient(90deg,#285c78,#34a096 25%,#e8cd5a 50%,#f08432 70%,#e13c3c 88%,#aa1e6e)" }}>
        <span className="absolute top-1/2 size-3.5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-bg bg-text" style={{ left: pct(db) }} />
      </div>
      <div className="mt-1.5 flex justify-between font-mono text-[10.5px] text-faint">
        <span>Library 40</span>
        <span>Office 55</span>
        <span>Busy street 70</span>
      </div>
      <p className="eyebrow mt-6 mb-2.5">Where it comes from</p>
      {sources.length === 0 ? (
        <p className="text-[13px] text-muted">Nothing loud close by in this model.</p>
      ) : (
        <ul className="space-y-2.5">
          {sources.map((s) => (
            <li key={s.group + s.label} className="grid grid-cols-[minmax(0,1fr)_120px_44px] items-center gap-3 text-[13px]">
              <span className="min-w-0 truncate">
                <span className="text-text/90">{s.label}</span>
                <span className="text-faint"> · {GROUP_LABEL[s.group]}</span>
              </span>
              <span className="h-1 overflow-hidden rounded-full bg-line-strong">
                <span className="block h-full rounded-full bg-text/70" style={{ width: `${Math.max(6, (s.db / max) * 100)}%` }} />
              </span>
              <span className="text-right font-mono text-[11.5px] text-muted">{Math.round(s.db)} dB</span>
            </li>
          ))}
        </ul>
      )}
    </Section>
  )
}

type Me = { lat: number; lng: number } | null

function Getting({ listing: l, me }: { listing: Listing; me: Me }) {
  const stops = STOPS.map((st) => ({ ...st, m: meters(l.lat, l.lng, st.lat, st.lng) }))
    .sort((a, b) => a.m - b.m)
    .slice(0, 4)
  const places = DESTINATIONS.map((d) => {
    const m = meters(l.lat, l.lng, d.lat, d.lng)
    const walk = walkMinutes(m)
    const bus = walkMinutes(stops[0].m) + Math.round(m / 400) + 4
    return { ...d, m, walk, bus, mode: walk > 25 ? "bus" : "walk" }
  })
  const pts = [...stops.map((s) => ({ ...s, kind: "stop" as const })), ...places.filter((p) => p.m < 2200).map((p) => ({ ...p, kind: "place" as const }))]
  const reach = Math.max(...pts.map((p) => p.m), 300)
  const W = 300
  const H = 230
  const kx = 111320 * Math.cos((l.lat * Math.PI) / 180)
  const proj = (lat: number, lng: number) => {
    const dx = (lng - l.lng) * kx
    const dy = (lat - l.lat) * 110540
    const d = Math.hypot(dx, dy) || 1
    const r = Math.sqrt(d / reach)
    return { x: W / 2 + (dx / d) * r * (W / 2 - 40), y: H / 2 - (dy / d) * r * (H / 2 - 24) }
  }
  return (
    <Section title="Getting around" aside={<span className="font-mono text-[11px] text-faint">WALK · BUS</span>}>
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-[300px_minmax(0,1fr)]">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full rounded-md border border-line bg-bg">
          {[0.33, 0.66, 1].map((r) => (
            <ellipse key={r} cx={W / 2} cy={H / 2} rx={(W / 2 - 34) * r} ry={(H / 2 - 22) * r} fill="none" stroke="rgba(255,255,255,0.05)" />
          ))}
          {pts.map((p) => {
            const q = proj(p.lat, p.lng)
            return (
              <g key={p.name}>
                <line x1={W / 2} y1={H / 2} x2={q.x} y2={q.y} stroke={p.kind === "stop" ? "rgba(120,190,255,0.55)" : "rgba(255,255,255,0.22)"} strokeDasharray={p.kind === "stop" ? "" : "3 3"} />
                <circle cx={q.x} cy={q.y} r={p.kind === "stop" ? 3.5 : 3} fill={p.kind === "stop" ? "#78beff" : "#d9d9d9"} />
                {p.kind === "stop" && (
                  <text x={q.x + 6} y={q.y + 3} fontSize="9" fill="rgba(160,205,255,0.9)">
                    {walkMinutes(p.m)} min
                  </text>
                )}
                <title>{p.name}</title>
              </g>
            )
          })}
          <circle cx={W / 2} cy={H / 2} r={6} fill="#ededed" stroke="#1b1b1b" strokeWidth={2} />
        </svg>
        <div className="min-w-0">
          <ul className="divide-y divide-line">
            {places.map((p) => (
              <li key={p.name} className="flex items-baseline justify-between gap-3 py-1.5 text-[13px]">
                <span className="text-text/90">{p.name}</span>
                <span className="font-mono text-[12px] text-muted">
                  {p.mode === "walk" ? `${p.walk} min walk` : `${p.bus} min by bus`}
                </span>
              </li>
            ))}
            {me && (
              <li className="flex items-baseline justify-between gap-3 py-1.5 text-[13px]">
                <span className="text-text/90">From where you are</span>
                <span className="font-mono text-[12px] text-muted">{walkMinutes(meters(l.lat, l.lng, me.lat, me.lng))} min walk</span>
              </li>
            )}
          </ul>
          <p className="eyebrow mt-4 mb-1.5">Nearest stops</p>
          <ul className="space-y-1.5">
            {stops.map((st) => (
              <li key={st.name} className="flex items-baseline justify-between gap-3 text-[12.5px]">
                <span className="min-w-0 truncate">
                  <span className="text-text/90">{st.name}</span>
                  <span className="text-faint"> · {st.routes}</span>
                </span>
                <span className="shrink-0 font-mono text-[11.5px] text-muted">{walkMinutes(st.m)} min</span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </Section>
  )
}

function Costs({ listing: l }: { listing: Listing }) {
  const c = costs(l)
  return (
    <Section title="What you will actually pay" aside={<span className="font-mono text-[11px] text-faint">EST.</span>}>
      <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
        <dl className="space-y-1.5 text-[13px]">
          <Row label="Rent" value={usd(l.rent)} />
          {c.lines.map((x) => (
            <Row key={x.label} label={x.label} value={x.monthly ? usd(x.monthly) : "Included"} />
          ))}
          <div className="flex items-baseline justify-between border-t border-line pt-2 font-medium">
            <dt>All in, monthly</dt>
            <dd className="tabular">{usd(c.allIn)}</dd>
          </div>
        </dl>
        <div>
          <p className="eyebrow mb-2">Due up front</p>
          <dl className="space-y-1.5 text-[13px]">
            {c.fees.map((x) => (
              <Row key={x.label} label={x.label} value={x.once ? usd(x.once) : "None"} />
            ))}
          </dl>
          <p className="eyebrow mt-4 mb-2">Utilities in rent</p>
          <div className="flex flex-wrap gap-1.5">
            {l.utilitiesIncluded.length ? l.utilitiesIncluded.map((u) => <Tag key={u} tone="good">{u}</Tag>) : <Tag>Trash</Tag>}
          </div>
        </div>
      </div>
    </Section>
  )
}

function Market({ listing: l }: { listing: Listing }) {
  const m = market(l)
  const max = Math.max(...m.history.map((h) => h.rent))
  return (
    <Section title="Price history and demand" aside={<span className="font-mono text-[11px] text-faint">EST.</span>}>
      <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
        <div>
          <div className="flex h-24 items-end gap-2">
            {m.history.map((h) => (
              <div key={h.year} className="flex flex-1 flex-col items-center gap-1">
                <span className="font-mono text-[10px] text-faint">{(h.rent / 1000).toFixed(1)}K</span>
                <span className="w-full rounded-sm bg-text/25 last:bg-text/70" style={{ height: `${(h.rent / max) * 64}px` }} />
                <span className="font-mono text-[10px] text-muted">{String(h.year).slice(2)}</span>
              </div>
            ))}
          </div>
          <p className="mt-2 text-[12px] text-muted">Up about {m.rise}% a year</p>
        </div>
        <dl className="space-y-1.5 text-[13px]">
          <Row label="Demand" value={m.demand} />
          <Row label="Usually leased by" value={m.leasedBy} />
          <Row label="Waitlist" value={m.waitlist ? `${m.waitlist} people` : "None"} />
          <Row label="Rent freeze on renewal" value={m.freeze ? "Yes" : "No"} />
          <Row label="Subleasing" value={m.sublet} />
          {m.deal && <Row label="Current deal" value={m.deal} />}
        </dl>
      </div>
    </Section>
  )
}

function People({ listing: l }: { listing: Listing }) {
  const g = management(l)
  const c = community(l)
  return (
    <Section title="Management and neighborhood" aside={<span className="font-mono text-[11px] text-faint">EST.</span>}>
      <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
        <dl className="space-y-1.5 text-[13px]">
          <Row label={l.manager} value={`${g.rating} / 5`} />
          <Row label="Typical repair time" value={`${g.repairDays} day${g.repairDays > 1 ? "s" : ""}`} />
          <Row label="Condition" value={g.condition} />
          <Row label="Tenants like" value={g.praise} />
          <Row label="Tenants flag" value={g.complaint} />
        </dl>
        <dl className="space-y-1.5 text-[13px]">
          <Row label="Area" value={l.area} />
          <Row label="Feel" value={c.social} />
          <Row label="Safety" value={c.safety} />
          <Row label="Closest study spot" value={c.studySpot} />
        </dl>
      </div>
    </Section>
  )
}

function Lightbox({ photos, index, onIndex, onClose }: { photos: string[]; index: number; onIndex: (i: number) => void; onClose: () => void }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "ArrowRight") onIndex((index + 1) % photos.length)
      if (e.key === "ArrowLeft") onIndex((index - 1 + photos.length) % photos.length)
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [index, photos.length, onIndex])

  return (
    <div className="fade-in fixed inset-0 z-[60] flex flex-col bg-black/95" onMouseDown={(e) => e.stopPropagation()}>
      <div className="flex h-12 items-center justify-between px-5 text-[13px]">
        <span className="font-mono text-muted">
          {index + 1} / {photos.length}
        </span>
        <button type="button" onClick={onClose} className="text-muted hover:text-text">
          Close
        </button>
      </div>
      <div className="relative flex min-h-0 flex-1 items-center justify-center px-16">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={proxy(photos[index])} alt="" className="max-h-full max-w-full rounded-lg object-contain" />
        {[-1, 1].map((d) => (
          <button
            key={d}
            type="button"
            aria-label={d < 0 ? "Previous photo" : "Next photo"}
            onClick={() => onIndex((index + d + photos.length) % photos.length)}
            className={cn(
              "absolute top-1/2 grid size-10 -translate-y-1/2 place-items-center rounded-full border border-line-strong bg-bg/70 hover:bg-hover",
              d < 0 ? "left-4" : "right-4",
            )}
          >
            <svg width="14" height="14" viewBox="0 0 16 16" aria-hidden>
              <path d={d < 0 ? "M10 3 5 8l5 5" : "M6 3l5 5-5 5"} fill="none" stroke="currentColor" strokeWidth="1.6" />
            </svg>
          </button>
        ))}
      </div>
      <div className="scroll-thin flex gap-1.5 overflow-x-auto px-5 py-3">
        {photos.map((src, i) => (
          <button key={src} type="button" onClick={() => onIndex(i)} className={cn("shrink-0 overflow-hidden rounded", i === index ? "ring-2 ring-text" : "opacity-50 hover:opacity-80")}>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={proxy(src, "medium")} alt="" className="h-14 w-20 object-cover" />
          </button>
        ))}
      </div>
    </div>
  )
}

function tidy(text: string) {
  return text
    .replace(/Utilities (not )?included:?[^\n]*\n?/gi, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim()
}
