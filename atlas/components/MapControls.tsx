"use client"

import { cn } from "@/lib/cn"
import { SCENES } from "@/lib/format"
import type { Scene } from "@/lib/types"

const TICKS = [45, 55, 65, 75]

const SOURCES = [
  ["#e8d36a", "Rail crossings"],
  ["#f08a4b", "Bars and clubs"],
  ["#d36fd8", "Greek houses"],
  ["#69b7ff", "Hospital helipad"],
  ["#e5625c", "Stadium and arenas"],
  ["#ffb020", "Construction"],
] as const

export function MapControls({
  noise,
  onNoise,
  tilt,
  onTilt,
}: {
  noise: Scene | null
  onNoise: (s: Scene | null) => void
  tilt: boolean
  onTilt: (v: boolean) => void
}) {
  return (
    <>
      <div className="absolute top-3 left-3 z-10 flex gap-2">
        <button
          type="button"
          onClick={() => onNoise(noise ? null : "day")}
          aria-pressed={noise != null}
          className={cn(
            "flex h-8 items-center gap-2 rounded-md border px-3 text-[13px] shadow-[0_6px_20px_rgba(0,0,0,0.4)] backdrop-blur-md transition-colors",
            noise ? "border-text/40 bg-text text-bg" : "border-line-strong bg-bg/85 text-text hover:bg-hover",
          )}
        >
          <svg width="14" height="14" viewBox="0 0 16 16" aria-hidden>
            <path d="M2 6v4h3l4 3V3L5 6z" fill="currentColor" />
            <path d="M11 5.5a3.5 3.5 0 0 1 0 5M12.8 3.5a6 6 0 0 1 0 9" fill="none" stroke="currentColor" strokeWidth="1.3" />
          </svg>
          Noise map
        </button>
        <div className="flex rounded-md border border-line-strong bg-bg/85 p-0.5 shadow-[0_6px_20px_rgba(0,0,0,0.4)] backdrop-blur-md">
          {[true, false].map((v) => (
            <button
              key={String(v)}
              type="button"
              onClick={() => onTilt(v)}
              className={cn("h-7 rounded-[5px] px-2.5 font-mono text-[11px]", tilt === v ? "bg-hover text-text" : "text-muted")}
            >
              {v ? "3D" : "2D"}
            </button>
          ))}
        </div>
      </div>

      {noise && (
        <div className="rise-in absolute bottom-3 left-3 z-10 w-[300px] rounded-md border border-line-strong bg-bg/90 p-4 shadow-[0_20px_50px_rgba(0,0,0,0.5)] backdrop-blur-md">
          <div className="flex items-baseline justify-between">
            <p className="text-[13px] font-medium">Modeled noise</p>
            <p className="font-mono text-[10.5px] text-faint">dB(A) · TYPICAL HOUR</p>
          </div>
          <div className="mt-3 flex rounded-md border border-line-strong bg-surface p-0.5">
            {SCENES.map((s) => (
              <button
                key={s.id}
                type="button"
                onClick={() => onNoise(s.id)}
                className={cn(
                  "h-7 flex-1 rounded-[5px] text-[12px] transition-colors",
                  noise === s.id ? "bg-text font-medium text-bg" : "text-muted hover:text-text",
                )}
              >
                {s.label}
              </button>
            ))}
          </div>
          <p className="mt-2 text-[11.5px] text-muted">{SCENES.find((s) => s.id === noise)?.hint}</p>
          <div
            className="mt-3 h-2 rounded-full"
            style={{
              background:
                "linear-gradient(90deg, rgba(40,92,120,0.35) 0%, #34a096 22%, #e8cd5a 46%, #f08432 66%, #e13c3c 84%, #aa1e6e 100%)",
            }}
          />
          <div className="mt-1.5 flex justify-between font-mono text-[10.5px] text-faint">
            {TICKS.map((t) => (
              <span key={t}>{t}</span>
            ))}
          </div>
          <div className="mt-3 grid grid-cols-2 gap-x-3 gap-y-1.5 border-t border-line pt-3">
            {SOURCES.map(([color, label]) => (
              <span key={label} className="flex items-center gap-1.5 text-[11.5px] text-muted">
                <span className="size-2 rounded-full ring-1 ring-black" style={{ background: color }} />
                {label}
              </span>
            ))}
            <span className="flex items-center gap-1.5 text-[11.5px] text-muted">
              <span className="h-0.5 w-2.5 rounded bg-white/50" />
              Road traffic
            </span>
          </div>
          <p className="mt-3 text-[11px] leading-snug text-faint">
            Built from every road, rail line, bar, Greek house and venue in OpenStreetMap, with sound falling off over distance.
            It is a model, not a measurement.
          </p>
        </div>
      )}
    </>
  )
}
