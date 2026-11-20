"use client"

import { useEffect } from "react"
import type { Listing } from "@/lib/types"

export function TourOverlay({ listing, onClose }: { listing: Listing; onClose: () => void }) {
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose()
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [onClose])

  return (
    <div className="fade-in fixed inset-0 z-[70] flex flex-col bg-bg">
      <header className="flex h-12 shrink-0 items-center justify-between border-b border-line px-5">
        <div className="min-w-0">
          <p className="truncate text-[13px] font-medium">{listing.street}</p>
          <p className="font-mono text-[10.5px] text-faint">3D WALKTHROUGH</p>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="h-8 rounded-md border border-line-strong px-3 text-[13px] text-muted hover:bg-hover hover:text-text"
        >
          Close
        </button>
      </header>
      <iframe
        src={listing.tour?.url}
        title={`3D walkthrough of ${listing.street}`}
        className="min-h-0 w-full flex-1 border-0 bg-black"
        allow="fullscreen; xr-spatial-tracking"
      />
    </div>
  )
}
