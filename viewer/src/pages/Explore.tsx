import { useCallback, useMemo, useState } from "react"
import { useNavigate } from "react-router-dom"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Skeleton } from "@/components/ui/skeleton"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { ListingMap } from "@/components/map/ListingMap"
import { TourBadge } from "@/components/TourBadge"
import { Wordmark } from "@/components/Wordmark"
import { asset, money, useListings } from "@/lib/data"
import type { Listing } from "@/lib/types"
import { cn } from "@/lib/utils"

type Filter = "all" | "tour"

export default function Explore() {
  const { data } = useListings()
  const [hovered, setHovered] = useState<string | null>(null)
  const [filter, setFilter] = useState<Filter>("all")
  const navigate = useNavigate()

  const listings = useMemo(
    () => (data ?? []).filter((l) => filter === "all" || l.tour !== "insufficient"),
    [data, filter],
  )
  const open = useCallback((id: string) => navigate(`/l/${id}`), [navigate])

  return (
    <div className="flex h-full">
      <aside className="flex w-[420px] shrink-0 flex-col border-r bg-background">
        <header className="space-y-4 px-6 pt-6 pb-4">
          <Wordmark />
          <p className="text-sm text-muted-foreground">
            Walk through Ann Arbor rentals from their listing photos, even the ones without a tour.
          </p>
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground tabular">
              {data ? `${listings.length} listings` : " "}
            </span>
            <ToggleGroup
              type="single"
              size="sm"
              variant="outline"
              value={filter}
              onValueChange={(v) => v && setFilter(v as Filter)}
            >
              <ToggleGroupItem value="all">All</ToggleGroupItem>
              <ToggleGroupItem value="tour">Walkable</ToggleGroupItem>
            </ToggleGroup>
          </div>
        </header>
        <ScrollArea className="min-h-0 flex-1">
          <ul className="space-y-1 px-3 pb-6">
            {!data && Array.from({ length: 5 }, (_, i) => <CardSkeleton key={i} />)}
            {listings.map((l) => (
              <li key={l.id}>
                <ListingRow
                  listing={l}
                  active={hovered === l.id}
                  onHover={setHovered}
                  onOpen={open}
                />
              </li>
            ))}
          </ul>
        </ScrollArea>
      </aside>
      <main className="relative min-w-0 flex-1">
        <ListingMap listings={listings} hovered={hovered} onHover={setHovered} onSelect={open} />
      </main>
    </div>
  )
}

function ListingRow({
  listing: l,
  active,
  onHover,
  onOpen,
}: {
  listing: Listing
  active: boolean
  onHover: (id: string | null) => void
  onOpen: (id: string) => void
}) {
  return (
    <button
      onClick={() => onOpen(l.id)}
      onMouseEnter={() => onHover(l.id)}
      onMouseLeave={() => onHover(null)}
      className={cn(
        "flex w-full gap-4 rounded-lg p-3 text-left transition-colors",
        active ? "bg-secondary" : "hover:bg-secondary/60",
      )}
    >
      <img
        src={asset(l.id, l.cover)}
        alt=""
        loading="lazy"
        className="h-[84px] w-[112px] shrink-0 rounded-md bg-muted object-cover"
      />
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="flex items-start justify-between gap-2">
          <h3 className="truncate text-sm font-medium">{l.name}</h3>
          <span className="shrink-0 text-sm font-medium tabular">
            {money(l.rent)}
            {l.rentBasis === "per person" && <span className="text-muted-foreground">/pp</span>}
          </span>
        </div>
        <p className="mt-0.5 truncate text-xs text-muted-foreground">
          {l.area} · {l.beds} bd · {l.baths} ba{l.sqft ? ` · ${l.sqft} sq ft` : ""}
        </p>
        <div className="mt-auto pt-2">
          <TourBadge listing={l} />
        </div>
      </div>
    </button>
  )
}

function CardSkeleton() {
  return (
    <div className="flex gap-4 p-3">
      <Skeleton className="h-[84px] w-[112px] rounded-md" />
      <div className="flex-1 space-y-2 py-1">
        <Skeleton className="h-4 w-3/4" />
        <Skeleton className="h-3 w-1/2" />
      </div>
    </div>
  )
}
