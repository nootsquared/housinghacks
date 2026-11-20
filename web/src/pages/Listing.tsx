import { useEffect, useState } from "react"
import { Link, useParams } from "react-router-dom"
import { ArrowLeft, ExternalLink } from "lucide-react"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { TourBadge } from "@/components/TourBadge"
import { Tour, type TourMode } from "@/components/tour/Tour"
import { asset, money, useListing, useScene } from "@/lib/data"
import type { Listing, Scene } from "@/lib/types"
import { cn } from "@/lib/utils"

export default function ListingPage() {
  const { id } = useParams()
  const { listing, loading } = useListing(id)
  const { data: scene } = useScene(id, listing?.tour === "generated")
  const [mode, setMode] = useState<TourMode>("dollhouse")
  const [current, setCurrent] = useState(0)

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!scene || mode !== "walk") return
      if (e.key === "ArrowRight") setCurrent((c) => (c + 1) % scene.views.length)
      if (e.key === "ArrowLeft") setCurrent((c) => (c - 1 + scene.views.length) % scene.views.length)
      if (e.key === "Escape") setMode("dollhouse")
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [scene, mode])

  if (loading) return null
  if (!listing) return <NotFound />

  const go = (i: number) => {
    setCurrent(i)
    setMode("walk")
  }

  return (
    <div className="flex h-full">
      <main className="relative min-w-0 flex-1 overflow-hidden bg-[#efede8]">
        {listing.tour === "matterport" && listing.matterport && (
          <iframe
            src={listing.matterport}
            title="Matterport tour"
            className="absolute inset-0 size-full"
            allow="fullscreen; xr-spatial-tracking"
          />
        )}
        {listing.tour === "generated" && scene && (
          <Tour id={listing.id} scene={scene} mode={mode} current={current} onSelectView={go} />
        )}
        {listing.tour === "insufficient" && <Insufficient listing={listing} />}

        <div className="pointer-events-none absolute inset-x-0 top-0 flex items-center justify-between p-4">
          <Button asChild variant="outline" size="sm" className="pointer-events-auto bg-card/95 backdrop-blur">
            <Link to="/">
              <ArrowLeft />
              All listings
            </Link>
          </Button>
          {listing.tour === "generated" && scene && (
            <ToggleGroup
              type="single"
              size="sm"
              variant="outline"
              value={mode}
              onValueChange={(v) => v && setMode(v as TourMode)}
              className="pointer-events-auto bg-card/95 backdrop-blur"
            >
              <ToggleGroupItem value="dollhouse">Overview</ToggleGroupItem>
              <ToggleGroupItem value="walk">Walk through</ToggleGroupItem>
            </ToggleGroup>
          )}
        </div>

        {listing.tour === "generated" && scene && (
          <ViewStrip id={listing.id} scene={scene} current={mode === "walk" ? current : -1} onSelect={go} />
        )}
      </main>
      <Details listing={listing} />
    </div>
  )
}

function ViewStrip({
  id,
  scene,
  current,
  onSelect,
}: {
  id: string
  scene: Scene
  current: number
  onSelect: (i: number) => void
}) {
  return (
    <div className="absolute inset-x-0 bottom-0 flex justify-center p-4">
      <div className="flex max-w-full gap-1.5 overflow-x-auto rounded-xl bg-card/95 p-1.5 shadow-sm ring-1 ring-border backdrop-blur">
        {scene.views.map((v, i) => (
          <button
            key={v.id}
            onClick={() => onSelect(i)}
            className={cn(
              "relative h-14 w-20 shrink-0 overflow-hidden rounded-md ring-2 ring-transparent transition",
              i === current ? "ring-brand" : "opacity-80 hover:opacity-100",
            )}
          >
            <img src={asset(id, v.file)} alt="" className="size-full object-cover" />
          </button>
        ))}
      </div>
    </div>
  )
}

function Details({ listing: l }: { listing: Listing }) {
  return (
    <aside className="flex w-[360px] shrink-0 flex-col border-l bg-background">
      <ScrollArea className="min-h-0 flex-1">
        <div className="space-y-6 p-6">
          <div className="space-y-2">
            <TourBadge listing={l} />
            <h1 className="text-xl font-semibold tracking-tight">{l.name}</h1>
            <p className="text-sm text-muted-foreground">{l.address}</p>
          </div>
          <dl className="grid grid-cols-3 gap-px overflow-hidden rounded-lg bg-border ring-1 ring-border">
            <Stat label={l.rentBasis === "per person" ? "Rent / person" : "Rent"} value={money(l.rent)} />
            <Stat label="Beds · baths" value={`${l.beds} · ${l.baths}`} />
            <Stat label="Size" value={l.sqft ? `${l.sqft} ft²` : "—"} />
          </dl>
          {(l.utilities.length > 0 || l.amenities.length > 0 || l.parking) && (
            <>
              <Separator />
              <section className="space-y-3 text-sm">
                {l.utilities.length > 0 && <Row label="Included" value={l.utilities.join(", ")} />}
                {l.parking && <Row label="Parking" value={l.parking} />}
                {l.available && <Row label="Available" value={l.available} />}
                {l.leaseTerm && <Row label="Lease" value={l.leaseTerm} />}
                {l.amenities.length > 0 && <Row label="Amenities" value={l.amenities.join(", ")} />}
              </section>
            </>
          )}
          <Separator />
          <a
            href={l.source.url}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground"
          >
            {l.source.manager} · accessed {l.source.accessed}
            <ExternalLink className="size-3" />
          </a>
        </div>
      </ScrollArea>
    </aside>
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-card px-3 py-2.5">
      <dt className="text-[11px] text-muted-foreground">{label}</dt>
      <dd className="mt-0.5 text-sm font-medium tabular">{value}</dd>
    </div>
  )
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="grid grid-cols-[88px_1fr] gap-3">
      <span className="text-muted-foreground">{label}</span>
      <span>{value}</span>
    </div>
  )
}

function Insufficient({ listing }: { listing: Listing }) {
  return (
    <div className="absolute inset-0 grid place-items-center p-10">
      <div className="max-w-md space-y-4 text-center">
        <div className="flex justify-center gap-2">
          {listing.photos.slice(0, 4).map((p) => (
            <img key={p.file} src={asset(listing.id, p.file)} alt="" className="h-20 w-28 rounded-md object-cover" />
          ))}
        </div>
        <h2 className="text-lg font-semibold">Not enough to build a walkthrough</h2>
        <p className="text-sm text-muted-foreground">
          This listing has {listing.photos.length} photos and they don't show enough of the inside. Ask the
          landlord for photos of every room, or a video tour.
        </p>
      </div>
    </div>
  )
}

function NotFound() {
  return (
    <div className="grid h-full place-items-center">
      <Button asChild variant="outline">
        <Link to="/">Back to listings</Link>
      </Button>
    </div>
  )
}
