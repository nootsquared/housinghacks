import type { Listing } from "@/lib/types"
import { cn } from "@/lib/utils"

export function TourBadge({ listing, className }: { listing: Listing; className?: string }) {
  const used = listing.coverage?.used ?? listing.photos.length
  const text =
    listing.tour === "matterport"
      ? "Matterport tour"
      : listing.tour === "generated"
        ? `3D from ${used} photos`
        : "Too few photos for 3D"
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 text-xs",
        listing.tour === "insufficient" ? "text-muted-foreground" : "text-foreground",
        className,
      )}
    >
      <span
        className={cn(
          "size-1.5 rounded-full",
          listing.tour === "generated" && "bg-brand",
          listing.tour === "matterport" && "bg-foreground",
          listing.tour === "insufficient" && "bg-muted-foreground/40",
        )}
      />
      {text}
    </span>
  )
}
