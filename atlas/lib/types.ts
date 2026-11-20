export type Scene = "day" | "night" | "game"

export type NoiseSource = {
  group: "traffic" | "rail" | "crossing" | "bars" | "greek" | "stadium" | "helicopter" | "fire"
  label: string
  db: number
}

export type Listing = {
  id: string
  title: string | null
  street: string
  unit: string | null
  address: string
  zip: string | null
  lat: number
  lng: number
  area: string
  rent: number
  beds: number
  baths: number | null
  sqft: number | null
  status: "now" | "date" | "unavailable" | null
  availableOn: string | null
  description: string | null
  amenities: string[]
  appliances: string[]
  pets: string[]
  utilitiesIncluded: string[]
  utilitiesExcluded: string[]
  deposit: number | null
  applicationFee: number | null
  photos: string[]
  manager: string
  source: string
  accessed: string
  building: number | null
  tour: { kind: "generated" | "matterport"; url: string; matterport: string | null } | null
  floorplan: string | null
  noise: Record<Scene, number> & { sources: Record<Scene, NoiseSource[]> }
}

export type Building = {
  id: number
  ring: [number, number][]
  height: number | null
  kind: string | null
}

export type Atlas = {
  listings: Listing[]
  buildings: Building[]
  noise: {
    bounds: [[number, number], [number, number], [number, number], [number, number]]
    sources: { group: NoiseSource["group"]; label: string; lat: number; lng: number }[]
  }
}

export type Filters = {
  query: string
  minPrice: number | null
  maxPrice: number | null
  beds: number
  exactBeds: boolean
  baths: number
  moveIn: "any" | "now" | "summer" | "fall"
  pets: "any" | "cats" | "dogs"
  laundry: boolean
  furnished: boolean
  parking: boolean
  tour: boolean
  leased: boolean
  maxDb: number | null
}

export type Sort = "price-asc" | "price-desc" | "beds" | "sqft" | "quiet"

export type Bounds = { west: number; south: number; east: number; north: number }
