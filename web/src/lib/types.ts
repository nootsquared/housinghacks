export type TourKind = "matterport" | "generated" | "insufficient"

export type Photo = {
  file: string
  caption?: string | null
  room?: string | null
  kind?: string
}

export type Listing = {
  id: string
  name: string
  address: string
  lat: number
  lng: number
  area: string
  unit: string
  beds: number
  baths: number
  sqft: number | null
  rent: number | null
  rentBasis: "total" | "per person"
  available: string | null
  leaseTerm: string | null
  utilities: string[]
  amenities: string[]
  parking: string | null
  matterport: string | null
  floorplan: string | null
  photos: Photo[]
  source: { url: string; manager: string; accessed: string }
  notes?: string
  tour: TourKind
  cover: string
  coverage?: Coverage
}

export type Coverage = {
  score: number
  photos: number
  interior: number
  used: number
  rooms: { type: string; label: string; photos: number; seen: boolean }[]
  flags: string[]
}

export type Vec3 = [number, number, number]

export type View = {
  id: string
  file: string
  room: string
  position: Vec3
  quaternion: [number, number, number, number]
  fov: number
  aspect: number
  depth: string
}

export type Room = {
  id: string
  type: string
  label: string
  center: Vec3
  size: Vec3
  yaw: number
  confidence: "measured" | "estimated" | "inferred"
  views: string[]
  shell?: Shell
}

export type Surface = {
  name: string
  origin: Vec3
  u: Vec3
  v: Vec3
  texture: string
  coverage: number
  detected: boolean
}

export type Shell = {
  x: [number, number]
  z: [number, number]
  height: number
  surfaces: Surface[]
}

export type Scene = {
  id: string
  units: "m"
  rooms: Room[]
  views: View[]
  points?: string
  mode?: "single" | "multi"
  bounds: { min: Vec3; max: Vec3 }
}
