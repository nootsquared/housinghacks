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
  model?: RoomModel
  mesh?: RoomMesh
  models?: RoomObject[]
  staging?: StagedItem[]
  place?: { position: Vec3; rotation: number }
  cloneOf?: string
}

export type StagedItem = {
  kind: string
  center: Vec3
  size: Vec3
  rot?: number
  color?: string
  wall?: boolean
}

export type RoomObject = {
  file: string
  kind: string
  size: Vec3
  center: Vec3
  views: string[]
}

export type RoomMesh = {
  file: string
  triangles: number
  segments: { view: string | null; start: number; count: number }[]
}

export type Opening = {
  wall: WallId
  kind: "door" | "window"
  from: number
  to: number
  bottom: number
  top: number
  frame?: string
}

export type WallId = "n" | "s" | "e" | "w"

export type Furniture = {
  kind: string
  center: Vec3
  size: Vec3
  color: string
  accent?: string
  evidence: number
}

export type RoomModel = {
  decals?: Partial<Record<WallId, string>>
  fills?: Partial<Record<WallId | "floor", string>>
  /** opaque photo texture covering a whole wall or the floor; replaces paint, decals and fills */
  surfaces?: Partial<Record<WallId | "floor", string>>
  walls: Record<WallId, string>
  wallColor: string
  floor: { material: "wood" | "carpet" | "tile"; color: string; accent?: string; evidence: number; tile?: string; size?: number }
  openings: Opening[]
  furniture: Furniture[]
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
  floorTurn?: number
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
  doorways?: Doorway[]
  cameraHeight?: number
  floorArea?: number
}

export type Doorway = {
  rooms: string[]
  p0: [number, number]
  p1: [number, number]
  bottom: number
  top: number
  evidence?: string
  quality?: number
  bare?: boolean
}
