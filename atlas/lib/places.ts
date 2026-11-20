export type Place = { name: string; lat: number; lng: number }
export type Stop = Place & { routes: string }

export const LANDMARKS: Place[] = [
  { name: "The Diag", lat: 42.27685, lng: -83.73823 },
  { name: "Michigan Union", lat: 42.27526, lng: -83.74167 },
  { name: "Ross School of Business", lat: 42.27315, lng: -83.73805 },
  { name: "Michigan Medicine", lat: 42.28309, lng: -83.72866 },
  { name: "Kerrytown", lat: 42.28438, lng: -83.74585 },
]

export const STOPS: Stop[] = [
  { name: "Central Campus Transit Center", routes: "MBus Commuter, Bursley-Baits · TheRide 62, 64", lat: 42.27355, lng: -83.73555 },
  { name: "Blake Transit Center", routes: "TheRide downtown hub", lat: 42.27945, lng: -83.74845 },
  { name: "State & William", routes: "MBus Commuter · TheRide 4", lat: 42.27895, lng: -83.74075 },
  { name: "South U & East U", routes: "TheRide 62, 64", lat: 42.2739, lng: -83.73635 },
  { name: "North U & State", routes: "MBus · TheRide 4", lat: 42.27695, lng: -83.74055 },
  { name: "Forest & South U", routes: "TheRide 62 · MBus", lat: 42.27455, lng: -83.7347 },
  { name: "Kerrytown", routes: "TheRide 22", lat: 42.2842, lng: -83.7468 },
  { name: "Packard & State", routes: "TheRide 5, 62", lat: 42.2704, lng: -83.7415 },
  { name: "Geddes & Oxford", routes: "TheRide 64", lat: 42.2774, lng: -83.7323 },
  { name: "Huron & Division", routes: "TheRide 4, 22", lat: 42.2814, lng: -83.7481 },
  { name: "Washtenaw & Hill", routes: "TheRide 4", lat: 42.27315, lng: -83.7295 },
  { name: "Packard & Granger", routes: "TheRide 5", lat: 42.2668, lng: -83.7373 },
]
