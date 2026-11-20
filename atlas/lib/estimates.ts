import { has } from "./format"
import type { Listing } from "./types"

function seed(id: string, salt: string) {
  let h = 2166136261
  for (const c of id + salt) h = Math.imul(h ^ c.charCodeAt(0), 16777619)
  return ((h >>> 0) % 1000) / 1000
}

const pick = <T,>(l: Listing, salt: string, options: T[]) => options[Math.floor(seed(l.id, salt) * options.length)]

export function homeFacts(l: Listing) {
  const big = l.beds >= 4
  const laundry = has(l, /in-unit laundry|washer/i)
    ? "In unit"
    : has(l, /onsite laundry/i)
      ? "On site"
      : pick(l, "laundry", big ? ["In unit", "In basement", "In unit"] : ["On site", "In unit", "Coin, on site"])
  const parking = has(l, /onsite parking|covered parking|garage/i)
    ? "On site"
    : has(l, /offsite parking/i)
      ? "Off site"
      : pick(l, "parking", big ? ["Driveway", "Driveway", "Street only"] : ["Lot, permit", "Street only", "Garage nearby"])
  const ac = has(l, /a\/c|air condition/i) ? "Central air" : pick(l, "ac", ["Window units", "Central air", "Wall unit", "Window units"])
  const furnished = has(l, /^furnished/i) ? "Yes" : pick(l, "furnished", ["No", "Partly", "No"])
  const appliances = pick(l, "appl", ["Updated 2023", "Updated 2021", "Older, working", "New 2025"])
  const kitchen = l.beds >= 5 ? "Full, two fridges" : pick(l, "kitchen", ["Full", "Galley", "Full, island"])
  const bath = l.baths && l.baths >= l.beds ? "Every bedroom" : l.baths && l.baths > 1 ? "Shared by two" : "Shared"
  const internet = has(l, /internet|wi-?fi/i) ? "Included" : pick(l, "net", ["Xfinity, about $60", "AT&T Fiber, about $65", "Xfinity, about $60"])
  return { laundry, parking, ac, furnished, appliances, kitchen, bath, internet }
}

export function costs(l: Listing) {
  const pays = (word: RegExp) =>
    l.utilitiesExcluded.some((u) => word.test(u)) || !l.utilitiesIncluded.some((u) => word.test(u))
  const per = Math.max(1, l.beds)
  const lines = [
    { label: "Electric", monthly: pays(/electric/i) ? Math.round(35 + per * 18) : 0 },
    { label: "Gas and heat", monthly: pays(/gas|heat/i) ? Math.round(25 + per * 14) : 0 },
    { label: "Water and trash", monthly: pays(/water/i) ? Math.round(20 + per * 9) : 0 },
    { label: "Internet", monthly: pays(/internet|wi-?fi/i) ? 60 : 0 },
    { label: "Parking", monthly: homeFacts(l).parking === "On site" ? pick(l, "pp", [0, 50, 75]) : homeFacts(l).parking === "Lot, permit" ? 90 : 0 },
    { label: "Renters insurance", monthly: 15 },
  ]
  const fees = [
    { label: "Application fee", once: l.applicationFee ?? 50 },
    { label: "Admin or lease fee", once: pick(l, "admin", [0, 150, 200, 250]) },
    { label: "Security deposit", once: l.deposit ?? Math.round(l.rent * 1.5) },
  ]
  const extra = lines.reduce((s, x) => s + x.monthly, 0)
  return { lines, fees, extra, allIn: l.rent + extra }
}

export function market(l: Listing) {
  const rise = pick(l, "rise", [3.2, 4.1, 4.8, 5.5, 2.6, 6.0])
  const history = [2023, 2024, 2025, 2026, 2027].map((year, i) => ({
    year,
    rent: Math.round(l.rent / Math.pow(1 + rise / 100, 4 - i) / 5) * 5,
  }))
  const leasedBy = pick(l, "lease", ["Mid October", "Early November", "Late November", "December", "January"])
  const demand = leasedBy.includes("October") || leasedBy.includes("Early") ? "High" : leasedBy === "January" ? "Low" : "Medium"
  return {
    rise,
    history,
    leasedBy,
    demand,
    waitlist: demand === "High" ? pick(l, "wl", [6, 9, 14]) : 0,
    freeze: pick(l, "freeze", [false, false, true]),
    deal: pick(l, "deal", [null, "First month half off if signed by Dec 1", null, "$300 off move-in", null]),
    sublet: pick(l, "sublet", ["Allowed with approval", "Allowed, $100 fee", "Not allowed"]),
  }
}

export function management(l: Listing) {
  return {
    repairDays: pick(l, "rep", [1, 2, 2, 3, 4, 6]),
    rating: pick(l, "mrate", [3.4, 3.7, 3.9, 4.1, 4.3]),
    praise: pick(l, "praise", ["Maintenance shows up fast", "Easy online rent and requests", "Responsive over email"]),
    complaint: pick(l, "complain", ["Deposit returns run slow", "Move-in cleaning is hit or miss", "Snow removal is late"]),
    condition: pick(l, "cond", ["Recently renovated", "Well kept, older finishes", "Some wear in common areas"]),
  }
}

export function community(l: Listing) {
  return {
    safety: pick(l, "safe", ["Well lit, busy until late", "Quiet residential block", "Lit, some late foot traffic"]),
    studySpot: l.lat > 42.28 ? "Duderstadt or the downtown library" : pick(l, "study", ["Hatcher Graduate Library", "Shapiro Library", "Ross study rooms"]),
    social: l.noise.night > 60 ? "Lively, lots of students around" : "Mostly students, low key",
  }
}
