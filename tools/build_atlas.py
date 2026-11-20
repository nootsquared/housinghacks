"""Turn scraped listings and OSM data into what the atlas map loads.

Outputs atlas/public/atlas.json and atlas/public/noise/{day,night,game}.png.
Run with: uv run --project pipeline python tools/build_atlas.py
"""

import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.signal import fftconvolve

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "atlas" / "data"
PUBLIC = REPO / "atlas" / "public"
CURATED = REPO / "web" / "public" / "data" / "index.json"

S, W, N, E = 42.238, -83.785, 42.320, -83.652
CELL = 12.0
M_LAT = 110_540.0
M_LNG = 111_320.0 * math.cos(math.radians((S + N) / 2))
ROWS = int((N - S) * M_LAT / CELL)
COLS = int((E - W) * M_LNG / CELL)

ROAD_DB = {
    "motorway": 80,
    "motorway_link": 70,
    "trunk": 76,
    "trunk_link": 68,
    "primary": 71,
    "secondary": 65,
    "tertiary": 59,
}
ROAD_NIGHT = -7
RAIL_DB = 64
CROSSING_DB = 86
BAR_DB = 74
GREEK_DB = 72
HELI_DB = 84
FIRE_DB = 70
BUILD_DB = 76
VENUES = {
    "Michigan Stadium": 128,
    "Crisler Center": 104,
    "Yost Ice Arena": 100,
    "Ray Fisher Stadium": 98,
}

SCENES = {
    "day": {"traffic": 0, "rail": 0, "crossing": 0, "helicopter": 0, "fire": 0, "construction": 0},
    "night": {"traffic": ROAD_NIGHT, "rail": -2, "crossing": 0, "bars": 0, "greek": 0, "helicopter": -4, "fire": -2, "construction": -14},
    "game": {"traffic": 3, "rail": 0, "crossing": 0, "bars": 2, "greek": 4, "helicopter": 0, "fire": 0, "stadium": 0, "construction": -6},
}

FLOOR_DB = 42
ATTEN_M = 260.0


def xy(lat, lng):
    return (lng - W) * M_LNG, (lat - S) * M_LAT


def to_cell(lat, lng):
    x, y = xy(lat, lng)
    return int((N - S) * M_LAT / CELL - y / CELL), int(x / CELL)


def densify(coords, step):
    out = []
    for (a_lat, a_lng), (b_lat, b_lng) in zip(coords, coords[1:]):
        ax, ay = xy(a_lat, a_lng)
        bx, by = xy(b_lat, b_lng)
        length = math.hypot(bx - ax, by - ay)
        n = max(1, int(length / step))
        for i in range(n):
            t = i / n
            out.append((a_lat + (b_lat - a_lat) * t, a_lng + (b_lng - a_lng) * t))
    if coords:
        out.append(coords[-1])
    return out


def center(element):
    if "lat" in element:
        return element["lat"], element["lon"]
    if "center" in element:
        return element["center"]["lat"], element["center"]["lon"]
    if "geometry" in element:
        g = element["geometry"]
        return sum(p["lat"] for p in g) / len(g), sum(p["lon"] for p in g) / len(g)
    b = element["bounds"]
    return (b["minlat"] + b["maxlat"]) / 2, (b["minlon"] + b["maxlon"]) / 2


def sources():
    """Every source as (group, label, lat, lng, power at 10 m in linear units)."""
    osm = json.loads((DATA / "osm_noise.json").read_text())["elements"]
    out = []
    greek = re.compile(r"\b(Alpha|Beta|Gamma|Delta|Epsilon|Zeta|Theta|Kappa|Lambda|Sigma|Tau|Phi|Chi|Psi|Omega|Upsilon|Pi)\b.*\b(Alpha|Beta|Gamma|Delta|Epsilon|Zeta|Eta|Theta|Iota|Kappa|Lambda|Mu|Nu|Xi|Pi|Rho|Sigma|Tau|Upsilon|Phi|Chi|Psi|Omega)\b")
    for el in osm:
        tags = el.get("tags", {})
        name = tags.get("name")
        hw = tags.get("highway")
        if hw in ROAD_DB and "geometry" in el:
            pts = densify([(p["lat"], p["lon"]) for p in el["geometry"]], CELL)
            weight = 10 ** (ROAD_DB[hw] / 10) * CELL / (10 * math.pi)
            label = name or tags.get("ref") or hw.replace("_", " ")
            if tags.get("ref", "").startswith(("US", "M ", "I ")) and not name:
                label = tags["ref"]
            out += [("traffic", label, la, ln, weight) for la, ln in pts]
        elif tags.get("railway") == "rail" and "geometry" in el:
            pts = densify([(p["lat"], p["lon"]) for p in el["geometry"]], CELL)
            weight = 10 ** (RAIL_DB / 10) * CELL / (10 * math.pi)
            out += [("rail", "Rail line", la, ln, weight) for la, ln in pts]
        elif tags.get("railway") == "level_crossing":
            la, ln = center(el)
            out.append(("crossing", "Train horn at a crossing", la, ln, 10 ** (CROSSING_DB / 10)))
        elif tags.get("amenity") in ("bar", "pub", "nightclub"):
            la, ln = center(el)
            out.append(("bars", name or "Bar", la, ln, 10 ** (BAR_DB / 10)))
        elif tags.get("leisure") == "stadium" and name in VENUES:
            la, ln = center(el)
            out.append(("stadium", name, la, ln, 10 ** (VENUES[name] / 10)))
        elif tags.get("aeroway") == "helipad":
            la, ln = center(el)
            out.append(("helicopter", "UMMC hospital helipad", la, ln, 10 ** (HELI_DB / 10)))
        elif tags.get("landuse") == "construction" or tags.get("building") == "construction" or tags.get("highway") == "construction":
            la, ln = center(el)
            label = name or ("Road work" if tags.get("highway") else "Construction site")
            out.append(("construction", label, la, ln, 10 ** (BUILD_DB / 10)))
        elif tags.get("amenity") == "fire_station":
            la, ln = center(el)
            out.append(("fire", name or "Fire station", la, ln, 10 ** (FIRE_DB / 10)))
        elif name and greek.search(name) and "Piercing" not in name and "Rho Sigma" not in name and "Delta Phi" != name:
            la, ln = center(el)
            out.append(("greek", name, la, ln, 10 ** (GREEK_DB / 10)))
    return out


def kernel():
    radius = int(2400 / CELL)
    ys, xs = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    r = np.maximum(np.hypot(xs, ys) * CELL, 6.0)
    return (10.0 / r) ** 2 * np.exp(-r / ATTEN_M)


def falloff(meters):
    r = np.maximum(meters, 6.0)
    return (10.0 / r) ** 2 * np.exp(-r / ATTEN_M)


def grids(srcs):
    k = kernel()
    by_group = defaultdict(lambda: np.zeros((ROWS, COLS)))
    for group, _, la, ln, w in srcs:
        r, c = to_cell(la, ln)
        if 0 <= r < ROWS and 0 <= c < COLS:
            by_group[group][r, c] += w
    return {g: fftconvolve(power, k, mode="same").clip(min=0) for g, power in by_group.items()}


RAMP = [
    (45, (0, 0, 0, 0)),
    (50, (40, 92, 120, 70)),
    (56, (52, 160, 150, 125)),
    (62, (232, 205, 90, 160)),
    (68, (240, 132, 50, 185)),
    (74, (225, 60, 60, 205)),
    (82, (170, 30, 110, 220)),
]


def colorize(db):
    out = np.zeros(db.shape + (4,), dtype=np.float32)
    levels = [r[0] for r in RAMP]
    colors = np.array([r[1] for r in RAMP], dtype=np.float32)
    for ch in range(4):
        out[..., ch] = np.interp(db, levels, colors[:, ch])
    return out.astype(np.uint8)


def to_db(linear):
    return 10 * np.log10(linear + 10 ** (FLOOR_DB / 10))


def render(gr):
    (PUBLIC / "noise").mkdir(parents=True, exist_ok=True)
    for scene, shifts in SCENES.items():
        total = sum(gr[g] * 10 ** (shift / 10) for g, shift in shifts.items() if g in gr)
        Image.fromarray(colorize(to_db(total)), "RGBA").save(PUBLIC / "noise" / f"{scene}.png", optimize=True)


def point_noise(lat, lng, srcs_arr, labels, groups):
    la, ln, w = srcs_arr
    dx = (ln - lng) * M_LNG
    dy = (la - lat) * M_LAT
    contrib = w * falloff(np.hypot(dx, dy))
    out = {}
    for scene, shifts in SCENES.items():
        scale = np.zeros_like(contrib)
        for g, shift in shifts.items():
            scale[groups == g] = 10 ** (shift / 10)
        out[scene] = round(float(to_db((contrib * scale).sum())), 1)
    out["sources"] = {}
    for scene, shifts in SCENES.items():
        per_label = defaultdict(float)
        for i in np.nonzero(contrib > 10 ** (20 / 10))[0]:
            if groups[i] in shifts:
                per_label[(groups[i], labels[i])] += contrib[i] * 10 ** (shifts[groups[i]] / 10)
        top = sorted(per_label.items(), key=lambda kv: -kv[1])[:4]
        out["sources"][scene] = [
            {"group": g, "label": label, "db": round(10 * math.log10(v), 1)} for (g, label), v in top
        ]
    return out


def point_in_poly(lat, lng, ring):
    inside = False
    for (a_lng, a_lat), (b_lng, b_lat) in zip(ring, ring[1:]):
        if (a_lat > lat) != (b_lat > lat):
            x = a_lng + (lat - a_lat) * (b_lng - a_lng) / (b_lat - a_lat)
            if lng < x:
                inside = not inside
    return inside


def edge_distance(lat, lng, ring):
    px, py = xy(lat, lng)
    best = float("inf")
    for (a_lng, a_lat), (b_lng, b_lat) in zip(ring, ring[1:]):
        ax, ay = xy(a_lat, a_lng)
        bx, by = xy(b_lat, b_lng)
        dx, dy = bx - ax, by - ay
        t = 0 if dx == dy == 0 else max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
        best = min(best, math.hypot(px - ax - t * dx, py - ay - t * dy))
    return best


def polygon_area(ring):
    pts = [xy(lat, lng) for lng, lat in ring]
    return abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(pts, pts[1:]))) / 2


def footprints():
    els = json.loads((DATA / "osm_buildings.json").read_text())["elements"]
    out = []
    seen = set()
    for el in els:
        if el["id"] in seen or "geometry" not in el:
            continue
        seen.add(el["id"])
        ring = [[p["lon"], p["lat"]] for p in el["geometry"]]
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        tags = el.get("tags", {})
        height = None
        m = re.match(r"[\d.]+", str(tags.get("height", "")))
        if m:
            height = float(m.group(0))
        if height is None and tags.get("building:levels"):
            try:
                height = float(tags["building:levels"]) * 3.3 + 1.5
            except ValueError:
                pass
        if height is None:
            area = polygon_area(ring)
            height = 8.5 if area < 220 else 11 if area < 600 else 16 if area < 1500 else 24
        out.append({"id": el["id"], "ring": ring, "height": round(height, 1), "kind": tags.get("building")})
    return out


def match_building(lat, lng, buildings):
    for b in buildings:
        if point_in_poly(lat, lng, b["ring"]):
            return b
    near = min(buildings, key=lambda b: edge_distance(lat, lng, b["ring"]), default=None)
    if near and edge_distance(lat, lng, near["ring"]) < 18:
        return near
    return None


def area_name(lat, lng):
    if lat > 42.2825 and lng > -83.735:
        return "North Campus"
    if lat > 42.2795 and lng < -83.742:
        return "Kerrytown" if lat > 42.2815 else "Downtown"
    if lat > 42.2805:
        return "Old West Side" if lng < -83.755 else "Kerrytown"
    if lng < -83.752:
        return "Old West Side" if lat > 42.272 else "Southwest"
    if lat < 42.266:
        return "South Campus" if lng > -83.75 else "Southwest"
    if lng > -83.728:
        return "Burns Park" if lat < 42.272 else "East Campus"
    if lat < 42.2745:
        return "South U" if lng > -83.74 else "South Campus"
    return "Central Campus" if lng > -83.745 else "Downtown"


def clean_address(raw):
    street = raw.split(", Ann Arbor")[0]
    street = re.sub(r"\s+", " ", street).strip()
    unit = None
    m = re.match(r"^(.*?)(?:,?\s+(?:Unit|Apt)\s*#?\s*([\w-]+)|\s+-\s+(?:Unit\s*)?#?\s*([\w-]+))$", street, flags=re.I)
    if m:
        street, unit = m.group(1), m.group(2) or m.group(3)
    street = street.title() if street.isupper() else street
    street = re.sub(r"\b(Ne|Nw|Se|Sw)\b", lambda x: x.group(1).upper(), street)
    return street.strip(" ,-"), unit


def parse_available(s):
    if not s:
        return None, None
    if "Unavailable" in s:
        return "unavailable", None
    if "Now" in s:
        return "now", None
    m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", s)
    if m:
        mo, d, y = map(int, m.groups())
        y = y + 2000 if y < 100 else y
        return "date", f"{y:04d}-{mo:02d}-{d:02d}"
    return None, None


def utilities(row):
    inc = list(row.get("utilitiesIncluded") or [])
    exc = list(row.get("utilitiesExcluded") or [])
    desc = row.get("description") or ""
    m = re.search(r"Utilities included:?\s*([^\n]+)", desc, flags=re.I)
    if m and not inc:
        inc = [x.strip() for x in m.group(1).split(",") if x.strip()]
    m = re.search(r"Utilities not included:?\s*([^\n]+)", desc, flags=re.I)
    if m and not exc:
        exc = [x.strip() for x in m.group(1).split(",") if x.strip()]
    return inc, exc


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def main():
    raw = json.loads((DATA / "listings.json").read_text())
    raw = [r for r in raw if r["beds"] is not None and r["rent"] and r["rent"] >= 400 and "Garage" not in (r.get("title") or "")]
    srcs = sources()
    print(f"{len(srcs)} noise source samples")
    gr = grids(srcs)
    render(gr)
    srcs_arr = (
        np.array([s[2] for s in srcs]),
        np.array([s[3] for s in srcs]),
        np.array([s[4] for s in srcs]),
    )
    labels = np.array([s[1] for s in srcs], dtype=object)
    groups = np.array([s[0] for s in srcs], dtype=object)

    buildings = footprints()
    curated = {slug(c["address"].split(",")[0]): c for c in json.loads(CURATED.read_text())}
    listings = []
    noise_cache = {}
    ids = set()
    for row in raw:
        street, unit = clean_address(row["address"])
        key = (round(row["lat"], 6), round(row["lng"], 6))
        if key not in noise_cache:
            noise_cache[key] = point_noise(row["lat"], row["lng"], srcs_arr, labels, groups)
        b = match_building(row["lat"], row["lng"], buildings)
        status, date_ = parse_available(row.get("available"))
        inc, exc = utilities(row)
        base = slug(f"{street} {unit or ''}")
        lid = base
        n = 2
        while lid in ids:
            lid = f"{base}-{n}"
            n += 1
        ids.add(lid)
        cur = curated.get(slug(street))
        tour = None
        if cur and cur["tour"] in ("generated", "matterport"):
            tour = {"kind": cur["tour"], "url": f"http://localhost:5173/l/{cur['id']}", "matterport": cur.get("matterport")}
        listings.append(
            {
                "id": lid,
                "title": row.get("title"),
                "street": street,
                "unit": unit,
                "address": f"{street}{', Unit ' + unit if unit else ''}, Ann Arbor, MI",
                "zip": (re.search(r"\b(481\d\d)\b", row["address"]) or [None, None])[1],
                "lat": row["lat"],
                "lng": row["lng"],
                "area": area_name(row["lat"], row["lng"]),
                "rent": row["rent"],
                "beds": row["beds"],
                "baths": row["baths"],
                "sqft": row["sqft"],
                "status": status,
                "availableOn": date_,
                "description": row.get("description"),
                "amenities": row.get("amenities") or [],
                "appliances": row.get("appliances") or [],
                "pets": row.get("pets") or [],
                "utilitiesIncluded": inc,
                "utilitiesExcluded": exc,
                "deposit": row.get("deposit"),
                "applicationFee": row.get("applicationFee"),
                "photos": row.get("photos") or ([row["cover"]] if row.get("cover") else []),
                "manager": row["manager"],
                "source": row["url"],
                "accessed": row["accessed"],
                "building": b["id"] if b else None,
                "tour": tour,
                "floorplan": f"/media/{cur['id']}/floorplan.jpg" if cur and cur.get("floorplan") else None,
                "noise": noise_cache[key],
            }
        )

    for cur in json.loads(CURATED.read_text()):
        if cur["tour"] not in ("generated", "matterport"):
            continue
        if any(l["tour"] and cur["id"] in l["tour"]["url"] for l in listings):
            continue
        street, unit = clean_address(cur["address"])
        b = match_building(cur["lat"], cur["lng"], buildings) if buildings else None
        listings.append(
            {
                "id": cur["id"],
                "title": cur["name"],
                "street": street,
                "unit": None,
                "address": f"{street}, Ann Arbor, MI",
                "zip": "48104",
                "lat": cur["lat"],
                "lng": cur["lng"],
                "area": area_name(cur["lat"], cur["lng"]),
                "rent": cur["rent"],
                "beds": cur["beds"],
                "baths": cur["baths"],
                "sqft": cur["sqft"],
                "status": "date" if cur.get("available") else None,
                "availableOn": None,
                "description": cur.get("notes"),
                "amenities": cur.get("amenities") or [],
                "appliances": [],
                "pets": [],
                "utilitiesIncluded": cur.get("utilities") or [],
                "utilitiesExcluded": [],
                "deposit": None,
                "applicationFee": None,
                "photos": [f"/media/{cur['id']}/{p['file']}" for p in cur["photos"][:24]],
                "manager": cur["source"]["manager"],
                "source": cur["source"]["url"],
                "accessed": cur["source"]["accessed"],
                "building": b["id"] if b else None,
                "tour": {
                    "kind": cur["tour"],
                    "url": f"http://localhost:5173/l/{cur['id']}",
                    "matterport": cur.get("matterport"),
                },
                "floorplan": f"/media/{cur['id']}/{cur['floorplan']}" if cur.get("floorplan") else None,
                "noise": point_noise(cur["lat"], cur["lng"], srcs_arr, labels, groups),
            }
        )

    used = {l["building"] for l in listings if l["building"]}
    shapes = [
        {"id": b["id"], "ring": b["ring"], "height": b["height"], "kind": b["kind"]}
        for b in buildings
        if b["id"] in used
    ]
    sources_out = []
    for group, label, la, ln, _ in srcs:
        if group in ("bars", "greek", "stadium", "helicopter", "crossing", "fire", "construction"):
            sources_out.append({"group": group, "label": label, "lat": round(la, 6), "lng": round(ln, 6)})
    out = {
        "listings": listings,
        "buildings": shapes,
        "noise": {"bounds": [[W, N], [E, N], [E, S], [W, S]], "sources": sources_out},
    }
    (PUBLIC / "atlas.json").write_text(json.dumps(out, separators=(",", ":")))
    matched = sum(1 for l in listings if l["building"])
    print(f"{len(listings)} listings, {matched} on a real outline, {len(shapes)} buildings")


if __name__ == "__main__":
    main()
