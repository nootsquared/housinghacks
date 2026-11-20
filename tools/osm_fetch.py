"""Fetch OpenStreetMap noise sources and building outlines for central Ann Arbor.

noise  -> atlas/data/osm_noise.json
build  -> atlas/data/osm_buildings.json (outlines around each scraped listing)
"""

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "atlas" / "data"
API = "https://overpass-api.de/api/interpreter"
BBOX = (42.235, -83.80, 42.315, -83.67)


def overpass(query: str):
    body = urllib.parse.urlencode({"data": query}).encode()
    for attempt in range(4):
        try:
            req = urllib.request.Request(API, data=body, headers={"User-Agent": "housing-map/0.1"})
            with urllib.request.urlopen(req, timeout=240) as r:
                return json.load(r)
        except Exception as error:
            print(f"overpass retry {attempt}: {error}", file=sys.stderr)
            time.sleep(10 * (attempt + 1))
    raise SystemExit("overpass failed")


def noise():
    s, w, n, e = BBOX
    b = f"({s},{w},{n},{e})"
    q = f"""
[out:json][timeout:200];
(
  way["highway"~"^(motorway|motorway_link|trunk|trunk_link|primary|secondary|tertiary)$"]{b};
  way["railway"~"^(rail)$"]{b};
  node["railway"="level_crossing"]{b};
  nwr["amenity"~"^(bar|pub|nightclub)$"]{b};
  nwr["leisure"="stadium"]{b};
  nwr["aeroway"="helipad"]{b};
  nwr["amenity"="fire_station"]{b};
  nwr["name"~"(Alpha|Beta|Gamma|Delta|Epsilon|Zeta|Theta|Iota|Kappa|Lambda|Sigma|Tau|Phi|Chi|Psi|Omega) (Alpha|Beta|Gamma|Delta|Epsilon|Zeta|Eta|Theta|Iota|Kappa|Lambda|Mu|Nu|Xi|Pi|Rho|Sigma|Tau|Upsilon|Phi|Chi|Psi|Omega)"]{b};
  nwr["building"~"^(fraternity|sorority)$"]{b};
  nwr["landuse"="construction"]{b};
  nwr["building"="construction"]{b};
  way["highway"="construction"]{b};
);
out geom tags;
"""
    data = overpass(q)
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "osm_noise.json").write_text(json.dumps(data))
    print(f"noise: {len(data['elements'])} elements", file=sys.stderr)


def buildings():
    rows = json.loads((ROOT / "listings.json").read_text())
    pts = {(round(r["lat"], 6), round(r["lng"], 6)) for r in rows}
    parts = "".join(f'way["building"](around:45,{lat},{lng});' for lat, lng in pts)
    data = overpass(f"[out:json][timeout:200];({parts});out geom tags;")
    (ROOT / "osm_buildings.json").write_text(json.dumps(data))
    print(f"buildings: {len(data['elements'])} for {len(pts)} points", file=sys.stderr)


if __name__ == "__main__":
    {"noise": noise, "build": buildings}[sys.argv[1]]()
