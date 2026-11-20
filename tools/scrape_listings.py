"""Pull every Ann Arbor rental from property managers' public AppFolio pages.

Writes atlas/data/listings.json. Only manager-run pages are read, no aggregators.
"""

import html
import json
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

MANAGERS = {
    "varsitymanagement": "Varsity Management",
    "crmgmtinc": "Campus Realty",
    "oldtownrealty": "Old Town Realty",
    "barnes": "Barnes & Barnes",
    "jmzmanagement": "JMZ Management",
    "howardhanna": "Howard Hanna",
}

OUT = Path(__file__).resolve().parents[1] / "atlas" / "data" / "listings.json"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}


def get(url: str) -> str:
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                return r.read().decode("utf-8", "replace")
        except Exception:
            time.sleep(2 + attempt * 3)
    return ""


def flat(fragment: str) -> str:
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", fragment, flags=re.S)
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"<[^>]+>", "|", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return re.sub(r"(\s*\|\s*)+", "|", text)


def money(s: str | None):
    if not s:
        return None
    m = re.search(r"\$([\d,]+(?:\.\d+)?)", s)
    return round(float(m.group(1).replace(",", ""))) if m else None


def split_list(s: str | None):
    if not s:
        return []
    return [x.strip() for x in re.split(r",|\|", s) if x.strip() and x.strip() != "..."]


def field(text: str, label: str, stop: str = r"\|") -> str | None:
    m = re.search(rf"{label}:?\|?\s*(.+?){stop}", text)
    return m.group(1).strip() if m else None


def parse_card(card: str):
    text = flat(card)
    available = None
    m = re.search(r"\|(Available [^|]+|Unit Unavailable|Coming Soon[^|]*)\|", text)
    if m:
        available = m.group(1).strip()
    title = None
    tm = re.search(r'class="[^"]*listing-item__title[^"]*"[^>]*>(.*?)</', card, flags=re.S)
    if tm:
        title = html.unescape(re.sub(r"<[^>]+>", "", tm.group(1))).strip() or None
    return {
        "title": title,
        "available": available,
        "utilitiesIncluded": split_list(field(text, "Utilities included")),
        "utilitiesExcluded": split_list(field(text, "Utilities not included")),
    }


def parse_detail(page: str):
    text = flat(page)
    photos = []
    for m in re.finditer(r"https://images\.cdn\.appfolio\.com/[^\"' )]+?/large\.(?:jpg|jpeg|png|webp)", page):
        if m.group(0) not in photos:
            photos.append(m.group(0))
    desc = None
    dm = re.search(r'class="[^"]*listing-detail__description[^"]*"[^>]*>(.*?)</p>', page, flags=re.S)
    if dm:
        desc = html.unescape(re.sub(r"<br\s*/?>", "\n", dm.group(1)))
        desc = re.sub(r"<[^>]+>", "", desc).strip()
    def section(name: str):
        m = re.search(rf"\|{name}\|(.+?)\|(?:Rental Terms|Pet Policy|Amenities|Appliances|Apply Now|Contact|Schedule|Share|Similar)", text)
        return [x.strip() for x in m.group(1).split("|") if x.strip()] if m else []
    pets = section("Pet Policy")
    terms = section("Rental Terms")
    terms_text = "|".join(terms)
    lease = field("|" + terms_text + "|", "Lease Length") or field(text, "Lease Length")
    return {
        "photos": photos,
        "description": desc,
        "deposit": money(field("|" + terms_text + "|", "Deposit")),
        "applicationFee": money(field("|" + terms_text + "|", "Application Fee")),
        "lease": lease,
        "amenities": section("Amenities"),
        "appliances": section("Appliances"),
        "pets": pets,
    }


def scrape_manager(sub: str, name: str, today: str):
    base = f"https://{sub}.appfolio.com"
    page = get(f"{base}/listings")
    mm = re.search(r"markers: (\[.*?\])\s*,?\s*\n", page) or re.search(r"markers: (\[.*\])", page)
    markers = json.loads(mm.group(1)) if mm else []
    cards = {}
    for m in re.finditer(r'id="listing_(\d+)"(.*?)(?=id="listing_\d+"|$)', page, flags=re.S):
        cards[int(m.group(1))] = parse_card(m.group(2)[:20000])
    out = []
    for marker in markers:
        address = marker.get("address") or ""
        if "Ann Arbor" not in address:
            continue
        specs = marker.get("unit_specs") or ""
        beds = re.search(r"([\d.]+) bd", specs)
        baths = re.search(r"([\d.]+) ba", specs)
        sqft = re.search(r"([\d,]+) Sq", specs)
        studio = "Studio" in specs
        out.append(
            {
                "manager": name,
                "address": address,
                "lat": marker["latitude"],
                "lng": marker["longitude"],
                "rent": money(marker.get("rent_range")),
                "beds": 0 if studio else (float(beds.group(1)) if beds else None),
                "baths": float(baths.group(1)) if baths else None,
                "sqft": int(sqft.group(1).replace(",", "")) if sqft else None,
                "cover": marker.get("default_photo_url"),
                "url": base + marker["detail_page_url"],
                "accessed": today,
                **cards.get(marker.get("listing_id"), {}),
            }
        )
    return out


def main():
    today = date.today().isoformat()
    rows = []
    for sub, name in MANAGERS.items():
        found = scrape_manager(sub, name, today)
        print(f"{name}: {len(found)}", file=sys.stderr)
        rows.extend(found)
        time.sleep(1)

    def enrich(row):
        time.sleep(0.4)
        row.update(parse_detail(get(row["url"])))
        return row

    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(enrich, rows))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=1))
    print(f"wrote {len(rows)} listings to {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
