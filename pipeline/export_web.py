import argparse
import json
import shutil
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
LISTINGS = ROOT / "data" / "listings"
WEB = ROOT / "web" / "public" / "data"
MAX_SIDE = 1600


def save_photo(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return
    im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    im.save(dst, "JPEG", quality=84, optimize=True, progressive=True)


def tour_kind(listing: dict, out: Path) -> str:
    if (out / "scene.json").exists():
        return "generated"
    if listing.get("matterport"):
        return "matterport"
    return "insufficient"


def export(slug: str | None = None):
    WEB.mkdir(parents=True, exist_ok=True)
    index = []
    for d in sorted(p for p in LISTINGS.iterdir() if (p / "listing.json").exists()):
        listing = json.loads((d / "listing.json").read_text())
        out = WEB / listing["id"]
        if slug is None or slug == listing["id"]:
            for p in listing["photos"]:
                save_photo(d / p["file"], out / p["file"])
            if listing.get("floorplan"):
                save_photo(d / listing["floorplan"], out / listing["floorplan"])
        labels = d / "labels.json"
        if labels.exists():
            lab = json.loads(labels.read_text())
            for p in listing["photos"]:
                p.update({k: v for k, v in lab.get(p["file"], {}).items() if k in ("room", "kind")})
        coverage = d / "coverage.json"
        if coverage.exists():
            listing["coverage"] = json.loads(coverage.read_text())
        interior = [p for p in listing["photos"] if p.get("kind", "interior") == "interior"]
        listing["cover"] = (interior or listing["photos"])[0]["file"] if listing["photos"] else ""
        listing["tour"] = tour_kind(listing, out)
        index.append(listing)
    (WEB / "index.json").write_text(json.dumps(index, indent=1))
    print(f"exported {len(index)} listings")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("slug", nargs="?")
    a = ap.parse_args()
    export(a.slug)
