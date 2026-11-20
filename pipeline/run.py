import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "pipeline" / "out"
LISTINGS = ROOT / "data" / "listings"


def stage(name):
    print(f"[{name}]", flush=True)


def room_spec(listing_dir: Path, work: Path, auto: bool) -> dict:
    manual = listing_dir / "rooms.json"
    if manual.exists() and not auto:
        return json.loads(manual.read_text())
    from associate import associate

    result = associate(listing_dir)
    (work / "associate.json").write_text(json.dumps(result, indent=1))
    return {r["id"]: {k: v for k, v in r.items() if k != "id"} for r in result["rooms"]}


def reconstruct_rooms(listing_dir: Path, work: Path, spec: dict, force: bool) -> list[Path]:
    from recon import reconstruct

    runs = []
    for rid, r in spec.items():
        run = work / rid
        if force or not (run / "cameras.json").exists():
            files = [listing_dir / "photos" / f"{p}.jpg" for p in r["photos"]]
            stats = reconstruct(files, run)
            print(f"  {rid}: {stats['n_images']} photos in {stats['inference_s']:.1f}s", flush=True)
        runs.append(run)
    return runs


def plan_sheet(work: Path, sheet: Path):
    plan, rooms = work / "plan.json", work / "rooms_prelayout.json"
    if not plan.exists():
        return
    args = ["python3", str(ROOT / "tools" / "plan_view.py"), str(rooms), str(plan), "--out", str(sheet)]
    if (work / "doors.json").exists():
        args += ["--doors", str(work / "doors.json")]
    subprocess.run(args, check=False, capture_output=True)


def iso_sheet(lid: str, sheet: Path):
    url = f"http://localhost:5173/l/{lid}"
    subprocess.run(["node", str(ROOT / "tools" / "shot.mjs"), url, str(sheet), "wait=5000", "shot"],
                   cwd=ROOT / "tools", check=False, capture_output=True)


def run(lid: str, auto=False, force=False, sheets=True):
    from build_scene import WEB, build, room_specs
    from export_web import export

    listing_dir = LISTINGS / lid
    work = OUT / lid
    stages = work / "stages"
    stages.mkdir(parents=True, exist_ok=True)
    stage("rooms")
    spec = room_spec(listing_dir, work, auto)
    (work / "rooms.json").write_text(json.dumps(spec, indent=1))
    stage("reconstruct")
    runs = reconstruct_rooms(listing_dir, work, spec, force)
    stage("build")
    scene = build(listing_dir, room_specs(spec, runs), WEB / lid)
    print(f"  {len(scene['views'])} views, {len(scene['rooms'])} rooms, {scene.get('floorArea')} ft2", flush=True)
    stage("export")
    export(lid)
    if sheets:
        stage("sheets")
        plan_sheet(work, stages / "3-plan.png")
        iso_sheet(lid, stages / "4-overview.png")
        print(f"  {stages}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("listings", nargs="+")
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-sheets", action="store_true")
    a = ap.parse_args()
    for lid in a.listings:
        run(lid, a.auto, a.force, not a.no_sheets)
