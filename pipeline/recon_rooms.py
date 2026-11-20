import json
import sys
from pathlib import Path

from recon import reconstruct

listing = Path(sys.argv[1]).resolve()
rooms = json.loads((listing / "rooms.json").read_text())
out = Path(__file__).parent / "out" / listing.name
only = set(sys.argv[2:])
for rid, r in rooms.items():
    if only and rid not in only:
        continue
    files = [listing / "photos" / f"{p}.jpg" for p in r["photos"]]
    stats = reconstruct(files, out / rid)
    print(rid, stats["n_images"], stats["inference_s"], flush=True)
