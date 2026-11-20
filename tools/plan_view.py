import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from plan import WALL, realize, wall_pos  # noqa: E402

FILL = {
    "living": "#f3e3c3", "kitchen": "#f6d6b0", "dining": "#f3e3c3", "hallway": "#e4e4e4", "hall": "#e4e4e4",
    "entry": "#e4e4e4", "bedroom": "#cfe0f3", "bathroom": "#cdeee6", "office": "#e3d7f0", "laundry": "#eeeeee",
}


def segment(wall, rect, a, b, outset=0.0):
    p = wall_pos(rect, wall) + (outset if wall in "es" else -outset)
    return ([a, b], [p, p]) if wall in "ns" else ([p, p], [a, b])


def subtract(lo, hi, cuts):
    pieces, cur = [], lo
    for a, b in sorted(cuts):
        if a > cur:
            pieces.append((cur, min(a, hi)))
        cur = max(cur, b)
    if cur < hi:
        pieces.append((cur, hi))
    return [(a, b) for a, b in pieces if b - a > 0.01]


def door_arc(ax, wall, rect, a, b, color):
    w = b - a
    p = wall_pos(rect, wall)
    inward = -1 if wall in "es" else 1
    if wall in "ns":
        ax.add_patch(Arc((a, p), 2 * w, 2 * w, theta1=0 if inward > 0 else -90, theta2=90 if inward > 0 else 0,
                         color=color, lw=0.8))
        ax.plot([a, a], [p, p + inward * w], color=color, lw=0.8)
    else:
        ax.add_patch(Arc((p, a), 2 * w, 2 * w, theta1=0 if inward > 0 else 90, theta2=90 if inward > 0 else 180,
                         color=color, lw=0.8))
        ax.plot([p, p + inward * w], [a, a], color=color, lw=0.8)


def render(rooms, doors, solution, out, title=None):
    placed = realize(rooms, doors, solution)
    types = {r["id"]: r["type"] for r in rooms}
    conns = solution.get("connections", [])
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect("equal")
    ax.invert_yaxis()
    centers = {}
    cuts = {rid: {w: [] for w in "nesw"} for rid in placed}
    for c in conns:
        (x0, z0), (x1, z1) = c["door"]
        for rid in (c["a"], c["b"]):
            r = placed[rid].rect
            for w in "nesw":
                p = wall_pos(r, w)
                if w in "ns" and abs(z0 - p) < WALL and abs(z1 - p) < WALL:
                    cuts[rid][w].append(tuple(sorted((x0, x1))))
                if w in "ew" and abs(x0 - p) < WALL and abs(x1 - p) < WALL:
                    cuts[rid][w].append(tuple(sorted((z0, z1))))
    for rid, P in sorted(placed.items()):
        x0, x1, z0, z1 = P.rect
        centers[rid] = ((x0 + x1) / 2, (z0 + z1) / 2)
        ax.add_patch(Rectangle((x0, z0), x1 - x0, z1 - z0, facecolor=FILL.get(types[rid], "#eeeeee"), edgecolor="none"))
        for s in P.slots():
            if s.kind == "door" and not any(a < s.b and b > s.a for a, b in cuts[rid][s.wall]):
                cuts[rid][s.wall].append((s.a, s.b))
                door_arc(ax, s.wall, P.rect, s.a, s.b, "#c0392b")
            elif s.kind == "closet":
                cuts[rid][s.wall].append((s.a, s.b))
                xs, zs = segment(s.wall, P.rect, s.a, s.b, WALL / 2)
                ax.plot(xs, zs, color="#8e6b3d", lw=3, ls=(0, (2, 1)), solid_capstyle="butt")
        for w in "nesw":
            lo, hi = P.rect[0:2] if w in "ns" else P.rect[2:4]
            for a, b in subtract(lo - WALL / 2, hi + WALL / 2, cuts[rid][w]):
                xs, zs = segment(w, P.rect, a, b, WALL / 2)
                ax.plot(xs, zs, color="#333333", lw=3.2, solid_capstyle="butt")
        for s in P.windows():
            xs, zs = segment(s.wall, P.rect, s.a, s.b, WALL / 2)
            ax.plot(xs, zs, color="#1f77d0", lw=5, solid_capstyle="butt")
        area = (x1 - x0) * (z1 - z0)
        label = f"{rid}\n{x1 - x0:.1f}x{z1 - z0:.1f} m\n{area:.1f} m²  t{solution[rid]['turn']}"
        ax.text(*centers[rid], label, ha="center", va="center", fontsize=8)
    for c in conns:
        (x0, z0), (x1, z1) = c["door"]
        mid = ((x0 + x1) / 2, (z0 + z1) / 2)
        color = {"door": "#27ae60", "wild": "#2e86c1", "virtual": "#e67e22"}[c["kind"]]
        ax.plot([x0, x1], [z0, z1], color=color, lw=5, alpha=0.8, solid_capstyle="butt")
        for rid in (c["a"], c["b"]):
            ax.plot([centers[rid][0], mid[0]], [centers[rid][1], mid[1]], color=color, lw=1, ls="--")
    ent = solution.get("entrance")
    if ent:
        (x0, z0), (x1, z1) = ent["door"]
        ax.plot([x0, x1], [z0, z1], color="#8e44ad", lw=6, solid_capstyle="butt")
        ax.annotate("entry", ((x0 + x1) / 2, (z0 + z1) / 2), xytext=(6, 6), textcoords="offset points",
                    color="#8e44ad", fontsize=8)
    ax.autoscale_view()
    ax.margins(0.08)
    sb = solution.get("score_breakdown", {})
    parts = [f"{k}={v:.2f}" for k, v in sb.items() if k != "total" and abs(v) > 1e-6]
    head = (title + "  " if title else "") + f"score {sb.get('total', 0):.2f}"
    body = "\n".join(", ".join(parts[i:i + 6]) for i in range(0, len(parts), 6))
    ax.set_title(head + "\n" + body, fontsize=8)
    ax.text(0.01, 0.01, "green: matched doors  blue: door into flexible hall  orange: guessed door\n"
            "red arc: unmatched door  brown dashes: closet  blue wall: window  purple: entry",
            transform=ax.transAxes, fontsize=7, color="#555555")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("rooms")
    ap.add_argument("solution")
    ap.add_argument("--doors")
    ap.add_argument("--out", default="plan.png")
    ap.add_argument("--title")
    args = ap.parse_args()
    rooms = json.loads(Path(args.rooms).read_text())
    doors = json.loads(Path(args.doors).read_text()) if args.doors else {}
    render(rooms, doors, json.loads(Path(args.solution).read_text()), args.out, args.title)
