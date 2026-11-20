import sys
from pathlib import Path
from PIL import Image, ImageDraw

d = Path(sys.argv[1])
out = sys.argv[2]
files = sorted((d / "photos").glob("*.jpg"))
cols, tw, th = 8, 220, 150
rows = (len(files) + cols - 1) // cols
sheet = Image.new("RGB", (cols * tw, rows * th), "white")
dr = ImageDraw.Draw(sheet)
for i, f in enumerate(files):
    im = Image.open(f).convert("RGB")
    im.thumbnail((tw - 4, th - 18))
    x, y = (i % cols) * tw, (i // cols) * th
    sheet.paste(im, (x + 2, y + 16))
    dr.text((x + 4, y + 2), f.stem, fill="black")
sheet.save(out, quality=80)
