# Openhouse

Ann Arbor rentals on a map, with 3D walkthroughs built from listing photos.

```
app/ components/ lib/ public/   the map site (Next.js), host this
data/                            scraped listings and map data
viewer/                          3D tour viewer (Vite), embedded in the listing modal
pipeline/                        Python: scraping, map data, photos to 3D
```

Run it

```
npm install && npm run dev                 # map on :3000
npm install --prefix viewer && npm run dev --prefix viewer -- --port 5173
```

Rebuild map data: `uv run --project pipeline python pipeline/build_atlas.py`
