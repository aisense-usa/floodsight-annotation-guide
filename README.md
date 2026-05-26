# FloodSight Annotation Guide

Interactive onboarding guide for annotating drone orthomosaics of smallholder
farmland in Nepal's Terai (NAAMII Agri AI / FAO FloodSight dataset).

**Live:** https://aisense-usa.github.io/floodsight-annotation-guide/

## What's inside
- **How to decide** — a flowchart for picking the right class.
- **Rules & Steps** — the core rules in plain language.
- **Videos** — two screen recordings of the annotation tool.
- **Classes** — interactive examples for every class (raw vs annotated, per-class isolate), pulled from verified tiles.
- **Maps** — pan/zoom index maps of each site with the tile grid.
- **Compare**, **Do & Don't**, **Hard cases**, **Checklist**.

Light / dark theme. Self-contained `index.html` (Lora font + class-example tiles embedded as base64); videos and site maps live under `assets/`.

## Rebuilding
The page is generated:
```
python3 build_guide.py
```
Source data (not in this repo): FloodSight 2 COCO export + the per-site
`tile_index_map_3800.png` + the two screen recordings. `build_guide.py` embeds
the class examples and references `assets/videos/*.mp4` and `assets/maps/*.jpg`.

## Hosting notes
Served by GitHub Pages from `main` / root. All assets are plain commits under
GitHub's 100 MB/file limit (Pages does not serve Git LFS).
