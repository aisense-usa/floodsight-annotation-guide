#!/usr/bin/env python3
"""Build an interactive Nepal Farm annotation guide from the FloodSight 2
verified COCO export. Self-contained tabbed HTML: per-class example tiles with
raw|annotated view, overlay on/off, isolate-one-class, compare mode, plus a
Do/Don't gallery of verified infographics. Plain-manual style, light + deep-navy
dark theme. Read-only over the dataset; writes annotation_guide_v0.4.html.
"""
import json, os, base64, io
from collections import defaultdict
import numpy as np
import cv2
from PIL import Image

COCO   = "/Users/jc/Downloads/FloodSight 2/train/_annotations.coco.json"
IMGDIR = "/Users/jc/Downloads/FloodSight 2/train"
DLDIR  = "/Users/jc/Downloads"
OUT    = "/Users/jc/Downloads/floodsight-annotation-guide/index.html"

# Onboarding videos hosted on Google Drive (embedded via /preview iframe)
VIDEOS = [
 {"title": "Annotation Example 1", "desc": "A complete walkthrough of annotating tiles, start to finish.",
  "drive": "1r9JH2PTy91cnloy-kHa5twPbflCkDPjV"},
 {"title": "Annotation Example 2", "desc": "A shorter look at the annotation tool and how to use it.",
  "drive": "1z7Jtgngo6o2g8lCsHBuY1xzbKPfw0KnY"},
]
# Maps: existing public XYZ slippy-map tiles (FAO BIPAD) on GCS, zoom 10-19, pre+post.
TILE_BASE = "https://storage.googleapis.com/fao-bipad/tiles"
MAPS = [  # bounds = [[south,west],[north,east]] from manifest clean-tile centroids
 {"code": "HK", "name": "Hanumannagar Kankalini", "bounds": [[26.5416, 86.8934], [26.5755, 86.9193]]},
 {"code": "HR", "name": "Harinagar",              "bounds": [[26.5203, 87.1345], [26.5448, 87.1530]]},
 {"code": "SS", "name": "Sunsari Saptakoshi",     "bounds": [[26.7049, 86.9523], [26.7219, 86.9718]]},
]
MANIFEST_3800 = {  # for the red 3800-tile grid overlay
 "HK": "/Volumes/PortableSSD/AI Sense/Nepal/GeoAI/Nepal_Flood/HK/pre_flood/manifest_3800.json",
 "HR": "/Volumes/PortableSSD/AI Sense/Nepal/GeoAI/Nepal_Flood/HR/pre_flood/manifest_3800.json",
 "SS": "/Volumes/PortableSSD/AI Sense/Nepal/GeoAI/Nepal_Flood/SS/pre_flood/manifest_3800.json",
}

OUT_PX   = 520
JPEG_Q   = 84
SIMPLIFY = 2.0
N_PER    = 3

# The 6 verified Do/Don't infographics (deduped by md5). kind: tool | compare
DODONT = [
 {"file": "ChatGPT Image May 25, 2026 at 11_28_03 PM.png", "kind": "tool",
  "title": "Use the right tool",
  "take": "Polygon is preferred. Smart Select when it genuinely helps, Bounding Box only for box shapes. Avoid the Brush, it makes messy, inconsistent labels."},
 {"file": "ChatGPT Image May 25, 2026 at 11_23_12 PM.png", "kind": "compare",
  "title": "Only target classes, no gaps",
  "take": "Trace clean boundaries on target classes only. Never brush vegetation, and never leave a required area empty."},
 {"file": "ChatGPT Image May 25, 2026 at 11_23_19 PM.png", "kind": "compare",
  "title": "Keep it simple and clean",
  "take": "No vegetation brushing, no overlapping regions. Read the raw image first, then annotate only the segments that are needed."},
 {"file": "ChatGPT Image May 25, 2026 at 11_23_26 PM.png", "kind": "compare",
  "title": "Patchy is not marsh",
  "take": "A patchy-looking field is still a GROWING_FIELD. Follow the ridge structure, not the surface texture."},
 {"file": "ChatGPT Image May 25, 2026 at 11_23_29 PM.png", "kind": "compare",
  "title": "Follow the ridge segments",
  "take": "Trace the boundary that aligns with the true ridges. A loose boundary that ignores the ridges is wrong."},
 {"file": "purple_road_wrong_annotation.png", "kind": "compare",
  "title": "Do not mislabel a field edge as ROAD",
  "take": "A purple line was tagged ROAD where there is no road surface. Relabel it as the field boundary instead."},
]
DODONT_MAXW = 1180

# Class colors (exact Roboflow palette the user picked) + simple guidance.
# [color, mode, def, cue, do, dont]
CLASS_INFO = {
 "GROWING_FIELD":   ["#75FBD1","draw","A field with healthy plants growing in it.","Green and leafy, clearly growing.","draw each field on its own, up to its edge","joining two fields into one"],
 "BARREN_FIELD":    ["#EF8733","draw","A field with an edge, but only bare soil. Nothing growing.","Brown, empty soil inside a field edge.","empty or harvested fields","mixing it up with GROUND (ground has no field edge)"],
 "NONCROP_FIELD":   ["#52B4E6","draw","A field with an edge and some growth, but not a clear crop and not wet.","Clearly a field, but not green crop, not bare, not wet.","fields that do not fit the other three","using it for plain ground with no edge"],
 "MARSH_FIELD":     ["#FFFF54","draw","A wet field. Water or black wet patches inside a field edge.","Shiny water or dark wet ground in a field.","wet or flooded fields","using it for open water with no field"],
 "HOUSE":           ["#D2FB50","draw","A building roof.","Square roof with sharp edges and shadow.","each building","adding the yard around it"],
 "ROAD":            ["#7B2AF5","draw","A road, track, or path.","A long line that joins places.","roads and paths","marking field edges as roads"],
 "WATER":           ["#F5C242","draw","Open water, like a river or pond.","Smooth, dark, no plants on top.","clear open water","including weedy marsh"],
 "GROUND":          ["#0000F5","draw","Bare earth next to a house. No field edge around it.","Open dirt or yard by a house.","bare ground near houses","using it for empty fields (that is BARREN_FIELD)"],
 "SAND":            ["#965635","draw","Soft, pale sand near water or a riverbed.","Light, soft-looking, near water.","river sand and flood sand","mixing it up with hard bare ground"],
 "TREES":           ["#EA33F7","draw","Trees or groups of trees.","Round leafy tops, tall, with shadow.","trees and tree groups","including low grass or bush"],
 "MARSH_VEGETATION":["#242424","explicit","Green plants on wet, muddy ground. This green you DO draw.","Green growth over water or wet soil.","plants on wet ground","leaving them blank like normal grass"],
 "VEGETATION":      ["#3778F5","nodraw","Plain grass and bush. Do NOT draw it. Leave it blank.","Normal grass or bush, not on wet ground.","nothing, leave it blank","drawing it"],
 "EDGECASE_IGNORE": ["#7F1786","explicit","Anything you cannot understand.","You truly cannot tell what it is.","mark confusing spots here","guessing, or leaving them blank"],
}
LORA_NORMAL = "/tmp/lora-normal.woff2"
LORA_ITALIC = "/tmp/lora-italic.woff2"
GROUPS = {
 "fields":  ["GROWING_FIELD","BARREN_FIELD","NONCROP_FIELD","MARSH_FIELD"],
 "built":   ["HOUSE","ROAD"],
 "water":   ["WATER"],
 "ground":  ["GROUND","SAND"],
 "veg":     ["TREES","MARSH_VEGETATION"],
 "ignore":  ["EDGECASE_IGNORE"],
}
GROUP_TITLES = {"fields":"Fields","built":"Built structures","water":"Water",
                "ground":"Ground and sand","veg":"Trees and marsh vegetation",
                "ignore":"Ignore"}
PLACEHOLDER = set()
VEG_RAW_N = 10   # raw vegetation example tiles for the top section

PILOT = "/Users/jc/Downloads/nepal-farm-pilot/train"   # 2 fully-annotated example tiles


# ---- COCO compressed-RLE decoder (pure python) ----
def rle_counts_from_str(s):
    s = s.encode("ascii") if isinstance(s, str) else s
    cnts, p, m = [], 0, len(s)
    while p < m:
        x, k, more = 0, 0, 1
        while more:
            c = s[p] - 48
            x |= (c & 0x1f) << (5 * k)
            more = c & 0x20
            p += 1; k += 1
            if not more and (c & 0x10):
                x |= (-1 << (5 * k))
        if len(cnts) > 2:
            x += cnts[-2]
        cnts.append(x)
    return cnts

def rle_to_mask(seg):
    h, w = seg["size"]
    cnts = rle_counts_from_str(seg["counts"])
    flat = np.zeros(h * w, dtype=np.uint8)
    idx, val = 0, 0
    for c in cnts:
        flat[idx:idx + c] = val
        idx += c; val ^= 1
    return flat.reshape((h, w), order="F")

def mask_to_polys(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        if cv2.contourArea(c) < 25:
            continue
        c = cv2.approxPolyDP(c, SIMPLIFY, True)
        if len(c) >= 3:
            out.append(c.reshape(-1, 2).astype(float).tolist())
    return out

def poly_from_seg(seg):
    if isinstance(seg, dict):
        return mask_to_polys(rle_to_mask(seg))
    polys = []
    for ring in seg:
        pts = np.array(ring, dtype=np.float32).reshape(-1, 2)
        if len(pts) >= 3:
            c = cv2.approxPolyDP(pts.reshape(-1,1,2), SIMPLIFY, True)
            polys.append(c.reshape(-1, 2).astype(float).tolist())
    return polys

def embed_image(path, max_w, q=82):
    im = Image.open(path).convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=q)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), im.width, im.height

def make_dzi(src_path, out_dir, name, tile=510, overlap=1, q=90):
    """Generate a Deep Zoom (.dzi + _files/) pyramid for OpenSeadragon. Returns (W,H,n_tiles)."""
    import math
    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(src_path).convert("RGB")
    W, H = im.size
    max_level = math.ceil(math.log2(max(W, H)))
    files_dir = os.path.join(out_dir, name + "_files")
    n = 0
    for level in range(max_level + 1):
        scale = 2 ** (max_level - level)
        lw, lh = max(1, math.ceil(W / scale)), max(1, math.ceil(H / scale))
        lim = im if (lw, lh) == (W, H) else im.resize((lw, lh), Image.LANCZOS)
        ld = os.path.join(files_dir, str(level)); os.makedirs(ld, exist_ok=True)
        cols, rows = math.ceil(lw / tile), math.ceil(lh / tile)
        for row in range(rows):
            for col in range(cols):
                x, y = col * tile, row * tile
                box = (max(0, x - overlap), max(0, y - overlap),
                       min(lw, x + tile + overlap), min(lh, y + tile + overlap))
                lim.crop(box).save(os.path.join(ld, f"{col}_{row}.jpg"), "JPEG", quality=q)
                n += 1
    with open(os.path.join(out_dir, name + ".dzi"), "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n'
                f'<Image TileSize="{tile}" Overlap="{overlap}" Format="jpg" '
                'xmlns="http://schemas.microsoft.com/deepzoom/2008">'
                f'<Size Width="{W}" Height="{H}"/></Image>')
    return W, H, n


def main():
    d = json.load(open(COCO))
    id2name = {c["id"]: c["name"] for c in d["categories"]}
    imgs = {i["id"]: i for i in d["images"]}

    per_img = defaultdict(list)
    cls_imgs = defaultdict(set)
    for a in d["annotations"]:
        cn = id2name[a["category_id"]]
        if cn == "objects-p8wr" or a["category_id"] == 0:
            continue
        im = imgs[a["image_id"]]
        W, H = im["width"], im["height"]
        polys = poly_from_seg(a["segmentation"])
        if not polys:
            continue
        bb = a["bbox"]; frac = (bb[2] * bb[3]) / (W * H)
        per_img[a["image_id"]].append((cn, polys, frac))
        cls_imgs[cn].add(a["image_id"])

    richness = {iid: len({c for c, _, _ in anns}) for iid, anns in per_img.items()}

    selected = {}
    for cn in CLASS_INFO:
        if cn in PLACEHOLDER:
            selected[cn] = []; continue
        cand = []
        for iid in cls_imgs.get(cn, []):
            fr = max(f for c, _, f in per_img[iid] if c == cn)
            cand.append((iid, fr, richness[iid]))
        good = [c for c in cand if 0.04 <= c[1] <= 0.78]
        pool = good if len(good) >= 2 else cand
        pool.sort(key=lambda c: (round(min(c[1], 0.6), 2), c[2]), reverse=True)
        picks, seen = [], set()
        for iid, fr, r in pool:
            if iid in seen: continue
            seen.add(iid); picks.append(iid)
            if len(picks) >= N_PER: break
        selected[cn] = picks

    needed = sorted({iid for v in selected.values() for iid in v})
    images_payload, sid = {}, {}
    for iid in needed:
        im = imgs[iid]; fn = im["file_name"]
        pim = Image.open(os.path.join(IMGDIR, fn)).convert("RGB")
        scale = OUT_PX / max(pim.width, pim.height)
        ow, oh = round(pim.width * scale), round(pim.height * scale)
        pim = pim.resize((ow, oh), Image.LANCZOS)
        buf = io.BytesIO(); pim.save(buf, "JPEG", quality=JPEG_Q)
        b64 = base64.b64encode(buf.getvalue()).decode()
        polys = []
        for cn, plist, fr in per_img[iid]:
            for ring in plist:
                polys.append({"cls": cn, "pts": [[round(x*scale,1), round(y*scale,1)] for x, y in ring]})
        key = "t%d" % iid; sid[iid] = key
        images_payload[key] = {"src": "data:image/jpeg;base64," + b64,
                               "w": ow, "h": oh, "polys": polys,
                               "name": fn.split(".rf.")[0]}

    examples = {cn: [{"img": sid[iid], "focus": cn} for iid in v]
                for cn, v in selected.items()}

    # 10 raw vegetation tiles (no overlay) for the top section: most veg, least else
    veg_score = {}
    for iid, anns in per_img.items():
        veg = sum(f for c, _, f in anns if c == "VEGETATION")
        other = sum(f for c, _, f in anns if c != "VEGETATION")
        if veg > 0:
            veg_score[iid] = (veg, -other)
    veg_ids = sorted(veg_score, key=lambda i: veg_score[i], reverse=True)[:VEG_RAW_N]
    veg_raw = [embed_image(os.path.join(IMGDIR, imgs[iid]["file_name"]),
                           560 if k < 4 else 340, q=82)[0]      # first 4 bigger (2x2 large)
               for k, iid in enumerate(veg_ids)]

    dodont = []
    for item in DODONT:
        path = os.path.join(DLDIR, item["file"])
        if not os.path.exists(path):
            print("  WARNING missing Do/Don't image:", item["file"]); continue
        src, w, h = embed_image(path, DODONT_MAXW, q=82)
        dodont.append({"title": item["title"], "take": item["take"],
                       "kind": item["kind"], "src": src, "w": w, "h": h})

    # complete-segment pilot tiles (fully annotated) for the Demos & Examples tab
    pilot = []
    pj = os.path.join(PILOT, "_annotations.coco.json")
    if os.path.exists(pj):
        pd = json.load(open(pj)); pid2n = {c["id"]: c["name"] for c in pd["categories"]}
        for im_rec in pd["images"]:
            W0 = im_rec["width"]
            pim = Image.open(os.path.join(PILOT, im_rec["file_name"])).convert("RGB")
            sc = 820 / max(pim.width, pim.height)
            ow, oh = round(pim.width*sc), round(pim.height*sc)
            pim2 = pim.resize((ow, oh), Image.LANCZOS)
            buf = io.BytesIO(); pim2.save(buf, "JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode()
            psc = ow / W0
            polys = []
            for a in pd["annotations"]:
                if a["image_id"] != im_rec["id"]: continue
                cn = pid2n[a["category_id"]]
                if cn not in CLASS_INFO: continue
                for ring in poly_from_seg(a["segmentation"]):
                    polys.append({"cls": cn, "pts": [[round(x*psc,1), round(y*psc,1)] for x, y in ring]})
            key = "pilot%d" % im_rec["id"]
            images_payload[key] = {"src": "data:image/jpeg;base64," + b64, "w": ow, "h": oh,
                                   "polys": polys, "name": im_rec["file_name"].split(".rf.")[0]}
            pilot.append({"img": key, "focus": "__all__"})

    # red 3800-tile grid (UTM bounds -> lat/lon) for the Leaflet overlay
    from pyproj import Transformer
    _tf = Transformer.from_crs("EPSG:32645", "EPSG:4326", always_xy=True)
    map_grid = {}
    for code, mpath in MANIFEST_3800.items():
        cells = []
        for t in json.load(open(mpath))["tiles"]:
            e0, n0, e1, n1 = t["bounds_utm"]
            w_, s_ = _tf.transform(e0, n0); e_, n_ = _tf.transform(e1, n1)
            cells.append([round(s_, 6), round(w_, 6), round(n_, 6), round(e_, 6), t["name"].split("_")[-1]])
        map_grid[code] = cells
        print(f"  grid {code}: {len(cells)} cells")

    data = {"images": images_payload, "examples": examples,
            "colors": {k: v[0] for k, v in CLASS_INFO.items()},
            "info": {k: {"mode": v[1], "def": v[2], "cue": v[3], "do": v[4], "dont": v[5]}
                     for k, v in CLASS_INFO.items()},
            "groups": GROUPS, "groupTitles": GROUP_TITLES,
            "placeholder": sorted(PLACEHOLDER), "dodont": dodont,
            "vegRaw": veg_raw, "vegColor": CLASS_INFO["VEGETATION"][0],
            "marshVeg": examples.get("MARSH_VEGETATION", []),
            "videos": VIDEOS, "maps": MAPS, "tileBase": TILE_BASE,
            "mapGrid": map_grid, "pilot": pilot}

    # embed Lora woff2 (variable weight) as base64 @font-face, self-contained
    def font_b64(p):
        return base64.b64encode(open(p, "rb").read()).decode()
    fonts = """
@font-face{font-family:'Lora';font-style:normal;font-weight:400 700;font-display:swap;
  src:url(data:font/woff2;base64,%s) format('woff2');}
@font-face{font-family:'Lora';font-style:italic;font-weight:400 700;font-display:swap;
  src:url(data:font/woff2;base64,%s) format('woff2');}
""" % (font_b64(LORA_NORMAL), font_b64(LORA_ITALIC))

    html = (HTML_TEMPLATE
            .replace("/*__FONTS__*/", fonts)
            .replace("/*__DATA__*/", json.dumps(data)))
    open(OUT, "w").write(html)

    print("Wrote", OUT, "(%.2f MB)" % (os.path.getsize(OUT)/1e6))
    print("Lora embedded:", round((os.path.getsize(LORA_NORMAL)+os.path.getsize(LORA_ITALIC))/1024), "KB woff2")
    print("Embedded unique tiles:", len(images_payload), "| Do/Don't:", len(dodont), "| veg-raw:", len(veg_raw))
    print("Group order:", " > ".join(GROUPS.keys()))
    for g, cls in GROUPS.items():
        for cn in cls:
            n = len(examples[cn])
            flag = "  <-- PLACEHOLDER" if cn in PLACEHOLDER else ("  <-- only %d" % n if n < 2 else "")
            print("  %-20s %d%s" % (cn, n, flag))


HTML_TEMPLATE = r"""<!DOCTYPE html><html lang="en" data-theme="light"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Nepal Farm Annotation Guide</title>
<style>
/*__FONTS__*/
:root{
  --serif:'Lora',Georgia,'Times New Roman',serif;
  --sans:'Lora',Georgia,serif;
  --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
}
html[data-theme="light"]{
  --bg:oklch(0.985 0.004 80); --raise:oklch(0.965 0.005 80); --field:oklch(0.955 0.006 80);
  --ink:oklch(0.25 0.015 75); --muted:oklch(0.48 0.012 75); --line:oklch(0.87 0.008 80);
  --accent:oklch(0.47 0.11 255); --good:oklch(0.48 0.11 150); --bad:oklch(0.52 0.15 28);
  --canvas:oklch(0.16 0.01 75);
}
html[data-theme="dark"]{
  --bg:oklch(0.19 0.03 255); --raise:oklch(0.235 0.03 255); --field:oklch(0.275 0.03 255);
  --ink:oklch(0.93 0.018 255); --muted:oklch(0.70 0.025 255); --line:oklch(0.33 0.03 255);
  --accent:oklch(0.82 0.11 250); --good:oklch(0.80 0.13 150); --bad:oklch(0.76 0.14 28);
  --canvas:oklch(0.14 0.025 255);
}
*{box-sizing:border-box}
body{font-family:var(--sans);background:var(--bg);color:var(--ink);margin:0;line-height:1.65;
  font-size:16px;-webkit-font-smoothing:antialiased;transition:background .3s ease,color .3s ease}
.wrap{max-width:1060px;margin:0 auto;padding:0 24px 120px}
.reading{max-width:68ch}
h1{font-family:var(--serif);font-weight:600;font-size:34px;letter-spacing:-.01em;line-height:1.15;margin:0 0 6px}
h2{font-family:var(--serif);font-weight:600;font-size:23px;letter-spacing:-.01em;margin:0 0 4px}
h3{font-family:var(--sans);font-weight:650;font-size:16px;margin:0}
p,li{font-size:16px}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
b,strong{font-weight:650}
code{font-family:var(--mono);background:var(--field);padding:1.5px 6px;border-radius:5px;font-size:13.5px}
.eyebrow{font-size:11.5px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);margin:0 0 6px}
.lede{font-size:17px;color:var(--muted);max-width:62ch;margin:6px 0 0}
.muted{color:var(--muted)}

/* hero */
.hero{padding:30px 0 18px}

/* sticky tab bar */
header{position:sticky;top:0;z-index:40;background:var(--bg);border-bottom:1px solid var(--line)}
.barwrap{max-width:1060px;margin:0 auto;padding:0 24px}
.brandrow{display:flex;align-items:center;gap:14px;padding:10px 0 8px}
.brand{font-family:var(--serif);font-weight:600;font-size:23px;letter-spacing:-.01em}
.brand .dot{color:var(--accent)}
.spacer{flex:1}
.toggle{font-size:12.5px;font-weight:600;color:var(--ink);background:transparent;border:1px solid var(--line);
  border-radius:8px;padding:6px 12px;cursor:pointer;display:flex;align-items:center;gap:7px;white-space:nowrap}
.toggle:hover{border-color:var(--accent)}
.toggle svg{width:15px;height:15px}
.tabs{display:flex;gap:2px;overflow-x:auto;scrollbar-width:none}
.tabs::-webkit-scrollbar{display:none}
.tab{font-family:var(--sans);font-size:14px;font-weight:600;color:var(--muted);background:transparent;border:0;
  padding:11px 15px;cursor:pointer;white-space:nowrap;border-bottom:2px solid transparent;margin-bottom:-1px}
.tab:hover{color:var(--ink)}
.tab.active{color:var(--ink);border-bottom-color:var(--accent)}

/* panels */
.panel{display:none;padding-top:34px}
.panel.active{display:block;animation:fade .25s ease}
@keyframes fade{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}
.panel > h1:first-child,.panel > h2:first-child{margin-top:0}

/* plain rule blocks: heading + hairline + text, no fills */
.block{margin:0 0 30px}
.block > h2,.block > h3{margin-bottom:8px}
.block .hr{height:1px;background:var(--line);margin:0 0 12px}
.block p{margin:0 0 10px}
.block p:last-child{margin-bottom:0}
.draw-note{color:var(--good);font-weight:650}
.lead-rule{margin:8px 0 36px}
.lead-rule h2{font-size:27px;margin-bottom:10px}

/* steps */
.steps{counter-reset:s;list-style:none;padding:0;margin:0}
.steps li{counter-increment:s;padding:13px 0 13px 44px;position:relative;border-top:1px solid var(--line)}
.steps li:first-child{border-top:0}
.steps li::before{content:counter(s);position:absolute;left:0;top:12px;width:26px;height:26px;border-radius:50%;
  border:1px solid var(--line);color:var(--accent);font-family:var(--serif);font-weight:600;font-size:14px;
  display:flex;align-items:center;justify-content:center}

/* class catalog */
.grp{margin:0 0 8px}
.grp-title{font-size:12px;font-weight:700;letter-spacing:.13em;text-transform:uppercase;color:var(--muted);
  margin:26px 0 14px;padding-bottom:7px;border-bottom:1px solid var(--line)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(470px,1fr));gap:30px 26px}
.cls{}
.cls .top{display:flex;align-items:center;gap:9px;margin-bottom:3px}
.sw{width:12px;height:12px;border-radius:3px;flex:0 0 auto;box-shadow:0 0 0 1px oklch(0 0 0 / .18) inset}
.cls-name{font-family:var(--mono);font-size:14.5px;font-weight:600}
.tag{font-size:10.5px;font-weight:700;text-transform:uppercase;letter-spacing:.05em;margin-left:auto}
.tag.draw{color:var(--good)} .tag.nodraw{color:var(--bad)} .tag.explicit{color:var(--accent)}
.def{font-size:14px;margin:5px 0 1px}
.cue{font-size:13px;color:var(--muted)}
.ud{display:flex;flex-direction:column;gap:2px;margin:8px 0 0;font-size:13px}
.ud span{display:flex;gap:8px}
.ud em{font-style:normal;font-weight:700;font-size:10px;letter-spacing:.06em;text-transform:uppercase;flex:0 0 42px;padding-top:2px}
.ud .do em{color:var(--good)} .ud .av em{color:var(--bad)}

/* viewer */
.viewer{margin-top:12px}
.panes{display:flex;gap:10px;flex-wrap:wrap}
.pane{flex:1 1 200px;min-width:170px}
.pane .plbl{font-size:10px;color:var(--muted);margin-bottom:4px;text-transform:uppercase;letter-spacing:.09em;font-weight:600}
canvas{width:100%;height:auto;border:1px solid var(--line);border-radius:7px;background:var(--canvas);display:block;cursor:zoom-in}
.ctrls{display:flex;align-items:center;gap:15px;flex-wrap:wrap;margin-top:9px;font-size:12.5px}
.ctrls label{display:flex;align-items:center;gap:6px;color:var(--muted);cursor:pointer}
.ctrls input[type=checkbox]{accent-color:var(--accent)}
.ctrls select{font-family:var(--mono);background:var(--field);color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:4px 7px;font-size:12px}
.thumbs{display:flex;gap:7px;margin-top:9px;flex-wrap:wrap}
.thumbs button{width:54px;height:54px;border-radius:7px;border:2px solid var(--line);background-size:cover;background-position:center;cursor:pointer;padding:0;opacity:.55;transition:opacity .2s,border-color .2s}
.thumbs button.active{border-color:var(--accent);opacity:1}
.tname{font-family:var(--mono);font-size:10px;color:var(--muted);margin-top:6px}
.ph{margin-top:11px;border:1.5px dashed var(--line);border-radius:9px;min-height:110px;display:flex;flex-direction:column;gap:4px;
  align-items:center;justify-content:center;text-align:center;color:var(--muted);font-size:12.5px;padding:16px}
.ph b{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--accent)}

/* compare */
.compare{display:flex;gap:22px;flex-wrap:wrap}
.cbox{flex:1 1 380px}
.cbox > select{font-family:var(--mono);width:100%;background:var(--field);color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:8px 10px;font-size:13.5px;font-weight:600;margin-bottom:12px}

/* vegetation top block */
.vegtop{display:flex;gap:26px;flex-wrap:wrap;margin:6px 0 34px;padding:22px;border:1px solid var(--line);border-radius:14px;background:var(--raise)}
.vegtop .vmain{flex:2 1 460px}
.vegtop .vaside{flex:1 1 290px;border-top:1px solid var(--line);padding-top:16px}
@media(min-width:780px){.vegtop .vaside{border-top:0;border-left:1px solid var(--line);padding-top:0;padding-left:26px}}
.vrule{font-size:21px;color:var(--bad);margin:0 0 8px}
.vsub{font-weight:650;margin:16px 0 9px;font-size:15px}
.vgrid{display:grid;grid-template-columns:repeat(6,1fr);gap:7px}
.vgrid img{width:100%;height:auto;border:1px solid var(--line);border-radius:6px;display:block;cursor:zoom-in}
.vgrid-big{display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin-bottom:10px}
.vgrid-big img{width:100%;height:auto;border:1px solid var(--line);border-radius:9px;display:block;cursor:zoom-in}
@media(max-width:620px){.vgrid{grid-template-columns:repeat(3,1fr)}.vgrid-big{grid-template-columns:repeat(2,1fr)}}

/* decision flowchart */
.obvious{margin:0 0 22px;padding:16px 18px;border:1px solid var(--line);border-radius:12px;background:var(--raise)}
.obvious .ohead{font-weight:650;margin-bottom:10px;font-size:15px}
.obvious .orow{display:flex;flex-wrap:wrap;gap:10px 18px}
.obvious .oitem{display:flex;align-items:center;gap:8px;font-size:14px}
.qcard{border:1px solid var(--line);border-radius:12px;padding:16px 20px;background:var(--raise);margin:0 0 16px;text-align:center}
.qcard .q{font-family:var(--serif);font-size:19px;font-weight:600}
.branches{display:flex;gap:18px;flex-wrap:wrap}
.branch{flex:1 1 320px;border:1px solid var(--line);border-radius:12px;padding:14px 18px}
.branch .bhead{font-weight:700;font-size:12.5px;letter-spacing:.05em;text-transform:uppercase;margin-bottom:6px}
.branch .bhead.no{color:var(--bad)} .branch .bhead.yes{color:var(--good)}
.outcome{display:flex;align-items:center;gap:10px;flex-wrap:wrap;padding:11px 0;border-top:1px solid var(--line)}
.outcome:first-of-type{border-top:0}
.outcome .cond{flex:1 1 180px;font-size:14.5px}
.outcome .arr{color:var(--muted);font-size:16px}
.chip{display:inline-flex;align-items:center;gap:7px;font-family:var(--mono);font-size:13px;font-weight:600;
  padding:5px 11px;border-radius:999px;border:1.6px solid var(--line);white-space:nowrap}
.chip .dot{width:11px;height:11px;border-radius:3px;box-shadow:0 0 0 1px rgba(0,0,0,.25) inset}
.chip.leave{font-family:var(--sans);font-style:italic;font-weight:500;color:var(--muted);border-style:dashed}
.flownote{margin-top:14px;padding:13px 18px;border:1px solid var(--line);border-radius:12px;font-size:14.5px;display:flex;align-items:center;gap:8px;flex-wrap:wrap}

/* videos */
.vids{display:grid;gap:30px;grid-template-columns:repeat(auto-fit,minmax(420px,1fr))}
.vid h3{margin-bottom:4px}
.vid p{margin:0 0 10px;color:var(--muted);font-size:14px}
.vframe{position:relative;width:100%;padding-top:56.25%;border:1px solid var(--line);border-radius:10px;overflow:hidden;background:#000}
.vframe iframe{position:absolute;inset:0;width:100%;height:100%;border:0}
@media(max-width:560px){.vids{grid-template-columns:1fr}}

/* maps (Leaflet) */
.mapctrl{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-left:auto;font-size:13px;color:var(--muted)}
.mapctrl .seg{display:flex;align-items:center;gap:5px;cursor:pointer}
.mapctrl input{accent-color:var(--accent)}
.mapbox{height:74vh;min-height:460px;border:1px solid var(--line);border-radius:12px;background:var(--field);overflow:hidden}
.leaflet-container{background:var(--field);font-family:var(--sans)}
.tnum{color:#fff;font:600 10px/14px var(--mono,monospace);text-align:center;text-shadow:0 0 2px #000,0 0 2px #000;pointer-events:none;white-space:nowrap}
.mapbar{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:12px}
.sitebtn{font-family:var(--sans);font-size:13.5px;font-weight:600;color:var(--ink);background:var(--raise);
  border:1px solid var(--line);border-radius:8px;padding:7px 13px;cursor:pointer}
.sitebtn.active{border-color:var(--accent);color:var(--accent)}
.zoombtn{margin-left:auto;display:flex;gap:6px}
.zoombtn button{width:34px;height:34px;border:1px solid var(--line);background:var(--raise);color:var(--ink);
  border-radius:8px;cursor:pointer;font-size:17px;font-family:var(--sans)}
.zoombtn button:hover{border-color:var(--accent)}
.mapview{position:relative;overflow:hidden;border:1px solid var(--line);border-radius:12px;background:var(--field);
  height:72vh;min-height:420px;cursor:grab;touch-action:none}
.mapview.drag{cursor:grabbing}
.mapview img{position:absolute;top:0;left:0;transform-origin:0 0;user-select:none;-webkit-user-drag:none;max-width:none}
.maphint{font-size:12.5px;color:var(--muted);margin-top:8px}

/* do/don't figures */
.figs{display:grid;gap:30px;max-width:730px;margin:0 auto}
figure.fig{margin:0}
.fig .meta{display:flex;align-items:baseline;gap:11px;margin-bottom:10px}
.fig .meta .fn{font-family:var(--serif);color:var(--accent);font-size:15px;font-weight:600}
.fig img{display:block;width:100%;height:auto;cursor:zoom-in;border:1px solid var(--line);border-radius:9px}
.fig figcaption{margin-top:10px;color:var(--muted);font-size:14px;max-width:74ch}

/* table */
table{border-collapse:collapse;width:100%;font-size:14.5px;margin-top:4px}
td,th{border-bottom:1px solid var(--line);padding:12px 14px 12px 0;text-align:left;vertical-align:top}
th{font-size:11.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);font-weight:700}
td:first-child{padding-right:24px;font-weight:600;width:34%}

/* checklist */
.check{list-style:none;padding:0;margin:0}
.check li{padding:13px 0 13px 30px;position:relative;font-size:15.5px;border-top:1px solid var(--line)}
.check li:first-child{border-top:0}
.check li::before{content:"";position:absolute;left:0;top:17px;width:13px;height:13px;border-radius:4px;border:1.5px solid var(--accent)}

footer{margin-top:54px;padding-top:18px;border-top:1px solid var(--line);color:var(--muted);font-size:12.5px}

/* lightbox */
.lb{position:fixed;inset:0;z-index:90;background:oklch(0.12 0.02 255 / .85);display:none;align-items:center;justify-content:center;
  padding:30px;cursor:zoom-out;opacity:0;transition:opacity .25s ease}
.lb.on{display:flex;opacity:1}
.lb img{max-width:96vw;max-height:92vh;border-radius:8px}
@media(max-width:560px){.grid{grid-template-columns:1fr}h1{font-size:27px}}
</style></head>
<body>

<header><div class="barwrap">
  <div class="brandrow">
    <span class="brand">Nepal Farm Annotation<span class="dot"> Guide</span></span>
    <span class="spacer"></span>
    <button class="toggle" id="themeBtn" aria-label="Toggle theme"></button>
  </div>
  <div class="tabs" id="tabs"></div>
</div></header>

<div class="wrap">

  <!-- HOW TO DECIDE -->
  <section class="panel" id="tab-decide">
    <h2>How to decide what to mark</h2>
    <p class="lede" style="margin-bottom:18px">Look at one area at a time. Follow the steps from the top. The coloured box is the class to use.</p>
    <div id="flow"></div>

    <div class="reading" style="margin-top:48px">
    <div class="lead-rule">
      <p class="eyebrow" style="color:var(--accent)">The most important rule</p>
      <h2>A boundary is the mud line, not the colour</h2>
      <p>Farms are split by thin <i>raised mud lines</i> called <b>Aali</b>. Always cut along the mud line. Two fields that touch at a mud line are <b>two parcels, not one</b>, <i>even if both sides look the same green</i>.</p>
    </div>

    <div class="block">
      <h2 style="color:var(--bad)">Do not draw plain grass or bush</h2>
      <div class="hr"></div>
      <p>Leave plain grass and bush <b>empty</b>. The computer fills it in later. <i>If you paint it, that is a mistake.</i></p>
      <p><span class="draw-note">You only draw two green things:</span> <b>TREES</b> (tall, with shadow) and <b>MARSH_VEGETATION</b> (green on wet, muddy ground).</p>
    </div>

    <div class="block">
      <h2>Label everything else. No gaps.</h2>
      <div class="hr"></div>
      <p>Anything you leave empty turns into <i>grass</i>. So a forgotten field, road, or house becomes a mistake. Draw every <b>field, house, road, water, channel, ground, and sand</b>.</p>
      <p>Truly cannot tell what it is? Mark it <b>EDGECASE_IGNORE</b>. <i>Never guess.</i></p>
    </div>

    <div class="block" style="margin-top:14px">
      <h2>How to annotate, step by step</h2>
      <div class="hr"></div>
      <ol class="steps">
        <li><b>Look first.</b> See where the fields, water, and houses are.</li>
        <li><b>Draw each field.</b> Use the <b>Polygon</b> tool. Trace one field at a time, along the mud line. <i>Never join two fields into one.</i></li>
        <li><b>Draw the rest.</b> Houses, roads, water, channels, ground, sand. <i>One shape for each.</i></li>
        <li><b>Trees and marsh plants.</b> <b>Smart Select</b> is fine here.</li>
        <li><b>Not sure?</b> Mark it <b>EDGECASE_IGNORE</b>. <i>Do not guess.</i></li>
        <li><b>Skip plain grass.</b> Leave it empty. Then check your work and save.</li>
      </ol>
      <p class="muted" style="margin-top:16px;font-size:15px"><b>Do not use the Brush tool.</b> <i>It makes messy labels.</i> See the tool guide under Do &amp; Don&rsquo;t.</p>
    </div>
    </div>
  </section>

  <!-- CLASSES -->
  <section class="panel" id="tab-classes">
    <h2>Class catalog</h2>
    <p class="lede" style="margin-bottom:8px">Verified examples. Left is the raw tile, right is annotated. Use <b>Show</b> to pick one class and see just that label on the tile, or pick <b>All classes</b>. Click a thumbnail to switch example, click an image to enlarge.</p>
    <div id="veg-top"></div>
    <div id="catalog"></div>
  </section>

  <!-- DEMOS & EXAMPLES -->
  <section class="panel" id="tab-videos">
    <h2>Demos and finished examples</h2>
    <p class="lede" style="margin-bottom:18px">Watch the demos, then study two finished tiles below. Use <b>Show</b> to see one class or all of them.</p>
    <div id="videos"></div>
    <h3 style="margin-top:46px">Finished tiles, fully annotated</h3>
    <p class="lede" style="margin:6px 0 14px">The same areas from the demo, completely labeled. Everything is marked except plain grass and bush.</p>
    <div id="pilot"></div>
  </section>

  <!-- MAPS -->
  <section class="panel" id="tab-maps">
    <h2>Site maps</h2>
    <p class="lede" style="margin-bottom:14px">The whole site with its tile grid. Find your tile here. Drag to move, scroll or use + / &minus; to zoom.</p>
    <div id="maps"></div>
  </section>

  <!-- COMPARE -->
  <section class="panel" id="tab-compare">
    <h2>Compare two classes</h2>
    <p class="lede" style="margin-bottom:18px">Pick any two classes to study the distinction side by side, for example GROWING_FIELD against BARREN_FIELD, or GROUND against SAND.</p>
    <div class="compare" id="compare"></div>
  </section>

  <!-- DO / DON'T -->
  <section class="panel" id="tab-examples">
    <h2>Do &amp; Don&rsquo;t</h2>
    <p class="lede" style="margin-bottom:22px">Reviewed cases from real tiles. Click any image to enlarge.</p>
    <div class="figs" id="figs"></div>

    <div class="reading" style="margin-top:48px">
      <h2>Hard cases</h2>
      <div class="hr" style="height:1px;background:var(--line);margin:12px 0 4px"></div>
      <table>
        <tr><th>Situation</th><th>What to do</th></tr>
        <tr><td>Two green fields look identical across a faint line</td><td>It is a mud line. Split into two fields. Zoom in, the mud line is a thin raised ridge.</td></tr>
        <tr><td>Bush vs trees</td><td>Trees are tall, leafy, with shadow, draw as <code>TREES</code>. Low bush or grass, leave it blank.</td></tr>
        <tr><td>Bush vs marsh plants</td><td>Green plants on wet ground, draw as <code>MARSH_VEGETATION</code>. Dry low bush, leave it blank.</td></tr>
        <tr><td>Open water vs marsh</td><td>Clear water is <code>WATER</code>. A wet, weedy area is <code>MARSH_FIELD</code> or <code>MARSH_VEGETATION</code>.</td></tr>
        <tr><td>Shadows or unclear</td><td>If you cannot tell, mark <code>EDGECASE_IGNORE</code>. Never guess, never leave it blank.</td></tr>
        <tr><td>A field runs off the tile edge</td><td>Label the part you can see, up to the edge. It joins the next tile later, do not invent beyond the image.</td></tr>
      </table>
    </div>
  </section>

  <footer>Examples drawn from verified FloodSight tiles. Built for the NAAMII Agri AI annotation team.</footer>
</div>

<div class="lb" id="lb"><img id="lbimg" alt="enlarged example"></div>

<link rel="stylesheet" href="assets/vendor/leaflet/leaflet.css">
<script src="assets/vendor/leaflet/leaflet.js"></script>
<script>
const D = /*__DATA__*/;
const TAGTXT = {draw:"draw", nodraw:"do not draw", explicit:"draw explicitly"};
const TABS = [["decide","Start here"],["videos","Demos & Examples"],["classes","Classes"],
              ["maps","Maps"],["compare","Compare"],["examples","Do & Don't"]];
let GID = 0;

/* tabs */
const tabsEl=document.getElementById('tabs');
function showTab(id){
  document.querySelectorAll('.panel').forEach(p=>p.classList.toggle('active',p.id==='tab-'+id));
  [...tabsEl.children].forEach(b=>b.classList.toggle('active',b.dataset.id===id));
  if(location.hash.slice(1)!==id) history.replaceState(null,'','#'+id);
  window.scrollTo(0,0);
  if(id==='maps' && window.__refitMap) requestAnimationFrame(window.__refitMap);
}
TABS.forEach(([id,label])=>{const b=document.createElement('button');b.className='tab';b.dataset.id=id;
  b.textContent=label;b.onclick=()=>showTab(id);tabsEl.appendChild(b);});
showTab((location.hash.slice(1) && document.getElementById('tab-'+location.hash.slice(1)))?location.hash.slice(1):'decide');
addEventListener('hashchange',()=>{const h=location.hash.slice(1);if(document.getElementById('tab-'+h))showTab(h);});

/* theme */
const root=document.documentElement, tb=document.getElementById('themeBtn');
const SUN='<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M2 12h2M20 12h2M5 5l1.5 1.5M17.5 17.5L19 19M19 5l-1.5 1.5M6.5 17.5L5 19"/></svg>';
const MOON='<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>';
function setTheme(t){root.setAttribute('data-theme',t);tb.innerHTML=(t==='light'?MOON:SUN)+(t==='light'?'Dark':'Light');localStorage.setItem('guideTheme',t);}
setTheme(localStorage.getItem('guideTheme') || (matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light'));
tb.onclick=()=>setTheme(root.getAttribute('data-theme')==='light'?'dark':'light');

/* lightbox */
const lb=document.getElementById('lb'), lbimg=document.getElementById('lbimg');
function zoom(src){lbimg.src=src;lb.classList.add('on');}
lb.onclick=()=>lb.classList.remove('on');
addEventListener('keydown',e=>{if(e.key==='Escape')lb.classList.remove('on');});

/* canvas paint. overlay always on. opts.only = class name or '__all__'. */
function paint(cv, rec, opts){
  const img=new Image();
  img.onload=()=>{cv.width=rec.w;cv.height=rec.h;const x=cv.getContext('2d');
    x.clearRect(0,0,rec.w,rec.h);x.drawImage(img,0,0,rec.w,rec.h);
    const all=opts.only==='__all__';
    for(const p of rec.polys){
      if(!all && p.cls!==opts.only)continue;
      const col=D.colors[p.cls]||'#fff';
      x.beginPath();p.pts.forEach((pt,i)=>i?x.lineTo(pt[0],pt[1]):x.moveTo(pt[0],pt[1]));x.closePath();
      x.fillStyle=hexA(col, opts.boost?0.55:0.28); x.fill();
      // contrast halo first, then the class colour on top -> always visible
      x.lineJoin='round';
      x.lineWidth=4.5; x.strokeStyle=haloFor(col); x.stroke();
      x.lineWidth=2.2; x.strokeStyle=col; x.stroke();
    }};
  img.src=rec.src;
}
function rgb(h){const n=parseInt(h.slice(1),16);return [n>>16&255,n>>8&255,n&255];}
function hexA(h,a){const [r,g,b]=rgb(h);return `rgba(${r},${g},${b},${a})`;}
function haloFor(h){const [r,g,b]=rgb(h);const L=(0.299*r+0.587*g+0.114*b)/255;
  return L>0.55?'rgba(0,0,0,.60)':'rgba(255,255,255,.88)';}

function viewer(exList, opts0){
  const id='v'+(GID++); const r=document.createElement('div'); r.className='viewer';
  const st=Object.assign({boost:false},opts0); let cur=0;
  r.innerHTML=`
   <div class="panes">
     <div class="pane"><div class="plbl">raw</div><canvas id="${id}r"></canvas></div>
     <div class="pane"><div class="plbl">annotated</div><canvas id="${id}a"></canvas></div>
   </div>
   <div class="ctrls">
     <label>Show <select id="${id}sel"></select></label>
   </div>
   <div class="tname" id="${id}nm"></div>
   <div class="thumbs" id="${id}th"></div>`;
  const cr=r.querySelector('#'+id+'r'),ca=r.querySelector('#'+id+'a');
  const sel=r.querySelector('#'+id+'sel'),nm=r.querySelector('#'+id+'nm'),th=r.querySelector('#'+id+'th');
  const classesOn=rec=>[...new Set(rec.polys.map(p=>p.cls))];
  function render(){const rec=D.images[exList[cur].img];
    paint(cr,rec,{only:'__none__'});                 // raw: nothing drawn
    paint(ca,rec,{only:sel.value,boost:sel.value!=='__all__'});
    nm.textContent=rec.name+'  ·  '+rec.w+'×'+rec.h;}
  function loadEx(i){cur=i;const rec=D.images[exList[i].img];const f=exList[i].focus;
    sel.innerHTML='<option value="__all__">All classes</option>'+
      classesOn(rec).map(c=>`<option value="${c}" ${c===f?'selected':''}>${c}</option>`).join('');
    [...th.children].forEach((b,j)=>b.classList.toggle('active',j===i));render();}
  sel.onchange=render;
  cr.onclick=ca.onclick=()=>zoom(D.images[exList[cur].img].src);
  exList.forEach((ex,i)=>{const b=document.createElement('button');
    b.style.backgroundImage=`url(${D.images[ex.img].src})`;b.onclick=()=>loadEx(i);th.appendChild(b);});
  if(exList.length<2)th.style.display='none';
  loadEx(0);return r;
}
function placeholder(cn){const d=document.createElement('div');d.className='ph';
  d.innerHTML=`<b>Example pending</b><span>${cn} example to be provided.</span>`;return d;}

/* decision flowchart (How to decide tab) */
const chip=(cls,leave)=> leave
  ? `<span class="chip leave">leave it blank</span>`
  : `<span class="chip" style="border-color:${D.colors[cls]}"><span class="dot" style="background:${D.colors[cls]}"></span>${cls}</span>`;
const flow=document.getElementById('flow');
if(flow){
  const out=(cond,cls,leave)=>`<div class="outcome"><span class="cond">${cond}</span><span class="arr">&rarr;</span>${chip(cls,leave)}</div>`;
  const obv=[['Building roof','HOUSE'],['Road or path','ROAD'],['Open water','WATER']];
  flow.innerHTML=`
   <div class="obvious"><div class="ohead">First, the easy ones. See any of these? Mark them right away:</div>
     <div class="orow">${obv.map(([l,c])=>`<span class="oitem">${l} <span class="arr">&rarr;</span> ${chip(c)}</span>`).join('')}</div>
   </div>
   <div class="qcard"><div class="q">Is there a field edge? &nbsp;(a raised mud line going around it)</div></div>
   <div class="branches">
     <div class="branch"><div class="bhead no">No edge</div>
       ${out('Bare earth right next to a house','GROUND')}
       ${out('Soft, pale sand near water','SAND')}
       ${out('Just plain grass or bush',null,true)}
     </div>
     <div class="branch"><div class="bhead yes">Yes, it is a field</div>
       ${out('Healthy plants clearly growing','GROWING_FIELD')}
       ${out('No plants, only bare soil','BARREN_FIELD')}
       ${out('Wet: black patches or water','MARSH_FIELD')}
       ${out('Has an edge and some growth, but none of the above','NONCROP_FIELD')}
     </div>
   </div>
   <div class="flownote">Green plants on <b>wet, muddy</b> ground? &rarr; ${chip('MARSH_VEGETATION')} (you draw this). Plain grass or bush &rarr; do not draw.</div>
   <div class="flownote">Cannot tell what it is? &rarr; ${chip('EDGECASE_IGNORE')} &nbsp; Never guess.</div>`;
}

/* vegetation top block (Classes tab, first) */
const vt=document.getElementById('veg-top');
if(vt){
  const wrap=document.createElement('div');wrap.className='vegtop';
  const main=document.createElement('div');main.className='vmain';
  main.innerHTML=`<h3 class="vrule">Do NOT mark VEGETATION</h3>
    <p>Mark <b>every other class</b>. Whatever you leave blank becomes vegetation, and <b>we mark it automatically</b>. <i>Never paint plain grass or bush.</i></p>
    <p class="vsub">This is what vegetation looks like. Leave it blank.</p>`;
  const mkimg=(src,cls)=>{const im=document.createElement('img');im.src=src;im.alt='vegetation';im.className=cls;im.onclick=()=>zoom(src);return im;};
  const big=document.createElement('div');big.className='vgrid-big';
  D.vegRaw.slice(0,4).forEach(s=>big.appendChild(mkimg(s)));
  const small=document.createElement('div');small.className='vgrid';
  D.vegRaw.slice(4).forEach(s=>small.appendChild(mkimg(s)));
  main.appendChild(big);main.appendChild(small);
  const aside=document.createElement('aside');aside.className='vaside';
  aside.innerHTML=`<p class="vsub">But DO draw this</p>
    <div class="top"><span class="sw" style="background:${D.colors['MARSH_VEGETATION']}"></span><span class="cls-name">MARSH_VEGETATION</span></div>
    <div class="cue" style="margin:4px 0 8px">Green growth on wet, muddy ground. The one green you outline.</div>`;
  if(D.marshVeg && D.marshVeg.length) aside.appendChild(viewer(D.marshVeg,{focus:'MARSH_VEGETATION'}));
  wrap.appendChild(main);wrap.appendChild(aside);vt.appendChild(wrap);
}

/* class catalog */
const cat=document.getElementById('catalog');
for(const grp in D.groups){
  const t=document.createElement('div');t.className='grp-title';t.textContent=D.groupTitles[grp];cat.appendChild(t);
  const g=document.createElement('div');g.className='grid';
  for(const cn of D.groups[grp]){
    const inf=D.info[cn],col=D.colors[cn];
    const cell=document.createElement('div');cell.className='cls';
    cell.innerHTML=`<div class="top"><span class="sw" style="background:${col}"></span>
      <span class="cls-name">${cn}</span></div>
      <div class="def">${inf.def}</div><div class="cue">Look for: ${inf.cue}</div>
      <div class="ud"><span class="do"><em>Do</em>${inf.do}</span><span class="av"><em>Avoid</em>${inf.dont}</span></div>`;
    const ex=D.examples[cn]||[];
    cell.appendChild((D.placeholder.includes(cn)||ex.length===0)?placeholder(cn):viewer(ex,{focus:cn}));
    g.appendChild(cell);
  }
  cat.appendChild(g);
}

/* compare */
const allCls=Object.keys(D.examples).filter(c=>(D.examples[c]||[]).length>0 && c!=='VEGETATION');
function compareBox(defCls){
  const box=document.createElement('div');box.className='cbox';
  const s=document.createElement('select');
  s.innerHTML=allCls.map(c=>`<option ${c===defCls?'selected':''}>${c}</option>`).join('');
  const slot=document.createElement('div');
  const load=c=>{slot.innerHTML='';slot.appendChild(viewer(D.examples[c],{focus:c}));};
  s.onchange=e=>load(e.target.value);box.appendChild(s);box.appendChild(slot);load(defCls);return box;
}
const cmp=document.getElementById('compare');
cmp.appendChild(compareBox(allCls.includes('GROWING_FIELD')?'GROWING_FIELD':allCls[0]));
cmp.appendChild(compareBox(allCls.includes('BARREN_FIELD')?'BARREN_FIELD':allCls[1]||allCls[0]));

/* videos (Google Drive embeds) */
const vids=document.getElementById('videos');
if(vids){vids.className='vids';
  D.videos.forEach(v=>{const d=document.createElement('div');d.className='vid';
    d.innerHTML=`<h3>${v.title}</h3><p>${v.desc}</p><div class="vframe"><iframe src="https://drive.google.com/file/d/${v.drive}/preview" allow="autoplay; fullscreen" allowfullscreen loading="lazy"></iframe></div>`;
    vids.appendChild(d);});}

/* finished pilot tiles (fully annotated), each in a raw/overlay viewer */
const pilotEl=document.getElementById('pilot');
if(pilotEl && D.pilot && D.pilot.length){
  D.pilot.forEach(ex=>{const wrap=document.createElement('div');wrap.style.marginBottom='30px';
    wrap.appendChild(viewer([ex],{focus:ex.focus}));pilotEl.appendChild(wrap);});
}

/* maps: Leaflet over existing GCS XYZ tiles (full res, only visible tiles load) + red 3800 grid */
const mapsEl=document.getElementById('maps');
if(mapsEl && D.maps && D.maps.length && window.L){
  const bar=document.createElement('div');bar.className='mapbar';
  D.maps.forEach((m,i)=>{const b=document.createElement('button');b.className='sitebtn'+(i===0?' active':'');
    b.textContent=m.name;b.dataset.i=i;bar.appendChild(b);});
  const ctrl=document.createElement('div');ctrl.className='mapctrl';
  ctrl.innerHTML='<label class="seg"><input type="radio" name="ph" value="pre" checked> pre-flood</label>'+
                 '<label class="seg"><input type="radio" name="ph" value="post"> post-flood</label>'+
                 '<label class="seg"><input type="checkbox" id="gridlbl"> tile numbers</label>';
  bar.appendChild(ctrl);
  const view=document.createElement('div');view.id='map';view.className='mapbox';
  const hint=document.createElement('div');hint.className='maphint';
  mapsEl.appendChild(bar);mapsEl.appendChild(view);mapsEl.appendChild(hint);
  let map=null,tile=null,gridL=null,lblL=null,cur=0,phase='pre',inited=false;
  function setTiles(){const c=D.maps[cur].code;
    if(tile)map.removeLayer(tile);
    tile=L.tileLayer(D.tileBase+'/'+c+'/'+phase+'/{z}/{x}/{y}.png',
      {minZoom:10,maxNativeZoom:19,maxZoom:19,tileSize:256,attribution:'NAXA / FAO'}).addTo(map);
    hint.textContent=D.maps[cur].name+' ('+phase+'-flood). Red boxes = the 3800x3800 tiles. Scroll to zoom, drag to move.';}
  function setGrid(){const c=D.maps[cur].code;
    if(gridL)map.removeLayer(gridL); if(lblL){map.removeLayer(lblL);}
    gridL=L.layerGroup(); lblL=L.layerGroup();
    (D.mapGrid[c]||[]).forEach(g=>{const s=g[0],w=g[1],n=g[2],e=g[3];
      L.rectangle([[s,w],[n,e]],{color:'#E0322E',weight:1,fill:false,interactive:false}).addTo(gridL);
      lblL.addLayer(L.marker([(s+n)/2,(w+e)/2],{interactive:false,
        icon:L.divIcon({className:'tnum',html:g[4],iconSize:[44,14]})}));});
    gridL.addTo(map); updateLabels();}
  function updateLabels(){if(!map)return;const want=ctrl.querySelector('#gridlbl').checked||map.getZoom()>=16;
    if(want){if(!map.hasLayer(lblL))lblL.addTo(map);}else if(map.hasLayer(lblL))map.removeLayer(lblL);}
  function load(i){cur=i;setTiles();setGrid();map.fitBounds(D.maps[i].bounds);
    [...bar.querySelectorAll('.sitebtn')].forEach(b=>b.classList.toggle('active',+b.dataset.i===i));}
  bar.querySelectorAll('.sitebtn').forEach(b=>b.onclick=()=>load(+b.dataset.i));
  ctrl.querySelectorAll('input[name=ph]').forEach(r=>r.onchange=()=>{phase=r.value;setTiles();});
  ctrl.querySelector('#gridlbl').onchange=updateLabels;
  function init(){if(inited)return;inited=true;
    map=L.map(view,{minZoom:10,maxZoom:19});map.on('zoomend',updateLabels);load(0);}
  window.__refitMap=()=>{ if(!inited)init(); else map.invalidateSize(); };
  if(document.getElementById('tab-maps').classList.contains('active')) requestAnimationFrame(window.__refitMap);
}

/* do / don't figures */
const figs=document.getElementById('figs');
D.dodont.forEach((f,i)=>{
  const el=document.createElement('figure');el.className='fig';
  el.innerHTML=`<div class="meta"><span class="fn">${String(i+1).padStart(2,'0')}</span><h3>${f.title}</h3></div>
    <img src="${f.src}" alt="${f.title}"><figcaption>${f.take}</figcaption>`;
  el.querySelector('img').onclick=()=>zoom(f.src);
  figs.appendChild(el);
});
</script>
</body></html>
"""

if __name__ == "__main__":
    main()
