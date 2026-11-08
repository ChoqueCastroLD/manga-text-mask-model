#!/usr/bin/env python3
"""Find the text and sound effects on a manga page and write them as a mask.

Three models: Shin1, Shin2 and Shin3. Shin1 and Shin2 come with the code. Shin3 is downloaded the first time it is used.

Command line:
    python manga_text_mask.py page.png -o out/                       # Shin1
    python manga_text_mask.py pages/*.png -o out/ --grow 2 --overlay --json
    python manga_text_mask.py page.png -o out/ --model shin2         # fewest false detections
    python manga_text_mask.py page.png -o out/ --model shin3         # most text and giant effects

Python:
    from manga_text_mask import MangaTextMask, read_page
    model = MangaTextMask()                   # or MangaTextMask("shin3")
    mask = model.mask(read_page("page.png"), grow=2)   # bool array, same size as the page
    mask, labels, regions = model.detect(read_page("page.png"))   # one id per word or effect
"""
import argparse
import glob
import hashlib
import json
import os
import time
import urllib.request

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
MODELS = {name: os.path.join(MODEL_DIR, name + ".onnx") for name in ("shin1", "shin2", "shin3")}
DEFAULT_MODEL, PRECISE_MODEL, SHIN3_MODEL = MODELS["shin1"], MODELS["shin2"], MODELS["shin3"]

# Shin3 is kept in a GitHub release, so the repository does not grow by another model.
SHIN3_URL = "https://github.com/ChoqueCastroLD/manga-text-mask-model/releases/download/shin3/shin3.onnx"
SHIN3_SHA256 = "a0dc1bf6ce533beec484f61b32986d0c3e67c8d30bd06b11b82782f974c0a3d6"

# The network works on pages of about this height. Other pages are resized for the network only:
# the result always comes back at the page's own size.
MIN_HEIGHT, MAX_HEIGHT = 1500, 2600

# Very large pages, such as double spreads, run as overlapping tiles to bound GPU memory.
TILE_SIZE, TILE_OVERLAP, TILE_ABOVE_AREA = 1536, 128, 6.5e6

# Filling the white body of outlined letters.
FILL_MIN_AREA, FILL_MAX_RATIO, FILL_MIN_WIDTH = 300, 3, 4

# Giant mode, used by models that ask for it (Shin3). A second pass at half the usual height shows a giant effect
# at the size the network knows best. That pass is merged only where the full-size pass found a piece at least
# GIANT_MIN_PIECE of the page tall and covers GIANT_MIN_COVER of it. It is never merged deep inside solid black.
GIANT_SCALE, GIANT_MIN_PIECE, GIANT_MIN_COVER, GIANT_DARK_DEPTH = 0.5, 0.06, 0.3, 8
# Then the white body of a giant letter is filled when the mask traces at least BODY_RING of the line around it.
BODY_RING, BODY_BRIGHT, BODY_MIN_SIDE, BODY_MAX_AREA, BODY_REACH, BODY_DOTS, BODY_GRID = 0.8, 200, 1 / 60, 0.12, 9, 4.0, 3

Image.MAX_IMAGE_PIXELS = None


def available_providers():
    """The fastest ONNX Runtime backends on this machine, with the CPU as the last resort."""
    preferred = ["CUDAExecutionProvider", "DmlExecutionProvider", "CoreMLExecutionProvider"]
    have = ort.get_available_providers()
    return [p for p in preferred if p in have] + ["CPUExecutionProvider"]


def fill_hollow(prob, img, thr=0.5):
    """Holes of the mask that are flat bright paper and much wider than the line around them become text.
    This is what fills the white body of a large outlined letter when the model traced its outline."""
    m = (prob >= thr).astype(np.uint8)
    if not m.any():
        return prob
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    k = (img.shape[0] / 1600) ** 2
    n, lab, st, _ = cv2.connectedComponentsWithStats(1 - m, connectivity=4)
    H, W = m.shape
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])).tolist())
    _, mlab, mst, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = prob.copy()
    for i in range(1, n):
        area = st[i, 4]
        if i in border or area < FILL_MIN_AREA * k:
            continue
        x, y, w, h = st[i, :4]
        hole = lab[y:y + h, x:x + w] == i
        ring = cv2.dilate(hole.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~hole
        around = mlab[y:y + h, x:x + w][ring]
        around = around[around > 0]
        if not around.size:
            continue
        j = np.bincount(around).argmax()
        if area > FILL_MAX_RATIO * mst[j, 4]:
            continue
        v = g[y:y + h, x:x + w][hole]
        if v.mean() < 200 or v.std() > 30:
            continue
        # An outlined letter is a thin line around a wide body. The counter of a bold letter (0, 6, O)
        # is about as wide as its stroke, and stays as it is.
        P = max(4, int(min(h, w) * 0.5))
        Y0, X0, Y1, X1 = max(0, y - P), max(0, x - P), min(H, y + h + P), min(W, x + w + P)
        hm = np.zeros((Y1 - Y0, X1 - X0), np.uint8)
        hm[y - Y0:y - Y0 + h, x - X0:x - X0 + w] = hole
        hole_width = 2 * cv2.distanceTransform(hm, cv2.DIST_L2, 3).max()
        dark = g[Y0:Y1, X0:X1] < 128
        dark |= cv2.cvtColor(np.ascontiguousarray(img[Y0:Y1, X0:X1]), cv2.COLOR_RGB2HSV)[..., 1] > 110
        ink = (dark & (mlab[Y0:Y1, X0:X1] == j)).astype(np.uint8)
        band = cv2.dilate(hm, np.ones((7, 7), np.uint8)).astype(bool) & (ink > 0)
        stroke = 2 * np.percentile(cv2.distanceTransform(ink, cv2.DIST_L2, 3)[band], 90) if band.sum() > 10 else 1e9
        if hole_width < FILL_MIN_WIDTH * stroke:
            continue
        out[y:y + h, x:x + w][hole] = np.maximum(out[y:y + h, x:x + w][hole], 0.75)
    return out


def merge_giant(prob, half, rgb, threshold=0.5):
    """Adds the half-size pass to the full-size probabilities, only around giant letters. Ordinary text is left alone:
    at half size a paragraph blurs into one blob that covers the gaps between letters and lines."""
    H = prob.shape[0]
    n, lab, st, _ = cv2.connectedComponentsWithStats((half >= threshold).astype(np.uint8), connectivity=8)
    if n <= 1:
        return prob
    m = prob >= threshold
    _, lab1, st1, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    big = np.maximum(st1[:, 2], st1[:, 3]) >= GIANT_MIN_PIECE * H
    big[0] = False
    has_big = np.zeros(n, bool)
    has_big[np.unique(lab[big[lab1]])] = True
    covered = np.bincount(lab[m].ravel(), minlength=n)
    hit = (covered >= GIANT_MIN_COVER * np.maximum(st[:, 4], 1)) & has_big
    hit[0] = False
    add = hit[lab]
    d = 2 * GIANT_DARK_DEPTH + 1
    dark = (rgb.mean(2) < 80).astype(np.uint8)
    add &= ~(cv2.erode(dark, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))) > 0)
    out = prob.copy()
    out[add] = np.maximum(prob[add], half[add])
    return out


def fill_bodies(labels, regions, rgb):
    """Fills the white body of giant letters from the page itself. A white area is a letter body when the mask traces
    most of the line around it. Holes inside a filled body are counters and keep showing the art. Halftone whites and
    grids of same-size squares (windows, tiles) are skipped. Each body joins the region of the line around it."""
    H, W = labels.shape
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    n, lab, st, _ = cv2.connectedComponentsWithStats((g >= BODY_BRIGHT).astype(np.uint8), connectivity=4)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BODY_REACH, BODY_REACH))
    near = cv2.dilate((labels > 0).astype(np.uint8), k) > 0
    amin, amax = (H * BODY_MIN_SIDE) ** 2, BODY_MAX_AREA * H * W
    blocked = np.zeros((H, W), bool)
    cells = []
    for i in sorted((i for i in range(1, n) if amin <= st[i, 4] <= amax), key=lambda i: -st[i, 4]):
        x, y, w, h, a = st[i]
        c = lab[y:y + h, x:x + w] == i
        if (blocked[y:y + h, x:x + w] & c).any():
            continue
        cs, _ = cv2.findContours(c.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        q = max(cs, key=len)[:, 0]
        px, py = q[:, 0] + x, q[:, 1] + y
        inside = (px > 0) & (py > 0) & (px < W - 1) & (py < H - 1)  # the edge of the page is nobody's line
        if inside.sum() < 20 or near[py[inside], px[inside]].mean() < BODY_RING:
            continue
        hull = np.zeros((h, w), np.uint8)
        cv2.drawContours(hull, cs, -1, 1, -1)
        _, _, hs, _ = cv2.connectedComponentsWithStats(((hull > 0) & ~c).astype(np.uint8), connectivity=8)
        if (hs[1:, 4] < 400).sum() * 1e4 / a > BODY_DOTS:
            continue
        blocked[y:y + h, x:x + w] |= (hull > 0) & ~c
        cells.append((i, x, y, w, h))
    square = [c for c in cells if min(c[3], c[4]) > 0.8 * max(c[3], c[4])]
    drop = set()
    for c in square:
        same = [j for j in square if abs(j[3] - c[3]) <= 0.2 * max(j[3], c[3]) and abs(j[4] - c[4]) <= 0.2 * max(j[4], c[4])]
        if len(same) >= BODY_GRID:
            drop.update(j[0] for j in same)
    cells = [c for c in cells if c[0] not in drop]
    if not cells:
        return labels, regions
    labels = labels.copy()
    by_id = {r["id"]: r for r in regions}
    for i, x, y, w, h in cells:
        X0, Y0, X1, Y1 = max(0, x - BODY_REACH), max(0, y - BODY_REACH), min(W, x + w + BODY_REACH), min(H, y + h + BODY_REACH)
        body = lab[Y0:Y1, X0:X1] == i
        around = labels[Y0:Y1, X0:X1][cv2.dilate(body.astype(np.uint8), k) > 0]
        around = around[around > 0]
        if not around.size:
            continue
        rid = int(np.bincount(around).argmax())
        new = body & (labels[Y0:Y1, X0:X1] == 0)
        labels[Y0:Y1, X0:X1][new] = rid
        r = by_id[rid]
        r["area"] += int(new.sum())
        r["box"] = [min(r["box"][0], int(x)), min(r["box"][1], int(y)), max(r["box"][2], int(x + w)), max(r["box"][3], int(y + h))]
    return labels, regions


def _group(st, keep, gap=0.7, max_gap=None):
    """Join components whose distance is small relative to the smaller one, so a word or an effect is one region."""
    if not keep:
        return []
    b = st[keep, :4].astype(np.float32)
    x0, y0, x1, y1 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    size = np.maximum(b[:, 2], b[:, 3])
    size = np.minimum(size, 4 * np.minimum(b[:, 2], b[:, 3]) + 6)  # long strokes count by their thickness
    dx = np.maximum(0, np.maximum(x0[:, None], x0[None]) - np.minimum(x1[:, None], x1[None]))
    dy = np.maximum(0, np.maximum(y0[:, None], y0[None]) - np.minimum(y1[:, None], y1[None]))
    limit = gap * np.minimum(size[:, None], size[None])
    if max_gap:
        limit = np.minimum(limit, max_gap)
    parent = list(range(len(keep)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in zip(*np.nonzero(np.triu(np.hypot(dx, dy) <= limit, 1))):
        a, c = find(i), find(j)
        if a != c:
            parent[a] = c
    groups = {}
    for i in range(len(keep)):
        groups.setdefault(find(i), []).append(keep[i])
    return list(groups.values())


def find_regions(prob, threshold=0.5, confident=0.8, min_area=12, min_region_area=60):
    """Probability map to regions. A component needs at least one confident pixel, and tiny regions are dropped.
    Returns a label map (0 = nothing, 1.. = region id) and a list of regions in manga reading order.
    A region's score is the mean probability of its pixels."""
    H = prob.shape[0]
    n, lab, st, _ = cv2.connectedComponentsWithStats((prob >= threshold).astype(np.uint8), connectivity=8)
    keep = [i for i in range(1, n) if st[i, 4] >= min_area]
    if keep and confident > threshold:
        core = np.zeros(n, bool)
        core[np.unique(lab[prob >= confident])] = True
        keep = [i for i in keep if core[i]]
    regions = []
    for g in _group(st, keep, max_gap=0.04 * H):
        area = int(st[g, 4].sum())
        if area < min_region_area:
            continue
        x0, y0 = int(st[g, 0].min()), int(st[g, 1].min())
        x1, y1 = int((st[g, 0] + st[g, 2]).max()), int((st[g, 1] + st[g, 3]).max())
        regions.append(dict(box=[x0, y0, x1, y1], area=area, components=g))
    # Top to bottom in bands, right to left inside a band.
    band = max(1, H // 14)
    regions.sort(key=lambda r: ((r["box"][1] + r["box"][3]) // 2 // band, -(r["box"][0] + r["box"][2]) // 2))
    lut = np.zeros(n, np.int32)
    for k, r in enumerate(regions, 1):
        r["id"] = k
        lut[r.pop("components")] = k
    labels = lut[lab]
    sums = np.bincount(labels.ravel(), weights=prob.ravel(), minlength=len(regions) + 1)
    for r in regions:
        r["score"] = round(float(sums[r["id"]] / r["area"]), 3)
    return labels, regions


def download_shin3(path=SHIN3_MODEL):
    """Fetches Shin3 from the project's GitHub release, and keeps it only if its checksum matches."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"downloading Shin3 (29 MB) from {SHIN3_URL}")
    part = path + ".part"
    digest = hashlib.sha256()
    with urllib.request.urlopen(SHIN3_URL) as r, open(part, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
            digest.update(chunk)
    if digest.hexdigest() != SHIN3_SHA256:
        os.remove(part)
        raise RuntimeError(f"the download does not match Shin3; get it by hand from {SHIN3_URL} into {path}")
    os.replace(part, path)


class MangaTextMask:
    def __init__(self, model_path=DEFAULT_MODEL, providers=None):
        """model_path: "shin1", "shin2", "shin3", or the path of an .onnx file."""
        model_path = MODELS.get(str(model_path).lower(), model_path)
        if model_path == SHIN3_MODEL and not os.path.exists(model_path):
            download_shin3(model_path)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(model_path, options, providers=providers or available_providers())
        self.input_name = self.session.get_inputs()[0].name
        # A model file can carry its own settings. Without them these are the defaults.
        settings = self.session.get_modelmeta().custom_metadata_map
        self.confident = float(settings.get("confident", 0.8))
        self.wide_page_max_height = int(settings.get("wide_page_max_height", 0))
        self.wide_page_min_width = int(settings.get("wide_page_min_width", 0))
        self.giant = settings.get("giant", "0") == "1"

    @property
    def backend(self):
        return self.session.get_providers()[0]

    def _run(self, rgb):
        x = rgb.astype(np.float32).transpose(2, 0, 1)[None] / 255
        return self.session.run(None, {self.input_name: x})[0][0, 0]

    def _logits(self, rgb):
        """The whole image at once. The network needs each side to be a multiple of 32."""
        h, w = rgb.shape[:2]
        canvas = np.full(((h + 31) // 32 * 32, (w + 31) // 32 * 32, 3), 255, np.uint8)
        canvas[:h, :w] = rgb
        return self._run(canvas)[:h, :w]

    def _logits_tiled(self, rgb, T):
        """Fixed-size tiles, blended with linear ramps over the overlaps."""
        h, w = rgb.shape[:2]
        total, weight = np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)
        ramp = np.minimum(np.arange(T) + 1, TILE_OVERLAP).astype(np.float32) / TILE_OVERLAP
        ramp = np.minimum(ramp, ramp[::-1])
        blend = np.outer(ramp, ramp)
        ys = list(range(0, max(h - T, 0) + 1, T - TILE_OVERLAP))
        xs = list(range(0, max(w - T, 0) + 1, T - TILE_OVERLAP))
        if ys[-1] + T < h:
            ys.append(h - T)
        if xs[-1] + T < w:
            xs.append(w - T)
        for y in ys:
            for x in xs:
                c = rgb[y:y + T, x:x + T]
                ch, cw = c.shape[:2]
                tile = np.full((T, T, 3), 255, np.uint8)
                tile[:ch, :cw] = c
                out = self._run(tile)
                total[y:y + ch, x:x + cw] += out[:ch, :cw] * blend[:ch, :cw]
                weight[y:y + ch, x:x + cw] += blend[:ch, :cw]
        return total / np.maximum(weight, 1e-6)

    def probabilities(self, rgb, tile=None):
        """rgb: uint8 array (H, W, 3). Returns a float array (H, W): how likely each pixel is text.
        tile: force tiles of this size in px; it must be a multiple of 32."""
        H, W = rgb.shape[:2]
        s = MIN_HEIGHT / H if H < MIN_HEIGHT else MAX_HEIGHT / H if H > MAX_HEIGHT else 1.0
        if self.wide_page_max_height and W >= self.wide_page_min_width and H * s > self.wide_page_max_height:
            s = self.wide_page_max_height / H
        h, w = round(H * s), round(W * s)
        x = rgb if s == 1 else cv2.resize(rgb, (w, h), interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
        T = tile or (TILE_SIZE if h * w > TILE_ABOVE_AREA else 0)
        logit = self._logits_tiled(np.ascontiguousarray(x), T) if T else self._logits(x)
        prob = 1 / (1 + np.exp(-logit))
        if s != 1:
            prob = cv2.resize(prob, (W, H), interpolation=cv2.INTER_LINEAR)
        prob = fill_hollow(prob, rgb)
        if self.giant:
            prob = merge_giant(prob, self._half_size(rgb), rgb)
        return prob

    def _half_size(self, rgb):
        """The page at GIANT_SCALE of its working height, in one piece, brought back to the page's size."""
        H, W = rgb.shape[:2]
        h = float(np.clip(H, MIN_HEIGHT, MAX_HEIGHT)) * GIANT_SCALE
        small = cv2.resize(rgb, (max(32, round(W * h / H)), max(32, round(h))), interpolation=cv2.INTER_AREA)
        prob = 1 / (1 + np.exp(-self._logits(small)))
        return cv2.resize(prob, (W, H), interpolation=cv2.INTER_LINEAR)

    def detect(self, rgb, threshold=0.5, grow=0, tile=None):
        """Returns (mask, labels, regions).
        mask: bool array (H, W), True on text and sound effects.
        labels: int array (H, W), the id of the region each pixel belongs to, 0 where there is none.
        regions: id, box [x0, y0, x1, y1], area and score of each word or effect, in manga reading order.
        grow: expand the mask and the labels outward by this many px, so no dark edge survives cleaning.
        Boxes, areas and scores describe the regions before growing."""
        labels, regions = find_regions(self.probabilities(rgb, tile), threshold, self.confident)
        if self.giant:
            labels, regions = fill_bodies(labels, regions, rgb)
        if grow:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
            labels = cv2.dilate(labels.astype(np.float32), k).astype(np.int32)
        return labels > 0, labels, regions

    def mask(self, rgb, threshold=0.5, grow=0, tile=None):
        return self.detect(rgb, threshold, grow, tile)[0]


def read_page(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def paint_overlay(rgb, mask, alpha=0.7):
    out = rgb.copy()
    out[mask] = (out[mask] * (1 - alpha) + np.array((255, 30, 60)) * alpha).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser(description="Find the text and sound effects on manga pages and write them as masks.")
    ap.add_argument("pages", nargs="+", help="image files; wildcards are fine")
    ap.add_argument("-o", "--out", default="out", help="output folder (default: out)")
    which = ap.add_mutually_exclusive_group()
    which.add_argument("--model", default="shin1", help="shin1 (default), shin2, shin3, or the path of an .onnx file")
    which.add_argument("--precise", action="store_true", help="same as --model shin2")
    ap.add_argument("--threshold", type=float, default=0.5, help="lower finds more and also marks more art (default: 0.5)")
    ap.add_argument("--grow", type=int, default=0, help="expand the mask outward by N px; 2 suits cleaning (default: 0)")
    ap.add_argument("--tile", type=int, default=None, help="force tiles of N px; try 1536 if the GPU runs out of memory")
    ap.add_argument("--cpu", action="store_true", help="do not use the GPU")
    ap.add_argument("--overlay", action="store_true", help="also write the page with the mask painted in red")
    ap.add_argument("--json", action="store_true", help="also write the id, box, area and score of each region")
    ap.add_argument("--labels", action="store_true", help="also write a 16-bit PNG whose pixels hold the region id")
    ap.add_argument("--prob", action="store_true", help="also write the probability map as a grey PNG (0-255)")
    a = ap.parse_args()

    pages = [p for spec in a.pages for p in (sorted(glob.glob(spec)) or [spec])]
    model = MangaTextMask("shin2" if a.precise else a.model, ["CPUExecutionProvider"] if a.cpu else None)
    print(f"running on {model.backend}")
    os.makedirs(a.out, exist_ok=True)
    for path in pages:
        rgb = read_page(path)
        t = time.perf_counter()
        m, labels, regions = model.detect(rgb, a.threshold, a.grow, a.tile)
        t = time.perf_counter() - t
        stem = os.path.join(a.out, os.path.splitext(os.path.basename(path))[0])
        Image.fromarray((m * 255).astype(np.uint8)).save(stem + "_mask.png")
        if a.overlay:
            Image.fromarray(paint_overlay(rgb, m)).save(stem + "_overlay.png")
        if a.labels:
            cv2.imwrite(stem + "_labels.png", labels.astype(np.uint16))
        if a.prob:
            prob = model.probabilities(rgb, a.tile)
            Image.fromarray((prob * 255).round().astype(np.uint8)).save(stem + "_prob.png")
        if a.json:
            with open(stem + ".json", "w", encoding="utf-8") as f:
                json.dump(dict(width=rgb.shape[1], height=rgb.shape[0], regions=regions), f, indent=1)
        print(f"{os.path.basename(path)}: {len(regions)} regions, {m.mean() * 100:.2f}% marked, {t:.2f} s")


if __name__ == "__main__":
    main()
