# Guide

- [Install](#install)
- [Command line](#command-line)
- [Python](#python)
- [Output files](#output-files)
- [Using the ONNX file directly](#using-the-onnx-file-directly)
- [Cleaning pages](#cleaning-pages)
- [Tiles and memory](#tiles-and-memory)
- [Troubleshooting](#troubleshooting)

## Install

You need Python 3 and exactly one ONNX Runtime package. The packages share a module name, so having two installed causes conflicts.

| Machine | Package | Backend | Tested |
|---|---|---|---|
| Windows, any GPU | `onnxruntime-directml` | DirectML | Yes |
| Linux, NVIDIA GPU | `onnxruntime-gpu` | CUDA | No |
| Mac | `onnxruntime` | CoreML | No |
| Anything, CPU only | `onnxruntime` | CPU | Yes, on Windows |

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install numpy opencv-python-headless Pillow onnxruntime-directml   # change the last one for your machine
```

`pip install -r requirements.txt` is the same thing with the CPU package.

The script picks the fastest backend it finds and prints its name when it starts. On Linux, `onnxruntime-gpu` also needs NVIDIA's CUDA libraries to be installed.

Shin1 and Shin2 come with the repository. Shin3 is in the [`shin3` release](https://github.com/ChoqueCastroLD/manga-text-mask-model/releases/tag/shin3), and the script downloads it to `model/shin3.onnx` the first time you ask for it. It checks the file's SHA-256 before keeping it. To work offline, download `shin3.onnx` from the release yourself and put it in `model/`.

## Command line

```bash
python manga_text_mask.py page.png -o out
python manga_text_mask.py "chapter/*.png" -o out --grow 2 --overlay --json
python manga_text_mask.py "chapter/*.png" -o out --grow 2 --model shin3
```

| Option | Default | What it does |
|---|---|---|
| `-o`, `--out` | `out` | Output folder |
| `--grow N` | 0 | Expand the mask outward by N px. Use 2 for cleaning, so no dark edge survives |
| `--threshold T` | 0.5 | Lower finds more and also marks more art |
| `--overlay` | off | Also write the page with the mask painted in red |
| `--json` | off | Also write the id, box, area and score of each region |
| `--labels` | off | Also write a PNG whose pixels hold the region id |
| `--prob` | off | Also write the probability map as a grey PNG |
| `--tile N` | automatic | Force tiles of N px. N must be a multiple of 32 |
| `--cpu` | off | Do not use the GPU |
| `--model NAME` | `shin1` | `shin1`, `shin2`, `shin3`, or the path of an `.onnx` file. See [which model](../README.md#which-model) |
| `--precise` | off | The same as `--model shin2`. Kept for older commands |

## Python

```python
from manga_text_mask import MangaTextMask, read_page

model = MangaTextMask()                # Shin1; or MangaTextMask("shin2"), MangaTextMask("shin3")
                                       # or MangaTextMask(providers=["CPUExecutionProvider"])
page = read_page("page.png")           # uint8 RGB array, (H, W, 3)

mask = model.mask(page, grow=2)        # bool array, (H, W)
mask, labels, regions = model.detect(page, grow=2)
prob = model.probabilities(page)       # float array, (H, W), 0 to 1
```

To work with one region at a time:

```python
for region in regions:
    only_this_one = labels == region["id"]
    x0, y0, x1, y1 = region["box"]
    print(region["id"], region["score"], only_this_one.sum())
```

## Output files

For `page.png`:

| File | Contents |
|---|---|
| `page_mask.png` | 8-bit, 255 on text and sound effects, 0 elsewhere |
| `page_overlay.png` | The page with the mask painted in red |
| `page_labels.png` | 16-bit, each pixel holds its region id, 0 where there is none. Most image viewers show it as black; read it with `cv2.imread(path, cv2.IMREAD_UNCHANGED)` |
| `page_prob.png` | 8-bit, the probability of each pixel times 255 |
| `page.json` | The page size and the list of regions |

```json
{
 "width": 1500,
 "height": 2150,
 "regions": [
  { "box": [981, 121, 1172, 209], "area": 3913, "id": 1, "score": 0.93 }
 ]
}
```

- **What a region is.** The network itself only gives a probability per pixel. Regions come afterwards: nearby marks are grouped by distance, relative to their size. A region is usually a word, a text block or a sound effect, but a giant effect can split in two and two close words can merge.
- **Order.** Ids run in manga reading order: top to bottom in bands, right to left inside a band.
- **`box`** is `[x0, y0, x1, y1]`, with the end coordinates exclusive.
- **`score`** is the mean probability of the region's pixels. It helps to sort regions for review, but it has not been calibrated, so a high score does not mean the region is certainly text.
- **`--grow`** changes the mask and the labels. Boxes, areas and scores describe the region before growing.

## Using the ONNX file directly

| | |
|---|---|
| Input | `image`, float32, shape `[1, 3, H, W]`, RGB, values 0 to 1 |
| Output | Logits, shape `[1, 1, H, W]`. Apply a sigmoid to get probabilities |
| Size | H and W must be multiples of 32. Pad with white (1.0) at the bottom and right |
| Opset | 20 |

The network works best on pages between 1500 and 2600 px tall, so the script resizes other pages for it and resizes the result back. The script also does two things after the network: it fills the white body of outlined letters whose outline was traced, and it drops marks with no pixel above 0.8. Without those steps, your results will differ from the numbers in the README.

All three models have this interface. Shin2 and Shin3 store their own settings in the file's metadata, and the script reads them:

| Model | Key | Value | Meaning |
|---|---|---|---|
| Shin2 | `confident` | 0.99 | Drop marks with no pixel above this, instead of 0.8 |
| Shin2 | `wide_page_max_height` | 2000 | Run wide pages at no more than this height |
| Shin2 | `wide_page_min_width` | 1200 | A page is wide from this width in px, measured before resizing |
| Shin3 | `confident` | 0.8 | The usual value |
| Shin3 | `giant` | 1 | Use giant mode, below |

**Giant mode.** Shin3 runs every page twice:

1. At its usual size.
2. In one piece at half that height, where a giant effect looks like letters of the size the network knows best.

The half-size result is added only to areas that meet all of these:

- It covers at least 30 % of the area.
- The full-size pass found a piece there at least 6 % of the page tall.
- The area is not more than 8 px inside solid black.

Ordinary text does not meet these, so it stays as the full-size pass found it. After the regions are built, the white body of a giant letter is filled from the page itself when the mask traces at least 80 % of the line around it. The fill skips halftone whites and grids of same-size squares. `merge_giant` and `fill_bodies` in the script are the whole method. Without them Shin3 is much weaker on giant effects.

## Cleaning pages

```bash
python examples/clean_page.py page.png clean.png
python examples/clean_page.py page.png clean.png --model shin3
```

This uses OpenCV's inpainting, which is fine on white paper and flat areas and rough on detailed art. Grow the mask by 2 px before inpainting, because the mask follows the letter closely and the antialiased edge of a letter can otherwise survive as a dark outline.

The model can mark art by mistake, so review the mask, or at least the overlay, before you keep a cleaned page.

## Tiles and memory

Pages larger than about 6.5 million pixels after resizing, such as double spreads, run automatically as overlapping tiles of 1536 px.

GPU backends can keep memory for each new page size. If a batch of mixed sizes runs out of GPU memory, force tiles for every page:

```bash
python manga_text_mask.py "chapter/*.png" -o out --tile 1536
```

Tiling can change the result compared with running the whole page at once, because each tile sees less of the page. The difference can be noticeable on giant letters: one effect went from 41 % to 30 % of its area covered in a test. With Shin3, `--tile` applies only to the full-size pass. The half-size pass always runs in one piece, because it is at most 1300 px tall.

## Troubleshooting

**It says `CPUExecutionProvider` and I have a GPU.** You have the wrong ONNX Runtime package, or two of them. Run `pip uninstall onnxruntime onnxruntime-directml onnxruntime-gpu`, then install only the one from the [table](#install).

**It runs out of GPU memory.** Use `--tile 1536`. If that is not enough, use a smaller tile such as 1024, or `--cpu`.

**It runs out of RAM on the CPU.** Peak use on pages up to 2132 × 3200 was about 7 GB with Shin1, 5 GB with Shin3 and 4 GB with Shin2. Use `--tile 1024` to lower it.

**The Shin3 download fails.** Download `shin3.onnx` from the [release](https://github.com/ChoqueCastroLD/manga-text-mask-model/releases/tag/shin3) and put it in `model/`.

**It marks art.** Try Shin2, which also covers less of each giant effect. If you use Shin1, Shin3 marks less art too. Or raise `--threshold`, for example to 0.7. This also loses some text. See the limits in the [README](../README.md#quality-and-limits).

**It misses letters.** Try Shin3. Lower `--threshold`, for example to 0.3. This also marks more art. Black letters with no outline are a known weak spot.
