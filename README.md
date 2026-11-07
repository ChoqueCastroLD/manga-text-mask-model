<div align="center">

# Manga Text Mask

**Find the text and sound effects on a manga page, and get them back as a mask.**

Two models, one script. Runs locally. No account, no upload.

[Quick start](#quick-start) · [Which model](#which-model) · [Examples](#examples) · [Speed](#speed-and-memory) · [Quality and limits](#quality-and-limits) · [Guide](docs/GUIDE.md)

<br>

<img src="docs/images/hero.jpg" alt="Two drawn manga pages, each next to the same page with the detected text painted red" width="100%">

</div>

<br>

The mask is built for cleaning: feed it to an inpainting tool to paint over the letters. It marks speech balloons, narration boxes, and sound effects drawn over the art, in Latin and Japanese script.

- **Same size as your page.** Any resolution goes in, and the mask comes back at that resolution.
- **Split into regions.** Each word or effect gets an id in manga reading order, a box, an area and a score.
- **Black and white first.** That is what the model is tuned for. Colour pages work less well.
- **Small.** ResNet-18 U-Nets of 59 MB and 29 MB that need only ONNX Runtime, on GPU or CPU.

This is a checkpoint and it makes mistakes, including on art. Read [Quality and limits](#quality-and-limits) before you run it on a batch you cannot check.

## Quick start

```bash
git clone https://github.com/ChoqueCastroLD/manga-text-mask-model
cd manga-text-mask-model
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python manga_text_mask.py examples/sample_page.png -o out --grow 2 --overlay --json
```

This installs the CPU version. [The guide](docs/GUIDE.md#install) lists the GPU packages for Windows, Linux and Mac.

From Python:

```python
from manga_text_mask import MangaTextMask, read_page

model = MangaTextMask()
mask, labels, regions = model.detect(read_page("page.png"), grow=2)

# mask     True on text and sound effects
# labels   the region id of each pixel, 0 where there is none
# regions  [{"id": 1, "box": [x0, y0, x1, y1], "area": 3913, "score": 0.93}, ...]
```

To remove the text from a page, there is a [complete example](examples/clean_page.py):

```bash
python examples/clean_page.py page.png clean.png
```

## Which model

| | `manga-text-mask` | `manga-text-mask-precise` |
|---|---|---|
| How to use it | The default | `--precise` |
| Better at | Covering giant effects, and a little more of the small text | Leaving art alone |
| False detections on the 44 test pages | 125 | 48 |
| Giant effect coverage in the tests | 53 % and 27 % | 41 % and 20 % |
| Time on a 2132 × 3200 page, GPU / CPU | 0.35 s / 5.3 s | 0.28 s / 3.2 s |
| File size | 59 MB | 29 MB |

Choose the precise model when a mark on the art costs you more than a letter left behind. Keep the default when you need as much of each giant effect as possible. Neither replaces a look at the result.

```bash
python manga_text_mask.py page.png -o out --grow 2 --overlay --precise
```

```python
from manga_text_mask import MangaTextMask, PRECISE_MODEL

model = MangaTextMask(PRECISE_MODEL)
```

The precise model is the same network with different weights, stored in half precision. It has stricter settings too: a mark needs a pixel at 0.99 instead of 0.8, and pages at least 1200 px wide run at no more than 2000 px tall. The file carries those settings, so the script applies them by itself.

## Examples

From left to right: the page, the mask painted over it in red, and the mask.

<img src="docs/images/example_english.jpg" alt="An action page with sound effects over buildings, its detected text in red, and the black and white mask" width="100%">

<img src="docs/images/example_japanese.jpg" alt="A quiet page with vertical Japanese text and sound effects, its detected text in red, and the black and white mask" width="100%">

Look closely and you will see errors on the second page: spots on the moon, lit windows in the skyline, and a piece of the balloon tail.

Cleaned with `examples/clean_page.py`. OpenCV's basic inpainting is fine on balloons and smears badly where a giant effect sits on art, so use a better inpainting model there. The mask also missed the dots after "valley", which stay on the page.

<img src="docs/images/cleaned.jpg" alt="The action page before and after removing the detected text" width="100%">

### A known failure

The same model on flat, geometric art. It takes large white shapes with a clean outline for giant letters, and it marks parts of the flat scenery.

<img src="docs/images/known_failure.jpg" alt="A robot and mountains drawn in flat shapes, with large parts of the art wrongly painted red" width="70%">

> **About these pictures.** They use the default model. All three pages are drawn by a script with invented text. They are not real manga and they are not a benchmark. The failure above comes from a first draft. The art of the other two pages was then redrawn with tone and hatching, closer to how manga looks, and it came out much better. So they show what the model can do, not how often it does it. For that, see the numbers below.

## Speed and memory

Measured on one desktop with an RTX 3060 Ti (8 GB), with both models on ONNX Runtime 1.24. Times are the median per page, after the first page.

| Page size | Default, GPU (DirectML) | Default, CPU | Precise, GPU (DirectML) | Precise, CPU |
|---|---:|---:|---:|---:|
| 800 × 1232 | 0.10 s | 1.6 s | 0.08 s | 1.5 s |
| 1080 × 1080 | 0.16 s | 2.5 s | 0.15 s | 2.3 s |
| 1725 × 2475 | 0.29 s | 6.0 s | 0.24 s | 2.9 s |
| 2132 × 3200 | 0.35 s | 5.3 s | 0.28 s | 3.2 s |

Splitting the mask into regions adds 0.02 to 0.13 s per page.

| Peak memory | Default, GPU (DirectML) | Default, CPU | Precise, GPU (DirectML) | Precise, CPU |
|---|---:|---:|---:|---:|
| System RAM | 0.7 GB | 7.3 GB | 0.6 GB | 4.1 GB |
| GPU memory | 5.4 GB | none | 2.4 GB | none |

Memory grows with each new page size the model sees, and only these four sizes were measured. A batch with many different sizes can use more, and `--tile 1536` keeps it bounded.

Only DirectML on Windows and the CPU on Windows have been tested. CUDA and CoreML should work through ONNX Runtime, but have not been tried.

## Quality and limits

Measured on real manga pages, through the script itself.

| Test | Default | Precise |
|---|---:|---:|
| Text boxes with at least half their text found, of 299 | 291 | 288 |
| Share of reviewed text pixels found | 93.5 % | 92.6 % |
| Share of marked pixels that are text | 91.9 % | 95.5 % |
| Pages with no false detection, of 44 | 20 | 29 |
| False detections on those 44 pages | 125 | 48 |
| Text-free panels left untouched, of 29 | 21 | 23 |
| Image marked on those panels, on average | 0.13 % | 0.07 % |
| Text-free covers left untouched, of 36 | 17 | 22 |
| Image marked on those covers, on average | 0.29 % | 0.23 % |
| Letter area covered, two giant outlined effects | 53 % and 27 % | 41 % and 20 % |
| Marked outside those letters | 1.0 % and 1.5 % | 1.0 % and 0.2 % |

Treat these as rough figures:

- The 44 pages and the text-free images were also used to choose between model versions and settings, so they flatter both models, and the precise one most of all. No untouched test set is left for it.
- The labels of the 44 pages miss some real sound effects, so some false detections are not real errors.
- The text-free images were not used in training. They ran with `--tile 1536`, in greyscale. Specks are not counted when deciding that an image was left untouched.

Two more checks compare the precise model with the default on pages not used to train it, reviewed by eye:

- On the first set, which also helped to choose the model, it removed 14 of 25 marks on art and 2 of 156 marks on text.
- On colour pages, checked after the model and its settings were fixed, only the low-confidence marks were reviewed. It removed 29 of 55 on art and also 10 of 77 on text. None of the 236 high-confidence marks was lost.

What it gets wrong:

- **Giant white letters are only partly filled.** Both models often find the outline and part of the body, and the precise model covers less, as the table above shows. Check these by eye.
- **Black letters with no outline can be missed.** One such giant effect in the tests got 0 % coverage.
- **It does touch art.** The usual victims are solid black areas, regular hatching and bars, and flat shapes with a clean outline. Do not run it unattended where a damaged panel matters.
- **Colour is secondary.** Expect worse results than on black and white.

## Files

| Path | What it is |
|---|---|
| `model/manga-text-mask.onnx` | The default model |
| `model/manga-text-mask-precise.onnx` | The model with fewer false detections |
| `manga_text_mask.py` | Command line tool and Python library |
| `examples/clean_page.py` | Removes the text from a page |
| `examples/sample_page.png` | A drawn page to try it on |
| `docs/GUIDE.md` | Install, options, output formats, troubleshooting |

## License

[Apache-2.0](LICENSE)
