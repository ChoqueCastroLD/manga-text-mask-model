#!/usr/bin/env python3
"""Remove the text from a page: find the mask, expand it by 2 px, and paint over it.

    python examples/clean_page.py page.png clean.png
    python examples/clean_page.py page.png clean.png --precise

OpenCV's inpainting keeps this example free of extra downloads. It works on white and flat areas and
is rough on detailed art, so check the result and use a better inpainting model for those areas.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from manga_text_mask import DEFAULT_MODEL, PRECISE_MODEL, MangaTextMask, read_page  # noqa: E402


def main():
    args = [a for a in sys.argv[1:] if a != "--precise"]
    if len(args) != 2:
        sys.exit(__doc__)
    page, output = args
    model = MangaTextMask(PRECISE_MODEL if "--precise" in sys.argv else DEFAULT_MODEL)
    rgb = read_page(page)
    mask = model.mask(rgb, grow=2)
    clean = cv2.inpaint(rgb, mask.astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)
    cv2.imwrite(output, clean[..., ::-1])
    print(f"{mask.mean() * 100:.2f}% of the page cleaned, saved to {output}")


if __name__ == "__main__":
    main()
