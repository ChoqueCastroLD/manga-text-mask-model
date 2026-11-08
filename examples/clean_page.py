#!/usr/bin/env python3
"""Remove the text from a page: find the mask, expand it by 2 px, and paint over it.

    python examples/clean_page.py page.png clean.png
    python examples/clean_page.py page.png clean.png --model shin3

OpenCV's inpainting keeps this example free of extra downloads. It works on white and flat areas and
is rough on detailed art, so check the result and use a better inpainting model for those areas.
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from manga_text_mask import MangaTextMask, read_page  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Remove the text from a manga page.")
    ap.add_argument("page")
    ap.add_argument("output")
    ap.add_argument("--model", default="shin1", help="shin1 (default), shin2, shin3, or the path of an .onnx file")
    ap.add_argument("--precise", action="store_true", help="same as --model shin2")
    a = ap.parse_args()
    model = MangaTextMask("shin2" if a.precise else a.model)
    rgb = read_page(a.page)
    mask = model.mask(rgb, grow=2)
    clean = cv2.inpaint(rgb, mask.astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)
    cv2.imwrite(a.output, clean[..., ::-1])
    print(f"{mask.mean() * 100:.2f}% of the page cleaned, saved to {a.output}")


if __name__ == "__main__":
    main()
