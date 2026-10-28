#!/usr/bin/env python3
"""Check a figure PDF the way the paper will use it: page size, embedded fonts (must be
TrueType/Type 42-style embedded, no Type 3), raster images (a vector figure has none),
and a rendering at the width the manuscript prints it.

    PYTHONPATH=/workspace/pylibs-pdfcheck python3.12 -I paper_g1/check_pdf.py FIG.pdf --print-width-in 5.2 --out preview.png

Needs PyMuPDF only (installed in an isolated --target dir; it is not a project dependency).
"""
import argparse
import json
import sys

import pymupdf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--print-width-in", type=float, required=True, help="width the manuscript prints it at")
    ap.add_argument("--out", required=True, help="PNG rendered at 200 dpi of the PRINTED size")
    a = ap.parse_args()
    doc = pymupdf.open(a.pdf)
    pg = doc[0]
    w_in, h_in = pg.rect.width / 72, pg.rect.height / 72
    fonts = sorted({(f[3], f[1]) for f in pg.get_fonts(full=True)})
    images = pg.get_images(full=True)
    scale = a.print_width_in / w_in
    pg.get_pixmap(matrix=pymupdf.Matrix(scale * 200 / 72, scale * 200 / 72)).save(a.out)
    rep = {"pdf": a.pdf, "page_in": [round(w_in, 2), round(h_in, 2)], "print_scale": round(scale, 3),
           "fonts": fonts, "type3_fonts": [f for f in fonts if f[1].lower() == "type3"],
           "raster_images": len(images), "vector": len(images) == 0}
    print(json.dumps(rep))
    sys.exit(0 if rep["vector"] and not rep["type3_fonts"] else 1)


if __name__ == "__main__":
    main()
