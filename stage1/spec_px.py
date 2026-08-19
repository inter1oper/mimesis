"""
Translate the physical rendering spec into pixels for a given projector.

Every rendering constraint in the brief is written in CSS pixels: 1-1.5px
strokes, 8-11px numerals, 28-40px serif. A pixel is not a unit of size until a
projector is chosen. At 4K across 3.5m one pixel is 0.91mm; at 1080p across the
same wall it is 1.82mm. The identical stylesheet produces two different pieces.

So the spec is anchored in millimetres on the wall, and the pixel values are
generated from the projector geometry rather than typed in by hand. Change the
projector, regenerate, and the work looks the same.

Millimetre anchors, derived from the brief's pixel values at the density they
were evidently written for (~0.65 mm/px):

    overlay stroke      0.65 - 1.0 mm    hairline, survives gloss, never fills
    overlay numerals    5 - 7 mm cap     legible at arm's length, mush at 4m
    reasoning trace     6.5 - 8.5 mm     telemetry, faster and dimmer
    answer serif        13 - 26 mm cap   the only thing meant to be read

Usage
-----
    python stage1/spec_px.py --projector 3840x2160 --span-mm 3500
    python stage1/spec_px.py --projector 1920x1080 --span-mm 3500 --out out/spec.json
"""
from __future__ import annotations

import argparse
import json
import pathlib

# name -> (min_mm, max_mm, kind). "cap" values are cap heights; CSS font-size is
# cap / 0.7 for a typical serif or mono.
SPEC_MM = {
    "overlay_stroke":      (0.65, 1.00, "length"),
    "overlay_numeral":     (5.0,  7.0,  "cap"),
    "reasoning_text":      (6.5,  8.5,  "cap"),
    "answer_serif":        (13.0, 26.0, "cap"),
    "answer_caret_width":  (0.65, 1.00, "length"),
    "blur_reveal_radius":  (3.5,  4.5,  "length"),
}
CAP_RATIO = 0.7          # cap height / font-size
TARGET_MEASURE = 65      # characters per line, comfortable for a slow reveal
AVG_ADVANCE_EM = 0.5     # average glyph advance for a text serif


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--projector", required=True, help="e.g. 3840x2160")
    ap.add_argument("--span-mm", required=True, type=float)
    ap.add_argument("--canvas-cm", nargs=2, type=float, default=[40.0, 30.0])
    ap.add_argument("--measure", type=int, default=TARGET_MEASURE)
    ap.add_argument("--out", type=pathlib.Path, default=None)
    args = ap.parse_args()

    pw, ph = (int(v) for v in args.projector.lower().split("x"))
    mmpx = args.span_mm / pw
    painting_px = (args.canvas_cm[0] * 10.0) / mmpx

    spec = {}
    for name, (lo, hi, kind) in SPEC_MM.items():
        f = (1.0 if kind == "length" else 1.0 / CAP_RATIO)
        spec[name] = {
            "mm": [lo, hi],
            "px": [round(lo / mmpx * f, 2), round(hi / mmpx * f, 2)],
            "kind": kind,
        }

    # Answer type sized so a column holds `measure` characters, then checked
    # against the physical anchor rather than the other way round.
    free_px = pw - 2 * painting_px
    col_px = free_px / 2
    type_px = col_px / (args.measure * AVG_ADVANCE_EM)
    cap_mm = type_px * CAP_RATIO * mmpx
    lo, hi = SPEC_MM["answer_serif"][0], SPEC_MM["answer_serif"][1]

    doc = {
        "schema": "mimesis.specpx.v1",
        "projector": {"raster_px": [pw, ph], "span_mm": args.span_mm,
                      "mm_per_px": round(mmpx, 4),
                      "raster_height_mm": round(ph * mmpx)},
        "painting_zone": {
            "px_wide": round(painting_px), "px_tall": round(painting_px * args.canvas_cm[1] / args.canvas_cm[0]),
            "fraction_of_width_both_panels": round(2 * painting_px / pw, 3),
        },
        "text_column": {
            "px_wide_if_filling": round(col_px),
            "measure_chars": args.measure,
            "answer_font_px": round(type_px, 1),
            "answer_cap_mm": round(cap_mm, 1),
            "comfortable_reading_distance_m": round(cap_mm * 200 / 1000, 1),
            "within_physical_anchor": bool(lo <= cap_mm <= hi),
            "note": (None if lo <= cap_mm <= hi else
                     f"cap {cap_mm:.1f}mm is outside the {lo}-{hi}mm anchor at a "
                     f"{args.measure}-char measure; either narrow the column and "
                     f"leave the surround black, or accept the larger type"),
        },
        "spec": spec,
        "rationale": ("Anchors are millimetres on the wall. Pixel values are "
                      "generated for this projector, not typed by hand, so "
                      "changing the projector does not silently change the work."),
    }
    out = args.out
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=2))
    print(json.dumps(doc, indent=2))


if __name__ == "__main__":
    main()
