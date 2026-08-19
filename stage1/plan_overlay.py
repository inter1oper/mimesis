"""
Stage 1a, step 4 — decide, from measured box sizes, what overlay detail each
face can actually carry on the wall.

The brief forbids adding decorative HUD elements the data does not support. The
same principle has a physical form: drawing a 478-point mesh into a face box
that renders 30 projector pixels wide produces a filled blob, not line work, and
filled regions bloom on gloss. Detail that cannot resolve is decoration.

So the tier is computed, never chosen by eye. Feed in the detection output and
the actual projector geometry; get back, per face, the rendered box size in
projector pixels and the highest tier it supports. Those numbers become
`min_box_px` on the overlay cues in the score.

Tiers, from the point spacing needed for a stroke to read as a stroke (~4 px):

    full_mesh    box >= 88 px    478-point tessellation
    contour      box >= 48 px    ~130 points: eye, lip, face oval
    landmarks    box >= 24 px    6 key points
    box_only     below that      bounding box, brackets and numerals

Usage
-----
    python stage1/plan_overlay.py --detection out/panel_a_detection.json \
        --projector 3840x2160 --span-mm 2200 --painting-zone-frac 0.182
"""
from __future__ import annotations

import argparse
import json
import pathlib

TIERS = [("full_mesh", 88), ("contour", 48), ("landmarks", 24), ("box_only", 0)]


def tier_for(px: float) -> str:
    for name, need in TIERS:
        if px >= need:
            return name
    return "box_only"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detection", required=True, type=pathlib.Path)
    ap.add_argument("--projector", required=True, help="e.g. 3840x2160")
    ap.add_argument("--span-mm", required=True, type=float,
                    help="Physical width the projector's full raster covers.")
    ap.add_argument("--painting-zone-frac", type=float, default=None,
                    help="Fraction of the projector's width the painting occupies. "
                         "Omit to derive it from canvas_cm and span-mm.")
    ap.add_argument("--out", type=pathlib.Path, default=None)
    args = ap.parse_args()

    det = json.loads(args.detection.read_text())
    pw, ph = (int(v) for v in args.projector.lower().split("x"))
    mm_per_px = args.span_mm / pw

    if args.painting_zone_frac is not None:
        zone_px = args.painting_zone_frac * pw
    else:
        canvas = det.get("canvas_cm")
        if not canvas:
            raise SystemExit("detection file has no canvas_cm; pass --painting-zone-frac")
        zone_px = (canvas[0] * 10.0) / mm_per_px

    faces = []
    for f in det["faces"]:
        x, y, w, h = f["box"]
        bw, bh = w * zone_px, h * zone_px * (det["painting_rect_px"][1] /
                                             det["painting_rect_px"][0])
        governing = min(bw, bh)
        t = tier_for(governing)
        faces.append({
            "face_id": f["face_id"],
            "box_px": [round(bw, 1), round(bh, 1)],
            "governing_px": round(governing, 1),
            "box_mm": [round(bw * mm_per_px, 1), round(bh * mm_per_px, 1)],
            "tier": t,
            "mesh_available": bool(f.get("mesh_ok")),
            "effective_tier": t if f.get("mesh_ok") else
                              ("box_only" if t in ("full_mesh", "contour", "landmarks") else t),
            "note": None if f.get("mesh_ok") else
                    "no converged mesh; nothing to draw above box_only regardless of size",
        })

    doc = {
        "schema": "mimesis.overlayplan.v1",
        "panel": det["panel"],
        "projector": {"raster_px": [pw, ph], "span_mm": args.span_mm,
                      "mm_per_px": round(mm_per_px, 4)},
        "painting_zone_px_wide": round(zone_px, 1),
        "tier_thresholds_px": dict(TIERS),
        "rationale": ("A stroke needs roughly 4 projector px between adjacent mesh "
                      "points to read as a stroke. Below that the mesh fills, and "
                      "fills bloom on gloss. Tier is computed from measured box "
                      "size, never chosen by eye."),
        "summary": {t: sum(1 for f in faces if f["effective_tier"] == t)
                    for t, _ in TIERS},
        "faces": faces,
    }
    out = args.out or args.detection.parent / f"panel_{det['panel'].lower()}_overlay_plan.json"
    out.write_text(json.dumps(doc, indent=2))
    print(json.dumps({k: doc[k] for k in
                      ("panel", "projector", "painting_zone_px_wide", "summary")}, indent=2))
    for f in faces:
        print(f'  {f["face_id"]}  {f["box_px"][0]:6.1f} x {f["box_px"][1]:6.1f} px  '
              f'({f["box_mm"][0]:5.1f} x {f["box_mm"][1]:5.1f} mm)  {f["effective_tier"]}')


if __name__ == "__main__":
    main()
