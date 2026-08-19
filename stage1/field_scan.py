"""
Stage 1a, step 6 — the detector response field, and blob tracking.

Two more measured layers, both of which give the overlay things to draw that a
tracker would actually draw.

Response field
--------------
A coarse grid over the canvas. Each cell holds the highest score the detector
produces for a window centred on it, at several window sizes. This is the
detector's opinion about every part of the painting, not only the places it
committed to a box -- a continuous field rather than a list.

It is what drives sweeping scan lines honestly: a line crossing the canvas can
read out the field beneath it, and cells light up because the detector actually
responded there. The field also shows the near-misses: regions that almost
became a face and did not.

Blob tracking
-------------
Connected-component analysis over three segmentations of the painting -- lit
skin, flame-bright, and dark mass. Per blob: centroid, area as a percentage of
canvas, bounding box, equivalent-circle radius, eccentricity and orientation
from image moments. These are the circles and the percentages.

Nothing here is a face detector. A blob is a region of paint that shares a
tonal range, which is the correct claim to make about it.

Usage
-----
    python stage1/field_scan.py --panel A --rectified out/panel_a_rectified.png
"""
from __future__ import annotations

import argparse
import json
import pathlib

import cv2
import numpy as np

MODELS = pathlib.Path(__file__).resolve().parent.parent / "models"
BLAZE = MODELS / "blaze_face_short_range.tflite"

GRID_W, GRID_H = 40, 30          # one cell per centimetre of canvas
WINDOW_FRACS = (0.18, 0.28)      # window sizes, as a fraction of the long edge
MIN_BLOB_FRAC = 3e-5
MAX_BLOB_FRAC = 8e-2


def response_field(rgb: np.ndarray) -> dict:
    import mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision

    H, W = rgb.shape[:2]
    field = np.zeros((GRID_H, GRID_W), np.float32)
    det = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
        base_options=mpp.BaseOptions(model_asset_path=str(BLAZE)),
        min_detection_confidence=0.05))
    try:
        for frac in WINDOW_FRACS:
            win = int(max(W, H) * frac)
            for gy in range(GRID_H):
                for gx in range(GRID_W):
                    cx = int((gx + 0.5) / GRID_W * W)
                    cy = int((gy + 0.5) / GRID_H * H)
                    x0, y0 = max(0, cx - win // 2), max(0, cy - win // 2)
                    x1, y1 = min(W, x0 + win), min(H, y0 + win)
                    crop = np.ascontiguousarray(rgb[y0:y1, x0:x1])
                    if crop.shape[0] < 32 or crop.shape[1] < 32:
                        continue
                    r = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop))
                    if r.detections:
                        s = max(float(d.categories[0].score) for d in r.detections
                                if d.categories)
                        field[gy, gx] = max(field[gy, gx], s)
    finally:
        det.close()

    flat = field[field > 0]
    return {
        "grid": [GRID_W, GRID_H],
        "cell_cm": [40.0 / GRID_W, 30.0 / GRID_H],
        "window_fracs": list(WINDOW_FRACS),
        "detector_min_confidence": 0.05,
        "values": [[round(float(v), 4) for v in row] for row in field],
        "stats": {
            "cells_with_response": int((field > 0).sum()),
            "cells_total": int(field.size),
            "max": round(float(field.max()), 4),
            "median_nonzero": round(float(np.median(flat)), 4) if flat.size else None,
            "p90_nonzero": round(float(np.percentile(flat, 90)), 4) if flat.size else None,
        },
        "note": ("Highest detector score for any window centred on each cell. A "
                 "high cell that never became a box is a near-miss: the detector "
                 "responded to that paint and did not commit."),
    }


def blobs_for(bgr: np.ndarray) -> list:
    H, W = bgr.shape[:2]
    area_px = float(H * W)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    segs = {
        # Lit skin: warm hue, present saturation, upper-mid brightness.
        "lit_skin": ((h >= 3) & (h <= 25) & (s >= 60) & (s <= 200) &
                     (v >= np.percentile(v, 72))),
        # Flame-bright: the top of the value range, any hue.
        "flame_bright": (v >= np.percentile(v, 99.0)),
        # Dark mass: suits, shadow, the ground the figures sit in.
        "dark_mass": (v <= np.percentile(v, 22)),
    }

    out = []
    for name, m in segs.items():
        mask = (m.astype(np.uint8)) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
        for i in range(1, n):
            x, y, w, hh, a = stats[i]
            if not (MIN_BLOB_FRAC * area_px <= a <= MAX_BLOB_FRAC * area_px):
                continue
            comp = (labels[y:y + hh, x:x + w] == i).astype(np.uint8)
            mo = cv2.moments(comp, binaryImage=True)
            if mo["m00"] <= 0:
                continue
            mu20, mu02, mu11 = mo["mu20"] / mo["m00"], mo["mu02"] / mo["m00"], mo["mu11"] / mo["m00"]
            common = np.sqrt(max(0.0, 4 * mu11 ** 2 + (mu20 - mu02) ** 2))
            l1, l2 = (mu20 + mu02 + common) / 2, (mu20 + mu02 - common) / 2
            ecc = float(np.sqrt(max(0.0, 1 - l2 / l1))) if l1 > 0 else 0.0
            theta = float(np.degrees(0.5 * np.arctan2(2 * mu11, mu20 - mu02)))
            cx, cy = cents[i]
            out.append({
                "blob_id": f"{name[:2].upper()}{len(out) + 1:03d}",
                "segment": name,
                "centroid": [round(cx / W, 6), round(cy / H, 6)],
                "box": [round(x / W, 6), round(y / H, 6), round(w / W, 6), round(hh / H, 6)],
                "area_pct_of_canvas": round(100.0 * a / area_px, 4),
                "equiv_radius_frac": round(float(np.sqrt(a / np.pi) / W), 6),
                "eccentricity": round(ecc, 4),
                "orientation_deg": round(theta, 2),
                "fill_ratio": round(float(a) / max(1.0, w * hh), 4),
            })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True, choices=["A", "B"])
    ap.add_argument("--rectified", required=True, type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("out"))
    ap.add_argument("--skip-field", action="store_true")
    args = ap.parse_args()

    bgr = cv2.imread(str(args.rectified), cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"cannot read {args.rectified}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    doc = {
        "schema": "mimesis.field.v1",
        "panel": args.panel,
        "painting_rect_px": [bgr.shape[1], bgr.shape[0]],
        "response_field": None if args.skip_field else response_field(rgb),
        "blobs": blobs_for(bgr),
    }
    counts = {}
    for b in doc["blobs"]:
        counts[b["segment"]] = counts.get(b["segment"], 0) + 1
    doc["blob_counts"] = counts
    (args.out / f"panel_{args.panel.lower()}_field.json").write_text(json.dumps(doc, indent=2))

    print(f"PANEL {args.panel}")
    if doc["response_field"]:
        st = doc["response_field"]["stats"]
        print(f"  response field {GRID_W}x{GRID_H}: {st['cells_with_response']}/{st['cells_total']} "
              f"cells responded, max {st['max']}, median {st['median_nonzero']}, p90 {st['p90_nonzero']}")
    print(f"  blobs: {counts}  total {len(doc['blobs'])}")
    big = sorted(doc["blobs"], key=lambda b: -b["area_pct_of_canvas"])[:5]
    for b in big:
        print(f"    {b['blob_id']} {b['segment']:12} {b['area_pct_of_canvas']:6.3f}%  "
              f"ecc={b['eccentricity']:.3f}  theta={b['orientation_deg']:7.2f}")


if __name__ == "__main__":
    main()
