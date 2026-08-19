"""
Stage 1a, step 3 — candle-flame regions, and flames with no face to attach to.

The brief asks specifically whether there is a region where a hand holds a
candle with no face to attach it to. That question is answerable from pixels
plus the Stage 1a face output, so it is answered here rather than asserted.

Method
------
A flame in these paintings is a small, near-saturated-bright, warm blob sitting
in a much darker neighbourhood. We threshold on that description, take
connected components, and record every candidate with its measured features.
Ceiling fixtures in the upper part of both canvases meet the brightness test
too, so they are *not* silently dropped: every blob is emitted with its
features and a rule-based `kind`, and the rule is written down in the output so
it can be argued with.

Orphan test: for each flame, the distance from the flame centroid to the
nearest detected face box, expressed in units of that face's box height.
Flames whose nearest face is further than `--orphan-threshold` box-heights away
are flagged `unattached`. This is a measurement with a stated threshold, not a
verdict; the raw distance is always present so the threshold can be moved.

Usage
-----
    python stage1/detect_flames.py --panel A \
        --rectified out/panel_a_rectified.png \
        --detection out/panel_a_detection.json \
        --out out/
"""
from __future__ import annotations

import argparse
import json
import pathlib

import cv2
import numpy as np

# All tunables surface in the output document.
V_PERCENTILE = 99.2      # brightness percentile that counts as "flame bright"
MIN_AREA_FRAC = 2.0e-6   # of painting area
MAX_AREA_FRAC = 3.0e-3
SURROUND_RATIO = 0.62    # blob must be this much brighter than its surround
WARM_HUE = (0, 45)       # OpenCV H range for candle warm; near-white passes on low S


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True, choices=["A", "B"])
    ap.add_argument("--rectified", required=True, type=pathlib.Path)
    ap.add_argument("--detection", required=True, type=pathlib.Path)
    ap.add_argument("--faces", choices=["consensus", "all"], default="consensus",
                    help="Which detections count as a face for the orphan test. "
                         "'consensus' uses only faces both detectors found, which "
                         "is the honest default: at a permissive threshold the "
                         "single-detector boxes are near the noise floor, and "
                         "letting them count means any candle lands near some box "
                         "and nothing is ever unattached.")
    ap.add_argument("--orphan-threshold", type=float, default=2.5,
                    help="nearest-face distance, in face-box heights, beyond which "
                         "a flame is flagged unattached")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("out"))
    args = ap.parse_args()

    bgr = cv2.imread(str(args.rectified), cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"cannot read {args.rectified}")
    H, W = bgr.shape[:2]
    area_px = float(H * W)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)

    thresh = float(np.percentile(v, V_PERCENTILE))
    mask = (v >= thresh).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)

    blobs = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if not (MIN_AREA_FRAC * area_px <= a <= MAX_AREA_FRAC * area_px):
            continue
        cx, cy = cents[i]
        comp = labels[y:y + h, x:x + w] == i
        # surround = a dilated ring around the blob
        pad = max(4, int(0.9 * max(w, h)))
        sy0, sy1 = max(0, y - pad), min(H, y + h + pad)
        sx0, sx1 = max(0, x - pad), min(W, x + w + pad)
        ring = v[sy0:sy1, sx0:sx1].copy()
        rm = np.ones(ring.shape, bool)
        rm[y - sy0:y - sy0 + h, x - sx0:x - sx0 + w] &= ~comp
        surround_v = float(ring[rm].mean()) if rm.any() else 0.0
        blob_v = float(v[y:y + h, x:x + w][comp].mean())
        if surround_v > 0 and (blob_v - surround_v) / max(blob_v, 1e-6) < SURROUND_RATIO * 0.5:
            contrast_ok = False
        else:
            contrast_ok = True
        hue = float(np.median(hsv[:, :, 0][y:y + h, x:x + w][comp]))
        sat = float(np.median(hsv[:, :, 1][y:y + h, x:x + w][comp]))
        aspect = h / max(w, 1)
        warm = (WARM_HUE[0] <= hue <= WARM_HUE[1]) or sat < 40
        # Rule, stated plainly: a flame is taller than it is wide, warm or
        # near-white, high local contrast, and not a wide horizontal fixture
        # in the top fifth of the canvas.
        top_fixture = (cy / H) < 0.2 and aspect < 1.2
        kind = ("ceiling_fixture" if top_fixture else
                "flame" if (aspect >= 1.0 and warm and contrast_ok) else "bright_other")
        blobs.append({
            "blob_id": f"{args.panel}F{len(blobs) + 1:02d}",
            "kind": kind,
            "centroid": [round(cx / W, 6), round(cy / H, 6)],
            "box": [round(x / W, 6), round(y / H, 6), round(w / W, 6), round(h / H, 6)],
            "area_frac": round(a / area_px, 8),
            "aspect_h_over_w": round(float(aspect), 3),
            "median_hue": round(hue, 1),
            "median_sat": round(sat, 1),
            "mean_value": round(blob_v, 1),
            "surround_mean_value": round(surround_v, 1),
        })

    det = json.loads(args.detection.read_text())
    all_faces = det["faces"]
    faces = ([f for f in all_faces if len(f.get("detectors", {})) >= 2]
             if args.faces == "consensus" else all_faces)
    for b in blobs:
        if b["kind"] != "flame":
            b["nearest_face"] = None
            continue
        cx, cy = b["centroid"]
        best = None
        for f in faces:
            fx, fy, fw, fh = f["box"]
            d = float(np.hypot(cx - (fx + fw / 2), cy - (fy + fh / 2)) / max(fh, 1e-6))
            if best is None or d < best[1]:
                best = (f["face_id"], d)
        b["nearest_face"] = None if best is None else {
            "face_id": best[0],
            "distance_in_face_heights": round(best[1], 3),
            "unattached": bool(best[1] > args.orphan_threshold),
        }

    flames = [b for b in blobs if b["kind"] == "flame"]
    orphans = [b for b in flames if b.get("nearest_face", {}) and b["nearest_face"]["unattached"]]
    doc = {
        "schema": "mimesis.flames.v1",
        "panel": args.panel,
        "painting_rect_px": [W, H],
        "parameters": {
            "v_percentile": V_PERCENTILE, "value_threshold": round(thresh, 1),
            "min_area_frac": MIN_AREA_FRAC, "max_area_frac": MAX_AREA_FRAC,
            "warm_hue_range": list(WARM_HUE), "orphan_threshold_face_heights":
                args.orphan_threshold,
            "faces_counted": args.faces,
            "faces_available": len(all_faces),
            "faces_used": len(faces),
            "kind_rule": ("ceiling_fixture: centroid above y=0.2 and aspect<1.2; "
                          "flame: aspect>=1.0 and (warm hue or sat<40) and high local "
                          "contrast; bright_other: everything else"),
        },
        "counts": {
            "blobs": len(blobs),
            "flames": len(flames),
            "ceiling_fixtures": sum(1 for b in blobs if b["kind"] == "ceiling_fixture"),
            "bright_other": sum(1 for b in blobs if b["kind"] == "bright_other"),
            "flames_unattached": len(orphans),
        },
        "blobs": blobs,
    }
    path = args.out / f"panel_{args.panel.lower()}_flames.json"
    path.write_text(json.dumps(doc, indent=2))
    print(json.dumps({k: doc[k] for k in ("panel", "counts")}, indent=2))
    for b in orphans:
        print(f"  UNATTACHED {b['blob_id']} at "
              f"({b['centroid'][0]:.3f}, {b['centroid'][1]:.3f})  "
              f'nearest {b["nearest_face"]["face_id"]} '
              f'{b["nearest_face"]["distance_in_face_heights"]} face-heights away')


if __name__ == "__main__":
    main()
