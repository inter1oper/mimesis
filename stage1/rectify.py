"""
Stage 1a, step 1 — rectify a photograph of a painting to the painting's own rectangle.

Why this exists
---------------
The brief requires every coordinate to be normalised 0-1 against *the painting's
own rectangle*, never against the source photo's pixel dimensions. The supplied
photographs are hand-held shots of an unframed canvas lying on a floor: the
canvas occupies part of the frame, sits at a slight angle, and is keystoned by
the camera. Normalising against the photo would bake that keystone into every
box and landmark, and the projector-side matrix3d warp would then be correcting
a distortion that is already half-corrected. So we rectify first, once, and
everything downstream lives in painting space.

Output of this step is the canonical painting raster: a plane image whose
extents are exactly the canvas edges, at a known px/cm scale.

Corner provenance is recorded. Automatic quad detection on a canvas
photographed against a floor is not reliable enough to trust silently, so the
QA overlay must be eyeballed and `--corners` used to override when auto fails.

Usage
-----
    python stage1/rectify.py \
        --photo data/paintings/panel_a_photo.jpg \
        --panel A \
        --width-cm 40 --height-cm 30 --px-per-cm 120 \
        --out out/

    # override auto-detected corners (photo pixel coords, TL TR BR BL)
        --corners "120,60 1980,55 1975,1480 118,1490"
"""
from __future__ import annotations

import argparse
import json
import pathlib

import cv2
import numpy as np


def order_quad(pts: np.ndarray) -> np.ndarray:
    """Return quad points ordered TL, TR, BR, BL."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype=np.float32,
    )


def auto_detect_quad(bgr: np.ndarray) -> tuple[np.ndarray | None, dict]:
    """
    Find the canvas quadrilateral.

    Strategy: the canvas is the single large bright-ish rectangle against a
    comparatively flat background. We work on a downscaled copy, take the
    strongest edges, close them so the canvas outline becomes one contour, and
    keep the largest 4-gon that covers a plausible fraction of the frame.

    Returns (quad_in_full_res_coords | None, diagnostics).
    """
    h, w = bgr.shape[:2]
    scale = 1000.0 / max(h, w)
    small = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    v = float(np.median(gray))
    lo, hi = int(max(0, 0.66 * v)), int(min(255, 1.33 * v))
    edges = cv2.Canny(gray, lo, hi)
    edges = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    )

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = small.shape[0] * small.shape[1]
    best, best_area = None, 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.25 * frame_area:
            continue
        peri = cv2.arcLength(c, True)
        for eps in (0.02, 0.03, 0.05):
            approx = cv2.approxPolyDP(c, eps * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx) and area > best_area:
                best, best_area = approx.reshape(4, 2).astype(np.float32), area
                break

    diag = {
        "frame_area_fraction": round(best_area / frame_area, 4) if best is not None else None,
        "canny_thresholds": [lo, hi],
        "contours_considered": len(contours),
    }
    if best is None:
        return None, diag
    return order_quad(best / scale), diag


def rectify(photo: pathlib.Path, quad: np.ndarray, out_w: int, out_h: int):
    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32
    )
    H = cv2.getPerspectiveTransform(order_quad(quad), dst)
    bgr = cv2.imread(str(photo), cv2.IMREAD_COLOR)
    warped = cv2.warpPerspective(bgr, H, (out_w, out_h), flags=cv2.INTER_CUBIC)
    return warped, H


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo", required=True, type=pathlib.Path)
    ap.add_argument("--panel", required=True, choices=["A", "B"])
    ap.add_argument("--width-cm", type=float, default=40.0)
    ap.add_argument("--height-cm", type=float, default=30.0)
    ap.add_argument("--px-per-cm", type=float, default=120.0)
    ap.add_argument(
        "--corners",
        default=None,
        help='Manual override, photo pixel coords: "x,y x,y x,y x,y" as TL TR BR BL.',
    )
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("out"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    bgr = cv2.imread(str(args.photo), cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"cannot read {args.photo}")

    if args.corners:
        pts = [tuple(float(v) for v in p.split(",")) for p in args.corners.split()]
        if len(pts) != 4:
            raise SystemExit("--corners needs exactly 4 'x,y' pairs")
        quad, method, diag = np.array(pts, dtype=np.float32), "manual", {}
    else:
        quad, diag = auto_detect_quad(bgr)
        method = "auto_quad"
        if quad is None:
            raise SystemExit(
                "Automatic canvas-quad detection failed. Re-run with --corners "
                "giving the four canvas corners in photo pixels (TL TR BR BL)."
            )

    out_w = int(round(args.width_cm * args.px_per_cm))
    out_h = int(round(args.height_cm * args.px_per_cm))
    warped, H = rectify(args.photo, quad, out_w, out_h)

    stem = f"panel_{args.panel.lower()}"
    rect_path = args.out / f"{stem}_rectified.png"
    cv2.imwrite(str(rect_path), warped)

    qa = bgr.copy()
    cv2.polylines(qa, [order_quad(quad).astype(np.int32)], True, (255, 255, 0), 6)
    for i, p in enumerate(order_quad(quad).astype(int)):
        cv2.circle(qa, tuple(p), 18, (0, 255, 255), -1)
        cv2.putText(qa, "TL TR BR BL".split()[i], tuple(p + 24), cv2.FONT_HERSHEY_SIMPLEX,
                    1.6, (0, 255, 255), 4)
    qa_path = args.out / f"{stem}_rectify_qa.jpg"
    cv2.imwrite(str(qa_path), qa, [cv2.IMWRITE_JPEG_QUALITY, 88])

    meta = {
        "panel": args.panel,
        "source_photo": str(args.photo),
        "source_photo_px": [int(bgr.shape[1]), int(bgr.shape[0])],
        "canvas_cm": [args.width_cm, args.height_cm],
        "px_per_cm": args.px_per_cm,
        "rectified_px": [out_w, out_h],
        "corner_method": method,
        "corners_photo_px": order_quad(quad).tolist(),
        "homography_photo_to_painting": H.tolist(),
        "auto_detect_diagnostics": diag,
        "rectified_image": str(rect_path),
        "qa_image": str(qa_path),
        "note": (
            "Verify qa_image before trusting downstream coordinates. If the drawn "
            "quad does not sit on the canvas edges, re-run with --corners."
        ),
    }
    meta_path = args.out / f"{stem}_rectify.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
