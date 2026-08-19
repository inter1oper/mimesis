"""
Stage 1a, step 0 — establish the detectors' false-positive floor.

The brief says the first number to look at is the detector's confidence on
Panel B faces, and that a high one is the strongest single data point. A
confidence is only strong relative to what the same detector scores on input
containing no faces at all. Without that null, "0.71 on Panel B" is a number
with no scale.

So: run the identical detection settings over face-free control input and
record the highest score each detector will produce anyway. Two controls:

  synthetic  procedurally generated abstract fields in the paintings' own
             tonal range -- dark ground, warm highlights, small bright specks.
             Reproducible from a seed, no image files needed.
  crop       optional, and the better control: face-free regions of the actual
             paintings, given as --crop panel:x,y,w,h in normalised painting
             coordinates. Same paint, same varnish, same camera, no faces.

Whatever comes out is the floor. A Panel B detection scoring below it is not
evidence of anything.

Usage
-----
    python stage1/calibrate_noise_floor.py --trials 12 --out out/
    python stage1/calibrate_noise_floor.py --rectified out/panel_a_rectified.png \
        --crop 0.02,0.60,0.18,0.30 --crop 0.80,0.05,0.18,0.20 --out out/
"""
from __future__ import annotations

import argparse
import json
import pathlib

import cv2
import numpy as np

import detect_faces as D  # noqa: E402  (same directory)


def synthetic(seed: int, w: int = 2400, h: int = 1800) -> np.ndarray:
    """Abstract field in the paintings' tonal range. No faces, by construction."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 26, np.uint8)
    for _ in range(rng.integers(18, 34)):
        c = (int(rng.integers(20, 90)), int(rng.integers(40, 130)), int(rng.integers(80, 210)))
        cv2.ellipse(img, (int(rng.integers(0, w)), int(rng.integers(0, h))),
                    (int(rng.integers(60, 340)), int(rng.integers(60, 340))),
                    float(rng.integers(0, 180)), 0, 360, c, -1)
    img = cv2.GaussianBlur(img, (0, 0), 9)
    for _ in range(rng.integers(20, 60)):  # candle-scale highlights
        cv2.circle(img, (int(rng.integers(0, w)), int(rng.integers(0, h))),
                   int(rng.integers(4, 14)), (238, 246, 252), -1)
    img = cv2.GaussianBlur(img, (0, 0), 2)
    return img


def scores_on(bgr: np.ndarray) -> dict:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    raws = D.run_blazeface(rgb) + D.run_yunet(bgr)
    out = {}
    for src in ("blazeface", "yunet"):
        s = [r.score for r in raws if r.source == src]
        out[src] = {"n": len(s), "max": round(max(s), 4) if s else None,
                    "p95": round(float(np.percentile(s, 95)), 4) if s else None}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--rectified", type=pathlib.Path, default=None)
    ap.add_argument("--crop", action="append", default=[],
                    help="Face-free control region, normalised 'x,y,w,h'. Repeatable.")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("out"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    controls = []
    for i in range(args.trials):
        controls.append({"control": "synthetic", "seed": i, **scores_on(synthetic(i))})

    if args.crop:
        if not args.rectified:
            raise SystemExit("--crop needs --rectified")
        img = cv2.imread(str(args.rectified), cv2.IMREAD_COLOR)
        H, W = img.shape[:2]
        for spec in args.crop:
            x, y, w, h = (float(v) for v in spec.split(","))
            sub = img[int(y * H):int((y + h) * H), int(x * W):int((x + w) * W)]
            controls.append({"control": "crop", "region": [x, y, w, h],
                             "source": str(args.rectified), **scores_on(sub)})

    floor = {}
    for src in ("blazeface", "yunet"):
        m = [c[src]["max"] for c in controls if c[src]["max"] is not None]
        floor[src] = {
            "false_positive_ceiling": round(max(m), 4) if m else None,
            "controls_with_any_detection": len(m),
            "controls_total": len(controls),
            "interpretation": (
                "Highest score this detector produced on input containing no faces, "
                "at the same thresholds used on the paintings. A painting detection "
                "at or below this score is not evidence of a face."
            ),
        }

    doc = {"schema": "mimesis.noisefloor.v1",
           "thresholds": {"blazeface": D.BLAZE_MIN_CONF, "yunet": D.YUNET_MIN_CONF},
           "floor": floor, "controls": controls}
    (args.out / "noise_floor.json").write_text(json.dumps(doc, indent=2))
    print(json.dumps({"thresholds": doc["thresholds"], "floor": floor}, indent=2))


if __name__ == "__main__":
    main()
