"""
Stage 1a, step 8 — detection stability, and the tracker state each face earns.

Why this exists
---------------
The reference look flashes. It is worth being precise about why real tracking
displays flash, because the answer decides whether flashing here is data or
decoration.

A multi-object tracker does not hold a steady box. Detections are unstable
frame to frame, so trackers run a state machine: a new track is *tentative* and
is not reported downstream until it survives n_init consecutive detections; it
is promoted to *confirmed*; and it is *deleted* after max_age frames without a
match. Tentative tracks appear and vanish. Confirmed tracks hold. That churn is
what the flashing in a real tracker display actually is.

These are paintings, not video. There is no next frame. But there is an
equivalent question with a real answer: how stable is each detection under
small changes that should not matter? Scale, rotation, exposure, recompression.
A face that survives every perturbation is confirmed. One that appears in half
of them is tentative, and in a tracker it would flicker. One that appears twice
in twenty-four flashes once and dies.

So the flash rate for each face is measured, not chosen. A box blinking on the
oil is blinking at the rate its own evidence supports.

Perturbations: 3 scales x 3 rotations x 3 exposures, plus JPEG recompression at
two qualities. 27 geometric-photometric passes, applied to a padded crop around
each consensus face, both detectors run on each.

Usage
-----
    python stage1/stability.py --out out/
"""
from __future__ import annotations

import argparse
import json
import pathlib

import cv2
import numpy as np

MODELS = pathlib.Path(__file__).resolve().parent.parent / "models"
BLAZE = MODELS / "blaze_face_short_range.tflite"
YUNET = MODELS / "face_detection_yunet_2023mar.onnx"

SCALES = (0.85, 1.0, 1.15)
ROTATIONS = (-3.0, 0.0, 3.0)
GAMMAS = (0.85, 1.0, 1.15)
JPEG_QUALITIES = (55, 92)
CROP_PAD = 0.6

# Tracker state machine thresholds, in the DeepSORT sense, expressed as a
# fraction of passes survived rather than consecutive frames.
CONFIRMED_AT = 0.80
TENTATIVE_AT = 0.35


def perturb(img: np.ndarray, scale: float, rot: float, gamma: float, jpeg: int | None):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), rot, scale)
    out = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_REPLICATE)
    if gamma != 1.0:
        lut = np.clip(((np.arange(256) / 255.0) ** (1.0 / gamma)) * 255.0, 0, 255).astype(np.uint8)
        out = cv2.LUT(out, lut)
    if jpeg is not None:
        ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, jpeg])
        if ok:
            out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("out"))
    ap.add_argument("--faces", choices=["consensus", "all"], default="all",
                    help="Consensus faces turn out to be perfectly stable, so the "
                         "interesting population is all merged detections: the "
                         "single-detector boxes near the noise floor are where the "
                         "flicker actually is.")
    args = ap.parse_args()

    import mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision

    blaze = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
        base_options=mpp.BaseOptions(model_asset_path=str(BLAZE)),
        min_detection_confidence=0.20))

    passes = [(s, r, g, None) for s in SCALES for r in ROTATIONS for g in GAMMAS]
    passes += [(1.0, 0.0, 1.0, q) for q in JPEG_QUALITIES]

    doc = {"schema": "mimesis.stability.v1",
           "faces_scope": None,
           "passes": {"scales": list(SCALES), "rotations_deg": list(ROTATIONS),
                      "gammas": list(GAMMAS), "jpeg_qualities": list(JPEG_QUALITIES),
                      "total": len(passes)},
           "state_machine": {
               "confirmed_at_hit_rate": CONFIRMED_AT,
               "tentative_at_hit_rate": TENTATIVE_AT,
               "reference": ("Track states follow the DeepSORT scheme: tentative "
                             "tracks are not reported until they survive n_init "
                             "detections, confirmed tracks hold, tracks are deleted "
                             "after max_age misses. Here the consecutive-frame "
                             "criterion is replaced by fraction of perturbation "
                             "passes survived, since a painting has no next frame."),
               "flash_rule": ("flash_hz is derived: a confirmed face does not flash, "
                              "a tentative face flashes at a rate proportional to how "
                              "often it drops out, a lost face appears once and dies. "
                              "The renderer must not invent a rate of its own."),
           },
           "panels": {}}
    doc["faces_scope"] = args.faces

    try:
        for P in ("A", "B"):
            p = P.lower()
            det = json.loads((args.out / f"panel_{p}_detection.json").read_text())
            img = cv2.imread(str(args.out / f"panel_{p}_rectified.png"), cv2.IMREAD_COLOR)
            H, W = img.shape[:2]
            rows = []
            for f in det["faces"]:
                if args.faces == "consensus" and len(f["detectors"]) < 2:
                    continue
                x, y, w, h = f["box"]
                x0 = int(max(0, (x - w * CROP_PAD) * W)); y0 = int(max(0, (y - h * CROP_PAD) * H))
                x1 = int(min(W, (x + w * (1 + CROP_PAD)) * W))
                y1 = int(min(H, (y + h * (1 + CROP_PAD)) * H))
                crop = img[y0:y1, x0:x1]
                if crop.size == 0:
                    continue
                ch, cw = crop.shape[:2]
                # Where the face centre sits inside the crop, in crop coords.
                fc = np.array([((x + w / 2) * W - x0) / cw, ((y + h / 2) * H - y0) / ch])

                hits = {"blazeface": [], "yunet": []}
                for (s, r, g, q) in passes:
                    pim = perturb(crop, s, r, g, q)
                    rgb = np.ascontiguousarray(cv2.cvtColor(pim, cv2.COLOR_BGR2RGB))
                    res = blaze.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                    best = 0.0
                    for d in res.detections:
                        bb = d.bounding_box
                        c = np.array([(bb.origin_x + bb.width / 2) / cw,
                                      (bb.origin_y + bb.height / 2) / ch])
                        if np.linalg.norm(c - fc) < 0.18 and d.categories:
                            best = max(best, float(d.categories[0].score))
                    hits["blazeface"].append(best)

                    yn = cv2.FaceDetectorYN.create(str(YUNET), "", (cw, ch), 0.30, 0.3, 5000)
                    yn.setInputSize((cw, ch))
                    _, faces = yn.detect(pim)
                    best = 0.0
                    if faces is not None:
                        for fr in faces:
                            c = np.array([(fr[0] + fr[2] / 2) / cw, (fr[1] + fr[3] / 2) / ch])
                            if np.linalg.norm(c - fc) < 0.18:
                                best = max(best, float(fr[-1]))
                    hits["yunet"].append(best)

                row = {"face_id": f["face_id"],
                       "consensus": len(f["detectors"]) >= 2,
                       "detectors_at_source": sorted(f["detectors"])}
                for k, v in hits.items():
                    arr = np.array(v)
                    seen = arr > 0
                    row[k] = {
                        "hit_rate": round(float(seen.mean()), 4),
                        "score_mean_when_seen": round(float(arr[seen].mean()), 4) if seen.any() else None,
                        "score_std_when_seen": round(float(arr[seen].std()), 4) if seen.any() else None,
                        "score_min_when_seen": round(float(arr[seen].min()), 4) if seen.any() else None,
                    }
                combined = max(row["blazeface"]["hit_rate"], row["yunet"]["hit_rate"])
                both = min(row["blazeface"]["hit_rate"], row["yunet"]["hit_rate"])
                row["hit_rate_either"] = round(combined, 4)
                row["hit_rate_both"] = round(both, 4)
                row["track_state"] = ("confirmed" if combined >= CONFIRMED_AT else
                                      "tentative" if combined >= TENTATIVE_AT else "lost")
                # Flash rate follows from dropout, and only from dropout.
                row["flash_hz"] = (0.0 if row["track_state"] == "confirmed" else
                                   round(6.0 * (1.0 - combined), 2))
                rows.append(row)
            doc["panels"][P] = rows
    finally:
        blaze.close()

    (args.out / "stability.json").write_text(json.dumps(doc, indent=2))

    for P, rows in doc["panels"].items():
        st = {}
        for r in rows:
            st[r["track_state"]] = st.get(r["track_state"], 0) + 1
        cs = {}
        for r in rows:
            k = ("consensus" if r["consensus"] else "single-detector") + "/" + r["track_state"]
            cs[k] = cs.get(k, 0) + 1
        print(f"\nPANEL {P}  ({len(rows)} detections, {doc['passes']['total']} passes each)")
        print(f"  states: {st}")
        print(f"  by origin: {cs}")
        for r in sorted(rows, key=lambda r: r["hit_rate_either"])[:14]:
            print(f"  {r['face_id']}  either={r['hit_rate_either']:.2f} both={r['hit_rate_both']:.2f}  "
                  f"BF={r['blazeface']['hit_rate']:.2f} YN={r['yunet']['hit_rate']:.2f}  "
                  f"{r['track_state']:9} flash={r['flash_hz']:4}Hz  "
                  f"{'consensus' if r['consensus'] else 'single'}")


if __name__ == "__main__":
    main()
