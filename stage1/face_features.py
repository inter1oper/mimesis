"""
Stage 1a, step 9 — per-feature geometry: eyes, brows, nose, mouth, irises.

The mesh has been sitting in detection.json as 478 undifferentiated points. The
canonical MediaPipe topology names them, so each feature can be given its own
box, centre and measurements, and the overlay can annotate an eye as an eye.

Per feature: bounding box normalised to the painting, centre, size in pixels and
in interocular units, and where it sits on the face as a fraction of face width
and height. Scale-free measures are given alongside pixel ones so a background
face and a foreground face can be compared.

Derived measures worth having:

  eye aspect ratio    vertical opening over horizontal width. Near zero is a
                      closed or squinting eye.
  mouth aspect ratio  the same for the inner lip contour; separates a closed
                      mouth from an open one without asking a classifier.
  iris ratio          iris diameter over interocular distance. In adults the
                      iris is close to a constant 11.7mm and the interocular
                      distance close to 63mm, so anatomy puts this near 0.186.
                      A painted face is free to depart from it. Measuring the
                      departure, and its spread within a panel, asks whether
                      the source held anatomy steady -- and the answer is not
                      knowable in advance, which is why it is worth measuring.
  gaze offset         iris centre minus eye centre, in eye widths. Which way
                      the eye is pointed inside its own socket.

Usage
-----
    python stage1/face_features.py --out out/
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

# Canonical MediaPipe FaceMesh index groups.
G = {
    "eye_r":   [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    "eye_l":   [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
    "brow_r":  [46, 53, 52, 65, 55, 70, 63, 105, 66, 107],
    "brow_l":  [276, 283, 282, 295, 285, 300, 293, 334, 296, 336],
    "nose":    [168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 327, 129, 358, 45, 275, 220, 440],
    "lips_out": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267,
                 0, 37, 39, 40, 185],
    "lips_in": [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312,
                13, 82, 81, 80, 191],
    "oval":    [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
                378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
                162, 21, 54, 103, 67, 109],
}
IRIS_R, IRIS_L = [469, 470, 471, 472], [474, 475, 476, 477]
IRIS_C_R, IRIS_C_L = 468, 473
# vertical / horizontal pairs for aspect ratios
EAR_R = ((159, 145), (33, 133))
EAR_L = ((386, 374), (362, 263))
MAR = ((13, 14), (78, 308))
ANATOMICAL_IRIS_RATIO = 11.7 / 63.0     # iris diameter over interocular, adult


def box_of(pts, ids, W, H):
    p = np.array([[pts[i][0], pts[i][1]] for i in ids if i < len(pts)])
    if not len(p):
        return None
    x0, y0 = p.min(axis=0)
    x1, y1 = p.max(axis=0)
    return {"box": [round(float(x0), 5), round(float(y0), 5),
                    round(float(x1 - x0), 5), round(float(y1 - y0), 5)],
            "centre": [round(float((x0 + x1) / 2), 5), round(float((y0 + y1) / 2), 5)],
            "w_px": round(float((x1 - x0) * W), 1), "h_px": round(float((y1 - y0) * H), 1)}


def dist(pts, a, b, W, H):
    p, q = pts[a], pts[b]
    return float(np.hypot((p[0] - q[0]) * W, (p[1] - q[1]) * H))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("out"))
    args = ap.parse_args()

    doc = {"schema": "mimesis.features.v1",
           "groups": {k: len(v) for k, v in G.items()},
           "anatomical_iris_ratio": round(ANATOMICAL_IRIS_RATIO, 4),
           "notes": {
               "iris_ratio": ("iris diameter over interocular distance. Anatomy in "
                              "adults sits near 0.186; a painted face need not."),
               "ear": "eye aspect ratio: vertical opening over horizontal width",
               "mar": "mouth aspect ratio, from the inner lip contour",
               "gaze_offset": "iris centre minus eye centre, in eye widths",
           },
           "panels": {}}

    for P in ("A", "B"):
        det = json.loads((args.out / f"panel_{P.lower()}_detection.json").read_text())
        W, H = det["painting_rect_px"]
        rows = []
        for f in det["faces"]:
            if len(f["detectors"]) < 2 or not f.get("landmarks"):
                continue
            pts = f["landmarks"]
            if len(pts) < 478:
                continue
            iod = dist(pts, 33, 263, W, H)
            feats = {}
            for name, ids in G.items():
                b = box_of(pts, ids, W, H)
                if not b:
                    continue
                fb = f["box"]
                b["w_iod"] = round(b["w_px"] / max(iod, 1e-6), 4)
                b["h_iod"] = round(b["h_px"] / max(iod, 1e-6), 4)
                b["at_face_frac"] = [
                    round((b["centre"][0] - fb[0]) / max(fb[2], 1e-9), 4),
                    round((b["centre"][1] - fb[1]) / max(fb[3], 1e-9), 4)]
                feats[name] = b

            ir_r = box_of(pts, IRIS_R, W, H)
            ir_l = box_of(pts, IRIS_L, W, H)
            for tag, ib, cid, eye in (("iris_r", ir_r, IRIS_C_R, "eye_r"),
                                      ("iris_l", ir_l, IRIS_C_L, "eye_l")):
                if not ib:
                    continue
                d_px = max(ib["w_px"], ib["h_px"])
                ib["diameter_px"] = round(d_px, 1)
                ib["iris_ratio"] = round(d_px / max(iod, 1e-6), 4)
                ib["ratio_vs_anatomy"] = round(
                    (d_px / max(iod, 1e-6)) / ANATOMICAL_IRIS_RATIO, 3)
                e = feats.get(eye)
                if e and cid < len(pts):
                    ib["gaze_offset_eyewidths"] = [
                        round((pts[cid][0] - e["centre"][0]) / max(e["box"][2], 1e-9), 3),
                        round((pts[cid][1] - e["centre"][1]) / max(e["box"][3], 1e-9), 3)]
                feats[tag] = ib

            def ratio(pair):
                (v0, v1), (h0, h1) = pair
                h = dist(pts, h0, h1, W, H)
                return round(dist(pts, v0, v1, W, H) / max(h, 1e-6), 4)

            rows.append({
                "face_id": f["face_id"],
                "interocular_px": round(iod, 1),
                "features": feats,
                "ear_r": ratio(EAR_R), "ear_l": ratio(EAR_L),
                "mar": ratio(MAR),
                "iris_ratio_mean": round(float(np.mean(
                    [feats[k]["iris_ratio"] for k in ("iris_r", "iris_l") if k in feats])), 4)
                if any(k in feats for k in ("iris_r", "iris_l")) else None,
            })
        doc["panels"][P] = rows

    (args.out / "face_features.json").write_text(json.dumps(doc, indent=2))

    for P, rows in doc["panels"].items():
        ir = [r["iris_ratio_mean"] for r in rows if r["iris_ratio_mean"]]
        print(f"\nPANEL {P}  {len(rows)} faces with full mesh")
        if ir:
            print(f"  iris ratio: mean {np.mean(ir):.4f}  sd {np.std(ir):.4f}  "
                  f"range {min(ir):.4f}-{max(ir):.4f}   "
                  f"(anatomy {ANATOMICAL_IRIS_RATIO:.4f})")
        print(f"  {'face':6}{'iod':>7}{'eyeR':>8}{'eyeL':>8}{'mouth':>8}{'irisR':>8}"
              f"{'gazeR':>16}")
        for r in rows:
            fr = r["features"]
            g = fr.get("iris_r", {}).get("gaze_offset_eyewidths")
            print(f"  {r['face_id']:6}{r['interocular_px']:>7.0f}"
                  f"{r['ear_r']:>8.3f}{r['ear_l']:>8.3f}{r['mar']:>8.3f}"
                  f"{fr.get('iris_r',{}).get('iris_ratio',0):>8.3f}"
                  f"{str(g):>16}")


if __name__ == "__main__":
    main()
