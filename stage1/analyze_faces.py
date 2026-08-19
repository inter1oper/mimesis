"""
Stage 1a, step 5 — per-face biometric readout.

The overlay wants what a biometric tracker puts on screen: named quantities with
percentages, angles, distances, match scores. The brief forbids inventing any of
it. So rather than draw decoration, this measures more, and the overlay draws
what comes out.

Everything here is computed from the painting. Per consensus face:

  blendshapes    52 named expression coefficients from MediaPipe FaceLandmarker,
                 0-100%. These are the percentage readouts.
  head_pose      yaw / pitch / roll in degrees, decomposed from the landmarker's
                 4x4 facial transformation matrix.
  geometry       interocular distance, box area as a fraction of canvas, aspect,
                 and how far the mesh's own extent departs from the detector's box.
  symmetry       bilateral symmetry residual: the mesh reflected across its own
                 fitted midline, matched back to itself, residual normalised by
                 interocular distance. Scale-free, comparable across faces and
                 across panels.
  photometry     luminance mean, standard deviation and range inside the face.
  embedding      128-d SFace descriptor, used only for the cross-panel matching
                 below; not itself projected.

Then the part that is about the piece rather than the pixels:

  cross_panel    cosine similarity between every Panel A face and every Panel B
                 face. For each invented face, which real classmate it most
                 resembles, and by how much. Two faces from different networks
                 pointed at the same canvas would not produce this; one face
                 recognition model comparing painted faces does.

Usage
-----
    python stage1/analyze_faces.py --out out/
"""
from __future__ import annotations

import argparse
import json
import pathlib

import cv2
import numpy as np

MODELS = pathlib.Path(__file__).resolve().parent.parent / "models"
LANDMARKER = MODELS / "face_landmarker.task"
SFACE = MODELS / "face_recognition_sface_2021dec.onnx"
PAD = 0.45


def euler_from_matrix(m: np.ndarray) -> dict:
    """Yaw/pitch/roll in degrees from the landmarker's 4x4 transform."""
    r = np.asarray(m, dtype=np.float64)[:3, :3]
    sy = float(np.hypot(r[0, 0], r[1, 0]))
    if sy > 1e-6:
        x = np.arctan2(r[2, 1], r[2, 2])
        y = np.arctan2(-r[2, 0], sy)
        z = np.arctan2(r[1, 0], r[0, 0])
    else:
        x, y, z = np.arctan2(-r[1, 2], r[1, 1]), np.arctan2(-r[2, 0], sy), 0.0
    return {"pitch_deg": round(float(np.degrees(x)), 2),
            "yaw_deg": round(float(np.degrees(y)), 2),
            "roll_deg": round(float(np.degrees(z)), 2)}


def symmetry_residual(pts: np.ndarray, interocular: float) -> dict:
    """
    Reflect the mesh across its own principal (vertical) axis and measure how
    well it lands back on itself.

    No hand-maintained table of left/right landmark pairs: the axis is fitted
    from the points, the whole set is mirrored, and each mirrored point is
    matched to its nearest original. Normalising by interocular distance makes
    the number comparable between a foreground face and one in the back row.
    """
    p = np.asarray(pts, dtype=np.float64)[:, :2]
    c = p.mean(axis=0)
    q = p - c
    # Principal axis of the mesh is its long (vertical) axis.
    _, _, vt = np.linalg.svd(q, full_matrices=False)
    axis = vt[0] / np.linalg.norm(vt[0])
    normal = np.array([-axis[1], axis[0]])
    # Reflect across the line through the centroid along `axis`.
    d = q @ normal
    mirrored = q - 2.0 * np.outer(d, normal)
    # Nearest-neighbour residual, mirrored set against the original.
    diff = mirrored[:, None, :] - q[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    nn = dist.min(axis=1)
    return {
        "residual_mean_iod": round(float(nn.mean() / max(interocular, 1e-9)), 4),
        "residual_p95_iod": round(float(np.percentile(nn, 95) / max(interocular, 1e-9)), 4),
        "note": ("mean distance from each mirrored landmark to the nearest real "
                 "landmark, in interocular units. Lower is more bilaterally "
                 "symmetric."),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("out"))
    ap.add_argument("--faces", choices=["consensus", "all"], default="consensus")
    args = ap.parse_args()

    import mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision

    lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=str(LANDMARKER)),
        num_faces=1, min_face_detection_confidence=0.2,
        output_face_blendshapes=True, output_facial_transformation_matrixes=True))
    sface = cv2.FaceRecognizerSF.create(str(SFACE), "")

    panels, embeddings = {}, {}
    for p in ("a", "b"):
        det = json.loads((args.out / f"panel_{p}_detection.json").read_text())
        img = cv2.imread(str(args.out / f"panel_{p}_rectified.png"), cv2.IMREAD_COLOR)
        H, W = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces_in = [f for f in det["faces"]
                    if args.faces == "all" or len(f.get("detectors", {})) >= 2]

        rows = []
        for f in faces_in:
            x, y, w, h = f["box"]
            x0 = int(max(0, (x - w * PAD) * W)); y0 = int(max(0, (y - h * PAD) * H))
            x1 = int(min(W, (x + w * (1 + PAD)) * W)); y1 = int(min(H, (y + h * (1 + PAD)) * H))
            crop = np.ascontiguousarray(img[y0:y1, x0:x1])
            if crop.size == 0:
                continue

            res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                     data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
            row = {"face_id": f["face_id"], "box": f["box"],
                   "detectors": {k: v["score"] for k, v in f["detectors"].items()}}

            if res.face_landmarks:
                pts = np.array([[p_.x * (x1 - x0) + x0, p_.y * (y1 - y0) + y0]
                                for p_ in res.face_landmarks[0]])
                # 33 / 263 are the outer eye corners in the canonical topology.
                iod = float(np.linalg.norm(pts[33] - pts[263]))
                mx0, my0 = pts.min(axis=0); mx1, my1 = pts.max(axis=0)
                row["geometry"] = {
                    "interocular_px": round(iod, 1),
                    "interocular_frac_of_canvas_w": round(iod / W, 5),
                    "box_area_frac_of_canvas": round(w * h, 6),
                    "box_aspect_h_over_w": round(h / max(w, 1e-9), 3),
                    "mesh_extent_px": [round(float(mx1 - mx0), 1), round(float(my1 - my0), 1)],
                    "mesh_wider_than_box": bool((mx1 - mx0) > w * W),
                }
                row["head_pose"] = (euler_from_matrix(res.facial_transformation_matrixes[0])
                                    if res.facial_transformation_matrixes else None)
                row["symmetry"] = symmetry_residual(pts, iod)
                row["blendshapes"] = {c.category_name: round(c.score * 100, 2)
                                      for c in res.face_blendshapes[0]} if res.face_blendshapes else None
                if row["blendshapes"]:
                    top = sorted(row["blendshapes"].items(), key=lambda kv: -kv[1])[:8]
                    row["blendshapes_top8"] = [{"name": k, "pct": v} for k, v in top]
            else:
                row.update({"geometry": None, "head_pose": None, "symmetry": None,
                            "blendshapes": None, "mesh_note": "mesh_did_not_converge"})

            bx0, by0 = int(x * W), int(y * H)
            bx1, by1 = int((x + w) * W), int((y + h) * H)
            patch = gray[by0:by1, bx0:bx1]
            row["photometry"] = None if patch.size == 0 else {
                "luma_mean": round(float(patch.mean()), 1),
                "luma_std": round(float(patch.std()), 1),
                "luma_min": int(patch.min()), "luma_max": int(patch.max()),
            }

            try:
                aligned = cv2.resize(img[by0:by1, bx0:bx1], (112, 112))
                emb = sface.feature(aligned).flatten()
                embeddings[f["face_id"]] = emb / (np.linalg.norm(emb) + 1e-9)
                row["embedding_ok"] = True
            except Exception:
                row["embedding_ok"] = False

            rows.append(row)
        panels[p.upper()] = rows

    # Hubness control. A descriptor can have "hub" vectors that sit near the
    # centroid and come up as the nearest neighbour of everything. Without this
    # check, one face being the best match for most of the other panel looks like
    # a finding when it is a property of the embedding space. Each face's mean
    # similarity to every other face, within and across panels, is recorded so a
    # match can be read against its own baseline.
    hubness = {}
    all_ids = list(embeddings)
    for i in all_ids:
        others = [float(embeddings[i] @ embeddings[j]) for j in all_ids if j != i]
        hubness[i] = {
            "mean_similarity_to_all": round(float(np.mean(others)), 4) if others else None,
            "times_best_match": 0,
        }

    # Cross-panel matching. Cosine similarity between every A face and every B face.
    cross = []
    a_ids = [r["face_id"] for r in panels.get("A", []) if r.get("embedding_ok")]
    b_ids = [r["face_id"] for r in panels.get("B", []) if r.get("embedding_ok")]
    for b in b_ids:
        sims = sorted(((float(embeddings[b] @ embeddings[a]), a) for a in a_ids), reverse=True)
        if sims:
            hubness[sims[0][1]]["times_best_match"] += 1
            base = hubness[sims[0][1]]["mean_similarity_to_all"] or 0.0
            cross.append({
                "excess_over_match_baseline": round(sims[0][0] - base, 4),
                "panel_b_face": b,
                "best_match_panel_a": sims[0][1],
                "cosine": round(sims[0][0], 4),
                "runner_up": ({"face_id": sims[1][1], "cosine": round(sims[1][0], 4)}
                              if len(sims) > 1 else None),
                "all": [{"face_id": a, "cosine": round(s, 4)} for s, a in sims],
            })

    doc = {
        "schema": "mimesis.faceanalysis.v1",
        "faces_scope": args.faces,
        "models": {"landmarker": LANDMARKER.name, "recognizer": SFACE.name},
        "measures": {
            "blendshapes": "52 named expression coefficients, percent",
            "head_pose": "degrees, from the landmarker's 4x4 transformation matrix",
            "symmetry": "mesh reflected across its own fitted axis, nearest-neighbour "
                        "residual in interocular units; lower is more symmetric",
            "cross_panel": "SFace cosine similarity. NOT identity: these are painted "
                           "faces, and a high score means the descriptor cannot "
                           "separate them, not that they are the same person.",
        },
        "sface_same_identity_threshold": {
            "cosine": 0.363,
            "note": ("The threshold OpenCV documents for SFace calling two "
                     "photographs the same person. Applied here to PAINTED faces, "
                     "which is out of the model's domain: shared palette, "
                     "brushwork and lighting inflate similarity between any two "
                     "faces from the same hand. A score above this says the "
                     "descriptor cannot separate them. It does not say they are "
                     "the same person, and the piece must not claim otherwise."),
        },
        "hubness_control": hubness,
        "panels": panels,
        "cross_panel": cross,
    }
    (args.out / "face_analysis.json").write_text(json.dumps(doc, indent=2))

    for pk, rows in panels.items():
        sym = [r["symmetry"]["residual_mean_iod"] for r in rows if r.get("symmetry")]
        print(f"\nPANEL {pk}: {len(rows)} faces, {len(sym)} with mesh")
        if sym:
            print(f"  symmetry residual (iod): median {np.median(sym):.4f}  "
                  f"min {min(sym):.4f}  max {max(sym):.4f}")
        for r in rows[:40]:
            g, hp = r.get("geometry"), r.get("head_pose")
            if not g:
                print(f"  {r['face_id']}  no mesh"); continue
            b = r.get("blendshapes_top8") or []
            print(f"  {r['face_id']}  iod={g['interocular_px']:6.1f}px  "
                  f"yaw={hp['yaw_deg']:7.2f} pitch={hp['pitch_deg']:7.2f} roll={hp['roll_deg']:7.2f}  "
                  f"sym={r['symmetry']['residual_mean_iod']:.4f}  "
                  f"top={b[0]['name']+' '+str(b[0]['pct'])+'%' if b else '-'}")
    print("\nCROSS-PANEL: each Panel B (invented) face and the Panel A (real) face it most resembles")
    for c in cross:
        ru = c["runner_up"]
        print(f"  {c['panel_b_face']} -> {c['best_match_panel_a']}  cos={c['cosine']:.4f}"
              + (f"   (2nd {ru['face_id']} {ru['cosine']:.4f})" if ru else ""))


if __name__ == "__main__":
    main()
