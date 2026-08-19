"""
Stage 1a, step 2 — face detection on the rectified painting.

Detectors
---------
Two architecturally independent detectors are run and reported separately. They
are never averaged into a single invented confidence; where they disagree, the
disagreement is preserved, because disagreement is the material.

  blazeface  MediaPipe BlazeFace (short range), run over an overlapping tile
             pyramid. BlazeFace resizes its input to 128x128 internally, so a
             face occupying 3% of a 4800px canvas is sub-pixel to it at full
             frame. Tiling at several scales is what makes small faces in a
             group scene reachable at all.
  yunet      OpenCV YuNet (libfacedetection). Anchor-free, trained on WIDER
             FACE, natively strong on small faces. Independent training data
             and architecture, so agreement between the two means something.

Landmarks come from MediaPipe FaceLandmarker (478 points) run per merged
detection on a padded crop. A box that the detector is confident about but on
which the mesh fails to converge is recorded as such; that is a real signal
about a face-like region that has no resolvable face geometry, not an error to
paper over.

Coordinates
-----------
Everything is normalised 0-1 against the rectified painting rectangle. Origin
top-left, x right, y down. No coordinate in this file's output refers to the
source photograph.

Usage
-----
    python stage1/detect_faces.py --panel A \
        --rectified out/panel_a_rectified.png \
        --rectify-meta out/panel_a_rectify.json \
        --out out/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from dataclasses import dataclass, field

import cv2
import numpy as np

MODELS = pathlib.Path(__file__).resolve().parent.parent / "models"
BLAZE = MODELS / "blaze_face_short_range.tflite"
LANDMARKER = MODELS / "face_landmarker.task"
YUNET = MODELS / "face_detection_yunet_2023mar.onnx"

# Detector thresholds are deliberately permissive. The piece is about what a
# detector will assert, including weakly; filtering hard at source would throw
# away the low-confidence assertions that are the most interesting on Panel B.
BLAZE_MIN_CONF = 0.20
YUNET_MIN_CONF = 0.30
MERGE_IOU = 0.40
MESH_PAD = 0.45  # crop padding around a box, as a fraction of box size


@dataclass
class Raw:
    source: str          # "blazeface" | "yunet"
    box: tuple           # x, y, w, h normalised to painting
    score: float
    scale_hint: str      # which tile/scale produced it, for auditability


@dataclass
class Face:
    face_id: str
    box: tuple
    detectors: dict = field(default_factory=dict)
    landmarks: list | None = None
    mesh_ok: bool = False
    mesh_note: str | None = None


def iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    return inter / (aw * ah + bw * bh - inter)


def run_blazeface(rgb: np.ndarray) -> list[Raw]:
    """Overlapping tile pyramid. Tile sizes are fractions of the long edge."""
    import mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision

    H, W = rgb.shape[:2]
    out: list[Raw] = []
    opts = vision.FaceDetectorOptions(
        base_options=mpp.BaseOptions(model_asset_path=str(BLAZE)),
        min_detection_confidence=BLAZE_MIN_CONF,
    )
    det = vision.FaceDetector.create_from_options(opts)
    try:
        for frac in (1.0, 0.5, 0.34, 0.25):
            tile = int(round(max(W, H) * frac))
            step = max(1, int(tile * 0.5))
            xs = list(range(0, max(1, W - tile + 1), step)) or [0]
            ys = list(range(0, max(1, H - tile + 1), step)) or [0]
            if xs[-1] + tile < W:
                xs.append(max(0, W - tile))
            if ys[-1] + tile < H:
                ys.append(max(0, H - tile))
            for ty in ys:
                for tx in xs:
                    crop = rgb[ty:ty + tile, tx:tx + tile]
                    if crop.size == 0:
                        continue
                    crop = np.ascontiguousarray(crop)
                    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop)
                    for d in det.detect(img).detections:
                        bb = d.bounding_box
                        out.append(Raw(
                            source="blazeface",
                            box=((tx + bb.origin_x) / W, (ty + bb.origin_y) / H,
                                 bb.width / W, bb.height / H),
                            score=float(d.categories[0].score) if d.categories else 0.0,
                            scale_hint=f"tile{frac:g}@{tx},{ty}",
                        ))
    finally:
        det.close()
    return out


def run_yunet(bgr: np.ndarray) -> list[Raw]:
    """YuNet at native resolution and 2x, which reaches smaller faces."""
    H, W = bgr.shape[:2]
    out: list[Raw] = []
    for mult in (1.0, 2.0):
        img = bgr if mult == 1.0 else cv2.resize(bgr, None, fx=mult, fy=mult,
                                                 interpolation=cv2.INTER_CUBIC)
        h, w = img.shape[:2]
        # YuNet's own input cap; oversized inputs are unstable, so cap the long edge.
        cap = 2000
        s = min(1.0, cap / max(h, w))
        if s < 1.0:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]
        det = cv2.FaceDetectorYN.create(str(YUNET), "", (w, h), YUNET_MIN_CONF, 0.3, 5000)
        det.setInputSize((w, h))
        _, faces = det.detect(img)
        if faces is None:
            continue
        for f in faces:
            x, y, bw, bh = f[0] / w, f[1] / h, f[2] / w, f[3] / h
            out.append(Raw(source="yunet", box=(float(x), float(y), float(bw), float(bh)),
                           score=float(f[-1]), scale_hint=f"x{mult:g}"))
    return out


def merge(raws: list[Raw]) -> list[list[Raw]]:
    """Greedy IoU clustering, highest score first. Returns clusters."""
    remaining = sorted(raws, key=lambda r: -r.score)
    clusters: list[list[Raw]] = []
    while remaining:
        seed = remaining.pop(0)
        cluster = [seed]
        rest = []
        for r in remaining:
            (cluster if iou(seed.box, r.box) >= MERGE_IOU else rest).append(r)
        clusters.append(cluster)
        remaining = rest
    return clusters


def mesh_for(rgb: np.ndarray, box, landmarker) -> tuple[list | None, str | None]:
    """Run FaceLandmarker on a padded crop; map 478 points back to painting space."""
    import mediapipe as mp

    H, W = rgb.shape[:2]
    x, y, w, h = box
    px, py = w * MESH_PAD, h * MESH_PAD
    x0 = int(max(0, (x - px) * W)); y0 = int(max(0, (y - py) * H))
    x1 = int(min(W, (x + w + px) * W)); y1 = int(min(H, (y + h + py) * H))
    if x1 - x0 < 24 or y1 - y0 < 24:
        return None, "crop_too_small"
    crop = np.ascontiguousarray(rgb[y0:y1, x0:x1])
    res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop))
    if not res.face_landmarks:
        return None, "mesh_did_not_converge"
    cw, ch = x1 - x0, y1 - y0
    pts = [[round((x0 + lm.x * cw) / W, 6), round((y0 + lm.y * ch) / H, 6), round(lm.z, 6)]
           for lm in res.face_landmarks[0]]
    return pts, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True, choices=["A", "B"])
    ap.add_argument("--rectified", required=True, type=pathlib.Path)
    ap.add_argument("--rectify-meta", required=True, type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("out"))
    ap.add_argument("--no-mesh", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    bgr = cv2.imread(str(args.rectified), cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"cannot read {args.rectified}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]
    rect_meta = json.loads(args.rectify_meta.read_text())

    raws = run_blazeface(rgb) + run_yunet(bgr)
    clusters = merge(raws)

    faces: list[Face] = []
    for i, cl in enumerate(clusters):
        by_src: dict[str, Raw] = {}
        for r in cl:
            if r.source not in by_src or r.score > by_src[r.source].score:
                by_src[r.source] = r
        # Consensus box: mean of each detector's best box, so neither detector's
        # framing convention silently wins.
        boxes = np.array([r.box for r in by_src.values()], dtype=float)
        box = tuple(round(float(v), 6) for v in boxes.mean(axis=0))
        faces.append(Face(
            face_id=f"{args.panel}{i + 1:02d}",
            box=box,
            detectors={k: {"score": round(v.score, 4), "box": [round(c, 6) for c in v.box],
                           "scale_hint": v.scale_hint} for k, v in by_src.items()},
        ))

    faces.sort(key=lambda f: (f.box[0], f.box[1]))
    for i, f in enumerate(faces):
        f.face_id = f"{args.panel}{i + 1:02d}"

    if not args.no_mesh:
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision
        lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=str(LANDMARKER)),
            num_faces=1, min_face_detection_confidence=BLAZE_MIN_CONF,
        ))
        try:
            for f in faces:
                pts, note = mesh_for(rgb, f.box, lm)
                f.landmarks, f.mesh_ok, f.mesh_note = pts, pts is not None, note
        finally:
            lm.close()

    doc = {
        "schema": "mimesis.detection.v1",
        "panel": args.panel,
        "painting_rect_px": [W, H],
        "canvas_cm": rect_meta.get("canvas_cm"),
        "source_photo_sha256": hashlib.sha256(
            pathlib.Path(rect_meta["source_photo"]).read_bytes()).hexdigest()[:16]
        if pathlib.Path(rect_meta["source_photo"]).exists() else None,
        "corner_method": rect_meta.get("corner_method"),
        "detectors": {
            "blazeface": {"model": BLAZE.name, "min_confidence": BLAZE_MIN_CONF,
                          "strategy": "overlapping tile pyramid, fracs 1.0/0.5/0.34/0.25, 50% stride"},
            "yunet": {"model": YUNET.name, "min_confidence": YUNET_MIN_CONF,
                      "strategy": "full frame at 1x and 2x, long edge capped at 2000px"},
            "landmarker": {"model": LANDMARKER.name, "points": 478,
                           "strategy": f"per-detection padded crop, pad={MESH_PAD}"},
        },
        "merge": {"method": "greedy IoU clustering", "iou": MERGE_IOU,
                  "box": "mean of each detector's best box in the cluster"},
        "counts": {
            "raw_blazeface": sum(1 for r in raws if r.source == "blazeface"),
            "raw_yunet": sum(1 for r in raws if r.source == "yunet"),
            "merged": len(faces),
            "both_detectors_agree": sum(1 for f in faces if len(f.detectors) == 2),
            "blazeface_only": sum(1 for f in faces if list(f.detectors) == ["blazeface"]),
            "yunet_only": sum(1 for f in faces if list(f.detectors) == ["yunet"]),
            "mesh_converged": sum(1 for f in faces if f.mesh_ok),
        },
        "faces": [{
            "face_id": f.face_id,
            "box": list(f.box),
            "detectors": f.detectors,
            "consensus": sorted(f.detectors),
            "mesh_ok": f.mesh_ok,
            "mesh_note": f.mesh_note,
            "landmarks": f.landmarks,
        } for f in faces],
    }
    path = args.out / f"panel_{args.panel.lower()}_detection.json"
    path.write_text(json.dumps(doc, indent=2))

    qa = bgr.copy()
    for f in faces:
        x, y, w, h = f.box
        p0 = (int(x * W), int(y * H)); p1 = (int((x + w) * W), int((y + h) * H))
        col = (255, 255, 0) if len(f.detectors) == 2 else (0, 165, 255)
        cv2.rectangle(qa, p0, p1, col, 3)
        best = max(f.detectors.values(), key=lambda d: d["score"])["score"]
        cv2.putText(qa, f"{f.face_id} {best:.2f}{'' if f.mesh_ok else ' NOMESH'}",
                    (p0[0], max(14, p0[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
        if f.landmarks:
            for lx, ly, _ in f.landmarks[::4]:
                cv2.circle(qa, (int(lx * W), int(ly * H)), 1, (0, 255, 128), -1)
    cv2.imwrite(str(args.out / f"panel_{args.panel.lower()}_detection_qa.jpg"), qa,
                [cv2.IMWRITE_JPEG_QUALITY, 88])

    print(json.dumps({k: doc[k] for k in ("panel", "painting_rect_px", "counts")}, indent=2))
    for f in doc["faces"]:
        s = {k: v["score"] for k, v in f["detectors"].items()}
        print(f'  {f["face_id"]}  box={[round(c,3) for c in f["box"]]}  {s}  '
              f'mesh={"ok" if f["mesh_ok"] else f["mesh_note"]}')


if __name__ == "__main__":
    main()
