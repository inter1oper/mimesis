"""
Stage 1a QA — a static render of the overlay vocabulary over the rectified paint.

Not the renderer. This exists so the density and the stroke weights can be
judged against the actual painting before any HTML is written, and so every
element can be checked back to its source. Everything drawn here comes from the
Stage 1 JSON; nothing is invented for the picture.

    python stage1/preview_overlay.py --panel B --out out/
"""
from __future__ import annotations

import argparse
import json
import pathlib

import cv2
import numpy as np

CYAN = (255, 230, 0)      # BGR
DIM = (150, 120, 0)
WARN = (60, 90, 255)
FONT = cv2.FONT_HERSHEY_PLAIN


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True, choices=["A", "B"])
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("out"))
    ap.add_argument("--scale", type=float, default=0.40)
    args = ap.parse_args()

    p = args.panel.lower()
    img = cv2.imread(str(args.out / f"panel_{p}_rectified.png"), cv2.IMREAD_COLOR)
    img = cv2.resize(img, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
    img = (img * 0.55).astype(np.uint8)          # the projector adds light, it cannot subtract
    H, W = img.shape[:2]

    det = json.loads((args.out / f"panel_{p}_detection.json").read_text())
    fld = json.loads((args.out / f"panel_{p}_field.json").read_text())
    fa = json.loads((args.out / "face_analysis.json").read_text())
    fl = json.loads((args.out / f"panel_{p}_flames.json").read_text())
    rows = {r["face_id"]: r for r in fa["panels"].get(args.panel, [])}

    # response field: cells above p90 only, as hairline squares
    rf = fld["response_field"]
    gw, gh = rf["grid"]
    vals = np.array(rf["values"])
    thr = rf["stats"]["p90_nonzero"] or 1.0
    for gy in range(gh):
        for gx in range(gw):
            if vals[gy, gx] < thr:
                continue
            x0, y0 = int(gx / gw * W), int(gy / gh * H)
            x1, y1 = int((gx + 1) / gw * W), int((gy + 1) / gh * H)
            cv2.rectangle(img, (x0, y0), (x1, y1), DIM, 1)

    # blob circles, equivalent radius, largest few per segment
    for seg, col in (("lit_skin", DIM), ("flame_bright", DIM)):
        bl = [b for b in fld["blobs"] if b["segment"] == seg]
        for b in sorted(bl, key=lambda b: -b["area_pct_of_canvas"])[:14]:
            cx, cy = b["centroid"]
            r = int(b["equiv_radius_frac"] * W)
            if r < 3:
                continue
            cv2.circle(img, (int(cx * W), int(cy * H)), r, col, 1)
            cv2.putText(img, f"{b['area_pct_of_canvas']:.2f}%",
                        (int(cx * W) + r + 3, int(cy * H)), FONT, 0.7, col, 1)

    # flames
    for b in fl["blobs"]:
        if b["kind"] != "flame":
            continue
        cx, cy = b["centroid"]
        nfc = b.get("nearest_face")
        orph = bool(nfc and nfc["unattached"])
        cv2.circle(img, (int(cx * W), int(cy * H)), 9, WARN if orph else DIM, 1)
        if orph:
            cv2.putText(img, "NO FACE", (int(cx * W) - 40, int(cy * H) - 14), FONT, 0.9, WARN, 1)

    # faces
    for f in det["faces"]:
        if len(f["detectors"]) < 2:
            continue
        fid = f["face_id"]
        x, y, w, h = f["box"]
        x0, y0, x1, y1 = int(x * W), int(y * H), int((x + w) * W), int((y + h) * H)
        col = CYAN if f["mesh_ok"] else WARN
        # corner brackets rather than a closed box
        k = max(6, int(0.22 * (x1 - x0)))
        for (ax, ay, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
            cv2.line(img, (ax, ay), (ax + dx * k, ay), col, 1)
            cv2.line(img, (ax, ay), (ax, ay + dy * k), col, 1)
        cv2.rectangle(img, (x0, y0), (x1, y1), col, 1)

        r = rows.get(fid, {})
        lines = [f"{fid}"]
        sc = f["detectors"]
        lines.append("BF %.3f  YN %.3f" % (sc.get("blazeface", {}).get("score", 0),
                                          sc.get("yunet", {}).get("score", 0)))
        if r.get("head_pose"):
            hp = r["head_pose"]
            lines.append("Y%+.1f P%+.1f R%+.1f" % (hp["yaw_deg"], hp["pitch_deg"], hp["roll_deg"]))
        if r.get("symmetry"):
            lines.append("SYM %.4f" % r["symmetry"]["residual_mean_iod"])
        if r.get("geometry"):
            lines.append("IOD %.0f" % r["geometry"]["interocular_px"])
        ty = y0 - 6 - 11 * len(lines)
        for i, t in enumerate(lines):
            cv2.putText(img, t, (x0, ty + 11 * i), FONT, 0.75, col, 1)

        # blendshape bars: name + percentage on a filled plate, top 4
        for i, bs in enumerate((r.get("blendshapes_top8") or [])[:4]):
            by = y1 + 12 + 11 * i
            if by > H - 4:
                break
            label = f"{bs['name'][:16]:16} {bs['pct']:5.1f}%"
            (tw, th), _ = cv2.getTextSize(label, FONT, 0.7, 1)
            cv2.rectangle(img, (x0, by - th - 2), (x0 + tw + 4, by + 2), (40, 34, 0), -1)
            cv2.rectangle(img, (x0, by - th - 2), (x0 + tw + 4, by + 2), col, 1)
            cv2.putText(img, label, (x0 + 2, by), FONT, 0.7, col, 1)
            bw = int((tw + 4) * bs["pct"] / 100.0)
            cv2.line(img, (x0, by + 3), (x0 + bw, by + 3), col, 1)

        # cross-panel link readout
        for c in fa["cross_panel"]:
            if c["panel_b_face"] == fid:
                thrid = fa["sface_same_identity_threshold"]["cosine"]
                mark = "  >THR" if c["cosine"] > thrid else ""
                cv2.putText(img, f"~{c['best_match_panel_a']} {c['cosine']:.3f}{mark}",
                            (x0, y1 + 10), FONT, 0.8, col, 1)

    cv2.putText(img, f"PANEL {args.panel}", (10, 18), FONT, 1.2, CYAN, 1)
    out = args.out / f"panel_{p}_overlay_preview.png"
    cv2.imwrite(str(out), img)
    print("wrote", out)


if __name__ == "__main__":
    main()
