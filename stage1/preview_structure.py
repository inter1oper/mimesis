"""
Stage 1a QA — the inferred-structure layer drawn over the paint.

Lines only: gaze rays and their convergence, the compositional skeleton, the
similarity graph, flame attribution, and the two detectors disagreeing. Every
line comes from structure.json.

    python stage1/preview_structure.py --panel A --out out/
"""
from __future__ import annotations

import argparse
import json
import pathlib

import cv2
import numpy as np

CYAN = (255, 230, 0)
DIM = (140, 110, 0)
WARN = (60, 90, 255)
HOT = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_PLAIN


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True, choices=["A", "B"])
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("out"))
    ap.add_argument("--scale", type=float, default=0.40)
    args = ap.parse_args()

    P, p = args.panel, args.panel.lower()
    img = cv2.imread(str(args.out / f"panel_{p}_rectified.png"), cv2.IMREAD_COLOR)
    img = cv2.resize(img, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
    img = (img * 0.45).astype(np.uint8)
    H, W = img.shape[:2]

    st = json.loads((args.out / "structure.json").read_text())
    fl = json.loads((args.out / f"panel_{p}_flames.json").read_text())
    node = {n["id"]: n for n in st["nodes"] if n["panel"] == P}
    def pt(i):
        c = node[i]["centroid"]
        return (int(c[0] * W), int(c[1] * H))

    E = st["edges"]

    # proximity: Delaunay faint, MST solid
    for e in E:
        if e["type"] != "proximity" or e.get("panel") != P:
            continue
        if e["from"] not in node or e["to"] not in node:
            continue
        cv2.line(img, pt(e["from"]), pt(e["to"]), CYAN if e["in_mst"] else DIM,
                 1, cv2.LINE_AA)
        if e["in_mst"]:
            m = ((pt(e["from"])[0] + pt(e["to"])[0]) // 2,
                 (pt(e["from"])[1] + pt(e["to"])[1]) // 2)
            cv2.putText(img, f"{e['distance_canvas']:.3f}", m, FONT, 0.7, CYAN, 1)

    # similarity above the same-identity threshold, within this panel
    for e in E:
        if e["type"] != "similarity" or not e["above_same_identity_threshold"]:
            continue
        if e["from"] not in node or e["to"] not in node:
            continue
        a, b = pt(e["from"]), pt(e["to"])
        for t in np.arange(0, 1, 0.05):     # dashed, no easing
            if int(t * 20) % 2:
                continue
            q0 = (int(a[0] + (b[0] - a[0]) * t), int(a[1] + (b[1] - a[1]) * t))
            q1 = (int(a[0] + (b[0] - a[0]) * (t + 0.05)),
                  int(a[1] + (b[1] - a[1]) * (t + 0.05)))
            cv2.line(img, q0, q1, HOT, 1, cv2.LINE_AA)
        m = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2 - 6)
        cv2.putText(img, f"cos {e['cosine']:.3f}", m, FONT, 0.8, HOT, 1)

    # attribution: flame -> face
    fpos = {b["blob_id"]: b["centroid"] for b in fl["blobs"]}
    for e in E:
        if e["type"] != "attribution" or e.get("panel") != P:
            continue
        if e["from"] not in fpos or e["to"] not in node:
            continue
        c = fpos[e["from"]]
        a = (int(c[0] * W), int(c[1] * H))
        col = WARN if e["unattached"] else DIM
        cv2.line(img, a, pt(e["to"]), col, 1, cv2.LINE_AA)
        cv2.putText(img, f"{e['distance_face_heights']:.2f}h",
                    (a[0] + 5, a[1] - 5), FONT, 0.7, col, 1)

    # gaze rays and convergence
    conv = st["per_panel"][P]["gaze_convergence"]
    for e in E:
        if e["type"] != "gaze" or e["from"] not in node:
            continue
        a = pt(e["from"])
        d = np.array(e["direction"], dtype=float)
        n = np.linalg.norm(d)
        if n < 1e-9:
            continue
        d = d / n
        b = (int(a[0] + d[0] * W * 0.55), int(a[1] + d[1] * W * 0.55))
        cv2.line(img, a, b, CYAN, 1, cv2.LINE_AA)
        cv2.putText(img, f"Y{e['yaw_deg']:+.0f}", (a[0] + 6, a[1] + 12), FONT, 0.7, CYAN, 1)
    if conv.get("converges"):
        c = conv["point"]
        cp = (int(c[0] * W), int(c[1] * H))
        for r in (10, 22, 34):
            cv2.circle(img, cp, r, HOT, 1, cv2.LINE_AA)
        cv2.line(img, (cp[0] - 44, cp[1]), (cp[0] + 44, cp[1]), HOT, 1)
        cv2.line(img, (cp[0], cp[1] - 44), (cp[0], cp[1] + 44), HOT, 1)
        cv2.putText(img, f"GAZE CONVERGENCE  res {conv['residual_mean']:.4f}  "
                         f"max {conv['residual_max']:.4f}  n={conv['rays']}",
                    (cp[0] + 48, cp[1] - 8), FONT, 0.9, HOT, 1)

    # detector disagreement vectors
    for e in E:
        if e["type"] != "disagreement" or e.get("panel") != P:
            continue
        a = (int(e["blazeface_centre"][0] * W), int(e["blazeface_centre"][1] * H))
        b = (int(e["yunet_centre"][0] * W), int(e["yunet_centre"][1] * H))
        cv2.line(img, a, b, WARN, 1, cv2.LINE_AA)
        cv2.circle(img, a, 3, WARN, 1); cv2.rectangle(img, (b[0]-3,b[1]-3),(b[0]+3,b[1]+3), WARN, 1)
        cv2.putText(img, f"d{e['centre_offset_canvas']*1000:.0f} IoU{e['box_iou']:.2f}",
                    (b[0] + 6, b[1] + 12), FONT, 0.7, WARN, 1)

    for i in node:
        cv2.circle(img, pt(i), 4, CYAN, 1)
        cv2.putText(img, i, (pt(i)[0] + 6, pt(i)[1] - 6), FONT, 0.9, CYAN, 1)

    d = st["per_panel"][P]["depth_order"]
    cv2.putText(img, f"PANEL {P}   nodes {len(node)}   "
                     f"MST {st['per_panel'][P]['mst_edges']}   "
                     f"depth span {d[-1]['relative_depth'] if d else 0:.2f}x",
                (10, 18), FONT, 1.2, CYAN, 1)
    out = args.out / f"panel_{p}_structure_preview.png"
    cv2.imwrite(str(out), img)
    print("wrote", out)


if __name__ == "__main__":
    main()
