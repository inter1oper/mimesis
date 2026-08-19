"""
Stage 1a, step 7 — inferred structure: the graph the measurements imply.

Boxes and percentages describe faces one at a time. The inferences live in the
relations between them, and those relations are computable. This emits a typed
graph — nodes and weighted edges — so the overlay can draw lines that mean
something rather than lines that look like something.

Edge types, all measured:

  gaze              Each face's forward axis, taken from the third column of the
                    landmarker's rotation matrix and projected into the picture
                    plane. Rays are then intersected in the least-squares sense:
                    where a group is looking at one thing, they converge, and the
                    residual says how tightly. A vigil photographed by one person
                    should converge near the camera. Whether the invented room
                    does is a question the data answers.
  similarity        SFace cosine between every pair of faces, within and across
                    panels. Edges above the same-identity threshold are marked.
  proximity         Delaunay triangulation over face centroids, plus its minimum
                    spanning tree. The compositional skeleton of the group: who
                    is beside whom, and the shortest structure that connects
                    everyone.
  depth             Relative depth from interocular distance. Real interocular
                    distance varies little between adults, so in-image IOD is
                    roughly inversely proportional to distance. Gives a ranking
                    and a rough depth ratio, not metric depth.
  attribution       Flame to nearest consensus face, in face-height units. The
                    chain from a light source to the person supposedly holding it.
  disagreement      Where the two detectors put the same face differently: centre
                    offset and box IoU. A line whose length is the disagreement.
  containment       Which lit-skin blob sits inside which dark-mass blob. The
                    figure-in-suit hierarchy, from geometry alone.

Usage
-----
    python stage1/infer_structure.py --out out/
"""
from __future__ import annotations

import argparse
import json
import pathlib

import cv2
import numpy as np

MODELS = pathlib.Path(__file__).resolve().parent.parent / "models"
SFACE = MODELS / "face_recognition_sface_2021dec.onnx"
SAME_ID_COSINE = 0.363


def ray_convergence(origins: np.ndarray, dirs: np.ndarray) -> dict:
    """
    Least-squares intersection of 2D rays.

    Each ray contributes the squared perpendicular distance from the solution to
    that ray; minimising the sum has a closed form. The residual is the mean
    perpendicular distance from the solution to the rays, in canvas units, and
    is what says whether "they are all looking at one thing" is supported.
    """
    A = np.zeros((2, 2))
    b = np.zeros(2)
    for o, d in zip(origins, dirs):
        n = np.linalg.norm(d)
        if n < 1e-9:
            continue
        d = d / n
        P = np.eye(2) - np.outer(d, d)      # projector onto the ray's normal
        A += P
        b += P @ o
    if abs(np.linalg.det(A)) < 1e-9:
        return {"converges": False, "reason": "rays are parallel or too few"}
    p = np.linalg.solve(A, b)
    res = []
    for o, d in zip(origins, dirs):
        n = np.linalg.norm(d)
        if n < 1e-9:
            continue
        d = d / n
        v = p - o
        res.append(float(np.linalg.norm(v - (v @ d) * d)))
    return {
        "converges": True,
        "point": [round(float(p[0]), 4), round(float(p[1]), 4)],
        "residual_mean": round(float(np.mean(res)), 4),
        "residual_max": round(float(np.max(res)), 4),
        "rays": len(res),
        "point_inside_canvas": bool(0 <= p[0] <= 1 and 0 <= p[1] <= 1),
        "note": ("Residual is the mean perpendicular distance from the solution to "
                 "each gaze ray, in canvas widths. Small means the faces really do "
                 "point at one place; large means the solution is an artefact of "
                 "least squares and there is no shared object of attention."),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("out"))
    args = ap.parse_args()

    fa = json.loads((args.out / "face_analysis.json").read_text())
    sface = cv2.FaceRecognizerSF.create(str(SFACE), "")

    nodes, edges, per_panel = [], [], {}
    embeds, centroids = {}, {}

    for P in ("A", "B"):
        p = P.lower()
        det = json.loads((args.out / f"panel_{p}_detection.json").read_text())
        fl = json.loads((args.out / f"panel_{p}_flames.json").read_text())
        fld = json.loads((args.out / f"panel_{p}_field.json").read_text())
        img = cv2.imread(str(args.out / f"panel_{p}_rectified.png"), cv2.IMREAD_COLOR)
        H, W = img.shape[:2]
        rows = {r["face_id"]: r for r in fa["panels"].get(P, [])}
        cons = [f for f in det["faces"] if len(f["detectors"]) >= 2]

        # ---- nodes, embeddings, centroids
        for f in cons:
            fid = f["face_id"]
            x, y, w, h = f["box"]
            c = [x + w / 2, y + h / 2]
            centroids[fid] = c
            r = rows.get(fid, {})
            nodes.append({"id": fid, "kind": "face", "panel": P, "centroid": c,
                          "iod_px": (r.get("geometry") or {}).get("interocular_px"),
                          "mesh_ok": f["mesh_ok"]})
            try:
                a = cv2.resize(img[int(y * H):int((y + h) * H), int(x * W):int((x + w) * W)],
                               (112, 112))
                e = sface.feature(a).flatten()
                embeds[fid] = e / (np.linalg.norm(e) + 1e-9)
            except Exception:
                pass

        # ---- gaze rays from the landmarker's rotation matrix
        origins, dirs, ray_ids = [], [], []
        for f in cons:
            r = rows.get(f["face_id"])
            hp = (r or {}).get("head_pose")
            if not hp:
                continue
            yaw, pitch = np.radians(hp["yaw_deg"]), np.radians(hp["pitch_deg"])
            # Forward axis projected into the picture plane. x from yaw, y from pitch.
            d = np.array([np.sin(yaw), -np.sin(pitch)])
            if np.linalg.norm(d) < 1e-6:
                d = np.array([0.0, 1e-6])
            origins.append(centroids[f["face_id"]]); dirs.append(d)
            ray_ids.append(f["face_id"])
            edges.append({"type": "gaze", "from": f["face_id"],
                          "direction": [round(float(d[0]), 4), round(float(d[1]), 4)],
                          "yaw_deg": hp["yaw_deg"], "pitch_deg": hp["pitch_deg"]})
        conv = (ray_convergence(np.array(origins), np.array(dirs))
                if len(origins) >= 2 else {"converges": False, "reason": "fewer than two rays"})

        # ---- proximity: Delaunay over centroids, then its MST
        ids = [f["face_id"] for f in cons]
        prox = []
        if len(ids) >= 3:
            sub = cv2.Subdiv2D((0, 0, 1001, 1001))
            for i in ids:
                sub.insert((float(centroids[i][0] * 1000), float(centroids[i][1] * 1000)))
            pts = {i: (centroids[i][0] * 1000, centroids[i][1] * 1000) for i in ids}
            def near(px, py):
                return min(ids, key=lambda i: (pts[i][0] - px) ** 2 + (pts[i][1] - py) ** 2)
            seen = set()
            for t in sub.getTriangleList():
                tri = [near(t[0], t[1]), near(t[2], t[3]), near(t[4], t[5])]
                for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                    if a == b:
                        continue
                    k = tuple(sorted((a, b)))
                    if k in seen:
                        continue
                    seen.add(k)
                    d = float(np.hypot(centroids[a][0] - centroids[b][0],
                                       centroids[a][1] - centroids[b][1]))
                    prox.append({"type": "proximity", "panel": P, "from": k[0], "to": k[1],
                                 "distance_canvas": round(d, 4), "in_mst": False})
        # MST over the Delaunay edges (Kruskal)
        parent = {i: i for i in ids}
        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]; a = parent[a]
            return a
        for e in sorted(prox, key=lambda e: e["distance_canvas"]):
            ra, rb = find(e["from"]), find(e["to"])
            if ra != rb:
                parent[ra] = rb; e["in_mst"] = True
        edges.extend(prox)

        # ---- depth from interocular distance
        iods = {i: (rows.get(i, {}).get("geometry") or {}).get("interocular_px")
                for i in ids}
        have = {k: v for k, v in iods.items() if v}
        depth = []
        if have:
            ref = max(have.values())
            for k, v in sorted(have.items(), key=lambda kv: -kv[1]):
                depth.append({"face_id": k, "iod_px": v,
                              "relative_depth": round(ref / v, 3)})
        # ---- attribution: flame -> nearest consensus face
        for b in fl["blobs"]:
            if b["kind"] != "flame" or not b.get("nearest_face"):
                continue
            nf = b["nearest_face"]
            edges.append({"type": "attribution", "panel": P, "from": b["blob_id"],
                          "to": nf["face_id"],
                          "distance_face_heights": nf["distance_in_face_heights"],
                          "unattached": nf["unattached"]})

        # ---- disagreement between the two detectors on the same face
        for f in cons:
            bb = f["detectors"]["blazeface"]["box"]; yb = f["detectors"]["yunet"]["box"]
            cb = np.array([bb[0] + bb[2] / 2, bb[1] + bb[3] / 2])
            cy = np.array([yb[0] + yb[2] / 2, yb[1] + yb[3] / 2])
            x1, y1 = max(bb[0], yb[0]), max(bb[1], yb[1])
            x2 = min(bb[0] + bb[2], yb[0] + yb[2]); y2 = min(bb[1] + bb[3], yb[1] + yb[3])
            inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            iou = inter / (bb[2] * bb[3] + yb[2] * yb[3] - inter) if inter > 0 else 0.0
            edges.append({"type": "disagreement", "panel": P, "from": f["face_id"],
                          "blazeface_centre": [round(float(v), 5) for v in cb],
                          "yunet_centre": [round(float(v), 5) for v in cy],
                          "centre_offset_canvas": round(float(np.linalg.norm(cb - cy)), 5),
                          "box_iou": round(float(iou), 4)})

        # ---- containment: lit skin inside dark mass
        skin = [b for b in fld["blobs"] if b["segment"] == "lit_skin"]
        dark = sorted([b for b in fld["blobs"] if b["segment"] == "dark_mass"],
                      key=lambda b: -b["area_pct_of_canvas"])[:40]
        for s in sorted(skin, key=lambda b: -b["area_pct_of_canvas"])[:40]:
            cx, cy2 = s["centroid"]
            for dm in dark:
                x, y, w, h = dm["box"]
                if x <= cx <= x + w and y <= cy2 <= y + h:
                    edges.append({"type": "containment", "panel": P,
                                  "from": s["blob_id"], "to": dm["blob_id"],
                                  "child_area_pct": s["area_pct_of_canvas"],
                                  "parent_area_pct": dm["area_pct_of_canvas"]})
                    break

        per_panel[P] = {"gaze_convergence": conv, "gaze_ray_faces": ray_ids,
                        "depth_order": depth,
                        "proximity_edges": len(prox),
                        "mst_edges": sum(1 for e in prox if e["in_mst"])}

    # ---- similarity edges, every pair, both panels
    ids = list(embeds)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            cos = float(embeds[a] @ embeds[b])
            edges.append({"type": "similarity", "from": a, "to": b,
                          "cosine": round(cos, 4),
                          "cross_panel": a[0] != b[0],
                          "above_same_identity_threshold": bool(cos > SAME_ID_COSINE)})

    sims = [e for e in edges if e["type"] == "similarity"]
    doc = {
        "schema": "mimesis.structure.v1",
        "same_identity_cosine": SAME_ID_COSINE,
        "edge_types": ["gaze", "similarity", "proximity", "attribution",
                       "disagreement", "containment"],
        "per_panel": per_panel,
        "summary": {
            "nodes": len(nodes),
            "edges": len(edges),
            "similarity_edges_above_threshold": sum(1 for e in sims if e["above_same_identity_threshold"]),
            "cross_panel_edges_above_threshold": sum(
                1 for e in sims if e["cross_panel"] and e["above_same_identity_threshold"]),
        },
        "nodes": nodes,
        "edges": edges,
    }
    (args.out / "structure.json").write_text(json.dumps(doc, indent=2))

    print(json.dumps(doc["summary"], indent=2))
    for P, v in per_panel.items():
        g = v["gaze_convergence"]
        print(f"\nPANEL {P}")
        if g.get("converges"):
            print(f"  gaze: {g['rays']} rays -> ({g['point'][0]:.3f}, {g['point'][1]:.3f})  "
                  f"residual mean {g['residual_mean']:.4f} max {g['residual_max']:.4f}  "
                  f"inside canvas: {g['point_inside_canvas']}")
        else:
            print(f"  gaze: {g.get('reason')}")
        print(f"  proximity: {v['proximity_edges']} Delaunay edges, {v['mst_edges']} in the MST")
        d = v["depth_order"]
        if d:
            print(f"  depth: nearest {d[0]['face_id']} (iod {d[0]['iod_px']}) -> "
                  f"furthest {d[-1]['face_id']} (iod {d[-1]['iod_px']}, "
                  f"{d[-1]['relative_depth']}x further)")


if __name__ == "__main__":
    main()
