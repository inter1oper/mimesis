"""
Stage 1a, step 10 — field layers: iso-luminance contours, gradient flow, blob
outlines, and the mesh's own triangulation.

The overlay so far draws objects the detector named. These are layers drawn
from the painting itself, at a density objects cannot reach. Everything here is
measured off the pixels; none of it is generative noise.

  iso_contours   marching-squares contours of the luminance channel at N evenly
                 spaced levels. This is the painting's own topography: where the
                 light sits, as closed curves. Dense by nature and entirely real.
  flow           the luminance gradient sampled on a grid, as vectors. Direction
                 is the direction of steepest change in the paint, magnitude is
                 how fast it changes. A brushstroke field, measured.
  blob_outlines  the actual connected-component boundaries behind the blob
                 circles, rather than a circle standing in for a shape.
  mesh_topology  Delaunay triangulation of the canonical 478-point face mesh,
                 computed once and shared by every face, so the wireframe can be
                 drawn as triangles rather than as loose points.

Usage
-----
    python stage1/fields.py --panel A --rectified out/panel_a_rectified.png
"""
from __future__ import annotations

import argparse
import json
import pathlib

import cv2
import numpy as np

ISO_LEVELS = 14
FLOW_GRID = (64, 48)
MAX_CONTOUR_POINTS = 60000


def iso_contours(gray: np.ndarray, levels: int, W: int, H: int):
    """Contours of constant luminance. The painting's topography."""
    g = cv2.GaussianBlur(gray, (0, 0), 2.2)
    lo, hi = np.percentile(g, 2), np.percentile(g, 99)
    out = []
    per_level = MAX_CONTOUR_POINTS // levels    # every level gets its share
    for i in range(levels):
        budget = per_level
        v = lo + (hi - lo) * (i + 0.5) / levels
        mask = (g >= v).astype(np.uint8) * 255
        cs, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        polys = []
        for c in cs:
            if cv2.arcLength(c, True) < 60:
                continue
            eps = 2.0
            a = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
            if len(a) < 4:
                continue
            if budget - len(a) < 0:
                break
            budget -= len(a)
            polys.append([[round(float(x) / W, 4), round(float(y) / H, 4)] for x, y in a])
        if polys:
            out.append({"level": round(float(v), 1),
                        "level_frac": round((v - lo) / max(hi - lo, 1e-6), 4),
                        "polylines": polys})
    return out


def flow_field(gray: np.ndarray, grid, W: int, H: int):
    """Luminance gradient on a grid: where the paint changes, and which way."""
    g = cv2.GaussianBlur(gray, (0, 0), 3.0).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=5)
    mag = np.hypot(gx, gy)
    mx = float(np.percentile(mag, 99.5)) or 1.0
    GW, GH = grid
    cells = []
    for j in range(GH):
        for i in range(GW):
            y0, y1 = int(j * H / GH), int((j + 1) * H / GH)
            x0, x1 = int(i * W / GW), int((i + 1) * W / GW)
            u = float(gx[y0:y1, x0:x1].mean())
            v = float(gy[y0:y1, x0:x1].mean())
            m = float(np.hypot(u, v) / mx)
            if m < 0.02:
                continue
            n = np.hypot(u, v) or 1.0
            cells.append({"c": [i, j],
                          "d": [round(u / n, 3), round(v / n, 3)],
                          "m": round(min(1.0, m), 3)})
    return {"grid": [GW, GH], "cells": cells,
            "note": "unit direction of the luminance gradient, magnitude normalised "
                    "to the 99.5th percentile"}


def blob_outlines(bgr: np.ndarray, W: int, H: int):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    segs = {
        "lit_skin": ((h >= 3) & (h <= 25) & (s >= 60) & (s <= 200) &
                     (v >= np.percentile(v, 72))),
        "flame_bright": (v >= np.percentile(v, 99.0)),
    }
    out = []
    for name, m in segs.items():
        mask = m.astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(cs, key=cv2.contourArea, reverse=True)[:70]:
            if cv2.contourArea(c) < 0.00004 * W * H:
                continue
            a = cv2.approxPolyDP(c, 1.6, True).reshape(-1, 2)
            if len(a) < 4:
                continue
            out.append({"segment": name,
                        "poly": [[round(float(x) / W, 4), round(float(y) / H, 4)]
                                 for x, y in a]})
    return out


def mesh_topology(detection: dict):
    """
    Delaunay triangulation of the canonical face mesh.

    The 478-point topology is fixed, so one triangulation computed from any
    converged face indexes every face. Computed rather than shipped as a table,
    so it cannot drift from the model that produced the points.
    """
    face = next((f for f in detection["faces"]
                 if f.get("landmarks") and len(f["landmarks"]) >= 468), None)
    if not face:
        return None
    pts = np.array([[p[0], p[1]] for p in face["landmarks"][:468]], dtype=np.float32)
    p = (pts - pts.min(axis=0)) / np.ptp(pts, axis=0)
    sub = cv2.Subdiv2D((0, 0, 1002, 1002))
    idx = {}
    for i, (x, y) in enumerate(p * 1000):
        k = (round(float(x), 2), round(float(y), 2))
        if k in idx:
            continue
        idx[k] = i
        sub.insert((float(x), float(y)))
    tris = []
    for t in sub.getTriangleList():
        ids = []
        for k in range(3):
            x, y = t[k * 2], t[k * 2 + 1]
            best, bd = None, 1e9
            for (kx, ky), i in idx.items():
                d = (kx - x) ** 2 + (ky - y) ** 2
                if d < bd:
                    best, bd = i, d
            if bd > 4.0:
                ids = None
                break
            ids.append(best)
        if ids and len(set(ids)) == 3:
            tris.append(ids)
    # unique undirected edges
    edges = sorted({tuple(sorted((a, b)))
                    for a, b, c in tris for a, b in ((a, b), (b, c), (c, a))})
    return {"source_face": face["face_id"], "triangles": len(tris),
            "edges": [list(e) for e in edges]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True, choices=["A", "B"])
    ap.add_argument("--rectified", required=True, type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("out"))
    args = ap.parse_args()

    bgr = cv2.imread(str(args.rectified), cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"cannot read {args.rectified}")
    H, W = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    det = json.loads((args.out / f"panel_{args.panel.lower()}_detection.json").read_text())

    doc = {"schema": "mimesis.fields.v1", "panel": args.panel,
           "painting_rect_px": [W, H],
           "iso_contours": iso_contours(gray, ISO_LEVELS, W, H),
           "flow": flow_field(gray, FLOW_GRID, W, H),
           "blob_outlines": blob_outlines(bgr, W, H),
           "mesh_topology": mesh_topology(det)}
    path = args.out / f"panel_{args.panel.lower()}_fields.json"
    path.write_text(json.dumps(doc, separators=(",", ":")))

    npts = sum(len(p) for lv in doc["iso_contours"] for p in lv["polylines"])
    nply = sum(len(lv["polylines"]) for lv in doc["iso_contours"])
    mt = doc["mesh_topology"]
    print(f"PANEL {args.panel}: {len(doc['iso_contours'])} iso levels, {nply} polylines, "
          f"{npts} points | flow {len(doc['flow']['cells'])} vectors | "
          f"outlines {len(doc['blob_outlines'])} | "
          f"mesh {mt['triangles'] if mt else 0} triangles, {len(mt['edges']) if mt else 0} edges | "
          f"{path.stat().st_size/1024:.0f} KB")


if __name__ == "__main__":
    main()
