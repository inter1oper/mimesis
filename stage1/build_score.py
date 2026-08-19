"""
Stage 1c — the timed score.

Structure
---------
The loop is five ANALYSIS PASSES. Each pass is a different kind of thinking
about the painting, and each carries a colour role, so colour reads as time:

    1 DETECT    white          boxes emerge; the detector commits or does not
    2 RESOLVE   white -> green geometry is attempted; boxes subdivide or fail
    3 MEASURE   green          quantities are taken; nodes appear beside faces
    4 RELATE    green -> red   faces are compared to each other and across panels
    5 DOUBT     red            the floor, the disagreements, the rejected boxes

Composition
-----------
The unit is not a labelled box. It is a BOX, a CIRCLE placed beside it, and a
CONNECTOR between them carrying the number and the sentence that justifies it.
Nothing floats: every value sits on the line that links the thing measured to
the measurement.

Node placement is solved here, not in the renderer: for each face, eight
candidate directions are scored against every other face box and every node
already placed, and the least-crowded one wins. That is why the labels stop
colliding.

Reasoning text
--------------
Each connector carries a `label`: a sentence in the machine's own terms, built
from the measurement it annotates. "MESH 478/478 CONVERGED". "COSINE 0.501 >
0.363 CANNOT SEPARATE". These are not model transcripts and are never presented
as such -- they are the pipeline's own reasoning about the paint, generated from
real values. When transcripts arrive the reasoning track carries the model's
words and these stay as the instrument layer beneath.

Usage
-----
    python stage1/build_score.py --out out/ --loop 900
    python stage1/build_score.py --selftest
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib

CHAR_MS = 21.0
CHAR_JITTER_MS = 3.5
SENTENCE_PAUSE_MS = 420.0
COMMA_PAUSE_MS = 130.0
BLUR_RESOLVE_MS = 120.0

# Pass name, window as a fraction of the loop, colour role.
PASSES = [
    ("DETECT",  0.00, 0.22, "obs"),
    ("RESOLVE", 0.19, 0.42, "obs"),
    ("MEASURE", 0.39, 0.62, "measure"),
    ("RELATE",  0.59, 0.83, "relate"),
    ("DOUBT",   0.80, 1.00, "doubt"),
]

SPECIMEN = [
    "Fourteen people stood in a room and someone photographed them.",
    "A description of that photograph was given to a machine, and the machine "
    "produced a room that had never held anyone.",
    "Both rooms were then painted, by the same hand, in the same hours.",
    "The detector finds faces in both.",
]


def char_onsets(text: str) -> list[float]:
    out, t = [], 0.0
    for i, ch in enumerate(text):
        out.append(round(t, 2))
        t += CHAR_MS + CHAR_JITTER_MS * (((i * 2654435761) % 1000) / 1000.0 - 0.5) * 2
        if ch in ".!?":
            t += SENTENCE_PAUSE_MS
        elif ch in ",;:":
            t += COMMA_PAUSE_MS
    return out


def window(name: str, loop: float) -> tuple[float, float]:
    for n, a, b, _ in PASSES:
        if n == name:
            return a * loop, b * loop
    raise KeyError(name)


def role(name: str) -> str:
    for n, _, _, r in PASSES:
        if n == name:
            return r
    return "obs"


def place_nodes(faces, aspect: float) -> dict:
    """
    Put one analysis circle beside each face, in the least crowded direction.

    Scored against every other face box and every node already placed, so
    connectors stay short and labels stop landing on each other. Solved once
    here rather than fought with at render time.
    """
    centres = [(f["box"][0] + f["box"][2] / 2, f["box"][1] + f["box"][3] / 2) for f in faces]
    placed, nodes = [], {}
    order = sorted(range(len(faces)), key=lambda i: -faces[i]["box"][2] * faces[i]["box"][3])
    for i in order:
        f = faces[i]
        x, y, w, h = f["box"]
        cx, cy = centres[i]
        reach = max(w, h) * 1.9 + 0.055
        best, best_score = None, -1e9
        for k in range(12):
            a = (k / 12.0) * 2 * math.pi
            nx = cx + math.cos(a) * reach
            ny = cy + math.sin(a) * reach / aspect
            if not (0.03 < nx < 0.97 and 0.05 < ny < 0.95):
                continue
            s = 0.0
            for j, (ox, oy) in enumerate(centres):
                if j == i:
                    continue
                s += min(0.35, math.hypot(nx - ox, (ny - oy) * aspect))
            for (px, py) in placed:
                s += 2.2 * min(0.30, math.hypot(nx - px, (ny - py) * aspect))
            s -= 1.4 * abs(math.hypot(nx - cx, (ny - cy) * aspect) - reach)
            if s > best_score:
                best, best_score = (nx, ny), s
        if best is None:
            best = (min(0.95, cx + reach), cy)
        placed.append(best)
        nodes[f["face_id"]] = [round(best[0], 5), round(best[1], 5)]
    return nodes


def build(panel: str, out: pathlib.Path, loop: float, lead: float,
          specimen: bool = False) -> dict:
    p = panel.lower()
    det = json.loads((out / f"panel_{p}_detection.json").read_text())
    fl = json.loads((out / f"panel_{p}_flames.json").read_text())
    plan = json.loads((out / f"panel_{p}_overlay_plan.json").read_text())
    stab = json.loads((out / "stability.json").read_text())
    stru = json.loads((out / "structure.json").read_text())
    fa = json.loads((out / "face_analysis.json").read_text())
    nf = json.loads((out / "noise_floor.json").read_text())
    claims_path = out / "claims.json"
    sessions = json.loads(claims_path.read_text()) if claims_path.exists() else []
    mine = [s_ for s_ in sessions if s_.get("panel") == panel]

    W, H = det["painting_rect_px"]
    aspect = W / H
    tier = {f["face_id"]: f for f in plan["faces"]}
    stabr = {r["face_id"]: r for r in stab["panels"].get(panel, [])}
    stab_index = {r["face_id"]: i for i, r in enumerate(stab["panels"].get(panel, []))}
    anal = {r["face_id"]: r for r in fa["panels"].get(panel, [])}
    anal_index = {r["face_id"]: i for i, r in enumerate(fa["panels"].get(panel, []))}
    faces = det["faces"]
    idx = {f["face_id"]: i for i, f in enumerate(faces)}
    cons = [f for f in faces if len(f["detectors"]) >= 2]
    cons = sorted(cons, key=lambda f: -max(v["score"] for v in f["detectors"].values()))
    nodes = place_nodes(cons, aspect)
    total_passes = stab["passes"]["total"]

    cues = []

    def add(cue, pass_name, src_file, pointer, t_in, t_out, **kw):
        c = {"cue_id": f"{panel}-{cue}-{len(cues):04d}", "cue": cue,
             "pass": pass_name, "role": role(pass_name),
             "t_in": round(t_in, 3), "t_out": round(t_out, 3),
             "source": {"file": src_file, "pointer": pointer}}
        c.update(kw)
        cues.append(c)

    # ------------------------------------------------------------------
    # DENSITY
    #
    # The reference images carry roughly fifteen marks. Attention moves; it
    # does not accumulate. So each face gets an ATTENTION WINDOW -- box,
    # circle, connector and one label appear together and retire together --
    # and the windows are spaced so only CONCURRENT of them overlap. The paint
    # stays visible, which is the point of projecting onto it.
    # ------------------------------------------------------------------
    CONCURRENT = 3.0

    def windows(items, pass_name):
        t0, t1 = window(pass_name, loop)
        n = max(1, len(items))
        step = (t1 - t0) / n
        hold = step * CONCURRENT
        for i, it in enumerate(items):
            a = t0 + i * step
            yield it, a, min(t1, a + hold)

    kw_by_face = {}
    for sess in mine:
        for cl in sess["answer_claims"]:
            k = cl.get("keyword")
            fid = (cl.get("anchor") or {}).get("face_id")
            if k and fid and fid not in kw_by_face:
                kw_by_face[fid] = (k, cl["unsupportable"])
    loose = [(sess_i, ci, cl) for sess_i, sess in
             ((sessions.index(x), x) for x in mine)
             for ci, cl in enumerate(sess["answer_claims"])
             if cl.get("keyword") and not (cl.get("anchor") or {}).get("face_id")]

    # ---------- PASS 1 DETECT: a box, a circle, one number ----------
    for f, a, b in windows(cons, "DETECT"):
        fid = f["face_id"]; i = idx[fid]
        d = f["detectors"]; bf, yn = d["blazeface"]["score"], d["yunet"]["score"]
        add("box_emerge", "DETECT", "detection", f"/faces/{i}", a, b,
            face_ids=[fid], grow_s=0.8)
        add("node", "DETECT", "detection", f"/faces/{i}", a + 0.5, b,
            face_ids=[fid], at=nodes[fid], r=0.017)
        add("connector", "DETECT", "detection", f"/faces/{i}", a + 0.5, b,
            face_ids=[fid], at=nodes[fid], grow_s=0.6)
        add("connector_label", "DETECT", "detection", f"/faces/{i}", a + 1.1, b,
            face_ids=[fid], at=nodes[fid], label=f"{max(bf, yn):.3f}")

    # ---------- PASS 2 RESOLVE: geometry, or the absence of it ----------
    for f, a, b in windows(cons, "RESOLVE"):
        fid = f["face_id"]; i = idx[fid]
        add("box_emerge", "RESOLVE", "detection", f"/faces/{i}", a, b,
            face_ids=[fid], grow_s=0.8)
        if f["mesh_ok"]:
            t_ = tier.get(fid, {}).get("effective_tier", "box_only")
            cue = {"full_mesh": "mesh_full", "contour": "mesh_contour",
                   "landmarks": "mesh_points"}.get(t_, "mesh_points")
            add(cue, "RESOLVE", "detection", f"/faces/{i}/landmarks", a + 0.5, b,
                face_ids=[fid])
            add("connector_label", "RESOLVE", "detection", f"/faces/{i}", a + 0.9, b,
                face_ids=[fid], at=nodes[fid], label=f"{len(f['landmarks'])}/478")
        else:
            add("box_content", "RESOLVE", "detection", f"/faces/{i}", a, b,
                face_ids=[fid], mode="fail", grow_s=0.7, alarm=True)
            add("connector_label", "RESOLVE", "detection", f"/faces/{i}", a + 0.5, b,
                face_ids=[fid], at=nodes[fid], alarm=True, label="no geometry")

    # ---------- PASS 3 MEASURE: the model's word, and one number ----------
    for f, a, b in windows(cons, "MEASURE"):
        fid = f["face_id"]; i = idx[fid]
        add("box_emerge", "MEASURE", "detection", f"/faces/{i}", a, b,
            face_ids=[fid], grow_s=0.8)
        r = anal.get(fid)
        if r and r.get("head_pose"):
            k = anal_index[fid]
            add("node", "MEASURE", "detection", f"/faces/{i}", a + 0.4, b,
                face_ids=[fid], at=nodes[fid], r=0.017)
            add("connector", "MEASURE", "detection", f"/faces/{i}", a + 0.4, b,
                face_ids=[fid], at=nodes[fid], grow_s=0.6)
            add("connector_label", "MEASURE", "face_analysis",
                f"/panels/{panel}/{k}/head_pose", a + 1.0, b,
                face_ids=[fid], at=nodes[fid],
                label=f"yaw {r['head_pose']['yaw_deg']:+.0f}")
        if fid in kw_by_face:
            w, unsup = kw_by_face[fid]
            add("claim_word", "MEASURE", "claims", "/0", a + 1.4, b,
                face_ids=[fid], label=w, unsupportable=unsup, alarm=unsup)

    # a few of the model's unanchored words, drifting on the canvas
    seed = 7
    for (si, ci, cl), a, b in windows(loose[:10], "MEASURE"):
        seed = (seed * 1103515245 + 12345) & 0x7fffffff
        add("claim_word", "MEASURE", "claims", f"/{si}/answer_claims/{ci}", a, b,
            at=[round(0.10 + (seed % 780) / 1000.0, 4),
                round(0.12 + ((seed >> 11) % 740) / 1000.0, 4)],
            label=cl["keyword"], unsupportable=cl["unsupportable"],
            alarm=cl["unsupportable"])

    # ---------- PASS 4 RELATE: only what crosses the threshold ----------
    E = stru["edges"]
    rel = [(n, e) for n, e in enumerate(E)
           if e["type"] == "similarity" and e["above_same_identity_threshold"]
           and e["from"] in nodes and e["to"] in nodes]
    for (n, e), a, b in windows(rel[:6], "RELATE"):
        for fid in (e["from"], e["to"]):
            add("box_emerge", "RELATE", "detection", f"/faces/{idx[fid]}", a, b,
                face_ids=[fid], grow_s=0.6)
        add("relation", "RELATE", "structure", f"/edges/{n}", a + 0.4, b,
            face_ids=[e["from"], e["to"]], grow_s=0.9, style="dashed",
            label=f"cos {e['cosine']:.3f}")
    gz = stru["per_panel"][panel]["gaze_convergence"]
    ga, gb = window("RELATE", loop)
    if gz.get("converges"):
        for e_i, e in [(n, e) for n, e in enumerate(E)
                       if e["type"] == "gaze" and e["from"] in nodes]:
            add("gaze_ray", "RELATE", "structure", f"/edges/{e_i}",
                ga + (gb - ga) * 0.55, gb, face_ids=[e["from"]], grow_s=1.4)
        add("convergence", "RELATE", "structure",
            f"/per_panel/{panel}/gaze_convergence",
            ga + (gb - ga) * 0.7, gb, at=gz["point"],
            label=f"res {gz['residual_mean']:.4f}")

    # ---------- PASS 5 DOUBT: what was thrown away ----------
    ghosts = [f for f in faces if len(f["detectors"]) < 2
              and (stabr.get(f["face_id"], {}).get("flash_hz", 0) or 0) > 0]
    da, db = window("DOUBT", loop)
    for f, a, b in windows(ghosts[:14], "DOUBT"):
        fid = f["face_id"]; i = idx[fid]
        add("ghost", "DOUBT", "detection", f"/faces/{i}", a, b, face_ids=[fid],
            state=stabr[fid]["track_state"],
            flash_from={"file": "stability", "pointer":
                        f"/panels/{panel}/{stab_index[fid]}", "field": "flash_hz"})
    orph = [(n, e) for n, e in enumerate(E)
            if e["type"] == "attribution" and e.get("panel") == panel and e["unattached"]]
    for (n, e), a, b in windows(orph, "DOUBT"):
        add("orphan", "DOUBT", "structure", f"/edges/{n}", a, b,
            face_ids=[e["to"]], blob_ids=[e["from"]], alarm=True, label="no face")
    add("floor", "DOUBT", "noise_floor", "/floor", da + (db - da) * 0.35, db,
        label=f"floor {nf['floor']['blazeface']['false_positive_ceiling']}")

    # ---------- the status block, always on ----------
    if mine:
        s0 = mine[0]
        add("status_block", "DETECT", "claims", f"/{sessions.index(s0)}", 0.0, loop,
            lines=[f"PANEL      {panel}",
                   f"DETECTED   {det['counts']['merged']}",
                   f"CONFIRMED  {det['counts']['both_detectors_agree']}",
                   f"GEOMETRY   {det['counts']['mesh_converged']}",
                   f"MODEL SAID {s0['headcount']['stated']}",
                   f"HEDGE      {s0['hedges']['per_hundred_words']}/100W"])

    cues.sort(key=lambda c: (c["t_in"], c["cue_id"]))

    answer, reasoning = [], []
    if mine:
        sess = mine[0]
        cl = [c2 for c2 in sess["answer_claims"] if len(c2["text"]) > 40]
        span = loop / max(1, len(cl))
        for i, c2 in enumerate(cl):
            answer.append({"cue_id": f"{panel}-ans-{i:03d}",
                           "t_in": round(i * span, 3),
                           "t_out": round((i + 1) * span, 3),
                           "text": c2["text"], "char_onsets_ms": char_onsets(c2["text"]),
                           "claim_ids": [c2["claim_id"]], "section": c2["section"],
                           "evidence": c2["evidence"],
                           "unsupportable": c2["unsupportable"]})
        rf = sess["source"].get("reasoning_file")
        if rf and pathlib.Path(rf).exists():
            raw = pathlib.Path(rf).read_text()
            lines = [ln.strip() for ln in raw.splitlines() if len(ln.strip()) > 3]
            # the trace leads the answer by lead_offset_s so deliberation is seen first
            rspan = loop / max(1, len(lines))
            for i, ln in enumerate(lines):
                t0 = max(0.0, i * rspan + lead)
                reasoning.append({"cue_id": f"{panel}-rsn-{i:03d}",
                                  "t_in": round(t0, 3),
                                  "t_out": round(min(loop, t0 + rspan * 2.4), 3),
                                  "text": ln, "line_interval_ms": 34,
                                  "section": "trace"})
    if specimen and not answer:
        span = loop / len(SPECIMEN)
        for i, text in enumerate(SPECIMEN):
            answer.append({"cue_id": f"{panel}-specimen-{i:02d}",
                           "t_in": round(i * span, 3), "t_out": round((i + 1) * span, 3),
                           "text": text, "char_onsets_ms": char_onsets(text),
                           "claim_ids": [], "section": "specimen"})

    return {
        "schema": "mimesis.score.v2",
        "panel": panel,
        "session_id": None,
        "loop_duration_s": loop,
        "clock": {"source": "performance.now", "audio_file": None,
                  "fallback": "performance.now"},
        "lead_offset_s": lead,
        "passes": [{"name": n, "t_in": round(a * loop, 2), "t_out": round(b * loop, 2),
                    "role": r} for n, a, b, r in PASSES],
        "nodes": nodes,
        "stability_passes": total_passes,
        "reveal": {"char_ms": CHAR_MS, "sentence_pause_ms": SENTENCE_PAUSE_MS,
                   "comma_pause_ms": COMMA_PAUSE_MS, "blur_resolve_ms": BLUR_RESOLVE_MS},
        "source_state": {
            "answer": ("model_output" if mine else
                       ("type_specimen_not_model_output" if specimen
                        else "awaiting_transcripts")),
            "reasoning": ("model_output" if reasoning else "not_supplied"),
            "sessions": [x["session_id"] for x in mine],
            "overlay": "complete",
            "note": ("Connector labels are the pipeline's own reasoning, generated "
                     "from measured values. They are not model transcripts and must "
                     "never be presented as such. When transcripts arrive the "
                     "reasoning track carries the model's words and these remain the "
                     "instrument layer beneath."),
        },
        "tracks": {"answer": answer, "reasoning": reasoning, "overlay": cues},
    }


def selftest() -> None:
    s = "Fourteen people, or about that. I cannot be certain."
    o = char_onsets(s)
    assert len(o) == len(s) and all(o[i] < o[i + 1] for i in range(len(o) - 1))
    i = s.index(".")
    assert o[i + 1] - o[i] > SENTENCE_PAUSE_MS
    assert char_onsets(s) == o
    fake = [{"face_id": f"X{i:02d}", "box": [0.1 + 0.2 * (i % 4), 0.15 * (i // 4), .09, .12]}
            for i in range(8)]
    nd = place_nodes(fake, 4 / 3)
    assert len(nd) == 8 and all(0 <= v[0] <= 1 and 0 <= v[1] <= 1 for v in nd.values())
    pts = list(nd.values())
    mind = min(math.hypot(a[0] - b[0], a[1] - b[1])
               for i, a in enumerate(pts) for b in pts[i + 1:])
    assert mind > 0.02, f"nodes too close: {mind}"
    print(f"selftest OK — onsets {len(o)} chars; node layout min separation {mind:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("out"))
    ap.add_argument("--loop", type=float, default=900.0)
    ap.add_argument("--lead", type=float, default=-4.0)
    ap.add_argument("--specimen", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest(); return
    selftest()
    for panel in ("A", "B"):
        doc = build(panel, args.out, args.loop, args.lead, args.specimen)
        (args.out / f"score_{panel.lower()}.json").write_text(json.dumps(doc, indent=2))
        kinds, per_pass = {}, {}
        for c in doc["tracks"]["overlay"]:
            kinds[c["cue"]] = kinds.get(c["cue"], 0) + 1
            per_pass[c["pass"]] = per_pass.get(c["pass"], 0) + 1
        print(f"panel {panel}: {len(doc['tracks']['overlay'])} cues, "
              f"{len(kinds)} types, nodes {len(doc['nodes'])}")
        print(f"   per pass: {per_pass}")


if __name__ == "__main__":
    main()
