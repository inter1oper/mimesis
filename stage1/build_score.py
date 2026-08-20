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

# Typing cadence for the answer stream. The answer is typed as ONE continuous
# stream rather than one cue per claim: giving each claim its own slot across a
# fifteen-minute loop meant every claim typed itself and then sat idle, which is
# what made it crawl. The stream types straight through, holds, and repeats.
CHAR_MS = 18.0
CHAR_JITTER_MS = 3.0
SENTENCE_PAUSE_MS = 200.0
COMMA_PAUSE_MS = 80.0
PARA_PAUSE_MS = 340.0
BLUR_RESOLVE_MS = 110.0
STREAM_HOLD_MS = 4000.0        # pause on the finished text before it restarts

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
    """Onset in ms for every character. Deterministic: the jitter is a function
    of the character index, so a retime is reproducible and a seek is exact."""
    out, t = [], 0.0
    for i, ch in enumerate(text):
        out.append(round(t, 1))
        t += CHAR_MS + CHAR_JITTER_MS * (((i * 2654435761) % 1000) / 1000.0 - 0.5) * 2
        if ch == "\n":
            t += PARA_PAUSE_MS
        elif ch in ".!?":
            t += SENTENCE_PAUSE_MS
        elif ch in ",;:":
            t += COMMA_PAUSE_MS
    return out


def strip_markdown(t: str) -> str:
    import re as _re
    t = _re.sub(r"^\s*[*\-]\s+", "", t)
    t = _re.sub(r"^#{1,6}\s*", "", t)
    t = _re.sub(r"\*\*(.+?)\*\*", r"\1", t)
    t = _re.sub(r"\*(.+?)\*", r"\1", t)
    return _re.sub(r"\s+", " ", t).strip()


def build_stream(claim_sessions, panel):
    """
    One continuous typed stream for the panel, with a segment table so the
    renderer can mark observed against supplied without re-parsing anything.

    Markdown markers are dropped here, for display only; claims.json keeps every
    claim verbatim with its offsets intact.
    """
    parts, segs, sec = [], [], None
    pos = 0
    for sess in claim_sessions:
        for c in sess["answer_claims"]:
            txt = strip_markdown(c["text"])
            if not txt:
                continue
            head = None
            if c.get("section") and c["section"] != sec:
                sec = c["section"]
                head = strip_markdown(sec)
            # A claim whose text IS its heading is emitted once, as the section,
            # and then skipped below -- not suppressed in both places.
            if head:
                parts.append(head + "\n")
                segs.append({"start": pos, "end": pos + len(head),
                             "kind": "section"})
                pos += len(head) + 1
            if strip_markdown(sec or "").lower() == txt.lower():
                continue
            parts.append(txt + "\n")
            segs.append({"start": pos, "end": pos + len(txt),
                         "kind": "claim", "evidence": c["evidence"],
                         "unsupportable": c["unsupportable"],
                         "claim_id": c["claim_id"], "session": sess["session_id"]})
            pos += len(txt) + 1
    text = "".join(parts)
    onsets = char_onsets(text)
    total = (onsets[-1] + CHAR_MS) if onsets else 0.0
    return {"text": text, "char_onsets_ms": onsets,
            "cycle_ms": round(total + STREAM_HOLD_MS, 1),
            "type_ms": round(total, 1),
            "blur_resolve_ms": BLUR_RESOLVE_MS,
            "hold_ms": STREAM_HOLD_MS,
            "segments": segs}


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
          specimen: bool = False, cycle: float = 45.0) -> dict:
    p = panel.lower()
    det = json.loads((out / f"panel_{p}_detection.json").read_text())
    fl = json.loads((out / f"panel_{p}_flames.json").read_text())
    plan = json.loads((out / f"panel_{p}_overlay_plan.json").read_text())
    stab = json.loads((out / "stability.json").read_text())
    stru = json.loads((out / "structure.json").read_text())
    fa = json.loads((out / "face_analysis.json").read_text())
    nf = json.loads((out / "noise_floor.json").read_text())
    ff_path = out / "face_features.json"
    feats = {}
    if ff_path.exists():
        ffd = json.loads(ff_path.read_text())
        feats = {r["face_id"]: (i, r) for i, r in enumerate(ffd["panels"].get(panel, []))}
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
    # does not accumulate. So each face gets an ATTENTION WINDOW -- box, a
    # leader line and one label appear together and retire together --
    # and the windows are spaced so only CONCURRENT of them overlap. The paint
    # stays visible, which is the point of projecting onto it.
    # ------------------------------------------------------------------
    CONCURRENT = 3.0

    def windows(items, pass_name):
        t0, t1 = window(pass_name, cycle)
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
            face_ids=[fid], grow_s=0.16)
        add("connector", "DETECT", "detection", f"/faces/{i}", a + 0.10, b,
            face_ids=[fid], at=nodes[fid], grow_s=0.12)
        add("connector_label", "DETECT", "detection", f"/faces/{i}", a + 0.22, b,
            face_ids=[fid], at=nodes[fid], label=f"{max(bf, yn):.3f}")

    # ---------- PASS 2 RESOLVE: geometry, or the absence of it ----------
    for f, a, b in windows(cons, "RESOLVE"):
        fid = f["face_id"]; i = idx[fid]
        add("box_emerge", "RESOLVE", "detection", f"/faces/{i}", a, b,
            face_ids=[fid], grow_s=0.16)
        if f["mesh_ok"]:
            t_ = tier.get(fid, {}).get("effective_tier", "box_only")
            cue = {"full_mesh": "mesh_full", "contour": "mesh_contour",
                   "landmarks": "mesh_points"}.get(t_, "mesh_points")
            add(cue, "RESOLVE", "detection", f"/faces/{i}/landmarks", a + 0.10, b,
                face_ids=[fid])
            add("connector_label", "RESOLVE", "detection", f"/faces/{i}", a + 0.20, b,
                face_ids=[fid], at=nodes[fid], label=f"{len(f['landmarks'])}/478")
            # take the face apart: eye, eye, nose, mouth, one after another
            if fid in feats:
                fi, fr = feats[fid]
                # most feature boxes carry no label, as in the reference; only
                # the two that would otherwise be unreadable get a number, and
                # they sit at opposite ends of the face so they cannot stack
                order = [("eye_r", f"eye {fr['ear_r']:.2f}"),
                         ("eye_l", None),
                         ("iris_r", None),
                         ("nose", None),
                         ("lips_in", f"mouth {fr['mar']:.2f}")]
                step = (b - a) / (len(order) + 1)
                for j, (key, lab) in enumerate(order):
                    if key not in fr["features"]:
                        continue
                    add("feature_box", "RESOLVE", "face_features",
                        f"/panels/{panel}/{fi}/features/{key}",
                        a + 0.24 + j * step, b, face_ids=[fid], feature=key,
                        label=lab, grow_s=0.10)
        else:
            add("box_content", "RESOLVE", "detection", f"/faces/{i}", a, b,
                face_ids=[fid], mode="fail", grow_s=0.14, alarm=True)
            add("connector_label", "RESOLVE", "detection", f"/faces/{i}", a + 0.10, b,
                face_ids=[fid], at=nodes[fid], alarm=True, label="no geometry")

    # ---------- PASS 3 MEASURE: the model's word, and one number ----------
    for f, a, b in windows(cons, "MEASURE"):
        fid = f["face_id"]; i = idx[fid]
        add("box_emerge", "MEASURE", "detection", f"/faces/{i}", a, b,
            face_ids=[fid], grow_s=0.16)
        r = anal.get(fid)
        if r and r.get("head_pose"):
            k = anal_index[fid]
            add("connector", "MEASURE", "detection", f"/faces/{i}", a + 0.09, b,
                face_ids=[fid], at=nodes[fid], grow_s=0.12)
            add("connector_label", "MEASURE", "face_analysis",
                f"/panels/{panel}/{k}/head_pose", a + 1.0, b,
                face_ids=[fid], at=nodes[fid],
                label=f"yaw {r['head_pose']['yaw_deg']:+.0f}")
        if fid in kw_by_face:
            w, unsup = kw_by_face[fid]
            add("claim_word", "MEASURE", "claims", "/0", a + 0.30, b,
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
                face_ids=[fid], grow_s=0.12)
        add("relation", "RELATE", "structure", f"/edges/{n}", a + 0.09, b,
            face_ids=[e["from"], e["to"]], grow_s=0.18, style="dashed",
            label=f"cos {e['cosine']:.3f}")
    gz = stru["per_panel"][panel]["gaze_convergence"]
    ga, gb = window("RELATE", cycle)
    if gz.get("converges"):
        for e_i, e in [(n, e) for n, e in enumerate(E)
                       if e["type"] == "gaze" and e["from"] in nodes]:
            add("gaze_ray", "RELATE", "structure", f"/edges/{e_i}",
                ga + (gb - ga) * 0.55, gb, face_ids=[e["from"]], grow_s=0.30)
        add("convergence", "RELATE", "structure",
            f"/per_panel/{panel}/gaze_convergence",
            ga + (gb - ga) * 0.7, gb, at=gz["point"],
            label=f"res {gz['residual_mean']:.4f}")

    # ---------- PASS 5 DOUBT: what was thrown away ----------
    ghosts = [f for f in faces if len(f["detectors"]) < 2
              and (stabr.get(f["face_id"], {}).get("flash_hz", 0) or 0) > 0]
    da, db = window("DOUBT", cycle)
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
        add("status_block", "DETECT", "claims", f"/{sessions.index(s0)}", 0.0, cycle,
            lines=[f"PANEL      {panel}",
                   f"DETECTED   {det['counts']['merged']}",
                   f"CONFIRMED  {det['counts']['both_detectors_agree']}",
                   f"GEOMETRY   {det['counts']['mesh_converged']}",
                   f"MODEL SAID {s0['headcount']['stated']}",
                   f"HEDGE      {s0['hedges']['per_hundred_words']}/100W"])

    cues.sort(key=lambda c: (c["t_in"], c["cue_id"]))

    answer, reasoning, stream = [], [], None
    if mine:
        sess = mine[0]
        stream = build_stream(mine, panel)
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
        "overlay_cycle_s": cycle,
        "clock": {"source": "performance.now", "audio_file": None,
                  "fallback": "performance.now"},
        "lead_offset_s": lead,
        "passes": [{"name": n, "t_in": round(a * cycle, 2), "t_out": round(b * cycle, 2),
                    "role": r} for n, a, b, r in PASSES],
        "nodes": nodes,
        "stability_passes": total_passes,
        "reveal": {"char_ms": CHAR_MS, "sentence_pause_ms": SENTENCE_PAUSE_MS,
                   "comma_pause_ms": COMMA_PAUSE_MS, "blur_resolve_ms": BLUR_RESOLVE_MS},
        "source_state": {
            "answer": ("model_output" if stream else
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
        "answer_stream": stream,
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
    ap.add_argument("--cycle", type=float, default=45.0,
                    help="Seconds for one full DETECT..DOUBT pass of the overlay. "
                         "The overlay repeats this many times inside the master "
                         "loop; the text stream keeps its own cycle. Both are read "
                         "off the same clock.")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest(); return
    selftest()
    for panel in ("A", "B"):
        doc = build(panel, args.out, args.loop, args.lead, args.specimen, args.cycle)
        (args.out / f"score_{panel.lower()}.json").write_text(json.dumps(doc, indent=2))
        kinds, per_pass = {}, {}
        for c in doc["tracks"]["overlay"]:
            kinds[c["cue"]] = kinds.get(c["cue"], 0) + 1
            per_pass[c["pass"]] = per_pass.get(c["pass"], 0) + 1
        print(f"panel {panel}: cycle {args.loop/args.cycle:.0f}x per loop, "
              f"{len(doc['tracks']['overlay'])} cues, "
              f"{len(kinds)} types, nodes {len(doc['nodes'])}")
        print(f"   per pass: {per_pass}")


if __name__ == "__main__":
    main()
