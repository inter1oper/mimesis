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

    def stagger(items, pass_name, hold_frac, lead_frac=0.55):
        t0, t1 = window(pass_name, loop)
        span = (t1 - t0) * lead_frac
        n = max(1, len(items))
        for i, it in enumerate(items):
            a = t0 + span * i / n
            yield it, a, min(t1, a + (t1 - t0) * hold_frac)

    # ---------- PASS 1 DETECT ----------
    for f, a, b in stagger(cons, "DETECT", 0.92):
        fid = f["face_id"]; i = idx[fid]
        d = f["detectors"]
        bf, yn = d["blazeface"]["score"], d["yunet"]["score"]
        floor_bf = nf["floor"]["blazeface"]["false_positive_ceiling"]
        # the armature persists for the whole loop; only its contents and its
        # labels change from pass to pass
        add("box_emerge", "DETECT", "detection", f"/faces/{i}", a, loop,
            face_ids=[fid], grow_s=0.9)
        add("node", "DETECT", "detection", f"/faces/{i}", a + 0.6, loop,
            face_ids=[fid], at=nodes[fid], r=0.019, value=f"{max(bf, yn):.3f}")
        add("connector", "DETECT", "detection", f"/faces/{i}", a + 0.6, loop,
            face_ids=[fid], at=nodes[fid], grow_s=0.7)
        add("connector_label", "DETECT", "detection", f"/faces/{i}", a + 1.3, b,
            face_ids=[fid], at=nodes[fid],
            label=f"{max(bf, yn):.3f}",
            label2=("confirmed" if min(bf, yn) > floor_bf else "under floor"))
        add("box_content", "DETECT", "detection", f"/faces/{i}", a + 0.9, b,
            face_ids=[fid], mode="confidence", value=max(bf, yn))

    # ---------- PASS 2 RESOLVE ----------
    for f, a, b in stagger(cons, "RESOLVE", 0.86):
        fid = f["face_id"]; i = idx[fid]
        t = tier.get(fid, {}).get("effective_tier", "box_only")
        n_pts = len(f.get("landmarks") or [])
        if f["mesh_ok"]:
            add("box_content", "RESOLVE", "detection", f"/faces/{i}", a, b,
                face_ids=[fid], mode="subdivide", cells=3, grow_s=0.8)
            add("connector_label", "RESOLVE", "detection", f"/faces/{i}", a + 0.4, b,
                face_ids=[fid], at=nodes[fid],
                label=f"{n_pts}/478",
                label2=t.replace("_", " "))
            cue = {"full_mesh": "mesh_full", "contour": "mesh_contour",
                   "landmarks": "mesh_points"}.get(t, "mesh_points")
            add(cue, "RESOLVE", "detection", f"/faces/{i}/landmarks", a + 0.9, loop,
                face_ids=[fid])
        else:
            add("box_content", "RESOLVE", "detection", f"/faces/{i}", a, loop,
                face_ids=[fid], mode="fail", grow_s=0.8, alarm=True)
            add("connector_label", "RESOLVE", "detection", f"/faces/{i}", a + 0.4, b,
                face_ids=[fid], at=nodes[fid], alarm=True,
                label="no geometry", label2="unresolved")

    # ---------- PASS 3 MEASURE ----------
    for f, a, b in stagger(cons, "MEASURE", 0.80):
        fid = f["face_id"]
        if fid not in anal:
            continue
        k = anal_index[fid]; r = anal[fid]
        rows = []
        if r.get("geometry"):
            rows.append((f"iod {r['geometry']['interocular_px']:.0f}",
                         "face_analysis", f"/panels/{panel}/{k}/geometry"))
        if r.get("head_pose"):
            hp = r["head_pose"]
            rows.append((f"yaw {hp['yaw_deg']:+.0f}  roll {hp['roll_deg']:+.0f}",
                         "face_analysis", f"/panels/{panel}/{k}/head_pose"))
        if r.get("symmetry"):
            rows.append((f"sym {r['symmetry']['residual_mean_iod']:.4f}",
                         "face_analysis", f"/panels/{panel}/{k}/symmetry"))
        for bs in (r.get("blendshapes_top8") or [])[:3]:
            rows.append((f"{bs['name'][:14]} {bs['pct']:.0f}%",
                         "face_analysis", f"/panels/{panel}/{k}/blendshapes_top8"))
        if r.get("photometry"):
            rows.append((f"luma {r['photometry']['luma_mean']:.0f}",
                         "face_analysis", f"/panels/{panel}/{k}/photometry"))
        step = (b - a) / max(1, len(rows))
        for j, (text, sf, sp) in enumerate(rows):
            add("connector_label", "MEASURE", sf, sp, a + j * step, b,
                face_ids=[fid], at=nodes[fid], row=j, label=text)
        bars = [(bs["name"], bs["pct"]) for bs in (r.get("blendshapes_top8") or [])[:4]]
        if bars:
            add("box_content", "MEASURE", "face_analysis",
                f"/panels/{panel}/{k}/blendshapes_top8", a, b,
                face_ids=[fid], mode="bars", bars=bars, grow_s=1.1)
        add("node_pulse", "MEASURE", "detection", f"/faces/{idx[fid]}", a, b,
            face_ids=[fid], at=nodes[fid], r=0.019, rows=len(rows))

    # ---------- PASS 4 RELATE ----------
    E = stru["edges"]
    rel = []
    for n, e in enumerate(E):
        if e["type"] == "similarity" and e["above_same_identity_threshold"]:
            a_in = e["from"] in nodes
            b_in = e["to"] in nodes
            if a_in and b_in:
                rel.append((n, e, "similarity"))
            elif (a_in or b_in) and e["cross_panel"]:
                rel.append((n, e, "cross"))
        elif e["type"] == "proximity" and e.get("panel") == panel and e["in_mst"]:
            rel.append((n, e, "mst"))
    for (n, e, kind), a, b in stagger(rel, "RELATE", 0.62):
        if kind == "mst":
            add("relation", "RELATE", "structure", f"/edges/{n}", a, b,
                face_ids=[e["from"], e["to"]], grow_s=0.6, style="solid",
                label=f"{e['distance_canvas']:.3f}")
        else:
            thr = stru["same_identity_cosine"]
            add("relation", "RELATE", "structure", f"/edges/{n}", a, b,
                face_ids=[e["from"], e["to"]], grow_s=0.9, style="dashed",
                cross=(kind == "cross"), alarm=(kind == "cross"),
                label=f"cos {e['cosine']:.3f}", label2="cannot separate")
    gz = stru["per_panel"][panel]["gaze_convergence"]
    ga, gb = window("RELATE", loop)
    for e_i, e in [(n, e) for n, e in enumerate(E)
                   if e["type"] == "gaze" and e["from"] in nodes]:
        add("gaze_ray", "RELATE", "structure", f"/edges/{e_i}",
            ga + (gb - ga) * 0.45, gb, face_ids=[e["from"]], grow_s=1.2)
    if gz.get("converges"):
        add("convergence", "RELATE", "structure", f"/per_panel/{panel}/gaze_convergence",
            ga + (gb - ga) * 0.6, gb, at=gz["point"],
            label=f"res {gz['residual_mean']:.4f}",
            label2=("shared attention" if gz["residual_mean"] < 0.09
                    else "no shared attention"))

    # ---------- PASS 5 DOUBT ----------
    ghosts = [f for f in faces if len(f["detectors"]) < 2
              and (stabr.get(f["face_id"], {}).get("flash_hz", 0) or 0) > 0]
    da, db = window("DOUBT", loop)
    for f in ghosts:
        fid = f["face_id"]; i = idx[fid]
        r = stabr[fid]
        add("ghost", "DOUBT", "detection", f"/faces/{i}", da, db, face_ids=[fid],
            hatch=True,
            state=r["track_state"],
            flash_from={"file": "stability", "pointer": f"/panels/{panel}/{stab_index[fid]}",
                        "field": "flash_hz"})
    dis = [(n, e) for n, e in enumerate(E)
           if e["type"] == "disagreement" and e.get("panel") == panel
           and e["box_iou"] < 0.75]
    for (n, e), a, b in stagger(dis, "DOUBT", 0.55):
        add("disagreement", "DOUBT", "structure", f"/edges/{n}", a, b,
            face_ids=[e["from"]], label=f"IoU {e['box_iou']:.2f}")
    orph = [(n, e) for n, e in enumerate(E)
            if e["type"] == "attribution" and e.get("panel") == panel and e["unattached"]]
    for (n, e), a, b in stagger(orph, "DOUBT", 0.7):
        add("orphan", "DOUBT", "structure", f"/edges/{n}", a, b,
            face_ids=[e["to"]], blob_ids=[e["from"]], alarm=True,
            label="no face", label2=f"{e['distance_face_heights']:.2f}h")
    add("floor", "DOUBT", "noise_floor", "/floor", da + (db - da) * 0.2, db,
        label=f"floor {nf['floor']['blazeface']['false_positive_ceiling']} / "
              f"{nf['floor']['yunet']['false_positive_ceiling']}",
        label2="below this is not evidence")
    add("tally", "DOUBT", "detection", "/counts", da + (db - da) * 0.45, db,
        label=f"{det['counts']['merged']} / "
              f"{det['counts']['both_detectors_agree']} / "
              f"{det['counts']['mesh_converged']}",
        label2=f"{len(ghosts)} rejected")

    # ---------- the model's own words, on the paint ----------
    # One or two words lifted verbatim from a claim, placed at the face the
    # claim refers to when it names a position, and scattered on the canvas
    # when it does not. The reference look prints single words; these are the
    # model's, not ours.
    words = []
    for sess in mine:
        si = sessions.index(sess)
        for ci, cl in enumerate(sess["answer_claims"]):
            if cl.get("keyword"):
                words.append((si, ci, cl))
    seed = 0
    for (si, ci, cl), a, b in stagger(words[:64], "MEASURE", 0.34, lead_frac=0.94):
        fid = (cl.get("anchor") or {}).get("face_id")
        pos = None
        if not fid:
            seed = (seed * 1103515245 + 12345) & 0x7fffffff
            pos = [round(0.08 + (seed % 840) / 1000.0, 4),
                   round(0.10 + ((seed >> 10) % 800) / 1000.0, 4)]
        add("claim_word", "MEASURE", "claims", f"/{si}/answer_claims/{ci}", a, b, face_ids=[fid] if fid else [], at=pos,
            label=cl["keyword"],
            evidence=cl["evidence"], unsupportable=cl["unsupportable"],
            alarm=cl["unsupportable"])

    # a status block, as on an instrument that is telling you it is struggling
    sa, sb = window("DETECT", loop)
    if mine:
        s0 = mine[0]
        add("status_block", "DETECT", "claims", f"/{sessions.index(s0)}", sa, loop,
            lines=[f"PANEL      {panel}",
                   f"SESSIONS   {len(mine)}",
                   f"CLAIMS     {sum(len(x['answer_claims']) for x in mine)}",
                   f"SEEN       {sum(1 for x in mine for c2 in x['answer_claims'] if c2['evidence']=='seen')}",
                   f"SUPPLIED   {sum(1 for x in mine for c2 in x['answer_claims'] if c2['evidence']=='supplied')}",
                   f"UNSUPPORT  {sum(1 for x in mine for c2 in x['answer_claims'] if c2['unsupportable'])}",
                   f"HEDGE/100W {s0['hedges']['per_hundred_words']}",
                   f"HEADCOUNT  {s0['headcount']['stated']}  DETECTOR {det['counts']['both_detectors_agree']}"])

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
