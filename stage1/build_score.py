"""
Stage 1c — the timed score.

Three tracks on one clock. Every time is absolute seconds from loop start, and
the renderer resolves state as a pure function of the clock: it asks what should
be visible at t, it never accumulates. That is what makes a fifteen-minute loop
non-drifting and a seek land exactly.

Text tracks
-----------
The answer and reasoning tracks are populated from parsed transcripts. No
transcripts have been supplied, so those tracks are emitted EMPTY, with
`source_state: "awaiting_transcripts"`. They are not filled with placeholder
prose. `char_onsets()` below is the real implementation and is exercised by
--selftest, so when transcripts arrive nothing about the renderer changes.

Overlay track
-------------
Fully populated from Stage 1. Ordering and timing are derived from the data
itself rather than chosen: cues are grouped into phases, and within a phase each
cue's offset follows its own measured rank -- confidence order for faces, area
order for blobs, distance order for attribution. When transcripts arrive these
same cues gain `trigger` bindings to reasoning spans and this timing becomes the
fallback for when no audio is playing.

Usage
-----
    python stage1/build_score.py --out out/ --loop 900
    python stage1/build_score.py --selftest
"""
from __future__ import annotations

import argparse
import json
import pathlib

# Reveal cadence, from the brief: characters land every 18-25ms, longer pause at
# sentence ends. Onsets are precomputed so reveal is a lookup, not a simulation.
CHAR_MS = 21.0
CHAR_JITTER_MS = 3.5          # deterministic, from character index -- never random
SENTENCE_PAUSE_MS = 420.0
COMMA_PAUSE_MS = 130.0
BLUR_RESOLVE_MS = 120.0

PHASES = [
    ("scan", 0.00, 0.14),        # response field sweep
    ("detect", 0.10, 0.38),      # boxes, brackets, numerals
    ("analyse", 0.34, 0.62),     # blendshapes, pose, symmetry, photometry
    ("structure", 0.58, 0.84),   # gaze, similarity, proximity, attribution
    ("verdict", 0.80, 1.00),     # counts, unattached flames, noise floor
]


def char_onsets(text: str) -> list[float]:
    """
    Onset in milliseconds for every character, relative to the cue's t_in.

    Deterministic: the jitter is a function of the character index, so the same
    text always produces the same score and a retime is reproducible. Sentence
    and clause ends get a longer pause, which is why this is precomputed rather
    than derived from a single cadence at render time.
    """
    out, t = [], 0.0
    for i, ch in enumerate(text):
        out.append(round(t, 2))
        t += CHAR_MS + CHAR_JITTER_MS * (((i * 2654435761) % 1000) / 1000.0 - 0.5) * 2
        if ch in ".!?":
            t += SENTENCE_PAUSE_MS
        elif ch in ",;:":
            t += COMMA_PAUSE_MS
    return out


def phase_window(name: str, loop: float) -> tuple[float, float]:
    for n, a, b in PHASES:
        if n == name:
            return a * loop, b * loop
    raise KeyError(name)


def spread(items: list, name: str, loop: float, hold: float):
    """Place n cues across a phase in the order given, each held for `hold`."""
    t0, t1 = phase_window(name, loop)
    n = max(1, len(items))
    step = (t1 - t0) / n
    for i, it in enumerate(items):
        yield it, round(t0 + i * step, 3), round(min(t1, t0 + i * step + hold), 3)


# A type specimen, used only when --specimen is passed, so the serif, the
# measure, the leading and the blur reveal can be judged before any transcript
# exists. It is text ABOUT the piece, never text attributed to a model, and the
# renderer labels it as a specimen on screen. It is not a placeholder for the
# answer track: without --specimen the track stays empty.
SPECIMEN = [
    "Fourteen people stood in a room and someone photographed them.",
    "A description of that photograph was given to a machine, and the machine "
    "produced a room that had never held anyone.",
    "Both rooms were then painted, by the same hand, in the same hours.",
    "The detector finds faces in both.",
]


def build(panel: str, out: pathlib.Path, loop: float, lead: float,
          specimen: bool = False) -> dict:
    p = panel.lower()
    det = json.loads((out / f"panel_{p}_detection.json").read_text())
    fl = json.loads((out / f"panel_{p}_flames.json").read_text())
    fld = json.loads((out / f"panel_{p}_field.json").read_text())
    plan = json.loads((out / f"panel_{p}_overlay_plan.json").read_text())
    stab = json.loads((out / "stability.json").read_text())
    stru = json.loads((out / "structure.json").read_text())
    fa = json.loads((out / "face_analysis.json").read_text())
    nf = json.loads((out / "noise_floor.json").read_text())

    tier = {f["face_id"]: f for f in plan["faces"]}
    stabr = {r["face_id"]: r for r in stab["panels"].get(panel, [])}
    anal = {r["face_id"]: r for r in fa["panels"].get(panel, [])}
    faces = det["faces"]
    cons = [f for f in faces if len(f["detectors"]) >= 2]
    cons_by_conf = sorted(cons, key=lambda f: -max(v["score"] for v in f["detectors"].values()))

    cues = []

    def add(cue, source_file, pointer, t_in, t_out, **kw):
        c = {"cue_id": f"{panel}-{cue}-{len(cues):04d}", "cue": cue,
             "t_in": t_in, "t_out": t_out,
             "source": {"file": source_file, "pointer": pointer}}
        c.update(kw)
        cues.append(c)

    # --- scan: response field sweep, then the cells that responded most
    t0, t1 = phase_window("scan", loop)
    add("response_field_sweep", "field", "/response_field", round(t0, 3), round(t1, 3),
        axis="x", duration_s=round(t1 - t0, 3),
        fields=["/response_field/values", "/response_field/stats/p90_nonzero"])
    vals = fld["response_field"]["values"]
    thr = fld["response_field"]["stats"]["p90_nonzero"] or 1.0
    hot = [(v, gx, gy) for gy, row in enumerate(vals) for gx, v in enumerate(row) if v >= thr]
    hot.sort(reverse=True)
    for (v, gx, gy), a, b in spread(hot[:120], "scan", loop, hold=loop * 0.05):
        add("response_field_cell", "field", f"/response_field/values/{gy}/{gx}", a, b,
            cell=[gx, gy], value=v)

    # --- detect: brackets, box, numerals, consensus flag, per face in confidence order
    for f, a, b in spread(cons_by_conf, "detect", loop, hold=loop * 0.62):
        i = faces.index(f)
        fid = f["face_id"]
        mb = tier.get(fid, {}).get("governing_px")
        add("corner_brackets", "detection", f"/faces/{i}", a, b, face_ids=[fid])
        add("face_box", "detection", f"/faces/{i}", round(a + 0.15, 3), b, face_ids=[fid])
        add("face_numerals", "detection", f"/faces/{i}", round(a + 0.35, 3), b,
            face_ids=[fid], fields=["/detectors/blazeface/score", "/detectors/yunet/score"],
            min_box_px=24)
        add("consensus_flag", "detection", f"/faces/{i}", round(a + 0.35, 3), b, face_ids=[fid])
        t = tier.get(fid, {}).get("effective_tier", "box_only")
        cue = {"full_mesh": "face_mesh", "contour": "face_mesh_contour",
               "landmarks": "face_landmarks_6"}.get(t)
        if cue and f.get("landmarks"):
            add(cue, "detection", f"/faces/{i}/landmarks", round(a + 0.5, 3), b,
                face_ids=[fid], min_box_px={"face_mesh": 88, "face_mesh_contour": 48,
                                            "face_landmarks_6": 24}[cue])
        if not f["mesh_ok"]:
            add("no_mesh_marker", "detection", f"/faces/{i}", round(a + 0.5, 3), b, face_ids=[fid])

    # rejected hypotheses: flash at their measured rate, right through the loop
    ghosts = [f for f in faces if len(f["detectors"]) < 2
              and (stabr.get(f["face_id"], {}).get("flash_hz", 0) or 0) > 0]
    for f in ghosts:
        i = faces.index(f); fid = f["face_id"]
        r = stabr[fid]
        j = [x["face_id"] for x in stab["panels"][panel]].index(fid)
        add("tentative_track" if r["track_state"] == "tentative" else "lost_track",
            "detection", f"/faces/{i}", 0.0, round(loop, 3), face_ids=[fid],
            flash_from={"file": "stability", "pointer": f"/panels/{panel}/{j}",
                        "field": "flash_hz"})

    # --- analyse: blendshapes, pose, symmetry, photometry
    for f, a, b in spread(cons_by_conf, "analyse", loop, hold=loop * 0.30):
        fid = f["face_id"]
        if fid not in anal:
            continue
        k = [r["face_id"] for r in fa["panels"][panel]].index(fid)
        r = anal[fid]
        if r.get("blendshapes_top8"):
            add("blendshape_stack", "face_analysis", f"/panels/{panel}/{k}/blendshapes_top8",
                a, b, face_ids=[fid], fields=[f"/{i}/pct" for i in range(4)], min_box_px=24)
        if r.get("head_pose"):
            add("head_pose_readout", "face_analysis", f"/panels/{panel}/{k}/head_pose", a, b,
                face_ids=[fid], fields=["/yaw_deg", "/pitch_deg", "/roll_deg"])
        if r.get("symmetry"):
            add("symmetry_readout", "face_analysis", f"/panels/{panel}/{k}/symmetry", a, b,
                face_ids=[fid], fields=["/residual_mean_iod"])
        if r.get("photometry"):
            add("photometry_readout", "face_analysis", f"/panels/{panel}/{k}/photometry", a, b,
                face_ids=[fid], fields=["/luma_mean", "/luma_std"])
        if r.get("geometry"):
            add("interocular_rule", "face_analysis", f"/panels/{panel}/{k}/geometry", a, b,
                face_ids=[fid], fields=["/interocular_px"])

    # --- structure: gaze, convergence, skeleton, similarity, attribution, disagreement
    E = stru["edges"]
    gaze = [(n, e) for n, e in enumerate(E) if e["type"] == "gaze" and e["from"].startswith(panel)]
    for (n, e), a, b in spread(gaze, "structure", loop, hold=loop * 0.24):
        add("gaze_ray", "structure", f"/edges/{n}", a, b, face_ids=[e["from"]],
            fields=["/direction", "/yaw_deg"])
    ts, te = phase_window("structure", loop)
    if stru["per_panel"][panel]["gaze_convergence"].get("converges"):
        add("gaze_convergence", "structure", f"/per_panel/{panel}/gaze_convergence",
            round(ts + (te - ts) * 0.45, 3), round(te, 3),
            fields=["/point", "/residual_mean", "/rays"])
    prox = [(n, e) for n, e in enumerate(E)
            if e["type"] == "proximity" and e.get("panel") == panel]
    for (n, e), a, b in spread(prox, "structure", loop, hold=loop * 0.20):
        add("mst_edge" if e["in_mst"] else "proximity_edge", "structure", f"/edges/{n}", a, b,
            face_ids=[e["from"], e["to"]], fields=["/distance_canvas"])
    sim = [(n, e) for n, e in enumerate(E) if e["type"] == "similarity"
           and e["above_same_identity_threshold"]
           and (e["from"].startswith(panel) or e["to"].startswith(panel))]
    for (n, e), a, b in spread(sim, "structure", loop, hold=loop * 0.18):
        add("cross_panel_link" if e["cross_panel"] else "similarity_edge",
            "structure", f"/edges/{n}", a, b, face_ids=[e["from"], e["to"]],
            fields=["/cosine"])
    attr = [(n, e) for n, e in enumerate(E)
            if e["type"] == "attribution" and e.get("panel") == panel]
    for (n, e), a, b in spread(attr, "structure", loop, hold=loop * 0.16):
        add("unattached_flame_marker" if e["unattached"] else "attribution_link",
            "structure", f"/edges/{n}", a, b, face_ids=[e["to"]], blob_ids=[e["from"]],
            fields=["/distance_face_heights"])
    dis = [(n, e) for n, e in enumerate(E)
           if e["type"] == "disagreement" and e.get("panel") == panel]
    for (n, e), a, b in spread(dis, "structure", loop, hold=loop * 0.14):
        add("disagreement_vector", "structure", f"/edges/{n}", a, b, face_ids=[e["from"]],
            fields=["/centre_offset_canvas", "/box_iou"])

    # --- verdict: the counts, and the floor they should be read against
    va, vb = phase_window("verdict", loop)
    add("count_readout", "detection", "/counts", round(va, 3), round(vb, 3),
        fields=["/merged", "/both_detectors_agree", "/mesh_converged"])
    add("noise_floor_rule", "noise_floor", "/floor", round(va + (vb - va) * 0.25, 3),
        round(vb, 3), fields=["/blazeface/false_positive_ceiling",
                              "/yunet/false_positive_ceiling"])
    big = sorted([b for b in fld["blobs"] if b["segment"] == "lit_skin"],
                 key=lambda b: -b["area_pct_of_canvas"])[:24]
    for b, a, bb in spread(big, "verdict", loop, hold=loop * 0.12):
        j = fld["blobs"].index(b)
        add("blob_circle", "field", f"/blobs/{j}", a, bb, blob_ids=[b["blob_id"]])
        add("blob_readout", "field", f"/blobs/{j}", a, bb, blob_ids=[b["blob_id"]],
            fields=["/area_pct_of_canvas"])

    cues.sort(key=lambda c: (c["t_in"], c["cue_id"]))

    answer = []
    if specimen:
        span = loop / len(SPECIMEN)
        for i, text in enumerate(SPECIMEN):
            onsets = char_onsets(text)
            answer.append({
                "cue_id": f"{panel}-specimen-{i:02d}",
                "t_in": round(i * span, 3),
                "t_out": round((i + 1) * span, 3),
                "text": text,
                "char_onsets_ms": onsets,
                "claim_ids": [],
                "section": "specimen",
            })
    return {
        "schema": "mimesis.score.v1",
        "panel": panel,
        "session_id": None,
        "loop_duration_s": loop,
        "clock": {"source": "performance.now", "audio_file": None,
                  "fallback": "performance.now"},
        "lead_offset_s": lead,
        "reveal": {"char_ms": CHAR_MS, "sentence_pause_ms": SENTENCE_PAUSE_MS,
                   "comma_pause_ms": COMMA_PAUSE_MS,
                   "blur_resolve_ms": BLUR_RESOLVE_MS},
        "source_state": {
            "answer": ("type_specimen_not_model_output" if specimen
                       else "awaiting_transcripts"),
            "reasoning": "awaiting_transcripts",
            "overlay": "complete",
            "note": ("Answer and reasoning are empty because no transcripts have been "
                     "supplied. They are not filled with placeholder prose. When "
                     "transcripts arrive these tracks populate and the overlay cues "
                     "gain trigger bindings to reasoning spans; nothing in the "
                     "renderer changes."),
        },
        "tracks": {"answer": answer, "reasoning": [], "overlay": cues},
    }


def selftest() -> None:
    s = "Fourteen people, or about that. I cannot be certain."
    o = char_onsets(s)
    assert len(o) == len(s), "one onset per character"
    assert all(o[i] < o[i + 1] for i in range(len(o) - 1)), "onsets strictly increasing"
    i = s.index(".")
    gap_at_stop = o[i + 1] - o[i]
    gap_plain = o[2] - o[1]
    assert gap_at_stop > SENTENCE_PAUSE_MS, "sentence end pauses"
    assert gap_plain < CHAR_MS + CHAR_JITTER_MS + 1, "plain characters keep cadence"
    assert char_onsets(s) == o, "deterministic"
    print(f"char_onsets selftest OK: {len(o)} chars, span {o[-1]:.0f}ms, "
          f"stop pause {gap_at_stop:.0f}ms, plain gap {gap_plain:.1f}ms")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("out"))
    ap.add_argument("--loop", type=float, default=900.0)
    ap.add_argument("--lead", type=float, default=-4.0)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--specimen", action="store_true",
                    help="Put a labelled type specimen in the answer track so the "
                         "serif and the blur reveal can be judged before transcripts "
                         "exist. Never model output, and labelled as such on screen.")
    args = ap.parse_args()
    if args.selftest:
        selftest(); return
    selftest()
    for panel in ("A", "B"):
        doc = build(panel, args.out, args.loop, args.lead, args.specimen)
        (args.out / f"score_{panel.lower()}.json").write_text(json.dumps(doc, indent=2))
        kinds = {}
        for c in doc["tracks"]["overlay"]:
            kinds[c["cue"]] = kinds.get(c["cue"], 0) + 1
        print(f"panel {panel}: {len(doc['tracks']['overlay'])} overlay cues over "
              f"{args.loop:.0f}s, {len(kinds)} cue types")
        for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]):
            print(f"    {k:26} {v}")


if __name__ == "__main__":
    main()
