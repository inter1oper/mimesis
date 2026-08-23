"""
Stage 1c — align each narration to its panel's typed stream.

Fitting the typing to the voice's total length makes them finish together but
says nothing about whether a given word appears as it is spoken. This does the
real thing: transcribe the recording with word-level timestamps, match the
recognised words against the transcript the voice is reading, and take the
character onsets from the audio itself.

Method
------
1. faster-whisper transcribes with word timestamps.
2. The recognised word sequence is matched to the stream's word sequence with
   difflib. Only exact matching blocks become ANCHORS -- a recognition error
   simply fails to anchor rather than dragging the timing with it.
3. Anchors are forced monotonic, then character onsets are interpolated between
   them. Regions the recogniser got wrong inherit the local speaking rate from
   the anchors either side, so they stay plausible instead of stalling.

The result is per-character timing measured from the voice. Coverage is
reported: it is the fraction of stream words that anchored, and it is the
number to judge the alignment by.

Usage
-----
    python stage1/align_voice.py --panel A --audio data/audio/panel_a_voice.mp3
"""
from __future__ import annotations

import argparse
import difflib
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


def norm(w: str) -> str:
    return re.sub(r"[^a-z0-9]", "", w.lower())


def stream_words(text: str):
    """Words of the stream with the character offset each one starts at."""
    return [(m.group(0), m.start()) for m in re.finditer(r"\S+", text)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True, choices=["A", "B"])
    ap.add_argument("--audio", required=True, type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("out"))
    ap.add_argument("--model", default="base")
    args = ap.parse_args()

    import build_score as B
    sessions = json.loads((args.out / "claims.json").read_text())
    mine = [s for s in sessions if s.get("panel") == args.panel]
    if not mine:
        raise SystemExit(f"no sessions for panel {args.panel}")

    # transcribe once, cache, so candidate sessions can be tested cheaply
    cache = args.out / f"asr_{args.panel.lower()}.json"
    if cache.exists():
        c = json.loads(cache.read_text())
        aw = [(w, t) for w, t in c["words"]]
        lang = c["language"]
    else:
        from faster_whisper import WhisperModel
        model = WhisperModel(args.model, device="cpu", compute_type="int8")
        segs, info = model.transcribe(str(args.audio), word_timestamps=True,
                                      vad_filter=False, beam_size=1)
        aw = []
        for s in segs:
            for w in (s.words or []):
                n = norm(w.word)
                if n:
                    aw.append((n, float(w.start)))
        lang = info.language
        cache.write_text(json.dumps({"language": lang, "words": aw}))
    if not aw:
        raise SystemExit("no words recognised")
    awn = [w for w, _ in aw]

    # Which session is the voice actually reading? A panel can carry more than
    # one transcript, and a six-minute recording cannot read fourteen thousand
    # characters. Score every candidate and take the evidence, rather than
    # assuming the concatenation is what was narrated.
    def coverage_of(sess_list):
        st = B.build_stream(sess_list, args.panel)
        w = stream_words(st["text"])
        wn = [norm(x) for x, _ in w]
        m = difflib.SequenceMatcher(a=wn, b=awn, autojunk=False)
        anchored = sum(size for _, _, size in m.get_matching_blocks())
        return anchored / max(1, len(w)), st, w, wn

    trials = [([x], x["session_id"]) for x in mine]
    if len(mine) > 1:
        trials.append((mine, "all sessions"))
    scored = [(coverage_of(sl)[0], sl, name) for sl, name in trials]
    scored.sort(reverse=True, key=lambda x: x[0])
    print(f"PANEL {args.panel} candidate coverage:")
    for cov, _, name in scored:
        print(f"    {name:22} {cov*100:5.1f}%")
    best_cov, chosen, chosen_name = scored[0]

    cov, stream, sw, sw_norm = coverage_of(chosen)
    text = stream["text"]

    sm = difflib.SequenceMatcher(a=sw_norm, b=awn, autojunk=False)
    anchors = []
    for i, j, size in sm.get_matching_blocks():
        for k in range(size):
            anchors.append((sw[i + k][1], aw[j + k][1] * 1000.0))
    # monotonic: a later character can never start earlier than an earlier one
    anchors.sort()
    mono, last = [], -1e9
    for ch, ms in anchors:
        if ms > last:
            mono.append((ch, ms)); last = ms
    anchors = mono
    if len(anchors) < 8:
        raise SystemExit(f"only {len(anchors)} anchors; alignment not trustworthy")

    # interpolate every character between anchors
    n = len(text)
    onsets = [0.0] * n
    first_ch, first_ms = anchors[0]
    last_ch, last_ms = anchors[-1]
    lead_rate = ((anchors[min(6, len(anchors) - 1)][1] - first_ms) /
                 max(1, anchors[min(6, len(anchors) - 1)][0] - first_ch))
    for i in range(0, first_ch):
        onsets[i] = max(0.0, first_ms - (first_ch - i) * lead_rate)
    ai = 0
    for i in range(first_ch, last_ch + 1):
        while ai + 1 < len(anchors) and anchors[ai + 1][0] <= i:
            ai += 1
        c0, m0 = anchors[ai]
        if ai + 1 < len(anchors):
            c1, m1 = anchors[ai + 1]
            f = (i - c0) / max(1, c1 - c0)
            onsets[i] = m0 + (m1 - m0) * f
        else:
            onsets[i] = m0
    tail_rate = ((last_ms - anchors[max(0, len(anchors) - 7)][1]) /
                 max(1, last_ch - anchors[max(0, len(anchors) - 7)][0]))
    for i in range(last_ch + 1, n):
        onsets[i] = last_ms + (i - last_ch) * max(8.0, tail_rate)

    doc = {
        "schema": "mimesis.alignment.v1",
        "panel": args.panel,
        "audio": str(args.audio),
        "model": args.model,
        "language": lang,
        "narrated_sessions": [x["session_id"] for x in chosen],
        "candidate_coverage": {name: round(c, 4) for c, _, name in scored},
        "stream_chars": n,
        "stream_words": len(sw),
        "recognised_words": len(aw),
        "anchors": len(anchors),
        "coverage": round(len(anchors) / max(1, len(sw)), 4),
        "first_anchor_ms": round(first_ms, 1),
        "last_anchor_ms": round(last_ms, 1),
        "char_onsets_ms": [round(o, 1) for o in onsets],
        "note": ("Character onsets measured from the recording. Anchors are exact "
                 "word matches between the recogniser and the transcript; "
                 "everything between them is interpolated at the local speaking "
                 "rate. Coverage is the fraction of transcript words that "
                 "anchored, and is the number to judge this by."),
    }
    (args.out / f"align_{args.panel.lower()}.json").write_text(json.dumps(doc))
    print(f"PANEL {args.panel}: narrated = {chosen_name}; {len(aw)} recognised, "
          f"{len(sw)} in transcript, {len(anchors)} anchors, "
          f"coverage {doc['coverage']*100:.1f}%, "
          f"voice spans {first_ms/1000:.1f}s..{last_ms/1000:.1f}s")


if __name__ == "__main__":
    main()
