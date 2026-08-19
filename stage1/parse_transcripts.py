"""
Stage 1b — parse each model-panel session into claims.

Splits the answer into sentence-level claims, labels each seen or supplied,
flags what the image cannot support, counts hedges per hundred words, extracts
the stated headcount, and anchors claims that name a specific position in the
frame to a detected face id from 1a.

Labelling
---------
Where the model labelled its own claim -- these transcripts carry explicit
"Observed" and "Inferred" headings -- that label is used and `labelled_by` is
"model". Only claims outside a labelled section are classified by our reading,
and those are marked "reader". The distinction is kept so the piece never
presents our judgement as the model's.

Unsupportable claims are a narrower category than "supplied": a claim is
unsupportable when the image cannot bear on it at all -- identity, nationality,
date, gender attributed without visible basis, or the medium of the source.

Short labels
------------
Each claim also yields a KEYWORD: one or two words lifted verbatim from the
claim. These are what the overlay prints beside a box. They are extracted, not
invented, so the words on the paint are the model's own.

Usage
-----
    python stage1/parse_transcripts.py --out out/
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re

HEDGES = [
    "appears", "appear", "appearing", "seems", "seem", "roughly", "approximately",
    "about", "perhaps", "possibly", "possible", "might", "may", "could",
    "unclear", "uncertain", "ambiguous", "likely", "probably", "suggests",
    "suggesting", "suggest", "implies", "imply", "implying", "somewhat",
    "slightly", "partly", "partially", "presumably", "apparently", "seemingly",
    "not clearly", "appears to be", "hard to", "difficult to",
]
# Claims the image cannot bear on at all, distinct from ordinary inference.
UNSUPPORTABLE = [
    (r"\bEast Asian\b", "attributes ethnicity, which the image cannot establish"),
    (r"\byoung men\b", "attributes age and gender as a group fact"),
    (r"\bwoman\b", "attributes gender to a specific figure"),
    (r"\bphotograph\b", "calls the image a photograph; the source is a painting"),
    (r"\bSouth Korea\b|\bHong Kong\b|\b1980s\b", "attributes place or period"),
    (r"\bclassmates\b|\breunion\b|\bwedding\b", "attributes an occasion"),
]
SEEN_HEAD = re.compile(r"observed|visible image|spatial map|inventory|physical", re.I)
SUPP_HEAD = re.compile(r"inferred|emotion|meaning|narrative|imagination|"
                       r"self-exam|examining|reconstruction", re.I)
# Position phrases that can anchor a claim to a face detected in 1a.
POSITIONS = [
    (r"lower[- ]left", (0.0, 0.55, 0.45, 1.0)),
    (r"image-left edge|extreme image-left|far image-left|far left", (0.0, 0.0, 0.22, 1.0)),
    (r"image-left|left foreground|on the left", (0.0, 0.0, 0.4, 1.0)),
    (r"lower[- ](right|center right)|lower-center right|bottom-right", (0.5, 0.55, 1.0, 1.0)),
    (r"far right|image-right edge|far image-right", (0.78, 0.0, 1.0, 1.0)),
    (r"image-right|on the right|right foreground", (0.6, 0.0, 1.0, 1.0)),
    (r"center-left|centre-left", (0.28, 0.0, 0.55, 1.0)),
    (r"center-right|centre-right|middle-right", (0.5, 0.0, 0.78, 1.0)),
    (r"\bcenter\b|\bcentre\b|central", (0.35, 0.0, 0.68, 1.0)),
    (r"upper right|upper-right", (0.6, 0.0, 1.0, 0.45)),
]
NUMWORDS = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,
            "nine":9,"ten":10,"eleven":11,"twelve":12,"dozen":12,"thirteen":13,
            "fourteen":14,"fifteen":15,"sixteen":16}
# Words worth printing on the paint: concrete, visual, or judgemental.
KEYWORD_POOL = re.compile(
    r"\b(chiaroscuro|claustrophobic|solemn|somber|sombre|ambiguous|ambiguity|"
    r"intimacy|intimate|communal|crowded|cropped|shadowed|highlight|flame|"
    r"candle|saucer|brushstrokes|glasses|smile|smiling|teeth|neutral|tension|"
    r"vulnerability|solidarity|grief|hope|nostalgia|mournful|celebratory|"
    r"vigil|memorial|protest|ceremony|reunion|huddle|brotherhood|belonging|"
    r"warmth|darkness|unclear|uncertain|obscured|hidden|blurred|overlap|"
    r"speculative|interpretation|projection|silence|withholds)\b", re.I)


def sentences(text: str):
    """Sentence spans over the raw text, keeping exact offsets."""
    out, i = [], 0
    for m in re.finditer(r"[^\n]*?[.!?](?=\s|$)|\n{2,}|[^\n]+(?=\n|$)", text):
        s, e = m.start(), m.end()
        frag = text[s:e].strip()
        if len(frag) < 25:
            continue
        lead = len(text[s:e]) - len(text[s:e].lstrip())
        out.append((s + lead, s + lead + len(frag), frag))
        i = e
    return out


def section_at(text: str, pos: int) -> str:
    heads = [(m.start(), m.group(0)) for m in re.finditer(r"^#{1,4} .*$|^\d+\. .*$",
                                                          text, re.M)]
    cur = ""
    for p, h in heads:
        if p <= pos:
            cur = h.strip("# ").strip()
        else:
            break
    return cur


def keyword_for(frag: str) -> str | None:
    ms = KEYWORD_POOL.findall(frag)
    if ms:
        return ms[0].lower()
    m = re.search(r"\b(no face|not visible|cannot|does not|withheld)\b", frag, re.I)
    return m.group(1).lower() if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcripts", type=pathlib.Path,
                    default=pathlib.Path("data/transcripts"))
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("out"))
    args = ap.parse_args()

    mapping = {"session1_forensic": "B", "session2_observed": "A",
               "session3_narrative": "B"}
    det = {p: json.loads((args.out / f"panel_{p.lower()}_detection.json").read_text())
           for p in ("A", "B")}

    docs = []
    for d in sorted(args.transcripts.iterdir()):
        if not d.is_dir():
            continue
        ans_f = d / "answer.txt"
        if not ans_f.exists():
            continue
        panel = mapping.get(d.name)
        text = ans_f.read_text()
        rsn_f = d / "reasoning.txt"
        rsn = rsn_f.read_text() if rsn_f.exists() else ""
        rsn_supplied = bool(rsn.strip()) and "[NOT SUPPLIED]" not in rsn

        faces = [f for f in det[panel]["faces"] if len(f["detectors"]) >= 2] if panel else []

        claims = []
        for n, (s, e, frag) in enumerate(sentences(text)):
            sec = section_at(text, s)
            if SEEN_HEAD.search(sec):
                ev, by = "seen", "model"
            elif SUPP_HEAD.search(sec):
                ev, by = "supplied", "model"
            else:
                ev, by = ("supplied" if re.search(
                    r"\b(evokes|suggests|invites|might|we may|imagine|feels?|"
                    r"atmosphere|meaning)\b", frag, re.I) else "seen"), "reader"

            unsup, why = False, None
            for pat, reason in UNSUPPORTABLE:
                if re.search(pat, frag):
                    unsup, why = True, reason
                    break

            anchor = None
            for pat, (x0, y0, x1, y1) in POSITIONS:
                if re.search(pat, frag, re.I):
                    cand = [f for f in faces
                            if x0 <= f["box"][0] + f["box"][2] / 2 <= x1
                            and y0 <= f["box"][1] + f["box"][3] / 2 <= y1]
                    if cand:
                        cand.sort(key=lambda f: -f["box"][2] * f["box"][3])
                        anchor = {"face_id": cand[0]["face_id"],
                                  "basis": re.search(pat, frag, re.I).group(0),
                                  "confidence": "inferred"}
                    break

            claims.append({
                "claim_id": f"C{n + 1:03d}", "text": frag, "char_span": [s, e],
                "evidence": ev, "labelled_by": by,
                "unsupportable": unsup, "unsupportable_reason": why,
                "anchor": anchor, "section": sec,
                "keyword": keyword_for(frag),
            })

        words = len(re.findall(r"\b\w+\b", text))
        terms = {}
        for h in HEDGES:
            c = len(re.findall(r"\b" + re.escape(h) + r"\b", text, re.I))
            if c:
                terms[h] = c
        hcount = sum(terms.values())

        stated, quote, hedged = None, None, False
        for m in re.finditer(r"([\w\-]+(?:\s+to\s+[\w\-]+)?)\s+(?:visible\s+)?"
                             r"(?:individuals|faces|people|figures|young men|men)",
                             text, re.I):
            raw = m.group(1).lower()
            nums = re.findall(r"\d+", raw)
            if nums:
                stated = int(nums[-1])
            elif raw.split()[-1] in NUMWORDS:
                stated = NUMWORDS[raw.split()[-1]]
            else:
                continue
            ls = text.rfind(".", 0, m.start()) + 1
            le = text.find(".", m.end())
            quote = text[ls:le + 1].strip()
            hedged = bool(re.search(r"roughly|about|approximately|around|or\s+\w+",
                                    text[max(0, m.start() - 40):m.end()], re.I))
            break

        docs.append({
            "schema": "mimesis.claims.v1",
            "session_id": d.name, "model": "UNSPECIFIED", "panel": panel,
            "panel_assignment": "inferred_from_content_not_supplied",
            "source": {"answer_file": str(ans_f),
                       "reasoning_file": str(rsn_f) if rsn_supplied else None,
                       "answer_word_count": words,
                       "reasoning_word_count": (len(re.findall(r"\b\w+\b", rsn))
                                                if rsn_supplied else None)},
            "answer_claims": claims,
            "hedges": {"terms": terms, "count": hcount,
                       "per_hundred_words": round(100.0 * hcount / max(1, words), 2),
                       "scope": "answer"},
            "headcount": {"stated": stated, "quote": quote, "hedged": hedged},
        })

    (args.out / "claims.json").write_text(json.dumps(docs, indent=2))

    print(f"{'session':22} {'panel':6} {'claims':>7} {'seen':>5} {'suppl':>6} "
          f"{'unsup':>6} {'anch':>5} {'hedge/100w':>11} {'headcount':>10}")
    for d in docs:
        c = d["answer_claims"]
        print(f"{d['session_id']:22} {d['panel'] or '?':6} {len(c):>7} "
              f"{sum(1 for x in c if x['evidence']=='seen'):>5} "
              f"{sum(1 for x in c if x['evidence']=='supplied'):>6} "
              f"{sum(1 for x in c if x['unsupportable']):>6} "
              f"{sum(1 for x in c if x['anchor']):>5} "
              f"{d['hedges']['per_hundred_words']:>11} "
              f"{str(d['headcount']['stated']):>10}")


if __name__ == "__main__":
    main()
