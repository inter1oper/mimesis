# Stage 1a results

Both canvases rectified to 4800 × 3600 (40 × 30 cm at 120 px/cm). Canvas corners
found automatically by scanning the dark-to-light transition along each side and
fitting a line per edge; the measured aspect came out 1.341 (A) and 1.326 (B)
against a true 1.333, which is the check that the corners are actually on the
canvas rather than merely plausible.

## Counts

| | Panel A (real classmates) | Panel B (AI source) |
|---|---|---|
| Raw BlazeFace boxes | 185 | 139 |
| Raw YuNet boxes | 24 | 18 |
| Merged detections | 58 | 34 |
| **Both detectors agree** | **12** | **8** |
| Consensus and above the strict floor | 11 | 7 |
| Mesh converged, among consensus | 10 of 12 | **8 of 8** |
| Detected flames | 26 | 13 |
| Flames with no face to attach to | 1 | 1 |

The merged counts of 58 and 34 are not face counts. BlazeFace at a 0.20
threshold asserts a face on almost anything; the noise floor run predicted this
and the paintings confirm it. The row that means something is the consensus row.

## The noise floor

What each detector scores on input containing no faces, at the same settings:

| Control | BlazeFace | YuNet |
|---|---|---|
| Synthetic abstract fields, 6 seeds | 0.68 | 0.56 |
| Face-free crops of Panel A's own paint, 2 regions | 0.56 | 0.33 |

The real-paint control is the fairer one, and it is much kinder: on actual
face-free oil, YuNet produced a single weak box at 0.33. The synthetic control
stays in the record as the conservative bar. "Above the strict floor" in the
table above means BlazeFace > 0.68 **and** YuNet > 0.56, the harsher of the two.

## The finding

**The AI-generated panel reads cleaner to the detectors than the real one.**

Every one of Panel B's eight consensus faces converged a 478-point landmark
mesh. Panel A managed ten of twelve. Panel B's top detections — B05 at
0.960/0.937, B02 at 0.938/0.931 — sit level with Panel A's best, and Panel B
carries the larger faces: three of its eight clear the `contour` tier at 4K/3.5 m,
while none of Panel A's twelve do.

So the answer to the question the brief put first — does the detector emit a
confident box for a face on Panel B — is yes, and not marginally. The room
nobody stood in produces face geometry that two independent detectors agree on
and that a landmark mesh resolves without complaint.

## Panel A's two failures are the more interesting half

Of Panel A's twelve consensus detections, two produced no mesh:

- **A08** (0.847 / 0.863) — a real painted face in the back row. Both detectors
  confident, geometry unresolvable. A face-shaped region with no face structure.
- **A17** (0.270 / 0.392) — **not a face at all. It is a candle flame.** Both
  detectors independently asserted a face on a lit candle, and agreed with each
  other about it.

A17 is worth dwelling on, because it damages a claim I made earlier. I said
cross-detector consensus was the finding hardest for a sceptic to dismiss. A17
shows two architecturally independent detectors, trained on different data,
hallucinating the same face on the same flame. Consensus narrows the error, it
does not eliminate it. What actually catches A17 is the noise floor: both scores
sit below it, which is exactly what the floor is for. The two mechanisms have to
work together, and the piece is more honest for showing where each one fails.

## Flames with no face

Both panels report exactly one unattached flame, and this only became visible
after restricting the test to consensus faces. Run against all 58 merged
detections, every candle on Panel A landed near some box and nothing was ever
unattached — the false positives were absorbing the orphans.

| | Blob | Position | Nearest consensus face | Distance |
|---|---|---|---|---|
| Panel A | AF155 | x 0.007, y 0.752 | A05 | 3.48 face-heights |
| Panel B | BF232 | x 0.994, y 0.893 | B18 | 2.65 face-heights |

Both sit at the extreme canvas edge — a hand entering frame whose owner is
cropped out. That is a real hand holding a candle with no face attached, but the
reason is composition, not hallucination, and the piece should not pretend
otherwise. Panel A's is the more orphaned of the two, which is the opposite of
the expected direction.

## Overlay tiers at 4K across 3.5 m

| Panel | Consensus faces | contour | landmarks | box only |
|---|---|---|---|---|
| A | 12 | 0 | 9 | 3 |
| B | 8 | 3 | 5 | 0 |

Largest face on either canvas is B02 at 61 × 64 projector px. The full
478-point mesh needs 88 px, so **at 3.5 m the full mesh does not exist on either
panel** — the meshes are measured and stored, but they cannot be drawn at full
density without filling. Reaching it would mean a span of about 2.4 m.

That is a real decision, not a detail: the meshes are the most striking thing in
the data, and at 3.5 m the overlay can only show their contours.

## Still open

- Truth counts per panel, from you. The detector's consensus counts are 12 and
  8; my own reading of the canvases is higher on both. `reconcile.json` needs
  your number and its basis.
- All four transcripts. Nothing in 1b or 1c can run without them.
