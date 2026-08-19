# Projection geometry — recomputed for 40 × 30 cm canvases

The earlier font note assumed a 2.5–3 m span and asked whether one projector
was enough. With the canvas size now known (40 × 30 cm, landscape, both
panels), the answer sharpens, and one of the constraints turns out to bind on
the *overlay* layer rather than the text layer.

## The table

| Config | mm per px | Painting width in px | 34 px serif cap height | Comfortable reading distance |
|---|---|---|---|---|
| 1 × 1080p across 2.5 m | 1.30 | 307 | 31.0 mm | 6.2 m |
| 1 × 1080p across 2.0 m | 1.04 | 384 | 24.8 mm | 5.0 m |
| **2 × 1080p across 2.5 m** | **0.65** | **614** | **15.5 mm** | **3.1 m** |
| 2 × 1080p across 3.0 m | 0.78 | 512 | 18.6 mm | 3.7 m |
| 1 × 4K across 2.5 m | 0.65 | 614 | 15.5 mm | 3.1 m |
| 1 × 4K across 3.0 m | 0.78 | 512 | 18.6 mm | 3.7 m |

Comfortable reading distance uses cap height ≈ D/200, the usual signage rule.

## Why one 1080p fails, and it is not the text

At 1.30 mm/px the 34 px serif is *more* legible, not less — 31 mm caps read
comfortably out to 6 m. The text layer survives a single projector fine.

What dies is the overlay. A 40 cm painting becomes 307 projector pixels wide.
Faces in these two paintings run roughly 5–13% of canvas width, so face boxes
land between 15 and 38 px. A 478-point landmark mesh inside a 15 px box is not
line work, it is a filled blob — which violates the no-fill rule that exists
because fills bloom on gloss. The surveillance layer would be reduced to plain
rectangles with numerals larger than the boxes they annotate.

So: **two projectors, extended desktop, one browser window spanning both.** The
reason is the mesh, not the serif.

## The seam

Two side-by-side projectors put a blend seam at desktop x = 1920, dead centre
of the wall. Lay the zones out so the seam falls in a gap, not across a
painting or a text column. With paintings inboard and text outboard the seam
sits between the two paintings, which is the one place a 2–3 mm registration
error is invisible.

A single 4K projector gives identical horizontal density with no seam and
double the vertical room (2160 px ≈ 1.4 m tall at 0.65 mm/px, versus 702 mm for
a 3840 × 1080 desktop). At 34 px / 1.65 leading that is 38 lines of answer text
instead of 19. If the budget reaches a gallery-bright 4K, take it. If not, two
1080p is genuinely fine.

## Overlay detail ladder

At 0.65 mm/px, with the painting 614 px wide:

| Face size on canvas | Box in projector px | Point spacing in a 478-pt mesh |
|---|---|---|
| ~12.5% of width (foreground) | 77 × 100 | 4.5 px |
| ~8% (mid-ground) | 49 × 64 | 2.9 px |
| ~5% (background) | 31 × 40 | 1.8 px |

A mesh needs roughly 4 px between adjacent points to read as strokes rather
than fill. That gives three tiers:

| Tier | Minimum box | Draw |
|---|---|---|
| ≥ 88 px | face ≥ 5.7 cm on canvas | full 478-point mesh |
| ≥ 48 px | face ≥ 3.1 cm | contour subset (~130 pts: eye, lip, face oval) |
| ≥ 24 px | face ≥ 1.6 cm | 6 key landmarks only |
| < 24 px | — | box and numerals only |

This is why `score.schema.json` carries `min_box_px` on every overlay cue: the
score decides what detail a face gets, from its measured box size, and the
renderer only obeys. Dropping detail on a small face is the honest move —
drawing a mesh that resolves to a blob would be adding decoration the data
cannot support, which is exactly what the brief rules out.

Face-size percentages above are eyeballed from the supplied photographs and are
placeholders until Stage 1a runs. The pipeline emits exact box sizes; the tier
assignment is then computed, not estimated.

## Font choice is less coupled to the score than feared

The worry was that substituting the serif late would change every line length
in the score. It will change line *breaking*, but not timing: `score.schema.json`
stores each answer cue as verbatim text plus a `char_onsets_ms` array — a
per-character onset in milliseconds. Character timing is font-independent.
Swapping Tiempos Text for Source Serif 4 rewraps the column and changes how
many lines a block occupies; it does not move a single onset.

So the font decision does not have to precede Stage 2. What has to precede
Stage 2 is the *column width*, because that is what the reveal is laid out
into. Fix the column geometry, and the face can change afterwards.
