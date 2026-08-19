# Projection geometry — one projector, 40 × 30 cm canvases

Settled: **one large projector**, single raster, single coordinate space, span
around 3–4 m. At that span the projector has to be 4K — see
`docs/PHYSICAL_SPEC.md`, where the 1080p overlay stroke works out sub-pixel. Four
zones (Panel A overlay, Panel A text, Panel B overlay, Panel B text) are laid
out inside that one raster, each still carrying its own `matrix3d` keystone,
because the paintings are physically hung and will not land on an ideal grid.

The only question left is what the pixel budget buys, because the two paintings
and the two text columns share it.

## The budget

Two 40 cm paintings plus two text columns across a span `S`. A 34 px serif has
roughly a 17 px average advance, so column characters-per-line follows directly.

| Config | mm/px | Painting (px wide) | Text column (px / chars) | Foreground face box | 34px cap | Lines |
|---|---|---|---|---|---|---|
| 1080p @ 2.5 m | 1.30 | 307 | 653 / 38 | 38 px | 31.0 mm | 19 |
| 1080p @ 2.2 m | 1.15 | 349 | 611 / 36 | 44 px | 27.3 mm | 19 |
| 1080p @ 1.8 m | 0.94 | 427 | 533 / 31 | 53 px | 22.3 mm | 19 |
| **4K @ 2.2 m** | **0.57** | **698** | **1222 / 72** | **87 px** | 13.6 mm | 39 |
| 4K @ 2.5 m | 0.65 | 614 | 1306 / 77 | 77 px | 15.5 mm | 39 |
| 4K @ 3.0 m | 0.78 | 512 | 1408 / 83 | 64 px | 18.6 mm | 39 |

Foreground face box assumes a face at ~12.5% of painting width, eyeballed from
the photographs. Stage 1a replaces that estimate with measured boxes.

## What the resolution decides

The overlay's landmark mesh needs roughly 4 projector px between adjacent points
to read as strokes. Below that it fills, and fills bloom on gloss. That sets
three thresholds on the rendered face box:

| Tier | Minimum box | Draws |
|---|---|---|
| `full_mesh` | 88 px | 478-point tessellation |
| `contour` | 48 px | ~130 points: eye, lip, face oval |
| `landmarks` | 24 px | 6 key points |
| `box_only` | — | box, corner brackets, numerals |

So:

- **4K at 2.2–2.5 m** puts foreground faces at 77–87 px — full mesh or close to
  it on the front rank, contour on the middle, landmarks on the back. The whole
  ladder is in play, and the text column is a comfortable 72–77 characters.
- **1080p at any span** caps foreground faces around 38–53 px. Full mesh never
  happens; the front rank gets contour at best, everything behind it gets
  landmarks or box-only. The text column narrows to 31–38 characters, which is
  a tight measure but reads fine for a slow character reveal.

Either works. The difference is whether the mesh exists as a layer at all. If
it is 1080p, design the surveillance layer around boxes, brackets and numerals
from the start and treat any mesh as a bonus on one or two faces — deliberately,
rather than discovering it on the wall the night before.

## The tier is computed, not chosen

`stage1/plan_overlay.py` takes the detection output plus the projector geometry
and emits, per face, the rendered box size in projector px and millimetres and
the highest tier it supports. Those numbers become `min_box_px` on the overlay
cues in the score.

    python stage1/plan_overlay.py --detection out/panel_a_detection.json \
        --projector 3840x2160 --span-mm 2200

A face whose mesh never converged is forced to `box_only` regardless of size —
there is nothing to draw. That is the same rule as everywhere else in this
project: absent data stays absent.

## Font choice does not gate Stage 2

The score stores each answer cue as verbatim text plus `char_onsets_ms`, one
onset per character. Character timing is font-independent, so swapping Tiempos
Text for Source Serif 4 rewraps the column without moving a single onset.

What does need fixing before Stage 2 is the **column width in pixels**, since
that is what the reveal lays out into — and from the table above, that follows
from the projector and the span. Decide those two; the typeface can change
afterwards.
