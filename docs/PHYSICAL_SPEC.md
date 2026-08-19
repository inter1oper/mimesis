# The rendering spec has to be in millimetres, not pixels

Every rendering constraint in the brief is written in CSS pixels — 1–1.5 px
strokes, 8–11 px numerals, 28–40 px serif. A pixel is not a unit of size until a
projector is chosen. Across a 3.5 m wall one pixel is 0.91 mm at 4K and 1.82 mm
at 1080p. The same stylesheet makes two different pieces.

So the spec is anchored on the wall and the pixel values are generated:

| Element | Anchor on the wall | Why |
|---|---|---|
| Overlay stroke | 0.65 – 1.0 mm | hairline; heavier fills and blooms on gloss |
| Overlay numerals | 5 – 7 mm cap | legible at arm's length, mush at 4 m — as intended |
| Reasoning trace | 6.5 – 8.5 mm cap | telemetry; faster and dimmer than the answer |
| Answer serif | 13 – 26 mm cap | the only thing meant to be read |
| Caret | 0.65 – 1.0 mm | thin, never a block |
| Blur reveal radius | 3.5 – 4.5 mm | ~6 px at the density the brief was written for |

`stage1/spec_px.py` converts these for any projector, so changing the projector
does not silently change the work.

## At 3.5 m, this decides the projector

**4K across 3.5 m** — 0.91 mm/px:

| Element | mm | → px |
|---|---|---|
| Overlay stroke | 0.65 – 1.0 | 0.71 – 1.10 |
| Overlay numerals | 5 – 7 | 7.8 – 11.0 |
| Reasoning trace | 6.5 – 8.5 | 10.2 – 13.3 |
| Answer serif | 13 – 26 | 20.4 – 40.8 |

Those land almost exactly on the brief's original pixel numbers — numerals
8–11, serif 28–40, strokes ~1. The brief was implicitly written for roughly this
density.

**1080p across 3.5 m** — 1.82 mm/px:

| Element | mm | → px |
|---|---|---|
| Overlay stroke | 0.65 – 1.0 | **0.36 – 0.55** |
| Overlay numerals | 5 – 7 | **3.9 – 5.5** |
| Reasoning trace | 6.5 – 8.5 | 5.1 – 6.7 |

The overlay stroke is sub-pixel. The thinnest line a projector can draw is one
pixel, which here is 1.82 mm — two to three times heavier than the spec, on
glossy oil, which is precisely the condition the spec exists to avoid. The
numerals are unrenderable at 4–5 px. This is not a compromise, it is a broken
layer.

**So: 4K, if the span is 3–4 m.** At 1080p the surveillance overlay would have
to be redesigned around much heavier marks, and on gloss that fights the paint.

## Column width, and why the wall should not be filled

At 4K/3.5 m the two 40 cm paintings occupy 22% of the raster. Letting the text
columns fill the remaining 78% gives 1481 px per column, which at a 65-character
measure forces the type to 45.6 px — a 29 mm cap, readable to 5.8 m, larger than
the anchor wants.

The fix is to stop filling. Black projected onto a dark wall is invisible, so
unused raster costs nothing:

| Answer cap | Type | 65-char column | Lit content | Black surround | Reads to |
|---|---|---|---|---|---|
| 18 mm | 28.2 px | 917 px | 71% | 564 px/side | 3.6 m |
| **22 mm** | **34.5 px** | **1121 px** | **81%** | **360 px/side** | **4.4 m** |
| 24 mm | 37.6 px | 1223 px | 87% | 258 px/side | 4.8 m |

The middle row is the one to build to. Type lands at 34.5 px — the centre of the
brief's own 28–40 px range — with a 22 mm cap that reads comfortably from 4.4 m,
and 360 px of black either side that nobody will ever see as absence.

## What this costs the overlay

At 4K/3.5 m the painting renders 439 px wide, putting foreground face boxes near
55 px. That is `contour` tier: the ~130-point eye, lip and face-oval subset, not
the full 478-point tessellation, which would need 88 px and a span closer to
2.2 m.

That is a real loss, and worth taking knowingly. Reaching full mesh means
shrinking the projected image to ~2.2 m, which drops the answer column to a
72-character line at 13.6 mm caps — readable only from 2.7 m. On a 3–4 m wall
the contour mesh is the better trade: the front rank still carries visible face
geometry, and the text stays readable from across the room.

`stage1/plan_overlay.py` assigns the tier per face from measured boxes once
detection has run, so this stops being an estimate.
