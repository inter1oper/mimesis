# Overlay vocabulary — every drawable element and the measurement behind it

The brief forbids filling the overlay with decorative HUD elements to match the
reference images more closely than the data supports. That rule holds. The way
to get a denser overlay is not to loosen it, it is to measure more — so Stage 1
now produces 29 distinct quantities, and each one earns a cue type.

## Per face

| Element | Draws | Source |
|---|---|---|
| `face_box` | bounding box | `detection.faces[].box` |
| `face_box_subdivision` | nested sub-boxes | box subdivided; only above `min_box_px` |
| `corner_brackets` | four corner ticks | box corners |
| `face_mesh` / `face_mesh_contour` / `face_landmarks_6` | mesh at the tier the box supports | `detection.faces[].landmarks`, tier from `overlay_plan` |
| `no_mesh_marker` | box marked as unresolved | `mesh_ok: false` |
| `face_numerals` | per-detector scores | `detection.faces[].detectors` |
| `consensus_flag` | one detector or two | `detection.faces[].consensus` |
| `noise_floor_rule` | horizontal rule at the false-positive ceiling | `noise_floor.floor` |
| **`blendshape_bar` / `blendshape_stack`** | **named percentage with a filled bar** | **`face_analysis` — 52 coefficients per face** |
| `head_pose_readout` | yaw / pitch / roll in degrees | `face_analysis.head_pose` |
| `symmetry_readout` | bilateral residual | `face_analysis.symmetry` |
| `photometry_readout` | luminance mean / σ / range | `face_analysis.photometry` |
| `interocular_rule` | measured rule between the eye corners | `face_analysis.geometry.interocular_px` |

The 52 blendshapes are the percentage readouts the reference look is built on —
`eyeSquintLeft 68.2%`, `browInnerUp 73.7%`, `mouthPressRight 41.3%`. Real named
quantities, measured per face, and there are enough of them to fill a column
beside every head without a single invented number.

## Across panels

| Element | Draws | Source |
|---|---|---|
| `cross_panel_link` | a line from a Panel B face to its nearest Panel A face | `face_analysis.cross_panel` |
| `cross_panel_readout` | cosine, and whether it clears the same-identity threshold | same |

## Field and blobs

| Element | Draws | Source |
|---|---|---|
| `response_field_cell` | one cell of the 40 × 30 response grid | `field.response_field.values` |
| `response_field_sweep` | a scan line reading out the field beneath it | same |
| `near_miss_marker` | a high-response cell that never became a box | field vs `detection.faces` |
| `blob_circle` | circle at the blob's equivalent radius | `field.blobs[].equiv_radius_frac` |
| `blob_ellipse` | ellipse from the blob's moments | `eccentricity`, `orientation_deg` |
| `blob_readout` | area as a percentage of canvas | `area_pct_of_canvas` |
| `flame_marker` / `unattached_flame_marker` | candle, and candle with no face | `flames.blobs` |

The response field is one cell per square centimetre of canvas: 1200 cells, each
holding the highest score the detector produces for a window centred there. Both
panels responded in **every single cell**, median 0.57. That is the material for
sweeping lines — a line crossing the canvas reads out real values, and the field
saturating is itself the finding.

Blob counts: Panel A has 424 blobs (182 lit skin, 28 flame-bright, 214 dark
mass), Panel B has 205 (99 / 21 / 85). Circles and percentages, all measured.

## Driven by the reasoning, in real time

`reasoning` cues carry `mentions`: for each face or blob the trace names, the
character offsets where it is named, and the phrase that grounds the reference.
Overlay cues carry an optional `trigger` binding them to a reasoning cue.

So when the trace reaches the words that describe a face, that face's box fires
— not on a timer running alongside the prose, but on the character span itself.
`on: "span_reveal"` fires at the exact character; `flash` gives a literal on/off
pattern in milliseconds, no easing and no glow.

Two things make this safe over a fifteen-minute loop. The binding is resolved in
Stage 1, not by string matching at render time. And `t_in`/`t_out` remain on
every cue as the outer window, so a bound cue still has a defined state when the
clock is scrubbed or no audio is present.

This is the part that needs the transcripts. The cue types, the schema and the
measurements are all in place; `mentions` cannot be populated until there is a
reasoning trace to parse.

## What is still not allowed

No element that does not appear in the table above. No scanlines, no glow, no
easing, no motion that is not in the score. If the overlay wants something new,
Stage 1 measures it first.
