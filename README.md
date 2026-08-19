# mimesis

Two-channel projection installation over two oil paintings, 40 × 30 cm each.

Panel A was painted from a photograph of fourteen real classmates at a
candlelight vigil. Panel B was painted from an AI-generated image made from a
written description of that same photograph.

Panel A is the **more crowded** canvas; Panel B, from the AI source, holds
**fewer and more individuated** figures. That is worth stating explicitly
because it inverts the intuition — the crowded, mushier canvas is the one with
real people behind it. An early reading of these photographs guessed the
opposite, from exactly that intuition, and was wrong.

Whether anyone was ever in Panel B's room is a question the models are asked to
sit with, not a fact this pipeline resolves.

Multimodal models were each shown one panel, cold. The installation projects a
surveillance overlay and the models' reasoning traces onto the oil, and their
polished answers onto the bare wall beside it.

## Rule

Everything projected derives from measured data. No placeholder numbers, no
invented face coordinates, no fabricated confidences. Where a value is not
available the field is `null` and the gap is reported. Where the detector and
the models disagree, the disagreement is preserved — it is the material.

## Stage 1 — Python analysis

```
stage1/calibrate_noise_floor.py   -> what the detectors score on input with no faces
stage1/rectify.py        photo of canvas  ->  rectified painting raster + homography
stage1/detect_faces.py   rectified raster ->  faces, boxes, confidences, 478-pt meshes
stage1/detect_flames.py  rectified raster ->  candle flames, and flames with no face
stage1/plan_overlay.py   detection + projector -> per-face overlay detail tier
stage1/spec_px.py        projector geometry    -> the rendering spec in pixels
stage1/parse_transcripts.py   (not written — waiting on transcript files)
stage1/build_score.py         (not written — waiting on transcripts)
stage1/reconcile.py           (not written — waiting on transcripts)
```

### Why rectification comes first

The brief requires coordinates normalised against the painting's own rectangle,
never the source photo's pixels. The supplied photographs are hand-held shots of
an unframed canvas on a floor, keystoned by the camera. Normalising against the
photo would bake that keystone into every box and landmark, and the projector's
`matrix3d` warp would then be correcting a distortion that had already been half
applied. So the canvas quad is found once, warped to a canonical 40 × 30 cm
raster at a known px/cm, and everything downstream lives in painting space.

Corner provenance is recorded (`auto_quad` or `manual`) and a QA image is
written. Automatic quad detection against a concrete floor is not reliable
enough to trust silently — check the QA image, override with `--corners`.

### Detection approach

Two architecturally independent detectors, reported separately and never
averaged into a single invented confidence:

- **MediaPipe BlazeFace** (short range), run over an overlapping tile pyramid at
  four scales. BlazeFace resizes its input to 128 × 128 internally, so a face
  occupying 5% of a 4800 px canvas is effectively sub-pixel at full frame.
  Tiling is what makes small faces in a group scene reachable at all.
- **OpenCV YuNet**, at 1× and 2×. Anchor-free, trained on WIDER FACE, different
  architecture and different training data — so agreement between the two means
  something, and disagreement is a reportable state rather than noise.

Thresholds are deliberately permissive (BlazeFace 0.20, YuNet 0.30). Filtering
hard at source would discard exactly the weak assertions that are most
interesting on Panel B.

Landmarks come from MediaPipe FaceLandmarker (478 points) run per merged
detection on a padded crop. A box the detector is confident about on which the
mesh fails to converge is recorded as `mesh_ok: false` with a reason. That is a
face-shaped region with no resolvable face geometry — content, not a bug.

### Flames with no face

The brief asks whether a hand holds a candle with no face to attach it to. That
is answerable from pixels, so it is measured rather than asserted. Bright warm
high-local-contrast blobs are found, classified by a stated rule (ceiling
fixtures are labelled, not silently dropped), and each flame's distance to the
nearest face box is reported in units of that face's box height — scale-free, so
it is comparable across two differently composed panels. `unattached` is a flag
against a stated threshold, and the raw distance is always present so the
threshold can be moved.

## Schemas

Proposed, for review before the renderer is written:

| File | Purpose |
|---|---|
| `schemas/detection.schema.json` | faces, boxes, per-detector confidences, meshes |
| `schemas/flames.schema.json` | flames, fixtures, nearest-face distances |
| `schemas/claims.schema.json` | one model-panel session, parsed into claims |
| `schemas/reconcile.schema.json` | the three-way count table per panel |
| `schemas/score.schema.json` | three timed tracks on one clock |

Two design decisions in `score.schema.json` are worth arguing with before it is
built on:

1. **Per-character onsets are precomputed.** Each answer cue carries
   `char_onsets_ms`, one entry per character. The renderer resolves state as a
   pure function of the current time — it looks up what should be visible at
   `t`, it never accumulates elapsed time. That is what makes a fifteen-minute
   loop non-drifting and a seek land exactly. It also means sentence-end pauses
   are retimed in JSON without touching code, and that a font substitution
   changes line breaking but not one onset.
2. **Every overlay cue must name its source.** `cue.source` is a file plus a
   JSON Pointer into it. A cue with no source is invalid. That is the mechanism
   that keeps decorative HUD elements out — a new cue type requires a new Stage
   1 measurement to back it.

## Projection geometry

One large projector, single raster, four keystoned zones. See
`docs/PROJECTION.md` for what the pixel budget buys — the resolution decides
whether the landmark mesh exists as a layer at all, and `stage1/plan_overlay.py`
computes each face's detail tier from its measured box rather than by eye.
`docs/PHYSICAL_SPEC.md` anchors every rendering constraint in millimetres on the
wall and generates the pixel values, so changing the projector cannot silently
change the work.

## Setup

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# needs system EGL/GLES for mediapipe:
apt-get install -y libegl1 libgles2 libgl1
```

Model files are committed under `models/` so the pipeline runs offline in a
gallery.

## Calibration

`stage1/calibrate_noise_floor.py` establishes what each detector scores on
input containing no faces, at the same thresholds used on the paintings. On the
first control run BlazeFace reached 0.68 and YuNet 0.57 on face-free abstract
fields. Any confidence quoted from Panel B has to be read against that. See
`docs/NOISE_FLOOR.md`.
