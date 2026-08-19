# Presentation layer

    mimesis_standalone.html   one self-contained file — open by double-click
    index.html + data/        the same renderer, fetching JSON over http

Use the standalone file in the gallery. `index.html` needs a server, because
browsers refuse `fetch()` from `file://`:

    python -m http.server -d render 8080

Rebuild the standalone after regenerating any JSON:

    python tools/make_standalone.py

## Running it

Open fullscreen. It starts immediately on the wall clock and loops every 900 s.

| key | |
|---|---|
| `c` | config mode — four drag handles per zone, keystone onto the painting |
| `e` | export alignment JSON (config mode) |
| `i` | import alignment JSON (config mode) |
| `r` | reset zones to the physical default (config mode) |

Alignment persists to `localStorage` and survives a restart. Export it anyway
before the opening — a cleared browser profile takes the alignment with it.

The default zone layout is the physically accurate one from
`docs/PHYSICAL_SPEC.md`: 4K across 3.5 m, two 40 cm paintings at 439 px each
occupying 22% of the raster, text columns holding a 65-character measure at a
22 mm cap, and the rest black.

## The clock

One clock for both panels. State is resolved as a pure function of it — the
renderer asks what should be visible at `t`, it never accumulates elapsed time.
That is what keeps a fifteen-minute loop from drifting and makes a seek land
exactly, including mid-character in the reveal.

With no audio it reads `performance.now()`. Set `clock.audio_file` in the score
and it reads `audio.currentTime` instead; no cue timing changes.

## What is real and what is waiting

The overlay track is complete: 401 cues on Panel A, 328 on Panel B, 28 cue
types, every one naming its Stage 1 measurement through `cue.source`. Flash
rates come from `stability.json` through `cue.flash_from` — the renderer cannot
invent one, and a confirmed face never flashes.

The answer and reasoning tracks are **empty** pending transcripts. Running
`build_score.py --specimen` puts a short type specimen in the answer track so
the serif, measure, leading and blur reveal can be judged; it is labelled
TYPE SPECIMEN — NOT MODEL OUTPUT on screen and is never attributed to a model.
Without that flag the track stays empty rather than holding placeholder prose.
