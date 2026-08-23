# Presentation layer

    mimesis_standalone.html   the show — open by double-click
    audio/                    the two narrations; MUST travel beside the html
    index.html + data/        the same renderer, fetching JSON over http

The html is self-contained apart from the audio. The two voices come to about
12 MB, which base64 inflates past what belongs inside one file, so they are
referenced. Keep `audio/` in the same folder as the html and it works from
`file://` with no server.

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

## The clocks

**Two channels, two clocks.** Each panel is driven by its own narration —
363 s for Panel A, 376 s for Panel B. The recordings are different lengths and
a panel's words have to match the voice describing that panel, so sharing one
clock would put one of them permanently out of step with its own audio.

Within a panel nothing can drift: the overlay cycle, the typed answer and the
reasoning trace all read the same `audio.currentTime`. Between panels they are
deliberately independent, which is what a two-channel installation is.

Three cycles run inside each panel's clock:

| | Panel A | Panel B |
|---|---|---|
| narration loop | 363 s | 376 s |
| answer stream | fitted to the voice | fitted to the voice |
| overlay cycle | 45 s, 8.1× per loop | 45 s, 8.4× per loop |

The typed answer is paced to the voice: the stream finishes as the narration
does. The shape of the cadence — the pauses at sentence and clause ends — is
preserved and only the tempo changes. Panel A's text is short for its recording
so it types at 3.07× the base cadence; Panel B's forensic transcript nearly
fills its own narration already, at 1.02×.

With no audio present the identical lookup runs against `performance.now()` and
no cue timing changes.

**Starting the voice.** Browsers refuse to start audio without a gesture. The
show waits on one keypress or click — the status line says so until it gets one
— and then both voices start together.

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
