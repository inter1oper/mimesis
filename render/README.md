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

**The typing is aligned to the voice, not merely fitted to it.**
`stage1/align_voice.py` transcribes each recording with word-level timestamps,
matches the recognised words against the transcript, and takes the character
onsets from the audio. A word appears as it is spoken. Coverage — the fraction
of transcript words that anchored to a recognised word — is **95%** on Panel A
and **97%** on Panel B. Regions the recogniser got wrong are interpolated at
the local speaking rate between the anchors either side, so they stay plausible
rather than stalling.

The aligner also decides WHICH transcript the voice is reading, from the
recording rather than from directory order. Panel B carries two sessions and
its voice reads only one:

| candidate | coverage |
|---|---|
| session3_narrative | **96.6%** |
| both sessions concatenated | 30.9% |
| session1_forensic | 2.8% |

So Panel B's spoken answer is `session3_narrative`. `session1_forensic` — the
figure-by-figure inventory — has no narration and is not the spoken text; it
still supplies claims, keywords and counts to the overlay.

With no audio present the identical lookup runs against `performance.now()` and
no cue timing changes.

**Left and right.** Panel A's voice is hard-panned left, Panel B's right,
matching where each painting hangs, so a listener standing in front of one
panel hears the voice describing it. Done with a Web Audio `StereoPannerNode`;
if Web Audio is unavailable both voices stay centre and the status line says
so. The audio elements remain the clock source either way.

**Starting the voice.** Browsers refuse to start audio without a gesture. The
show waits on one keypress or click — the status line says so until it gets one
— and then both voices start together and the audio context resumes.

## Re-running the alignment

    python stage1/align_voice.py --panel A --audio data/audio/panel_a_voice.mp3
    python stage1/align_voice.py --panel B --audio data/audio/panel_b_voice.mp3
    python stage1/build_score.py --out out --cycle 45 --audio-dir data/audio

The recogniser output is cached in `out/asr_*.json`, so re-running to try a
different session or a larger model does not re-transcribe.

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
