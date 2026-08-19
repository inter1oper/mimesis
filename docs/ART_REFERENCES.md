# Research: the style, and what makes flashing honest

## 1. The theoretical frame — Farocki's operational images

The term for what this installation projects already exists. Harun Farocki
coined **operational image** in 2000, in the first part of the *Eye/Machine*
trilogy: an image made not to be looked at but to be used — an instrument in
performing a task, usually by a machine, usually with no human viewer intended.
The trilogy (2001–2003) worked through electronic surveillance, mapping and
object recognition, and the machine vision of the 1990–91 Gulf War.

This matters for the piece beyond citation. Farocki's move was to take images
that were never meant to be seen and put them on a gallery wall, where their
indifference to a viewer becomes the subject. That is exactly what the
surveillance overlay does here: a face box and a confidence numeral are
operational images, produced for a threshold comparison and nothing else.
Projecting them onto oil paint — the most emphatically *not* operational image
there is — is the whole argument.

It also suggests restraint. Farocki's power comes from leaving operational
images looking operational. Prettying them up would have destroyed the work.

## 2. The artists already working this seam

- **Trevor Paglen** — *It Began as a Military Experiment* (2017) takes ten
  photographs from the mid-1990s DoD database used to develop face recognition;
  *They Took the Faces from the Accused and the Dead* uses mugshots gathered
  without consent. *Behold These Glorious Times!* intercuts training footage
  with the grids the machine actually processes.
- **Adam Harvey** — *CV Dazzle* (camouflage against face detection),
  *MegaPixels* / *Exposing.ai* (auditing the datasets themselves).
- **Zach Blas** — *Facial Weaponization Suite*, masks generated from aggregated
  facial data that no detector can resolve.

What distinguishes this project from all three: they work with photographs and
datasets. This one puts a detector on a *painting*, where every detection is
already a category error. That is the gap worth occupying, and it is why the
A17 result — both detectors calling a candle flame a face — is not a bug but the
thesis.

## 3. Thermal display conventions worth borrowing

From FLIR's own documentation:

- **White-hot / black-hot** are what laboratory and military users actually run.
  The saturated palettes (Ironbow, Rainbow HC, Lava) are for presentation. Your
  cyan-on-dark instinct is right; keep the field monochrome.
- **Isotherm** — a bright colour highlighting only a *band* of the measured
  range, against an otherwise neutral image. This is directly reusable: apply an
  isotherm to confidence, so only faces inside a confidence band light up, and
  the band sweeps. It is a real instrument convention driven by a real range.
- **MSX** — FLIR overlays edge detail from the visible image onto the thermal
  layer, producing a hybrid where you can tell which window the cold spot is in.
  The inverse is available here: overlay machine-derived structure onto the
  visible oil.

## 4. Why real tracking displays flash — and what that licenses

This is the part that decides whether flashing is data or decoration.

A multi-object tracker does not hold a steady box. Following the DeepSORT
scheme, tracks live in a state machine: a new track is **tentative** and is not
reported downstream until it survives `n_init` consecutive detections; it is
promoted to **confirmed**; it is **deleted** after `max_age` frames without a
match. Tentative tracks appear and vanish. Confirmed tracks hold. **The flicker
in a real tracker display is track birth and death, not a visual effect.**

Paintings have no next frame. But the equivalent question has a real answer: how
stable is each detection under small changes that should not matter? So
`stage1/stability.py` re-runs both detectors over 29 perturbation passes —
3 scales × 3 rotations × 3 exposures, plus JPEG recompression at two qualities —
and assigns each detection a track state from the fraction of passes it survives.

### What came back

| | Detections | Confirmed | Tentative | Lost | Flashing |
|---|---|---|---|---|---|
| Panel A | 58 | 27 | 15 | 16 | **31 (53%)** |
| Panel B | 34 | 21 | 6 | 7 | **13 (38%)** |

Flash rates run 1.24–6.00 Hz, derived from dropout and nothing else.

And the finding that makes the whole layer work:

**Not one consensus face flashes. On either panel. Zero.**

Every face both detectors agreed on survived all 29 perturbations intact. The
flashing belongs entirely to the machine's *rejected* hypotheses — the
single-detector boxes near the noise floor. The committed claims hold dead
still; the discarded ones flicker at up to 6 Hz.

That is a far better image than decorative blinking. Thirty-one ghost boxes
stuttering across Panel A while twelve confirmed faces sit motionless: the
overlay shows the machine's discarded guesses, which is the part of machine
perception that never reaches a user. Farocki's operational image, in the state
before it becomes operational.

Panel A generates proportionally more instability than Panel B — 53% against
38% — which is consistent with the gaze finding and inconsistent with the
per-face findings. The real painting is harder for the machine in the aggregate
and easier face by face.

## 5. Rules this research imposes

1. Flash rate comes from `stability.json`. The renderer never invents one.
2. Confirmed faces do not flash, ever. Stillness is the signal.
3. Keep the palette monochrome cyan. Isotherm is the only permitted second
   emphasis, and it must be driven by a measured band.
4. No easing on a flash. Real track death is a hard cut.
5. Leave the operational images looking operational.

## Sources

- [Operational image — Wikipedia](https://en.wikipedia.org/wiki/Operational_image)
- [Harun Farocki: Eye/Machine (artist's site)](https://www.harunfarocki.de/installations/2000s/2000/eye-machine.html)
- [Eye/Machine III — MACBA](https://www.macba.cat/en/obra/r2471-eye--machine-iii/)
- [Trevor Paglen, It Began as a Military Experiment — MoMA](https://www.moma.org/collection/works/275173)
- [Trevor Paglen, They Took the Faces… — FAMSF](https://www.famsf.org/stories/trevor-paglen-they-took-the-faces-from-the-accused-and-the-dead-sd18)
- [Adam Harvey, On Computer Vision](https://adam.harvey.studio/on-computer-vision/)
- [Surveillance, Bias and Control in the Age of Facial Recognition Software — Frieze](https://www.frieze.com/article/surveillance-bias-and-control-age-facial-recognition-software)
- [Picking a Thermal Color Palette — FLIR](https://www.flir.com/discover/industrial/picking-a-thermal-color-palette/)
- [Forward-looking infrared — Wikipedia](https://en.wikipedia.org/wiki/Forward-looking_infrared)
- [deep_sort — DeepWiki](https://deepwiki.com/nwojke/deep_sort)
- [Deep Dive into Fundamentals of DeepSORT](https://medium.com/@shasvatdesai/deep-dive-into-fundamentals-of-deepsort-for-object-tracking-f92ec7abfa7e)
