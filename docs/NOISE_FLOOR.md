# The detector's confidence number needs a null

Your plan says: when Stage 1 comes back, look first at the detector's confidence
on Panel B faces, and if it is high, that is the strongest single data point.

Before the paintings are even in, that plan has a problem worth knowing about.

## What the control run shows

`stage1/calibrate_noise_floor.py` runs the exact detection settings used on the
paintings over procedurally generated abstract fields in the paintings' tonal
range — dark ground, warm elliptical masses, small bright highlights. No faces,
by construction. Ten controls:

| Detector | Threshold | Highest score on face-free input | Controls that produced any detection |
|---|---|---|---|
| BlazeFace | 0.20 | **0.68** | 10 / 10 |
| YuNet | 0.30 | **0.57** | 9 / 10 |

Both detectors will assert a face, with what reads as ordinary confidence, on
input that contains nothing. BlazeFace reached 0.68 on blurred ellipses.

## What this does to the argument

A bare number — "the detector was 0.71 confident about a face on Panel B" — is
not the strong data point it looks like, because a wall label cannot tell the
viewer whether 0.71 is high. Against this control it is barely above noise.

The strong data point is the *comparison*, and there are three good ones, all
of which the pipeline produces:

1. **Confidence against a measured null.** Panel B's confidences beside the
   false-positive ceiling from face-free input. This is the honest version of
   the claim you wanted to make, and it survives a sceptical viewer.
2. **Cross-detector consensus.** Two independent architectures agreeing on a
   region is far harder to dismiss than one scoring high. `consensus` in
   `detection.schema.json` carries this per face; a `blazeface_only` face on
   Panel B says something quite different from a face both detectors found.
3. **Box confident, mesh absent.** A high-confidence box on which the 478-point
   landmark mesh fails to converge is a face-shaped region with no resolvable
   face geometry. That is the most legible finding of the three for a viewer
   standing in front of the work, and it needs no calibration to read.

## Caveat on this particular control

Blurred ellipses in a dark field are arguably an adversarial null — oval, soft,
roughly face-proportioned. It is a hard control, and the ceiling it produces is
probably pessimistic.

The control that actually matters is face-free regions of the real paintings:
same paint, same varnish, same camera, no faces. The script takes them:

    python stage1/calibrate_noise_floor.py \
        --rectified out/panel_a_rectified.png \
        --crop 0.02,0.60,0.18,0.30 --crop 0.80,0.05,0.18,0.20

Both controls get reported. Neither is thrown away for being inconvenient.
