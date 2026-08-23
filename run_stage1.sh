#!/usr/bin/env bash
# Stage 1a end to end, both panels. Run after the painting photographs are in
# data/paintings/. Everything it writes lands in out/.
#
#   ./run_stage1.sh                       # 4K across 3.5 m, the current plan
#   PROJECTOR=1920x1080 SPAN_MM=3000 ./run_stage1.sh
#
# Check out/panel_*_rectify_qa.jpg before trusting anything downstream. If the
# drawn quad is not sitting on the canvas edges, re-run rectify.py for that
# panel with --corners and then re-run this script.
set -euo pipefail

PY=${PY:-.venv/bin/python}
OUT=${OUT:-out}
PROJECTOR=${PROJECTOR:-3840x2160}
SPAN_MM=${SPAN_MM:-3500}
PX_PER_CM=${PX_PER_CM:-120}

mkdir -p "$OUT"

echo "== detector noise floor (what these settings score on input with no faces)"
"$PY" stage1/calibrate_noise_floor.py --trials 10 --out "$OUT"

for panel in A B; do
  lower=$(echo "$panel" | tr 'A-Z' 'a-z')
  photo=$(ls data/paintings/panel_${lower}_photo.* 2>/dev/null | head -1 || true)
  if [ -z "$photo" ]; then
    echo "!! no photo for panel $panel at data/paintings/panel_${lower}_photo.*" >&2
    continue
  fi

  echo "== panel $panel: rectify  ($photo)"
  "$PY" stage1/rectify.py --photo "$photo" --panel "$panel" \
      --width-cm 40 --height-cm 30 --px-per-cm "$PX_PER_CM" --out "$OUT" >/dev/null

  echo "== panel $panel: faces"
  "$PY" stage1/detect_faces.py --panel "$panel" \
      --rectified "$OUT/panel_${lower}_rectified.png" \
      --rectify-meta "$OUT/panel_${lower}_rectify.json" --out "$OUT"

  echo "== panel $panel: flames"
  "$PY" stage1/detect_flames.py --panel "$panel" \
      --rectified "$OUT/panel_${lower}_rectified.png" \
      --detection "$OUT/panel_${lower}_detection.json" --out "$OUT"

  echo "== panel $panel: overlay detail tiers at $PROJECTOR across ${SPAN_MM}mm"
  "$PY" stage1/plan_overlay.py --detection "$OUT/panel_${lower}_detection.json" \
      --projector "$PROJECTOR" --span-mm "$SPAN_MM"
done

echo "== rendering spec in pixels for $PROJECTOR across ${SPAN_MM}mm"
"$PY" stage1/spec_px.py --projector "$PROJECTOR" --span-mm "$SPAN_MM" \
    --out "$OUT/spec_px.json" >/dev/null
echo "wrote $OUT/spec_px.json"

echo
echo "Review these before Stage 2:"
echo "  $OUT/panel_*_rectify_qa.jpg      is the quad on the canvas edges?"
echo "  $OUT/panel_*_detection_qa.jpg    boxes, meshes, NOMESH flags"
echo "  $OUT/noise_floor.json            what counts as a confident detection"

# Steps 5 and 6 run once both panels have detections.
echo "== per-face biometric readout and cross-panel matching"
"$PY" stage1/analyze_faces.py --out "$OUT"
for panel in A B; do
  lower=$(echo "$panel" | tr 'A-Z' 'a-z')
  echo "== panel $panel: response field and blob tracking"
  "$PY" stage1/field_scan.py --panel "$panel" \
      --rectified "$OUT/panel_${lower}_rectified.png" --out "$OUT"
done

echo "== inferred structure: gaze, similarity, proximity, attribution, disagreement"
"$PY" stage1/infer_structure.py --out "$OUT"
for panel in A B; do
  "$PY" stage1/preview_overlay.py --panel "$panel" --out "$OUT"
  "$PY" stage1/preview_structure.py --panel "$panel" --out "$OUT"
done

echo "== detection stability and track states (drives flash rates)"
"$PY" stage1/stability.py --out "$OUT" --faces all

echo "== per-feature geometry: eyes, brows, nose, mouth, irises"
"$PY" stage1/face_features.py --out "$OUT"

if [ -d data/audio ]; then
  echo "== aligning each narration to its panel's transcript"
  for panel in A B; do
    lower=$(echo "$panel" | tr 'A-Z' 'a-z')
    v="data/audio/panel_${lower}_voice.mp3"
    [ -f "$v" ] && "$PY" stage1/align_voice.py --panel "$panel" --audio "$v" --out "$OUT"
  done
fi
