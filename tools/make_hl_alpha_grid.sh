#!/usr/bin/env bash

set -euo pipefail

ROOT="/home/mmai6k_jh/workspace/panorama/Infinity-Sphere/eval_outputs"
BASE_DIR="$ROOT/comparison_2b_HL_a1"
OUT="$ROOT/comparison_2b_HL_all_grid.jpg"

if ! command -v montage >/dev/null 2>&1; then
  echo "montage command not found (ImageMagick required)." >&2
  exit 1
fi

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT
TARGET_WIDTH=360

PROMPTS=(
  "00 0"
  "01 1234"
  "02 5536"
  "03 8650"
  "04 9902"
)

inputs=()
for spec in "${PROMPTS[@]}"; do
  read -r pidx seed <<<"$spec"
  inputs+=("$BASE_DIR/default_baseline_p${pidx}_s${seed}.jpg")
  inputs+=("$BASE_DIR/default_spherical_all_p${pidx}_s${seed}.jpg")
  for i in {1..9}; do
    inputs+=("$ROOT/comparison_2b_HL_a${i}/default_spherical_split_custom_r0.75_p${pidx}_s${seed}.jpg")
  done
done

for path in "${inputs[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing image: $path" >&2
    exit 1
  fi
done

resized_inputs=()
idx=0
for path in "${inputs[@]}"; do
  out="$TMPDIR/$(printf '%03d' "$idx").jpg"
  convert "$path" -resize "${TARGET_WIDTH}x" "$out"
  resized_inputs+=("$out")
  idx=$((idx + 1))
done

montage \
  "${resized_inputs[@]}" \
  -tile 11x5 \
  -geometry +8+8 \
  -background white \
  "$OUT"

echo "Saved grid to $OUT"
