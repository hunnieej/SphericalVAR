#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

# Edit these arrays for your sweep.
SCALE_GROUPS=(
  "1 2"
  "1 2 3"
  "1 2 3 4"
  "1 2 3 4 5"
  "2 3"
  "2 3 4"
  "2 3 4 5"
  "3"
  "3 4"
  "3 4 5"
  "4"
  "4 5"
)

HEAD_SPLIT_RATIOS=(0.50 0.75)
BAND_RATIOS=(0.25 0.50 0.75)

# Optional interpolation sweep. Use "default" to keep the model default.
INTERP_MODES=(nearest bicubic bilinear area)

# Base output directory. Each run appends its own suffix.
BASE_OUT_DIR="eval_outputs/comparison_2b_scale_sweep"

for scale_group in "${SCALE_GROUPS[@]}"; do
  scale_tag="$(tr ' ' '-' <<<"$scale_group")"

  for interp_mode in "${INTERP_MODES[@]}"; do
    interp_tag="$interp_mode"

    for head_ratio in "${HEAD_SPLIT_RATIOS[@]}"; do
      for band_ratio in "${BAND_RATIOS[@]}"; do
        out_dir="${BASE_OUT_DIR}/interp_${interp_tag}/scales_${scale_tag}_hs${head_ratio}_br${band_ratio}"

        cmd=(
          python run_comparison_2b.py
          --rope_scales ${scale_group}
          --out_dir "$out_dir"
        )

        if [[ "$interp_mode" != "default" ]]; then
          cmd+=(--interp_mode "$interp_mode")
        fi

        echo "Running: scales=[$scale_group] head_split_ratio=$head_ratio band_ratio=$band_ratio interp_mode=$interp_tag"
        HACK_HEAD_SPLIT_RATIO="$head_ratio" \
        HACK_BAND_RATIO="$band_ratio" \
        "${cmd[@]}"
      done
    done
  done
done
