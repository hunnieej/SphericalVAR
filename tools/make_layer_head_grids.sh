#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="${1:-/home/mmai6k_jh/workspace/panorama/Infinity-Sphere/probe_outputs/head_classification/spherical_split_final_step_heatmaps}"
OUT_NAME="head_grid_4x4.png"

if ! command -v montage >/dev/null 2>&1; then
  echo "montage command not found (ImageMagick required)." >&2
  exit 1
fi

shopt -s nullglob

count=0
for scale_dir in "$ROOT_DIR"/scene*/L*/scale*; do
  attn_heads=("$scale_dir"/head[0-9][0-9]_attn_mean.png)
  var_heads=("$scale_dir"/head[0-9][0-9]_query_var.png)

  if [[ ${#attn_heads[@]} -eq 16 ]]; then
    montage \
      "$scale_dir"/head[0-9][0-9]_attn_mean.png \
      -tile 4x4 \
      -geometry +6+6 \
      -background white \
      "$scale_dir/head_attn_mean_grid_4x4.png"
    count=$((count + 1))
    echo "[saved] $scale_dir/head_attn_mean_grid_4x4.png"
  else
    echo "[skip] $scale_dir attn_mean: expected 16 files, found ${#attn_heads[@]}"
  fi

  if [[ ${#var_heads[@]} -eq 16 ]]; then
    montage \
      "$scale_dir"/head[0-9][0-9]_query_var.png \
      -tile 4x4 \
      -geometry +6+6 \
      -background white \
      "$scale_dir/head_query_var_grid_4x4.png"
    count=$((count + 1))
    echo "[saved] $scale_dir/head_query_var_grid_4x4.png"
  else
    echo "[skip] $scale_dir query_var: expected 16 files, found ${#var_heads[@]}"
  fi
done

echo "Done. Saved $count layer-head grids."
