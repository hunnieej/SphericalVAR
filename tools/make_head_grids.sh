#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/home/mmai6k_jh/workspace/panorama/Infinity-Sphere/probe_outputs/attn_heatmaps_Lall_Sall/spherical_split"
OUT_NAME="head_grid_4x4.png"

if ! command -v montage >/dev/null 2>&1; then
  echo "montage command not found (ImageMagick required)." >&2
  exit 1
fi

shopt -s nullglob

count=0
for scale_dir in "$ROOT_DIR"/L*/scale*; do
  heads=("$scale_dir"/head[0-9][0-9].png)
  if [[ ${#heads[@]} -ne 16 ]]; then
    echo "[skip] $scale_dir: expected 16 head images, found ${#heads[@]}"
    continue
  fi

  montage \
    "$scale_dir"/head[0-9][0-9].png \
    -tile 4x4 \
    -geometry +8+8 \
    -background white \
    "$scale_dir/$OUT_NAME"

  echo "[saved] $scale_dir/$OUT_NAME"
  count=$((count + 1))
done

echo "Done. Saved $count grid images."
