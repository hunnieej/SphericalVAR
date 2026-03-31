#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="${1:-/home/mmai6k_jh/workspace/panorama/Infinity-Sphere/probe_outputs/head_classification}"

if ! command -v montage >/dev/null 2>&1; then
  echo "montage command not found (ImageMagick required)." >&2
  exit 1
fi

if ! command -v identify >/dev/null 2>&1 || ! command -v convert >/dev/null 2>&1; then
  echo "identify/convert command not found (ImageMagick required)." >&2
  exit 1
fi

shopt -s nullglob

count=0
for scene_root in "$ROOT_DIR"/*_heatmaps/scene*; do
  [[ -d "$scene_root" ]] || continue
  generated="$scene_root/generated.png"
  for scale_dir in "$scene_root"/L*/scale*; do
    [[ -d "$scale_dir" ]] || continue
    for suffix in attn_mean query_var relative; do
      heads=("$scale_dir"/head[0-9][0-9]_${suffix}.png)
      if [[ ${#heads[@]} -ne 16 ]]; then
        continue
      fi
      grid="$scale_dir/head_${suffix}_grid_4x4.png"
      montage \
        "$scale_dir"/head[0-9][0-9]_${suffix}.png \
        -tile 4x4 \
        -geometry +6+6 \
        -background white \
        "$grid"
      count=$((count + 1))
      if [[ -f "$generated" ]]; then
        width="$(identify -format '%w' "$grid")"
        convert "$generated" -resize "${width}x" "$grid" -gravity north -background white -append "${grid%.png}_with_generated.png"
        count=$((count + 1))
      fi
    done
  done
done

echo "Done. Saved $count scene grid/composite images."
