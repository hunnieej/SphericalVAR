#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="${1:-/home/mmai6k_jh/workspace/panorama/Infinity-Sphere/probe_outputs/head_classification}"

if ! command -v montage >/dev/null 2>&1 || ! command -v convert >/dev/null 2>&1 || ! command -v identify >/dev/null 2>&1; then
  echo "ImageMagick tools (montage/convert/identify) are required." >&2
  exit 1
fi

shopt -s nullglob

move_count=0
grid_count=0

for heatmap_root in "$ROOT_DIR"/*_heatmaps; do
  [[ -d "$heatmap_root" ]] || continue
  for sample_root in "$heatmap_root"/aggregate "$heatmap_root"/scene*; do
    [[ -d "$sample_root" ]] || continue
    mkdir -p "$sample_root/layers"
    for layer_dir in "$sample_root"/L*; do
      [[ -d "$layer_dir" ]] || continue
      target="$sample_root/layers/$(basename "$layer_dir")"
      if [[ ! -e "$target" ]]; then
        mv "$layer_dir" "$target"
        move_count=$((move_count + 1))
      fi
    done

    generated="$sample_root/generated.png"
    layers_root="$sample_root/layers"
    for suffix in attn_mean query_var relative; do
      inputs=()
      for layer_idx in $(seq -w 0 31); do
        img="$layers_root/L${layer_idx}/scale12/head_${suffix}_grid_4x4.png"
        [[ -f "$img" ]] || { inputs=(); break; }
        inputs+=("$img")
      done
      if [[ ${#inputs[@]} -eq 32 ]]; then
        out="$sample_root/head_${suffix}_layer_grids_4x8.png"
        montage "${inputs[@]}" -tile 4x8 -geometry +6+6 -background white "$out"
        grid_count=$((grid_count + 1))
        if [[ -f "$generated" ]]; then
          width="$(identify -format '%w' "$out")"
          convert "$generated" -resize "${width}x" "$out" -gravity north -background white -append "${out%.png}_with_generated.png"
          grid_count=$((grid_count + 1))
        fi
      fi
    done
  done
done

echo "Moved layer folders: $move_count"
echo "Created summary grids/composites: $grid_count"
