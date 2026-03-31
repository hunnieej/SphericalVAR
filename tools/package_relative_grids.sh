#!/usr/bin/env bash

set -euo pipefail

TARGET_DIR="${1:-}"

if [[ -z "$TARGET_DIR" ]]; then
  echo "Usage: bash tools/package_relative_grids.sh <scene_dir>" >&2
  exit 1
fi

if [[ ! -d "$TARGET_DIR" ]]; then
  echo "Directory not found: $TARGET_DIR" >&2
  exit 1
fi

shopt -s nullglob

TMP_DIR="$TARGET_DIR/relative_grids_for_tar"
mkdir -p "$TMP_DIR"

count=0
for layer_dir in "$TARGET_DIR"/L*; do
  [[ -d "$layer_dir" ]] || continue
  layer_name="$(basename "$layer_dir")"
  for scale_dir in "$layer_dir"/scale*; do
    [[ -d "$scale_dir" ]] || continue
    src="$scale_dir/head_relative_grid_4x4.png"
    [[ -f "$src" ]] || continue
    dst="$TMP_DIR/head_relative_grid_4x4_${layer_name}.png"
    cp "$src" "$dst"
    count=$((count + 1))
  done
done

if [[ "$count" -eq 0 ]]; then
  rmdir "$TMP_DIR" 2>/dev/null || true
  echo "No head_relative_grid_4x4.png files found under: $TARGET_DIR" >&2
  exit 1
fi

tar_path="${TARGET_DIR%/}/head_relative_grid_4x4_layers2.tar"
tar -cf "$tar_path" -C "$TMP_DIR" .
rm -rf "$TMP_DIR"

echo "Packaged $count files into $tar_path"
