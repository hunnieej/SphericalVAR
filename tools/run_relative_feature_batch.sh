#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

INPUT_ROOT="${1:-probe_outputs/head_classification/baseline_step_ranked_heatmaps}"
CONFIG_PATH="${2:-configs/relative_map_feature_config.yaml}"
OUTPUT_ROOT="${3:-analysis_outputs/relative_feature_batch}"
PYTHON_BIN="${PYTHON_BIN:-/home/mmai6k_jh/anaconda3/envs/infinity/bin/python}"

mkdir -p "$OUTPUT_ROOT"

declare -a TARGETS=()
if [[ -d "$INPUT_ROOT/aggregate" ]]; then
  TARGETS+=("aggregate")
fi
for scene_dir in "$INPUT_ROOT"/scene*; do
  [[ -d "$scene_dir" ]] || continue
  TARGETS+=("$(basename "$scene_dir")")
done

total=${#TARGETS[@]}
if [[ "$total" -eq 0 ]]; then
  echo "No aggregate/scene directories found under $INPUT_ROOT" >&2
  exit 1
fi

done_count=0
for name in "${TARGETS[@]}"; do
  in_dir="$INPUT_ROOT/$name"
  out_dir="$OUTPUT_ROOT/$name"
  "$PYTHON_BIN" tools/extract_relative_map_features.py \
    --input_dir "$in_dir" \
    --config "$CONFIG_PATH" \
    --output_dir "$out_dir" >/dev/null
  done_count=$((done_count + 1))
  printf '\rProcessed %d/%d: %s' "$done_count" "$total" "$name"
done

echo
echo "Saved batch outputs to $OUTPUT_ROOT"
