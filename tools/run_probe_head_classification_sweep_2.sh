#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

CONDITIONS=(baseline spherical_all spherical_split)
ROPE_SCALE_GROUPS=(
  "2 3"
  "2 3 4"
  "2 3 4 5"
)
HEAD_SPLIT_RATIOS=(0.25 0.50 0.75)
BAND_RATIOS=(0.25 0.50 0.75 1.0)

BASE_OUT_DIR="probe_outputs/head_classification"
MAX_PROMPTS=5

mkdir -p "$BASE_OUT_DIR"

format_ratio_tag() {
  local value="$1"
  printf '%s' "${value//./}"
}

format_scale_tag() {
  local raw="$1"
  if [[ "$raw" == "all" ]]; then
    printf '%s' "sall"
    return
  fi
  local tag="$(tr ' ' '-' <<<"$raw")"
  tag="${tag//coarsest/coarsest}"
  tag="${tag//finest/finest}"
  printf 's%s' "$tag"
}

render_progress() {
  local done_count="$1"
  local total_count="$2"
  local width=28
  local filled=0
  if [[ "$total_count" -gt 0 ]]; then
    filled=$(( done_count * width / total_count ))
  fi
  local empty=$(( width - filled ))
  local bar
  bar="$(printf '%*s' "$filled" '' | tr ' ' '#')$(printf '%*s' "$empty" '')"
  printf '\r[%s] %d/%d' "$bar" "$done_count" "$total_count"
}

declare -a TASKS=()

for condition in "${CONDITIONS[@]}"; do
  if [[ "$condition" == "baseline" ]]; then
    stem="${condition}_step"
    TASKS+=("${condition}|${stem}||")
    continue
  fi

  for rope_group in "${ROPE_SCALE_GROUPS[@]}"; do
    scale_tag="$(format_scale_tag "$rope_group")"

    if [[ "$condition" == "spherical_all" ]]; then
      for band_ratio in "${BAND_RATIOS[@]}"; do
        br_tag="$(format_ratio_tag "$band_ratio")"
        stem="${condition}_step_${scale_tag}_br${br_tag}"
        TASKS+=("${condition}|${stem}|${rope_group}|band=${band_ratio}")
      done
      continue
    fi

    for head_ratio in "${HEAD_SPLIT_RATIOS[@]}"; do
      hr_tag="$(format_ratio_tag "$head_ratio")"
      for band_ratio in "${BAND_RATIOS[@]}"; do
        br_tag="$(format_ratio_tag "$band_ratio")"
        stem="${condition}_step_${scale_tag}_hr${hr_tag}_br${br_tag}"
        TASKS+=("${condition}|${stem}|${rope_group}|head=${head_ratio}|band=${band_ratio}")
      done
    done
  done
done

total_tasks=${#TASKS[@]}
remaining_tasks=0
for task in "${TASKS[@]}"; do
  IFS='|' read -r _ stem _rest <<<"$task"
  out_json="${BASE_OUT_DIR}/${stem}.json"
  heatmap_dir="${BASE_OUT_DIR}/${stem}_heatmaps"
  if [[ ! -f "$out_json" || ! -d "$heatmap_dir" ]]; then
    remaining_tasks=$((remaining_tasks + 1))
  fi
done

echo "Total experiments: ${total_tasks}"
echo "Remaining experiments: ${remaining_tasks}"

completed=0
ran=0
skipped=0

for task in "${TASKS[@]}"; do
  IFS='|' read -r condition stem rope_group arg1 arg2 <<<"$task"
  out_json="${BASE_OUT_DIR}/${stem}.json"
  heatmap_dir="${BASE_OUT_DIR}/${stem}_heatmaps"
  log_file="${BASE_OUT_DIR}/${stem}.log"

  if [[ -f "$out_json" && -d "$heatmap_dir" ]]; then
    skipped=$((skipped + 1))
    completed=$((completed + 1))
    render_progress "$completed" "$total_tasks"
    continue
  fi

  cmd=(
    python tools/probe_head_classification.py
    --condition "$condition"
    --layers all
    --max_prompts "$MAX_PROMPTS"
    --output "$out_json"
  )

  if [[ "$condition" != "baseline" && -n "${rope_group:-}" ]]; then
    # shellcheck disable=SC2206
    rope_args=( $rope_group )
    cmd+=(--rope_scales "${rope_args[@]}")
  fi

  if [[ "$condition" == "spherical_all" ]]; then
    band_ratio="${arg1#band=}"
    cmd+=(--band_ratio "$band_ratio")
  elif [[ "$condition" == "spherical_split" ]]; then
    head_ratio="${arg1#head=}"
    band_ratio="${arg2#band=}"
    cmd+=(--head_split_ratio "$head_ratio" --band_ratio "$band_ratio")
  fi

  if "${cmd[@]}" >"$log_file" 2>&1; then
    ran=$((ran + 1))
  else
    echo
    echo "Failed: $stem"
    echo "Log: $log_file"
    tail -n 40 "$log_file" || true
    exit 1
  fi

  completed=$((completed + 1))
  render_progress "$completed" "$total_tasks"
done

echo
echo "Done. ran=${ran}, skipped=${skipped}, total=${total_tasks}"
