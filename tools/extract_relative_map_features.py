import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from analysis.relative_map_features import (
    RelativeFeatureConfig,
    build_feature_grid,
    build_histogram_svg,
    extract_features,
    load_config_dict,
    overlay_masks,
    read_grayscale_image,
    save_png_via_convert,
    save_svg_as_png,
)


REL_RE = re.compile(r"head(?P<head>\d+)_relative\.png$")
LAYER_RE = re.compile(r"L(?P<layer>\d+)$")
SCALE_RE = re.compile(r"scale(?P<scale>\d+)$")
SCENE_RE = re.compile(r"scene(?P<scene>\d+)_?(?P<slug>.*)$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract morphology features from relative attention maps"
    )
    parser.add_argument("--input_dir", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument(
        "--glob",
        type=str,
        default="**/head*_relative.png",
        help="Glob relative to input_dir",
    )
    return parser.parse_args()


def load_config(path: Path) -> RelativeFeatureConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return load_config_dict(data)


def parse_metadata(path: Path, input_dir: Path):
    rel = path.relative_to(input_dir)
    parts = rel.parts
    layer = head = scale = None
    sample_id = scene_id = prompt_text = seed = None
    for part in parts:
        m = LAYER_RE.match(part)
        if m:
            layer = int(m.group("layer"))
        m = SCALE_RE.match(part)
        if m:
            scale = int(m.group("scale"))
        m = SCENE_RE.match(part)
        if m:
            sample_id = part
            scene_id = int(m.group("scene"))
            prompt_text = m.group("slug") or part
        if part == "aggregate":
            sample_id = "aggregate"
            scene_id = None
            prompt_text = "aggregate"
    m = REL_RE.match(path.name)
    if m:
        head = int(m.group("head"))
    if layer is None or head is None:
        raise ValueError(f"Failed to parse layer/head from {path}")
    return {
        "sample_id": sample_id or rel.parts[0],
        "seed": seed,
        "scene_id": scene_id,
        "prompt_text": prompt_text,
        "layer": layer,
        "head": head,
        "scale": scale,
        "path": str(path),
    }


def write_csv(path: Path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_features(rows):
    keys = [
        "center_score",
        "horizontal_score",
        "vertical_score",
        "cross_score",
        "balance_score",
        "spread_score",
        "self_peak_score",
    ]
    summary = {}
    for key in keys:
        vals = np.array(
            [float(r[key]) for r in rows if np.isfinite(float(r[key]))],
            dtype=np.float64,
        )
        if vals.size == 0:
            summary[key] = None
            continue
        summary[key] = {
            "min": float(vals.min()),
            "max": float(vals.max()),
            "mean": float(vals.mean()),
            "std": float(vals.std()),
        }
    return summary


def aggregate_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(int(row["layer"]), int(row["head"]))].append(row)
    out = []
    for (layer, head), items in sorted(grouped.items()):
        agg = {"layer": layer, "head": head, "sample_count": len(items)}
        for key in [
            "center_score",
            "horizontal_score",
            "vertical_score",
            "cross_score",
            "balance_score",
            "spread_score",
            "self_peak_score",
        ]:
            vals = np.array([float(item[key]) for item in items], dtype=np.float64)
            agg[f"{key}_mean"] = float(np.nanmean(vals))
        out.append(agg)
    return out


def save_feature_visuals(
    output_dir: Path,
    cfg: RelativeFeatureConfig,
    per_head_rows,
    aggregate_rows_out,
    representative,
):
    vis_dir = output_dir / "visuals"
    vis_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in enumerate(representative[: cfg.sanity_check_count]):
        arr = read_grayscale_image(Path(row["path"]))
        overlay = overlay_masks(arr, cfg)
        save_png_via_convert(
            vis_dir
            / f"sanity_{idx:02d}_L{int(row['layer']):02d}_H{int(row['head']):02d}.png",
            overlay,
        )

    for key in [
        "center_score",
        "horizontal_score",
        "vertical_score",
        "cross_score",
        "balance_score",
        "spread_score",
        "self_peak_score",
    ]:
        values = [float(row[key]) for row in per_head_rows]
        svg = build_histogram_svg(values, f"{key} histogram")
        save_svg_as_png(vis_dir / f"hist_{key}.png", svg)

    for key in [
        "center_score_mean",
        "horizontal_score_mean",
        "vertical_score_mean",
        "cross_score_mean",
        "balance_score_mean",
        "spread_score_mean",
        "self_peak_score_mean",
    ]:
        value_map = {
            (int(row["layer"]), int(row["head"])): float(row[key])
            for row in aggregate_rows_out
        }
        grid = build_feature_grid(value_map, key)
        save_png_via_convert(vis_dir / f"grid_{key}.png", grid)

    topk_path = output_dir / "topk_heads.txt"
    with open(topk_path, "w", encoding="utf-8") as f:
        for key in [
            "center_score_mean",
            "horizontal_score_mean",
            "vertical_score_mean",
            "cross_score_mean",
            "balance_score_mean",
            "spread_score_mean",
        ]:
            ranked = sorted(
                aggregate_rows_out, key=lambda r: float(r[key]), reverse=True
            )[: cfg.top_k]
            f.write(f"## top {cfg.top_k} by {key}\n")
            for row in ranked:
                f.write(
                    f"L{int(row['layer']):02d} H{int(row['head']):02d}  {float(row[key]):.6f}\n"
                )
            f.write("\n")


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(Path(args.config))

    files = sorted(input_dir.glob(args.glob))
    rows = []
    skipped = []
    for path in files:
        try:
            meta = parse_metadata(path, input_dir)
            arr = read_grayscale_image(path)
            feats = extract_features(arr, cfg)
            if feats is None:
                skipped.append({"path": str(path), "reason": "invalid_or_zero_sum"})
                continue
            row = {
                "prompt_id": meta["scene_id"],
                "sample_id": meta["sample_id"],
                "seed": meta["seed"],
                "scene_id": meta["scene_id"],
                "prompt_text": meta["prompt_text"],
                "layer": meta["layer"],
                "head": meta["head"],
                **feats,
                "map_height": int(arr.shape[0]),
                "map_width": int(arr.shape[1]),
                "path": meta["path"],
            }
            rows.append(row)
        except Exception as exc:
            skipped.append({"path": str(path), "reason": str(exc)})

    per_head_csv = output_dir / "per_head_features.csv"
    write_csv(
        per_head_csv,
        rows,
        [
            "prompt_id",
            "sample_id",
            "seed",
            "scene_id",
            "prompt_text",
            "layer",
            "head",
            "center_score",
            "horizontal_score",
            "vertical_score",
            "cross_score",
            "balance_score",
            "spread_score",
            "self_peak_score",
            "map_height",
            "map_width",
            "path",
        ],
    )

    agg_rows = aggregate_rows(rows)
    agg_csv = output_dir / "aggregated_features.csv"
    write_csv(
        agg_csv,
        agg_rows,
        [
            "layer",
            "head",
            "center_score_mean",
            "horizontal_score_mean",
            "vertical_score_mean",
            "cross_score_mean",
            "balance_score_mean",
            "spread_score_mean",
            "self_peak_score_mean",
            "sample_count",
        ],
    )

    summary = {
        "config": cfg.__dict__,
        "input_dir": str(input_dir),
        "glob": args.glob,
        "processed_sample_count": len(rows),
        "processed_layer_head_pairs": len(
            {(int(r["layer"]), int(r["head"])) for r in rows}
        ),
        "skipped_count": len(skipped),
        "feature_summary": summarize_features(rows),
        "skipped": skipped[:50],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    representative = sorted(
        rows, key=lambda r: (str(r["sample_id"]), int(r["layer"]), int(r["head"]))
    )
    save_feature_visuals(output_dir, cfg, rows, agg_rows, representative)

    print(f"Saved per-head CSV to {per_head_csv}")
    print(f"Saved aggregated CSV to {agg_csv}")
    print(f"Saved summary JSON to {summary_path}")


if __name__ == "__main__":
    main()
