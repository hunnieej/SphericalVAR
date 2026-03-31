import argparse
import math
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from infinity.utils.dynamic_resolution import dynamic_resolution_h_w
from spherical_rope_infinity import SphericalRoPEInfinityPatcher
from tools.layer_head_spec_utils import resolve_layer_head_map
from tools.run_infinity import (
    gen_one_img,
    load_tokenizer,
    load_transformer,
    load_visual_tokenizer,
)


DEFAULT_PROMPTS = [
    "This is a 360 degree panorama image. The photo shows a breathtaking snowy mountain summit at sunrise, with golden light illuminating the peaks and valleys stretching endlessly in all directions.",
    "This is a 360 degree panorama image. The photo shows a modern city skyline at night, with glittering lights reflecting off a calm river and bridges connecting both banks.",
    "This is a 360 degree panorama image. The photo shows a tranquil tropical beach at sunset, with crystal-clear turquoise water, white sand, and palm trees.",
    "This is a 360 degree panorama image. The photo shows the interior of a grand cathedral with soaring gothic arches, stained glass windows casting colorful light on the stone floor.",
    "This is a 360 degree panorama image. The photo shows a quiet forest path in autumn, with golden and red leaves covering the ground.",
]

DEFAULT_SEEDS = [1234, 5536, 8650, 9902, 0]
# DEFAULT_SEEDS = [0, 1234, 5536, 8650, 9902]


def parse_args():
    parser = argparse.ArgumentParser(description="HACK-style head classification probe")
    parser.add_argument(
        "--condition",
        default="baseline",
        choices=["baseline", "spherical_all", "spherical_split"],
    )
    parser.add_argument(
        "--pn", default="1M", choices=list(dynamic_resolution_h_w[0.5].keys())
    )
    parser.add_argument(
        "--h_div_w",
        type=float,
        default=0.5,
        choices=sorted(dynamic_resolution_h_w.keys()),
    )
    parser.add_argument(
        "--layers", nargs="*", default=["all"], help="Layer indices or 'all'"
    )
    parser.add_argument(
        "--prompt",
        action="append",
        default=None,
        help="Prompt to include; can be specified multiple times",
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default="",
        help="Text file with one prompt per line",
    )
    parser.add_argument("--max_prompts", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=900)
    parser.add_argument("--top_p", type=float, default=0.97)
    parser.add_argument(
        "--cfg_insertion_layer",
        nargs="*",
        type=int,
        default=[0],
        help="Match run_comparison_2b default CFG insertion behavior",
    )
    parser.add_argument("--gt_leak", type=int, default=0)
    parser.add_argument("--head_split_ratio", type=float, default=1.0)
    parser.add_argument("--band_ratio", type=float, default=0.5)
    parser.add_argument(
        "--layer_head_spec",
        type=str,
        default="",
        help="Exact spherical head selection, e.g. '0:5,6;6:6'",
    )
    parser.add_argument(
        "--layer_head_spec_file",
        type=str,
        default="",
        help="JSON/txt file describing exact spherical head selection",
    )
    parser.add_argument(
        "--rope_scales",
        nargs="*",
        default=None,
        help="Scales for spherical RoPE (ints, finest, coarsest, all)",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.5,
        help="Global quantile threshold for structural heads",
    )
    parser.add_argument(
        "--alpha",
        nargs="*",
        type=float,
        default=None,
        help="Structural top-ratio(s), e.g. 0.1 0.2 0.3",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="probe_outputs/head_classification/head_classification.json",
    )
    parser.add_argument(
        "--save_heatmaps",
        type=int,
        default=1,
        choices=[0, 1],
        help="Save final-step attention images together with the JSON",
    )
    return parser.parse_args()


def load_prompts(args):
    prompts = []
    if args.prompt:
        prompts.extend(args.prompt)
    if args.prompts_file:
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            prompts.extend(line.strip() for line in f if line.strip())
    if not prompts:
        prompts = list(DEFAULT_PROMPTS)
    return prompts[: args.max_prompts]


def resolve_prompt_seeds(num_prompts: int, base_seed: int):
    seeds = []
    for idx in range(num_prompts):
        if idx < len(DEFAULT_SEEDS):
            seeds.append(DEFAULT_SEEDS[idx])
        else:
            seeds.append(base_seed + idx)
    return seeds


def alpha_tag(alpha: float) -> str:
    return f"a{int(round(alpha * 100)):03d}"


def build_layer_head_map(items):
    out = {}
    for item in items:
        layer_idx = int(item["layer"][1:])
        out.setdefault(str(layer_idx), []).append(int(item["head"]))
    for key in out:
        out[key] = sorted(set(out[key]))
    return out


def write_alpha_configs(base_output: Path, aggregated, alphas):
    if not alphas:
        return []
    total = len(aggregated)
    written = []
    for alpha in alphas:
        if alpha <= 0 or alpha > 1:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        k = max(1, int(math.ceil(total * alpha)))
        structural = aggregated[:k]
        contextual = aggregated[k:]
        structural_map = build_layer_head_map(structural)
        contextual_map = build_layer_head_map(contextual)
        stem = base_output.stem
        struct_path = base_output.with_name(
            f"{stem}_{alpha_tag(alpha)}_structural_heads.json"
        )
        context_path = base_output.with_name(
            f"{stem}_{alpha_tag(alpha)}_contextual_heads.json"
        )
        summary_path = base_output.with_name(
            f"{stem}_{alpha_tag(alpha)}_split_summary.json"
        )
        with open(struct_path, "w", encoding="utf-8") as f:
            json.dump(structural_map, f, indent=2)
        with open(context_path, "w", encoding="utf-8") as f:
            json.dump(contextual_map, f, indent=2)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "alpha": alpha,
                    "num_total_heads": total,
                    "num_structural_heads": k,
                    "num_contextual_heads": total - k,
                    "structural_config": structural_map,
                    "contextual_config": contextual_map,
                },
                f,
                indent=2,
            )
        written.append((alpha, struct_path, context_path, summary_path))
    return written


def resolve_rope_scales(rope_scales, scale_schedule):
    if not rope_scales:
        return None
    total = len(scale_schedule)
    if len(rope_scales) == 1 and rope_scales[0].lower() == "all":
        return None
    resolved = []
    for token in rope_scales:
        lowered = token.lower()
        if lowered == "finest":
            resolved.append(total - 1)
        elif lowered == "coarsest":
            resolved.append(0)
        else:
            idx = int(token)
            if idx < 0 or idx >= total:
                raise ValueError(
                    f"rope scale index {idx} out of range (0..{total - 1})"
                )
            resolved.append(idx)
    return sorted(set(resolved))


def prepare_model(args):
    class VaeArgs:
        vae_type = 32
        vae_path = "pretrained/infinity_vae_d32reg.pth"
        apply_spatial_patchify = 0

    class ModelArgs(VaeArgs):
        model_path = "pretrained/infinity_2b_reg.pth"
        checkpoint_type = "torch"
        model_type = "infinity_2b"
        rope2d_each_sa_layer = 1
        rope2d_normalized_by_hw = 2
        use_scale_schedule_embedding = 0
        pn = args.pn
        use_bit_label = 1
        add_lvl_embeding_only_first_block = 0
        text_channels = 2048
        use_flex_attn = 0
        bf16 = 1
        cache_dir = "/dev/shm"
        enable_model_cache = 0

    print("Loading tokenizer and text encoder...")
    text_tokenizer, text_encoder = load_tokenizer("google/flan-t5-xl")
    print("Loading VAE...")
    vae = load_visual_tokenizer(VaeArgs())
    print("Loading Infinity...")
    infinity = load_transformer(vae, ModelArgs())
    return infinity, vae, text_tokenizer, text_encoder


def gather_self_attention_modules(infinity):
    modules = []
    for block in infinity.unregistered_blocks:
        if hasattr(block, "sa"):
            modules.append(block.sa)
        elif hasattr(block, "attn"):
            modules.append(block.attn)
    return modules


def select_modules(modules, layer_specs):
    if len(layer_specs) == 1 and layer_specs[0].lower() == "all":
        layer_indices = list(range(len(modules)))
    else:
        layer_indices = [int(v) for v in layer_specs]
    selected = []
    total = len(modules)
    for idx in layer_indices:
        actual = idx if idx >= 0 else total + idx
        if actual < 0 or actual >= total:
            raise ValueError(f"Layer index {idx} is out of range for {total} layers")
        selected.append((actual, modules[actual]))
    return selected


def configure_capture(selected_modules, final_scale_idx):
    for layer_idx, module in selected_modules:
        module.configure_attn_capture(
            True, target_scales=(final_scale_idx,), layer_id=f"L{layer_idx:02d}"
        )


def clear_capture(selected_modules):
    for _, module in selected_modules:
        module.configure_attn_capture(False)


def collect_capture(selected_modules):
    records = []
    for layer_idx, module in selected_modules:
        module_records = module.pop_attn_capture_records()
        if module_records:
            records.append(module_records[-1])
    return records


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    arr = arr - arr.min()
    maxv = float(arr.max())
    if maxv > 0:
        arr = arr / maxv
    return np.clip(np.round(arr * 255.0), 0, 255).astype(np.uint8)


def save_pgm(path: Path, arr: np.ndarray):
    arr_u8 = normalize_to_uint8(arr)
    h, w = arr_u8.shape
    with open(path, "wb") as f:
        f.write(f"P5\n{w} {h}\n255\n".encode("ascii"))
        f.write(arr_u8.tobytes())


def save_png(path: Path, arr: np.ndarray):
    tmp_pgm = path.with_suffix(".pgm")
    save_pgm(tmp_pgm, arr)
    subprocess.run(["convert", str(tmp_pgm), str(path)], check=True)
    tmp_pgm.unlink(missing_ok=True)


def save_ppm(path: Path, arr: np.ndarray):
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected RGB image shaped (H, W, 3), got {arr.shape}")
    h, w = arr.shape[:2]
    with open(path, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
        f.write(arr.tobytes())


def compute_scale_spans(scale_schedule):
    spans = []
    start = 0
    for pt, ph, pw in scale_schedule:
        length = pt * ph * pw
        spans.append((start, start + length))
        start += length
    return spans


def slugify(text: str, max_len: int = 48) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:max_len] or "prompt"


def save_heatmap_set(layer_maps, root_dir: Path, scale_idx: int, scale_span, scale_hw):
    root_dir.mkdir(parents=True, exist_ok=True)
    start, end = scale_span
    final_h, final_w = scale_hw
    for layer_id in sorted(layer_maps.keys()):
        layer_dir = root_dir / layer_id / f"scale{scale_idx:02d}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        mean_arr = layer_maps[layer_id]["attn_mean"]
        var_arr = layer_maps[layer_id]["query_var"]
        relative_arr = layer_maps[layer_id].get("relative_map")
        num_heads = mean_arr.shape[0]
        for head_idx in range(num_heads):
            mean_map = mean_arr[head_idx][start:end].reshape(final_h, final_w)
            var_map = var_arr[head_idx][start:end].reshape(final_h, final_w)
            save_png(layer_dir / f"head{head_idx:02d}_attn_mean.png", mean_map)
            save_png(layer_dir / f"head{head_idx:02d}_query_var.png", var_map)
            if relative_arr is not None:
                rel_map = relative_arr[head_idx]
                save_png(layer_dir / f"head{head_idx:02d}_relative.png", rel_map)


def build_layer_grids(root_dir: Path, scale_idx: int):
    scale_name = f"scale{scale_idx:02d}"
    sample_dir = root_dir / "L00" / scale_name
    if not sample_dir.exists():
        return
    image_names = sorted(
        p.name for p in sample_dir.glob("*.png") if "layers_grid" not in p.name
    )
    for image_name in image_names:
        inputs = []
        for layer_idx in range(32):
            image_path = root_dir / f"L{layer_idx:02d}" / scale_name / image_name
            if not image_path.exists():
                inputs = []
                break
            inputs.append(str(image_path))
        if not inputs:
            continue
        out_path = root_dir / f"{image_name[:-4]}_layers_grid_4x8.png"
        subprocess.run(
            [
                "montage",
                *inputs,
                "-tile",
                "4x8",
                "-geometry",
                "+6+6",
                "-background",
                "white",
                str(out_path),
            ],
            check=True,
        )


def build_head_grids(root_dir: Path, scale_idx: int):
    scale_name = f"scale{scale_idx:02d}"
    for layer_dir in sorted(root_dir.glob("L*")):
        scale_dir = layer_dir / scale_name
        if not scale_dir.exists():
            continue
        for suffix in ("attn_mean", "query_var", "relative"):
            inputs = []
            for head_idx in range(16):
                image_path = scale_dir / f"head{head_idx:02d}_{suffix}.png"
                if not image_path.exists():
                    inputs = []
                    break
                inputs.append(str(image_path))
            if not inputs:
                continue
            out_path = scale_dir / f"head_{suffix}_grid_4x4.png"
            subprocess.run(
                [
                    "montage",
                    *inputs,
                    "-tile",
                    "4x4",
                    "-geometry",
                    "+6+6",
                    "-background",
                    "white",
                    str(out_path),
                ],
                check=True,
            )


def save_generated_image(path: Path, img):
    if isinstance(img, torch.Tensor):
        arr = img.detach().cpu().numpy()
    else:
        arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = arr[..., ::-1]
    tmp_ppm = path.with_suffix(".ppm")
    save_ppm(tmp_ppm, arr)
    subprocess.run(["convert", str(tmp_ppm), str(path)], check=True)
    tmp_ppm.unlink(missing_ok=True)


def get_image_size(path: Path):
    out = subprocess.check_output(
        ["identify", "-format", "%w %h", str(path)], text=True
    ).strip()
    width, height = out.split()
    return int(width), int(height)


def build_generated_composites(root_dir: Path, generated_path: Path):
    grid_paths = sorted(root_dir.glob("*_layers_grid_4x8.png"))
    grid_paths.extend(sorted(root_dir.glob("L*/scale*/*_grid_4x4.png")))
    for grid_path in grid_paths:
        out_path = grid_path.with_name(f"{grid_path.stem}_with_generated.png")
        grid_w, _ = get_image_size(grid_path)
        subprocess.run(
            [
                "convert",
                str(generated_path),
                "-resize",
                f"{grid_w}x",
                str(grid_path),
                "-gravity",
                "north",
                "-background",
                "white",
                "-append",
                str(out_path),
            ],
            check=True,
        )


def main():
    args = parse_args()
    prompts = load_prompts(args)
    prompt_seeds = resolve_prompt_seeds(len(prompts), args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    heatmap_root = output_path.parent / f"{output_path.stem}_heatmaps"
    aggregate_root = heatmap_root / "aggregate"

    infinity, vae, tokenizer, text_encoder = prepare_model(args)
    scale_schedule = dynamic_resolution_h_w[args.h_div_w][args.pn]["scales"]
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
    scale_spans = compute_scale_spans(scale_schedule)
    final_scale_idx = len(scale_schedule) - 1
    final_h, final_w = (
        scale_schedule[final_scale_idx][1],
        scale_schedule[final_scale_idx][2],
    )
    final_start, final_end = scale_spans[final_scale_idx]
    target_rope_scales = resolve_rope_scales(args.rope_scales, scale_schedule)
    layer_head_map = resolve_layer_head_map(
        args.layer_head_spec, args.layer_head_spec_file
    )

    patcher = None
    if args.condition != "baseline":
        patcher = SphericalRoPEInfinityPatcher(
            infinity,
            alpha_w=1.0,
            alpha_h=0.0,
            head_split_ratio=(
                None
                if args.condition == "spherical_all" or layer_head_map is not None
                else args.head_split_ratio
            ),
            spherical_band_ratio=args.band_ratio,
            target_scales=target_rope_scales,
            layer_head_map=layer_head_map,
        )
        patcher.apply()

    selected_modules = select_modules(
        gather_self_attention_modules(infinity), args.layers
    )
    configure_capture(selected_modules, final_scale_idx)

    scores_by_layer = {}
    mean_maps_by_layer = {}
    var_maps_by_layer = {}
    relative_maps_by_layer = {}
    try:
        for pi, prompt in enumerate(prompts):
            prompt_tag = f"scene{pi:02d}_{slugify(prompt)}"
            seed = prompt_seeds[pi]
            print(f"[{pi + 1}/{len(prompts)}] seed={seed} {prompt[:80]}")
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                img = gen_one_img(
                    infinity,
                    vae,
                    tokenizer,
                    text_encoder,
                    prompt,
                    cfg_list=args.cfg,
                    tau_list=args.tau,
                    scale_schedule=scale_schedule,
                    g_seed=seed,
                    vae_type=32,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    cfg_insertion_layer=args.cfg_insertion_layer,
                    gt_leak=args.gt_leak,
                    gt_ls_Bl=None,
                    sampling_per_bits=1,
                )
            records = collect_capture(selected_modules)
            per_prompt_maps = {}
            for record in records:
                layer_id = record["layer_id"]
                scores_by_layer.setdefault(layer_id, []).append(
                    record["query_var_score"][0].numpy().tolist()
                )
                attn_mean_np = record["attn_mean"][0].numpy()
                query_var_np = record["query_var"][0].numpy()
                relative_np = record["relative_map"][0].numpy()
                mean_maps_by_layer.setdefault(layer_id, []).append(attn_mean_np)
                var_maps_by_layer.setdefault(layer_id, []).append(query_var_np)
                relative_maps_by_layer.setdefault(layer_id, []).append(relative_np)
                per_prompt_maps[layer_id] = {
                    "attn_mean": attn_mean_np,
                    "query_var": query_var_np,
                    "relative_map": relative_np,
                }
            if args.save_heatmaps:
                per_prompt_root = heatmap_root / prompt_tag
                per_prompt_root.mkdir(parents=True, exist_ok=True)
                save_generated_image(per_prompt_root / "generated.png", img)
                save_heatmap_set(
                    per_prompt_maps,
                    per_prompt_root,
                    final_scale_idx,
                    (final_start, final_end),
                    (final_h, final_w),
                )
                build_layer_grids(per_prompt_root, final_scale_idx)
                build_head_grids(per_prompt_root, final_scale_idx)
                build_generated_composites(
                    per_prompt_root, per_prompt_root / "generated.png"
                )
    finally:
        clear_capture(selected_modules)
        if patcher is not None:
            patcher.restore()

    aggregated = []
    global_scores = []
    for layer_id in sorted(scores_by_layer.keys()):
        arr = np.array(scores_by_layer[layer_id], dtype=np.float32)
        mean_scores = arr.mean(axis=0)
        std_scores = arr.std(axis=0)
        for head_idx, (mean_score, std_score) in enumerate(
            zip(mean_scores, std_scores)
        ):
            item = {
                "layer": layer_id,
                "head": head_idx,
                "mean_query_variance": float(mean_score),
                "std_query_variance": float(std_score),
            }
            aggregated.append(item)
            global_scores.append(mean_score)

    threshold = float(
        np.quantile(np.array(global_scores, dtype=np.float32), args.quantile)
    )
    for item in aggregated:
        item["label"] = (
            "structural" if item["mean_query_variance"] >= threshold else "contextual"
        )

    aggregated.sort(key=lambda x: x["mean_query_variance"], reverse=True)
    for rank, item in enumerate(aggregated, start=1):
        item["rank"] = rank
    output = {
        "condition": args.condition,
        "pn": args.pn,
        "h_div_w": args.h_div_w,
        "num_prompts": len(prompts),
        "final_scale_idx": final_scale_idx,
        "rope_scales": target_rope_scales,
        "layer_head_map": layer_head_map,
        "threshold_quantile": args.quantile,
        "threshold_value": threshold,
        "alpha_values": args.alpha,
        "generation": {
            "cfg": args.cfg,
            "tau": args.tau,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "cfg_insertion_layer": args.cfg_insertion_layer,
            "gt_leak": args.gt_leak,
            "prompt_seeds": prompt_seeds,
        },
        "results": aggregated,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Saved head classification to {output_path}")

    alpha_files = write_alpha_configs(output_path, aggregated, args.alpha)
    for alpha, struct_path, context_path, summary_path in alpha_files:
        print(
            f"Saved alpha={alpha:.2f} configs → {struct_path.name}, {context_path.name}, {summary_path.name}"
        )

    if args.save_heatmaps:
        aggregate_maps = {}
        for layer_id in sorted(mean_maps_by_layer.keys()):
            aggregate_maps[layer_id] = {
                "attn_mean": np.mean(
                    np.stack(mean_maps_by_layer[layer_id], axis=0), axis=0
                ),
                "query_var": np.mean(
                    np.stack(var_maps_by_layer[layer_id], axis=0), axis=0
                ),
                "relative_map": np.mean(
                    np.stack(relative_maps_by_layer[layer_id], axis=0), axis=0
                ),
            }
        save_heatmap_set(
            aggregate_maps,
            aggregate_root,
            final_scale_idx,
            (final_start, final_end),
            (final_h, final_w),
        )
        build_layer_grids(aggregate_root, final_scale_idx)
        build_head_grids(aggregate_root, final_scale_idx)
        print(f"Saved heatmaps to {heatmap_root}")


if __name__ == "__main__":
    main()
