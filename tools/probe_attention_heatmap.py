"""
Capture self-attention heatmaps for selected layers/heads of Infinity.

Usage example:
    python tools/probe_attention_heatmap.py \
        --condition baseline \
        --prompt "This is a panorama image..." \
        --output_dir probe_outputs/attn_heatmaps

This script runs a single inference with classifier-free guidance disabled
(`cfg_list=1.0`) so that B=1, captures the self-attention tensors of the
selected layers at the finest scale, averages attention over all query tokens,
and saves the resulting key-distribution heatmaps as PNG images.

Only heatmaps are written (no rendered images).
"""

import argparse
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from infinity.utils.dynamic_resolution import dynamic_resolution_h_w
import infinity.models.basic as basic_module
from spherical_rope_infinity import SphericalRoPEInfinityPatcher
from tools.run_infinity import (
    gen_one_img,
    load_tokenizer,
    load_visual_tokenizer,
    load_transformer,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Infinity attention heatmap probe")
    parser.add_argument(
        "--condition",
        default="baseline",
        choices=["baseline", "spherical_all", "spherical_split"],
        help="Which RoPE setting to run with",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "This is a panorama image. The photo shows a breathtaking snowy "
            "mountain summit at sunrise, with golden light illuminating the peaks."
        ),
        help="Prompt used for probing",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for CFG sampling"
    )
    parser.add_argument(
        "--pn",
        default="1M",
        choices=list(dynamic_resolution_h_w[0.5].keys()),
        help="Resolution bucket (pn)",
    )
    parser.add_argument(
        "--h_div_w",
        type=float,
        default=0.5,
        choices=sorted(dynamic_resolution_h_w.keys()),
        help="Aspect ratio template",
    )
    parser.add_argument(
        "--layers",
        nargs="*",
        default=["-4", "-3", "-2", "-1"],
        help="Layer indices to capture (use integers or 'all')",
    )
    parser.add_argument(
        "--max_heads",
        type=int,
        default=None,
        help="Limit number of heads per layer to visualize",
    )
    parser.add_argument(
        "--scale",
        default="finest",
        help="Single scale index to capture (int or 'finest')",
    )
    parser.add_argument(
        "--scales",
        nargs="*",
        default=None,
        help="Multiple scale tokens (ints, 'finest', 'coarsest', 'all')",
    )
    parser.add_argument(
        "--scale_range",
        nargs=2,
        type=int,
        default=None,
        metavar=("START", "END"),
        help="Inclusive range of scale indices",
    )
    parser.add_argument(
        "--output_dir",
        default="probe_outputs/attn_heatmaps",
        help="Directory to store PNGs",
    )
    parser.add_argument(
        "--head_split_ratio",
        type=float,
        default=0.25,
        help="Head split ratio when condition requires it",
    )
    parser.add_argument(
        "--band_ratio",
        type=float,
        default=0.5,
        help="Spherical band ratio for non-baseline conditions",
    )
    parser.add_argument(
        "--rope_scales",
        nargs="*",
        default=None,
        help="Scale indices to apply spherical RoPE to (default: all scales)",
    )
    return parser.parse_args()


def resolve_scale_index(scale_schedule, arg_scale):
    if arg_scale == "finest":
        return len(scale_schedule) - 1
    try:
        idx = int(arg_scale)
    except ValueError as exc:
        raise ValueError(f"Invalid --scale value: {arg_scale}") from exc
    if idx < 0 or idx >= len(scale_schedule):
        raise ValueError(
            f"Scale index {idx} out of range (0..{len(scale_schedule) - 1})"
        )
    return idx


def resolve_scale_token(token: str, schedule_len: int) -> int:
    lowered = token.lower()
    if lowered == "finest":
        return schedule_len - 1
    if lowered == "coarsest":
        return 0
    if lowered == "all":
        raise ValueError("'all' is only valid when used alone in --scales")
    try:
        idx = int(token)
    except ValueError as exc:
        raise ValueError(f"Invalid scale specifier: {token}") from exc
    if idx < 0 or idx >= schedule_len:
        raise ValueError(f"Scale index {idx} out of range (0..{schedule_len - 1})")
    return idx


def build_target_scales(args, scale_schedule):
    total = len(scale_schedule)
    if args.scales:
        if len(args.scales) == 1 and args.scales[0].lower() == "all":
            return list(range(total))
        return sorted({resolve_scale_token(tok, total) for tok in args.scales})
    if args.scale_range:
        start, end = args.scale_range
        if start < 0 or end < 0 or start >= total or end >= total:
            raise ValueError(
                f"Scale range {start}-{end} out of bounds (0..{total - 1})"
            )
        if start > end:
            start, end = end, start
        return list(range(start, end + 1))
    return [resolve_scale_index(scale_schedule, args.scale)]


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


def select_modules(modules, layer_indices):
    total = len(modules)
    selected = []
    for idx in layer_indices:
        actual = idx if idx >= 0 else total + idx
        if actual < 0 or actual >= total:
            raise ValueError(f"Layer index {idx} is out of range for {total} layers")
        mod = modules[actual]
        mod._capture_id = f"L{actual:02d}"
        selected.append(mod)
    return selected


def build_capture_hook(target_modules, target_scales):
    capture_entries = []

    original_apply_rope = basic_module.SelfAttention._apply_rope
    capture_enabled = {"flag": False}
    target_set = set(target_modules)
    target_scale_set = set(target_scales)

    def patched_apply_rope(self, q, k, scale_schedule, rope2d_freqs_grid, scale_ind):
        q_out, k_out = original_apply_rope(
            self, q, k, scale_schedule, rope2d_freqs_grid, scale_ind
        )
        if (
            capture_enabled["flag"]
            and self in target_set
            and scale_ind in target_scale_set
        ):
            capture_entries.append(
                {
                    "layer_id": getattr(self, "_capture_id", "unknown"),
                    "scale_ind": scale_ind,
                    "q": q_out.detach().cpu(),
                    "k": k_out.detach().cpu(),
                }
            )
        return q_out, k_out

    def enable():
        capture_enabled["flag"] = True

    def disable():
        capture_enabled["flag"] = False

    basic_module.SelfAttention._apply_rope = patched_apply_rope
    return capture_entries, original_apply_rope, enable, disable


def restore_apply_rope(original_fn):
    basic_module.SelfAttention._apply_rope = original_fn


def attention_heatmap_from_qk(q, k, ph, pw, max_heads=None):
    """Compute averaged attention heatmaps for each head."""
    # q, k: tensors on CPU (H, L, head_dim)
    q = q.float()
    k = k.float()
    num_heads, L, head_dim = q.shape
    assert L == ph * pw, "Sequence length does not match spatial dims"
    head_range = (
        range(num_heads) if max_heads is None else range(min(max_heads, num_heads))
    )
    maps = []
    for h in head_range:
        scores = torch.matmul(q[h], k[h].transpose(0, 1))
        attn = torch.softmax(scores, dim=-1)
        avg = attn.mean(dim=0)  # average over queries
        grid = avg.reshape(ph, pw)
        maps.append((h, grid.numpy()))
    return maps


def save_heatmap(grid, out_path, title=None):
    plt.figure(figsize=(4, 3))
    plt.imshow(grid, cmap="magma")
    plt.colorbar(fraction=0.046, pad=0.04)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    args = parse_args()
    output_root = Path(args.output_dir) / args.condition
    output_root.mkdir(parents=True, exist_ok=True)

    infinity, vae, tokenizer, text_encoder = prepare_model(args)

    # Prepare scale schedule
    scale_schedule = dynamic_resolution_h_w[args.h_div_w][args.pn]["scales"]
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
    target_scales = build_target_scales(args, scale_schedule)
    target_rope_scales = resolve_rope_scales(args.rope_scales, scale_schedule)
    scale_dims = {
        idx: (scale_schedule[idx][1], scale_schedule[idx][2])
        for idx in range(len(scale_schedule))
    }

    # Apply RoPE patcher if needed
    patcher = None
    if args.condition != "baseline":
        patcher = SphericalRoPEInfinityPatcher(
            infinity,
            alpha_w=1.0,
            alpha_h=0.0,
            head_split_ratio=(
                None if args.condition == "spherical_all" else args.head_split_ratio
            ),
            spherical_band_ratio=args.band_ratio,
            target_scales=target_rope_scales,
        )
        patcher.apply()

    # Select target layers
    sa_modules = gather_self_attention_modules(infinity)
    if len(args.layers) == 1 and args.layers[0].lower() == "all":
        layer_indices = list(range(len(sa_modules)))
    else:
        try:
            layer_indices = [int(val) for val in args.layers]
        except ValueError as exc:
            raise ValueError(
                f"Layers must be integers or 'all', got: {args.layers}"
            ) from exc
    selected_modules = select_modules(sa_modules, layer_indices)

    # Build capture hook
    capture_entries, original_apply_rope, enable_capture, disable_capture = (
        build_capture_hook(selected_modules, target_scales)
    )

    text_prompt = args.prompt
    cfg_value = 1.0  # disable CFG to keep B=1 for manageable tensors

    try:
        enable_capture()
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            _ = gen_one_img(
                infinity,
                vae,
                tokenizer,
                text_encoder,
                text_prompt,
                cfg_list=cfg_value,
                tau_list=1.0,
                scale_schedule=scale_schedule,
                g_seed=args.seed,
                vae_type=32,
                top_k=900,
                top_p=0.97,
                cfg_insertion_layer=[-5],
                sampling_per_bits=1,
            )
    finally:
        disable_capture()
        restore_apply_rope(original_apply_rope)
        if patcher is not None:
            patcher.restore()

    if not capture_entries:
        print("No attention tensors captured. Check layer indices and scale selection.")
        return

    # Aggregate by layer id (keep the last capture per layer)
    latest_by_layer_scale = {}
    for entry in capture_entries:
        key = (entry["layer_id"], entry["scale_ind"])
        latest_by_layer_scale[key] = entry

    print(
        f"Captured {len(latest_by_layer_scale)} layer-scale tensors. Generating heatmaps..."
    )

    for (layer_id, scale_idx), entry in sorted(latest_by_layer_scale.items()):
        q = entry["q"][0]  # (H, L, head_dim)
        k = entry["k"][0]
        ph, pw = scale_dims[scale_idx]
        heatmaps = attention_heatmap_from_qk(q, k, ph, pw, args.max_heads)
        layer_dir = output_root / layer_id / f"scale{scale_idx:02d}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        for head_idx, grid in heatmaps:
            out_path = layer_dir / f"head{head_idx:02d}.png"
            title = f"{layer_id} scale {scale_idx} head {head_idx:02d}"
            save_heatmap(grid, out_path, title=title)
    print(f"Saved heatmaps to {output_root}")


if __name__ == "__main__":
    main()
