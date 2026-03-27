"""
Improved RoPE Comparison: Baseline vs SphericalV1 (Global) vs SphericalV2 (Scale-Adaptive)

Purpose:
  Compare three approaches to seam continuity in panoramic ERP generation:
  1. Baseline: Standard RoPE
  2. Spherical v1 (global): All scales use upw-based spherical freq (alpha_w=1.0)
  3. Spherical v2 (scale-adaptive): Coarse→fine interpolation of alpha (0→1)

Usage:
  cd /home/mmai6k_02/anaconda3/workspace/mnt/infinity
  conda activate infinity
  python ../run_comparison_v2.py

  OR:
  cd /home/mmai6k_02/anaconda3/workspace/mnt/infinity
  conda run -n infinity python ../run_comparison_v2.py

Output:
  - eval_outputs/comparison_v2/metrics.json (detailed per-sample metrics)
  - eval_outputs/comparison_v2/summary.json (per-condition statistics)
  - eval_outputs/comparison_v2/*.jpg (generated images)
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import cv2
import numpy as np
import torch
from torch.cuda.amp import autocast

from tools.run_infinity import (
    gen_one_img, load_tokenizer, load_visual_tokenizer, load_transformer,
)
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w

# Import spherical RoPE patchers
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spherical_rope_infinity import SphericalRoPEInfinityPatcher
from spherical_rope_infinity_v2 import ScaleAwareSphericalRoPEInfinityPatcher

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

# Handle path: script may run from either mnt/ or mnt/infinity/
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MNT_DIR = _SCRIPT_DIR if os.path.basename(_SCRIPT_DIR) == "mnt" else os.path.dirname(_SCRIPT_DIR)

PRETRAINED = os.path.join(_MNT_DIR, "pretrained")
OUT_DIR = os.path.join(_MNT_DIR, "eval_outputs", "comparison_v2")
os.makedirs(OUT_DIR, exist_ok=True)

PROMPTS = [
    "This is a panorama image. The photo shows a breathtaking snowy mountain summit at sunrise, with golden light illuminating the peaks and valleys stretching endlessly in all directions.",
    "This is a panorama image. The photo shows a modern city skyline at night, with glittering lights reflecting off a calm river and bridges connecting both banks.",
    "This is a panorama image. The photo shows a tranquil tropical beach at sunset, with crystal-clear turquoise water, white sand, and palm trees.",
    "This is a panorama image. The photo shows the interior of a grand cathedral with soaring gothic arches, stained glass windows casting colorful light on the stone floor.",
    "This is a panorama image. The photo shows a quiet forest path in autumn, with golden and red leaves covering the ground.",
]
SEEDS = [0, 1234, 5536, 8650, 9902]

# Conditions to test
CONDITIONS = [
    ("baseline", None),
    ("spherical_v1_aw1.0", ("v1", 1.0)),
    ("spherical_v2_linear", ("v2_linear", None)),
    ("spherical_v2_exp", ("v2_exp", None)),
]


def seam_ssim(img: np.ndarray, s: int = 32) -> float:
    """Seam SSIM: structural similarity of left & right s-pixel strips."""
    from skimage.metrics import structural_similarity as ssim
    left = img[:, :s, :].astype(np.float32)
    right = img[:, -s:, :].astype(np.float32)
    score, _ = ssim(left, right, full=True, channel_axis=2, data_range=255.0)
    return float(score)


def seam_cont(img: np.ndarray) -> float:
    """Seam continuity: ratio of seam gradient to interior gradient."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    seam_grad = np.abs(gray[:, 0].astype(float) - gray[:, -1].astype(float)).mean()
    dx = np.abs(np.diff(gray, axis=1))
    interior_grad = dx[:, 1:-1].mean()
    if interior_grad < 1e-6:
        return float('nan')
    return float(seam_grad / interior_grad)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ──────────────────────────────────────────────────────────────────────────
    # MODEL LOADING
    # ──────────────────────────────────────────────────────────────────────────
    print("\nLoading T5...")
    text_tokenizer, text_encoder = load_tokenizer("google/flan-t5-xl")

    print("Loading VAE...")

    class Args:
        vae_type = 16
        vae_path = os.path.join(PRETRAINED, "infinity_vae_d16.pth")
        apply_spatial_patchify = 0

    vae = load_visual_tokenizer(Args())

    print("Loading Infinity 125M...")

    class Args2(Args):
        model_path = os.path.join(PRETRAINED, "infinity_125M_256x256.pth")
        checkpoint_type = "torch"
        model_type = "infinity_layer12"
        rope2d_each_sa_layer = 1
        rope2d_normalized_by_hw = 2
        use_scale_schedule_embedding = 0
        pn = "0.06M"
        use_bit_label = 1
        add_lvl_embeding_only_first_block = 0
        text_channels = 2048
        use_flex_attn = 0
        bf16 = 1
        cache_dir = "/dev/shm"
        enable_model_cache = 0

    model = load_transformer(vae, Args2())

    # Scale schedule for generation
    scale_schedule = dynamic_resolution_h_w[1.0]["0.06M"]["scales"]
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]

    print(f"Scale schedule: {scale_schedule}")
    print(f"Output directory: {OUT_DIR}\n")

    # ──────────────────────────────────────────────────────────────────────────
    # GENERATION & EVALUATION LOOP
    # ──────────────────────────────────────────────────────────────────────────
    metrics = []
    patchers = {}  # Cache patchers

    for cond_name, cond_spec in CONDITIONS:
        print(f"\n{'='*70}")
        print(f"Condition: {cond_name}")
        print(f"{'='*70}")

        # Setup patcher if needed
        patcher = None
        if cond_spec is not None:
            version, param = cond_spec
            if (version, param) not in patchers:
                if version == "v1":
                    patchers[(version, param)] = SphericalRoPEInfinityPatcher(
                        model, alpha_w=param
                    )
                elif version.startswith("v2"):
                    sched = version.split("_")[1]
                    patchers[(version, param)] = ScaleAwareSphericalRoPEInfinityPatcher(
                        model, alpha_schedule=sched
                    )
            patcher = patchers[(version, param)]

        # Apply patcher
        if patcher is not None:
            patcher.apply()
        else:
            # Restore baseline
            if patchers:
                for p in patchers.values():
                    p.restore()

        # Generation loop
        for p_idx, (prompt, seed) in enumerate(zip(PROMPTS, SEEDS)):
            print(f"  [{p_idx+1}/{len(PROMPTS)}] seed={seed}, prompt='{prompt[:50]}...'")

            start_time = time.time()
            try:
                with autocast(dtype=torch.bfloat16):
                    with torch.no_grad():
                        img = gen_one_img(
                            model,
                            vae,
                            text_tokenizer,
                            text_encoder,
                            prompt,
                            g_seed=seed,
                            gt_leak=0,
                            gt_ls_Bl=None,
                            cfg_list=3.0,
                            tau_list=1.0,
                            scale_schedule=scale_schedule,
                            cfg_insertion_layer=[0],
                            vae_type=16,
                            sampling_per_bits=1,
                        )
            except Exception as e:
                print(f"    ERROR: {e}")
                continue

            elapsed = time.time() - start_time
            img_np = img.cpu().numpy()

            # Compute metrics
            ssim_score = seam_ssim(img_np)
            cont_score = seam_cont(img_np)

            # Save image
            save_name = f"{cond_name}_p{p_idx:02d}_s{seed}.jpg"
            save_path = os.path.join(OUT_DIR, save_name)
            cv2.imwrite(save_path, img_np)

            metric_item = {
                "condition": cond_name,
                "prompt_idx": p_idx,
                "seed": seed,
                "elapsed": round(elapsed, 2),
                "seam_ssim": round(ssim_score, 4),
                "seam_cont": round(cont_score, 4),
                "save_path": save_path,
            }
            metrics.append(metric_item)

            print(f"    SSIM={ssim_score:.4f} | CONT={cont_score:.4f} | "
                  f"time={elapsed:.2f}s")

    # ──────────────────────────────────────────────────────────────────────────
    # SAVE RESULTS
    # ──────────────────────────────────────────────────────────────────────────
    metrics_path = os.path.join(OUT_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✅ Saved metrics: {metrics_path}")

    # Compute summary statistics
    summary = {}
    for cond_name, _ in CONDITIONS:
        cond_metrics = [m for m in metrics if m["condition"] == cond_name]
        if cond_metrics:
            ssim_vals = [m["seam_ssim"] for m in cond_metrics if not np.isnan(m["seam_ssim"])]
            cont_vals = [m["seam_cont"] for m in cond_metrics if not np.isnan(m["seam_cont"])]

            if ssim_vals:
                summary[cond_name] = {
                    "seam_ssim_mean": round(np.mean(ssim_vals), 4),
                    "seam_ssim_std": round(np.std(ssim_vals), 4),
                    "seam_ssim_max": round(np.max(ssim_vals), 4),
                    "seam_ssim_min": round(np.min(ssim_vals), 4),
                }
            if cont_vals:
                summary[cond_name]["seam_cont_mean"] = round(np.mean(cont_vals), 4)
                summary[cond_name]["seam_cont_std"] = round(np.std(cont_vals), 4)

    summary_path = os.path.join(OUT_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Saved summary: {summary_path}")

    # Print summary table
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}")
    print(f"{'Condition':<30} | {'SSIM Mean±Std':<20} | {'CONT Mean±Std':<20}")
    print("-" * 70)
    for cond_name, _ in CONDITIONS:
        if cond_name in summary:
            s = summary[cond_name]
            ssim_str = f"{s['seam_ssim_mean']:.4f}±{s['seam_ssim_std']:.4f}"
            cont_str = f"{s['seam_cont_mean']:.4f}±{s['seam_cont_std']:.4f}"
            print(f"{cond_name:<30} | {ssim_str:<20} | {cont_str:<20}")

    print(f"\n✅ All done! Results in: {OUT_DIR}")


if __name__ == "__main__":
    main()
