"""
Baseline vs Spherical RoPE 빠른 비교 스크립트

실행법:
  cd /home/mmai6k_02/anaconda3/workspace/mnt/infinity
  PYTHONPATH=. conda run -n infinity python run_comparison.py

결과: eval_outputs/comparison/ 에 이미지 저장 + metrics.json
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
    add_common_arguments,
)
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w
from spherical_rope_infinity import SphericalRoPEInfinityPatcher

# ── 설정 ──────────────────────────────────────────────────────────────────────
PRETRAINED = "pretrained"
OUT_DIR    = "eval_outputs/comparison"
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_KWARGS = dict(
    depth=12, embed_dim=768, num_heads=8,
    drop_path_rate=0.1, mlp_ratio=4, block_chunks=4,
)

PROMPTS = [
    "This is a panorama image. The photo shows a breathtaking snowy mountain summit at sunrise, with golden light illuminating the peaks and valleys stretching endlessly in all directions.",
    "This is a panorama image. The photo shows a modern city skyline at night, with glittering lights reflecting off a calm river and bridges connecting both banks.",
    "This is a panorama image. The photo shows a tranquil tropical beach at sunset, with crystal-clear turquoise water, white sand, and palm trees.",
    "This is a panorama image. The photo shows the interior of a grand cathedral with soaring gothic arches, stained glass windows casting colorful light on the stone floor.",
    "This is a panorama image. The photo shows a quiet forest path in autumn, with golden and red leaves covering the ground.",
]
SEEDS = [0, 1234, 5536, 8650, 9902]


def seam_ssim(img: np.ndarray, s: int = 32) -> float:
    from skimage.metrics import structural_similarity as ssim
    left  = img[:, :s,  :].astype(np.float32)
    right = img[:, -s:, :].astype(np.float32)
    score, _ = ssim(left, right, full=True, channel_axis=2, data_range=255.0)
    return float(score)


def seam_cont(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    seam_grad = np.abs(gray[:, 0].astype(float) - gray[:, -1].astype(float)).mean()
    dx = np.abs(np.diff(gray, axis=1))
    interior_grad = dx[:, 1:-1].mean()
    if interior_grad < 1e-6:
        return float('nan')
    return float(seam_grad / interior_grad)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── 모델 로딩 ─────────────────────────────────────────────────────────
    class Args:
        vae_type = 16
        vae_path = f"{PRETRAINED}/infinity_vae_d16.pth"
        apply_spatial_patchify = 0

    print("Loading T5...")
    text_tokenizer, text_encoder = load_tokenizer("google/flan-t5-xl")

    print("Loading VAE...")
    vae = load_visual_tokenizer(Args())

    print("Loading Infinity 125M...")

    class Args2(Args):
        model_path = f"{PRETRAINED}/infinity_125M_256x256.pth"
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

    infinity = load_transformer(vae, Args2())

    # scale_schedule
    scale_schedule = dynamic_resolution_h_w[1.0]['0.06M']['scales']
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]

    # ── Spherical RoPE Patcher ─────────────────────────────────────────────
    patcher = SphericalRoPEInfinityPatcher(infinity, alpha_w=1.0)

    # ── 생성 루프 ─────────────────────────────────────────────────────────
    all_results = []

    for cond_name, use_sph in [("baseline", False), ("spherical_aw1.0", True)]:
        if use_sph:
            patcher.apply()
        else:
            patcher.restore()

        for pi, (prompt, seed) in enumerate(zip(PROMPTS, SEEDS)):
            print(f"\n[{cond_name}] prompt {pi+1}/{len(PROMPTS)}, seed={seed}")

            t0 = time.time()
            with autocast(dtype=torch.bfloat16):
                with torch.no_grad():
                    img = gen_one_img(
                        infinity, vae, text_tokenizer, text_encoder,
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
            elapsed = time.time() - t0

            img_np = img.cpu().numpy()
            save_path = f"{OUT_DIR}/{cond_name}_p{pi:02d}_s{seed}.jpg"
            cv2.imwrite(save_path, img_np)

            ssim_val = seam_ssim(img_np)
            cont_val = seam_cont(img_np)

            result = {
                "condition": cond_name,
                "prompt_idx": pi,
                "seed": seed,
                "elapsed": round(elapsed, 2),
                "seam_ssim": round(ssim_val, 4),
                "seam_cont": round(cont_val, 4) if not np.isnan(cont_val) else None,
                "save_path": save_path,
            }
            all_results.append(result)
            print(f"  seam_ssim={ssim_val:.4f}  seam_cont={cont_val:.4f}  time={elapsed:.1f}s")

    patcher.restore()

    # ── 요약 ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    for cond in ["baseline", "spherical_aw1.0"]:
        res = [r for r in all_results if r["condition"] == cond]
        mean_ssim = np.mean([r["seam_ssim"] for r in res])
        mean_cont = np.nanmean([r["seam_cont"] for r in res if r["seam_cont"] is not None])
        print(f"{cond:<20}  seam_ssim={mean_ssim:.4f}  seam_cont={mean_cont:.4f}")

    out_path = f"{OUT_DIR}/metrics.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
