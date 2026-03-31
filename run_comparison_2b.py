"""
2B + 1:2 비율 Baseline vs Spherical RoPE 비교 스크립트

실행법:
  cd /home/mmai6k_02/anaconda3/workspace/mnt/infinity
  PYTHONPATH=. conda run -n infinity python run_comparison_2b.py --interp_mode nearest

결과: eval_outputs/comparison_2b/ 에 이미지 저장 + metrics.json
"""

import argparse
import json, os, sys, time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import cv2
import numpy as np
import torch
import yaml
from torch.cuda.amp import autocast

from tools.run_infinity import (
    gen_one_img,
    load_tokenizer,
    load_visual_tokenizer,
    load_transformer,
)
from tools.layer_head_spec_utils import resolve_layer_head_map
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w
from spherical_rope_infinity import SphericalRoPEInfinityPatcher

# ── 설정 ─────────────────────────────────────────────────────────────────────
PRETRAINED = "pretrained"
OUT_DIR = "eval_outputs/comparison_2b_HL_a1"
H_DIV_W = 0.5  # 1:2 파노라마 비율
PN = "1M"  # 368×736px  (속도 확인용; 720×1440px 원하면 "1M"으로 변경)
VAE_TYPE = 32
os.makedirs(OUT_DIR, exist_ok=True)

PROMPTS = [
    "This is a panorama image. The photo shows a breathtaking snowy mountain summit at sunrise, with golden light illuminating the peaks and valleys stretching endlessly in all directions.",
    "This is a panorama image. The photo shows a spectacular fireworks display exploding in the night sky above a modern city skyline, with glittering lights reflecting off a calm river and bridges connecting both banks.",
    "This is a panorama image. The photo shows a tranquil tropical beach at sunset, with crystal-clear turquoise water, white sand, and palm trees.",
    "This is a panorama image. The photo shows the interior of a grand cathedral with soaring gothic arches, stained glass windows casting colorful light on the stone floor.",
    "This is a panorama image. The photo shows a quiet forest path in autumn, with golden and red leaves covering the ground.",
]
SEEDS = [0, 1234, 5536, 8650, 9902]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=str,
        default=OUT_DIR,
        help="Base output directory for images and metrics",
    )
    parser.add_argument(
        "--interp_mode",
        type=str,
        default="",
        choices=["", "nearest", "bilinear", "bicubic", "trilinear", "area"],
        help="Override next-scale latent upsampling mode",
    )
    parser.add_argument(
        "--interp_down_mode",
        type=str,
        default="",
        choices=["", "nearest", "bilinear", "bicubic", "trilinear", "area"],
        help="Optional downsampling override for VAE multiscale paths",
    )
    parser.add_argument(
        "--rope_scales",
        nargs="*",
        default=None,
        help="Scale indices to apply spherical RoPE to (default: all scales)",
    )
    parser.add_argument(
        "--head_split_ratio",
        type=float,
        default=float(os.environ.get("HACK_HEAD_SPLIT_RATIO", 0.5)),
        help="Head split ratio for spherical_split",
    )
    parser.add_argument(
        "--band_ratio",
        type=float,
        default=float(os.environ.get("HACK_BAND_RATIO", 0.5)),
        help="Band ratio for spherical_split",
    )
    parser.add_argument(
        "--all_head_band_ratio",
        type=float,
        default=float(os.environ.get("HACK_ALL_HEAD_BAND_RATIO", 0.5)),
        help="Band ratio for spherical_all",
    )
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
    return parser.parse_args()


def build_experiment_metadata(cli_args, target_rope_scales, layer_head_map):
    metadata = {
        "interp_mode": cli_args.interp_mode or "default",
        "interp_down_mode": cli_args.interp_down_mode or "default",
        "rope_scales": target_rope_scales if target_rope_scales is not None else "all",
        "head_split_ratio": cli_args.head_split_ratio,
        "band_ratio": cli_args.band_ratio,
        "all_head_band_ratio": cli_args.all_head_band_ratio,
        "layer_head_spec": cli_args.layer_head_spec or None,
        "layer_head_spec_file": cli_args.layer_head_spec_file or None,
        "layer_head_map": layer_head_map,
    }
    return metadata


def save_experiment_metadata(out_dir, metadata):
    yaml_path = Path(out_dir) / "experiment_config.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(metadata, f, sort_keys=False, allow_unicode=False)
    return str(yaml_path)


def resolve_rope_scales(scale_schedule, rope_scales):
    if not rope_scales:
        return None
    if len(rope_scales) == 1 and rope_scales[0].lower() == "all":
        return None
    total = len(scale_schedule)
    resolved = []
    for token in rope_scales:
        if token.lower() == "finest":
            resolved.append(total - 1)
        elif token.lower() == "coarsest":
            resolved.append(0)
        else:
            idx = int(token)
            if idx < 0 or idx >= total:
                raise ValueError(
                    f"rope scale index {idx} out of range (0..{total - 1})"
                )
            resolved.append(idx)
    return sorted(set(resolved))


def seam_ssim(img: np.ndarray, s: int = 32) -> float:
    """좌우 s픽셀 스트립 SSIM."""
    from skimage.metrics import structural_similarity as ssim

    left = img[:, :s, :].astype(np.float32)
    right = img[:, -s:, :].astype(np.float32)
    score, _ = ssim(left, right, full=True, channel_axis=2, data_range=255.0)
    return float(score)


def seam_cont(img: np.ndarray) -> float:
    """seam gradient / 내부 gradient 비율 (낮을수록 좋음)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    seam_grad = np.abs(gray[:, 0].astype(float) - gray[:, -1].astype(float)).mean()
    interior_grad = np.abs(np.diff(gray, axis=1))[:, 1:-1].mean()
    if interior_grad < 1e-6:
        return float("nan")
    return float(seam_grad / interior_grad)


def build_comparison_grid(
    all_results, conditions, cond_display_map, out_dir, interp_tag
):
    grouped = {
        (item["prompt_idx"], item["seed"], item["condition"]): item["save_path"]
        for item in all_results
    }
    display_order = [cond_display_map[name] for name, _ in conditions]
    rows = []
    for pi, seed in enumerate(SEEDS):
        tiles = []
        for display in display_order:
            path = grouped.get((pi, seed, display))
            if path is None:
                continue
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                continue
            label_h = 36
            tile = np.full(
                (img.shape[0] + label_h, img.shape[1], 3), 255, dtype=np.uint8
            )
            tile[label_h:, :, :] = img
            cv2.putText(
                tile,
                display,
                (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
            tiles.append(tile)
        if not tiles:
            continue
        row = cv2.hconcat(tiles)
        row_label_h = 34
        row_canvas = np.full(
            (row.shape[0] + row_label_h, row.shape[1], 3), 255, dtype=np.uint8
        )
        row_canvas[row_label_h:, :, :] = row
        cv2.putText(
            row_canvas,
            f"prompt {pi:02d}  seed {seed}",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        rows.append(row_canvas)
    if not rows:
        return None
    grid = cv2.vconcat(rows)
    grid_path = f"{out_dir}/{interp_tag}_comparison_grid.jpg"
    cv2.imwrite(grid_path, grid)
    return grid_path


def main():
    cli_args = parse_args()
    interp_tag = cli_args.interp_mode or "default"
    base_out_dir = cli_args.out_dir
    out_dir = (
        base_out_dir if not cli_args.interp_mode else f"{base_out_dir}_{interp_tag}"
    )
    os.makedirs(out_dir, exist_ok=True)

    # ── 모델 로딩 ─────────────────────────────────────────────────────────
    class VaeArgs:
        vae_type = VAE_TYPE
        vae_path = f"{PRETRAINED}/infinity_vae_d32reg.pth"
        apply_spatial_patchify = 0
        interp_mode = cli_args.interp_mode
        interp_down_mode = cli_args.interp_down_mode

    print("Loading T5...")
    text_tokenizer, text_encoder = load_tokenizer("google/flan-t5-xl")

    print("Loading VAE (d32reg)...")
    vae = load_visual_tokenizer(VaeArgs())

    print("Loading Infinity 2B...")

    class ModelArgs(VaeArgs):
        model_path = f"{PRETRAINED}/infinity_2b_reg.pth"
        checkpoint_type = "torch"
        model_type = "infinity_2b"
        rope2d_each_sa_layer = 1
        rope2d_normalized_by_hw = 2
        use_scale_schedule_embedding = 0
        pn = PN
        use_bit_label = 1
        add_lvl_embeding_only_first_block = 0
        text_channels = 2048
        use_flex_attn = 0
        bf16 = 1
        cache_dir = "/dev/shm"
        enable_model_cache = 0

    infinity = load_transformer(vae, ModelArgs())

    # ── scale_schedule (1:2, 1M) ──────────────────────────────────────────
    scale_schedule_raw = dynamic_resolution_h_w[H_DIV_W][PN]["scales"]
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule_raw]
    target_rope_scales = resolve_rope_scales(scale_schedule, cli_args.rope_scales)
    layer_head_map = resolve_layer_head_map(
        cli_args.layer_head_spec, cli_args.layer_head_spec_file
    )
    finest = scale_schedule[-1]
    print(
        f"Scale schedule: {len(scale_schedule)} scales, finest {finest[1] * 16}x{finest[2] * 16}px"
    )
    metadata_path = save_experiment_metadata(
        out_dir,
        build_experiment_metadata(cli_args, target_rope_scales, layer_head_map),
    )
    print(f"Saved → {metadata_path}")

    # ── Spherical RoPE Patcher ────────────────────────────────────────────
    PATCHER_SETTINGS = {
        "spherical_all": dict(
            head_split_ratio=None, band_ratio=cli_args.all_head_band_ratio
        ),
        "spherical_split": dict(
            head_split_ratio=None
            if layer_head_map is not None
            else cli_args.head_split_ratio,
            band_ratio=cli_args.band_ratio,
            layer_head_map=layer_head_map,
        ),
    }
    patchers = {
        name: SphericalRoPEInfinityPatcher(
            infinity,
            alpha_w=1.0,
            alpha_h=0.0,
            head_split_ratio=settings["head_split_ratio"],
            spherical_band_ratio=settings.get("band_ratio", 1.0),
            target_scales=target_rope_scales,
            layer_head_map=settings.get("layer_head_map"),
        )
        for name, settings in PATCHER_SETTINGS.items()
    }

    # ── 생성 & 평가 루프 ──────────────────────────────────────────────────
    all_results = []

    CONDITIONS = [
        ("baseline", None),
        ("spherical_all", PATCHER_SETTINGS["spherical_all"]),
        ("spherical_split", PATCHER_SETTINGS["spherical_split"]),
    ]

    cond_display_map = {}
    active_patcher = None
    for cond_name, cond_cfg in CONDITIONS:
        if cond_cfg and cond_cfg.get("layer_head_map") is not None:
            cond_tag = f"{cond_name}_custom_r{cond_cfg['band_ratio']:.2f}"
        elif cond_cfg and cond_cfg.get("head_split_ratio") is not None:
            cond_tag = f"{cond_name}_r{cond_cfg['head_split_ratio']:.2f}_r{cond_cfg['band_ratio']:.2f}"
        else:
            cond_tag = cond_name
        cond_display_map[cond_name] = cond_tag
        if active_patcher is not None:
            active_patcher.restore()
            active_patcher = None

        if cond_cfg is None:
            pass
        else:
            active_patcher = patchers[cond_name]
            active_patcher.apply()

        for pi, (prompt, seed) in enumerate(zip(PROMPTS, SEEDS)):
            print(f"\n[{cond_tag}] prompt {pi + 1}/{len(PROMPTS)}, seed={seed}")

            t0 = time.time()
            with autocast(dtype=torch.bfloat16):
                with torch.no_grad():
                    img = gen_one_img(
                        infinity,
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
                        vae_type=VAE_TYPE,
                        sampling_per_bits=1,
                    )
            elapsed = time.time() - t0

            img_np = img.cpu().numpy()
            h, w = img_np.shape[:2]
            save_path = f"{out_dir}/{interp_tag}_{cond_tag}_p{pi:02d}_s{seed}.jpg"
            cv2.imwrite(save_path, img_np)

            ssim_val = seam_ssim(img_np)
            cont_val = seam_cont(img_np)

            result = {
                "condition": cond_tag,
                "prompt_idx": pi,
                "seed": seed,
                "resolution": f"{w}x{h}",
                "elapsed": round(elapsed, 2),
                "seam_ssim": round(ssim_val, 4),
                "seam_cont": round(cont_val, 4) if not np.isnan(cont_val) else None,
                "interp_mode": interp_tag,
                "save_path": save_path,
            }
            all_results.append(result)
            print(
                f"  {w}x{h}px  seam_ssim={ssim_val:.4f}  seam_cont={cont_val:.4f}  {elapsed:.1f}s"
            )

    if active_patcher is not None:
        active_patcher.restore()

    # ── 요약 ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    for cond in [name for name, _ in CONDITIONS]:
        display = cond_display_map[cond]
        res = [r for r in all_results if r["condition"] == display]
        mean_ssim = np.mean([r["seam_ssim"] for r in res])
        mean_cont = np.nanmean(
            [r["seam_cont"] for r in res if r["seam_cont"] is not None]
        )
        print(f"{display:<22}  seam_ssim={mean_ssim:.4f}  seam_cont={mean_cont:.4f}")

    out_path = f"{out_dir}/metrics.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {out_path}")

    grid_path = build_comparison_grid(
        all_results, CONDITIONS, cond_display_map, out_dir, interp_tag
    )
    if grid_path is not None:
        print(f"Saved → {grid_path}")


if __name__ == "__main__":
    main()
