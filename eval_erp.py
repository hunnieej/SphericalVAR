"""
Step 6: ERP (Equirectangular Projection) 파노라마 평가 파이프라인

평가 메트릭:
  - seam_ssim:     좌우 32px 스트립 SSIM (높을수록 seam 연속)
  - seam_cont:     seam gradient / 내부 gradient 비율 (낮을수록 좋음)
  - cubemap_clip:  ERP → 6 cubemap face, CLIP per face

실행법:
  conda activate infinity
  cd /home/mmai6k_02/anaconda3/workspace/mnt/infinity
  python eval_erp.py \
    --model_path pretrained/infinity_125M_256x256.pth \
    --vae_path   pretrained/infinity_vae_d32reg.pth \
    --t5_path    google/flan-t5-xl \
    --pn         0.06M \
    --model_type infinity_layer12 \
    --out_dir    eval_outputs/erp_baseline

  # Spherical RoPE 비교:
  python eval_erp.py ... --use_spherical_rope 1
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.cuda.amp import autocast

# ─────────────────────────────────────────────────────────────────────────────
# 평가 프롬프트 & 시드
# ─────────────────────────────────────────────────────────────────────────────

PROMPTS = [
    "This is a panorama image. The photo shows a breathtaking snowy mountain summit at sunrise, with golden light illuminating the peaks and valleys stretching endlessly in all directions.",
    "This is a panorama image. The photo shows a modern city skyline at night, with glittering lights reflecting off a calm river and bridges connecting both banks.",
    "This is a panorama image. The photo shows a tranquil tropical beach at sunset, with crystal-clear turquoise water, white sand, and palm trees swaying in the breeze.",
    "This is a panorama image. The photo shows the interior of a grand cathedral with soaring gothic arches, stained glass windows casting colorful light on the stone floor.",
    "This is a panorama image. The photo shows a quiet forest path in autumn, with golden and red leaves covering the ground and sunlight filtering through the canopy.",
]
SEEDS = [0, 1234, 5536, 8650, 9902]


# ─────────────────────────────────────────────────────────────────────────────
# 메트릭: seam_ssim
# ─────────────────────────────────────────────────────────────────────────────

def seam_ssim(img: np.ndarray, s: int = 32) -> float:
    """
    좌우 s픽셀 스트립의 SSIM.
    img: HxWx3 uint8
    """
    from skimage.metrics import structural_similarity as ssim
    left  = img[:, :s,  :].astype(np.float32)
    right = img[:, -s:, :].astype(np.float32)
    score, _ = ssim(left, right, full=True, channel_axis=2, data_range=255.0)
    return float(score)


def seam_cont(img: np.ndarray) -> float:
    """
    seam gradient / 내부 gradient 비율.
    낮을수록 seam이 내부와 구별되지 않음 → 좋음.
    img: HxWx3 uint8
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    H, W = gray.shape

    # seam 열 간 gradient (마지막 열 → 첫 열)
    seam_grad = np.abs(gray[:, 0].astype(float) - gray[:, -1].astype(float)).mean()

    # 내부 horizontal gradient (seam 제외)
    dx = np.abs(np.diff(gray, axis=1))       # HxW-1
    interior_grad = dx[:, 1:-1].mean()        # seam 근처 1px 제외

    if interior_grad < 1e-6:
        return float('nan')
    return float(seam_grad / interior_grad)


# ─────────────────────────────────────────────────────────────────────────────
# 메트릭: cubemap_clip
# ─────────────────────────────────────────────────────────────────────────────

def erp_to_cubemap_faces(img: np.ndarray, face_size: int = 224):
    """
    ERP(H×W×3) → 6 cubemap faces (각 face_size × face_size × 3).
    Returns: dict with keys 'F','R','B','L','U','D'
    """
    from PIL import Image as PILImage
    try:
        import equilib
        pil = PILImage.fromarray(img)
        # equilib 사용 시
        # (간단한 대체: py360convert)
    except ImportError:
        pass

    # py360convert 사용 (더 흔함)
    try:
        import py360convert
        # HxWx3 float [0,1] 필요
        img_f = img.astype(np.float32) / 255.0
        # e2c: erp to cubemap (dict of faces)
        cube = py360convert.e2c(img_f, face_w=face_size, mode='bilinear', cube_format='dict')
        # cube: {'F':..,'R':..,'B':..,'L':..,'U':..,'D':..}
        faces = {k: (v * 255).clip(0, 255).astype(np.uint8) for k, v in cube.items()}
        return faces
    except ImportError:
        # Fallback: 단순 분할 (equatorial 4-face)
        H, W, _ = img.shape
        face_w = W // 4
        faces = {}
        for i, name in enumerate(['F', 'R', 'B', 'L']):
            faces[name] = cv2.resize(img[:H//2, i*face_w:(i+1)*face_w], (face_size, face_size))
        faces['U'] = cv2.resize(img[:H//4, :], (face_size, face_size))
        faces['D'] = cv2.resize(img[3*H//4:, :], (face_size, face_size))
        return faces


def cubemap_clip(img: np.ndarray, prompt: str, clip_model=None, clip_processor=None, face_size: int = 224) -> dict:
    """
    ERP → cubemap faces → CLIP score per face.

    Returns:
        dict with 'clip_equatorial' (F/R/B/L mean), 'clip_polar' (U/D mean), 'clip_all'
    """
    if clip_model is None:
        return {"clip_equatorial": None, "clip_polar": None, "clip_all": None}

    faces = erp_to_cubemap_faces(img, face_size=face_size)

    from PIL import Image as PILImage
    import torch

    scores = {}
    for name, face_arr in faces.items():
        pil_face = PILImage.fromarray(face_arr)
        inputs = clip_processor(
            text=[prompt], images=[pil_face], return_tensors="pt", padding=True
        ).to(clip_model.device)
        with torch.no_grad():
            out = clip_model(**inputs)
        scores[name] = float(out.logits_per_image.item())

    equatorial = np.mean([scores[k] for k in ['F', 'R', 'B', 'L']])
    polar      = np.mean([scores[k] for k in ['U', 'D']])
    all_faces  = np.mean(list(scores.values()))

    return {
        "clip_per_face":    scores,
        "clip_equatorial":  float(equatorial),
        "clip_polar":       float(polar),
        "clip_all":         float(all_faces),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 메인 평가 루프
# ─────────────────────────────────────────────────────────────────────────────

def run_eval(args):
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from tools.run_infinity import (
        load_tokenizer, load_visual_tokenizer, load_transformer, gen_one_img
    )
    from infinity.utils.dynamic_resolution import dynamic_resolution_h_w

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 모델 로딩 ─────────────────────────────────────────────────────────
    print("Loading T5...")
    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.t5_path)

    print("Loading VAE...")
    vae = load_visual_tokenizer(args)

    print("Loading Infinity transformer...")
    infinity = load_transformer(vae, args)

    # ── Spherical RoPE 적용 ───────────────────────────────────────────────
    if args.use_spherical_rope:
        from spherical_rope_infinity import SphericalRoPEInfinityPatcher
        patcher = SphericalRoPEInfinityPatcher(
            infinity,
            alpha_w=args.alpha_w,
            alpha_h=args.alpha_h,
        )
        patcher.apply()
        tag = f"sph_aw{args.alpha_w:.1f}"
    else:
        tag = "baseline"

    # ── scale schedule ────────────────────────────────────────────────────
    scale_schedule = dynamic_resolution_h_w[args.h_div_w_template][args.pn]['scales']
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]

    # ── 옵션: CLIP 모델 ───────────────────────────────────────────────────
    clip_model = clip_processor = None
    if args.use_clip:
        from transformers import CLIPModel, CLIPProcessor
        clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").cuda()
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # ── 생성 & 평가 ───────────────────────────────────────────────────────
    all_results = []

    for prompt_idx, (prompt, seed) in enumerate(zip(PROMPTS, SEEDS)):
        print(f"\n[{prompt_idx+1}/{len(PROMPTS)}] seed={seed}")
        print(f"  prompt: {prompt[:80]}...")

        with autocast(dtype=torch.bfloat16):
            with torch.no_grad():
                img_tensor = gen_one_img(
                    infinity, vae, text_tokenizer, text_encoder,
                    prompt,
                    g_seed=seed,
                    gt_leak=0,
                    gt_ls_Bl=None,
                    cfg_list=args.cfg,
                    tau_list=args.tau,
                    scale_schedule=scale_schedule,
                    cfg_insertion_layer=[args.cfg_insertion_layer],
                    vae_type=args.vae_type,
                    sampling_per_bits=args.sampling_per_bits,
                )

        img_np = img_tensor.cpu().numpy()  # HxWx3 uint8 (from gen_one_img)

        # 저장
        save_path = out_dir / f"{tag}_p{prompt_idx:02d}_s{seed}.png"
        cv2.imwrite(str(save_path), img_np)

        # 메트릭
        ssim_val  = seam_ssim(img_np)
        cont_val  = seam_cont(img_np)
        clip_vals = cubemap_clip(img_np, prompt, clip_model, clip_processor)

        result = {
            "tag":      tag,
            "prompt_idx": prompt_idx,
            "seed":     seed,
            "img_path": str(save_path),
            "seam_ssim":  ssim_val,
            "seam_cont":  cont_val,
            **clip_vals,
        }
        all_results.append(result)
        print(f"  seam_ssim={ssim_val:.4f}  seam_cont={cont_val:.4f}")
        if clip_vals["clip_equatorial"] is not None:
            print(f"  clip_equatorial={clip_vals['clip_equatorial']:.2f}  clip_polar={clip_vals['clip_polar']:.2f}")

    # ── 요약 저장 ─────────────────────────────────────────────────────────
    results_path = out_dir / f"results_{tag}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Results saved → {results_path}")
    print(f"  Mean seam_ssim: {np.mean([r['seam_ssim'] for r in all_results]):.4f}")
    print(f"  Mean seam_cont: {np.nanmean([r['seam_cont'] for r in all_results]):.4f}")

    return all_results


def main():
    from tools.run_infinity import add_common_arguments

    parser = argparse.ArgumentParser(description="ERP evaluation for Infinity + Spherical RoPE")
    add_common_arguments(parser)
    parser.add_argument('--t5_path',           type=str,   default='google/flan-t5-xl')
    parser.add_argument('--out_dir',           type=str,   default='eval_outputs/erp')
    parser.add_argument('--use_spherical_rope', type=int,  default=0, choices=[0, 1])
    parser.add_argument('--alpha_w',           type=float, default=1.0)
    parser.add_argument('--alpha_h',           type=float, default=0.0)
    parser.add_argument('--use_clip',          type=int,   default=0, choices=[0, 1])
    args = parser.parse_args()

    # cfg/tau 파싱
    args.cfg = list(map(float, str(args.cfg).split(',')))
    if len(args.cfg) == 1:
        args.cfg = args.cfg[0]

    run_eval(args)


if __name__ == "__main__":
    main()
