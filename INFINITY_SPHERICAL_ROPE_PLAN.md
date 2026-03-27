# Infinity × Spherical RoPE — 작업 계획 문서

**작성일:** 2026-03-20
**목적:** 새 서버(ada6000)에서 Claude Code를 이어서 실행하기 위한 인수인계 문서

---

## 배경 — SphereFlow 프로젝트

FLUX.1-Kontext-dev로 ERP(Equirectangular Projection) 파노라마를 생성할 때:
1. **Seam 불연속** → Spherical RoPE Type A (α=1.0, all-layers)로 해결
2. **Pole 왜곡** → Latitude Adapter 훈련 중 (현재 GPU 0/1에서 돌아가는 중)

Spherical RoPE의 핵심 아이디어:
```
표준 RoPE의 u축 고주파 pair에서, 주파수를 정수 wavenumber로 반올림:
  k_d = max(1, round(freq_d × W / 2π))
  φ_A(u) = k_d × 2π × u / W
→ seam(Δu = W-1)의 위상차 = 인접(Δu = 1)의 위상차 → seam 연속성 확보
```

**이 아이디어를 Infinity 모델에도 적용하자는 것이 이번 작업의 목표.**

---

## 왜 Infinity인가?

- Infinity는 VAR(Visual AutoRegressive) 계열 multi-scale 자기회귀 모델
- FLUX와 마찬가지로 2D RoPE 사용 (height/width 독립 주파수)
- FLUX(diffusion) + Infinity(autoregressive) 두 모델에서 같은 원리가 동작하면
  논문 contribution이 크게 강화됨
- "생성 패러다임에 무관하게 analytic PE 보정이 일반적으로 적용됨"을 주장 가능

---

## Infinity RoPE 구조 (이미 분석 완료)

**파일:** `infinity/models/basic.py` — `precompute_rope2d_freqs_grid()` 함수

```python
# 현재 구현
inv_freq = 1.0 / (base ** (arange(0, half_dim, 2) / half_dim))
freqs_height = outer(arange(max_h), inv_freq)   # [max_h, half_dim]
freqs_width  = outer(arange(max_w), inv_freq)   # [max_w, half_dim]
freqs_grid   = cat([freqs_h_expanded, freqs_w_expanded], dim=-1)  # [max_h, max_w, dim]
cache = stack([cos(freqs_grid), sin(freqs_grid)], dim=0)  # (2, max_h, max_w, dim)
```

**Multi-scale 구조:**
Infinity는 scale별로 W_k가 다름 (1M 이미지 기준 13 scales: W_k = 1, 2, ..., 64)

**Spherical RoPE 적용 시 핵심 포인트:**
```
k_d^(k) = round(inv_freq_d × W_k / 2π)    ← scale마다 W_k가 달라짐

→ Scale-adaptive: 코스 scale (W_k=1,2)은 k_d≈0 → 보정 거의 없음 (맞음)
                  파인 scale (W_k=64)은 k_d 의미있는 값 → seam 보정 최대
```

---

## 해야 할 작업 순서

### Step 1: 환경 셋업 (ada6000 서버)

```bash
# 1. 레포 클론 (이미 /home/mmai5k_00/jw/mount/infinity/ 에 있을 수 있음)
cd /path/to/workspace
git clone https://github.com/FoundationVision/Infinity.git infinity

# 2. conda env 생성
conda create -n infinity python=3.10 -y

# 3. PyTorch 설치 (ada6000은 A6000 = sm_86, torch 2.5.1+cu124 호환)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# 4. 나머지 의존성
pip install easydict typed-argument-parser seaborn kornia gputil colorama omegaconf \
    pandas timm==0.9.6 decord transformers pytz wandb imageio einops openai \
    "httpx==0.20.0" opencv-python psutil

# 5. flash_attn
pip install flash-attn --no-build-isolation
```

### Step 2: 모델 다운로드

```bash
mkdir -p infinity/pretrained

python -c "
from huggingface_hub import hf_hub_download

# Probe용 (125M, 빠름)
hf_hub_download('FoundationVision/Infinity', 'infinity_125M_256x256.pth',
                local_dir='infinity/pretrained')

# VAE (필수)
hf_hub_download('FoundationVision/Infinity', 'infinity_vae_d32reg.pth',
                local_dir='infinity/pretrained')

# 본 실험용 (2B, 8.8GB) — probe 완료 후 다운로드
# hf_hub_download('FoundationVision/Infinity', 'infinity_2b_reg.pth',
#                 local_dir='infinity/pretrained')
"

# T5 text encoder (transformers 자동 캐시 — 약 10GB)
python -c "from transformers import AutoTokenizer, T5EncoderModel; \
           AutoTokenizer.from_pretrained('google/flan-t5-xl'); \
           T5EncoderModel.from_pretrained('google/flan-t5-xl')"
```

### Step 3: 기존 inference 동작 확인

`infinity/tools/run_infinity.py`의 로딩 패턴 참고:
```python
from infinity.models.bsq_vae.vae import vae_model
from infinity.models.infinity import Infinity

# VAE
vae = vae_model('pretrained/infinity_vae_d32reg.pth',
                schedule_mode='dynamic', codebook_dim=32,
                codebook_size=2**32, test_mode=True, patch_size=16).cuda().half().eval()

# Transformer (125M 기준)
# → tools/run_infinity.py 참고하여 Infinity(...) 생성자 인자 확인
```

먼저 `tools/run_infinity.py` 또는 `tools/reproduce.py`로 베이스라인 ERP 파노라마 생성이 되는지 확인.
# Command
```bash
python tools/run_infinity.py   --pn 1M   --model_type infinity_2b   --model_path pretrained/infinity_2b_reg.pth  --text_encoder_ckpt /home/mmai6k_jh/.cache/huggingface/hub/models--google--flan-t5-xl --use_flex_attn 0 --h_div_w_template 0.5 --prompt "This is a 360-degree panorama image. The photo shows a breathtaking snowy mountain summit at sunrise."  
```

---

### Step 4: Probe 실험 — 어떤 freq dimension이 u축 담당인지 확인

FLUX에서 했던 것과 동일한 방식으로 Infinity의 RoPE freq 분석.

**목표:** `infinity/models/basic.py`의 `precompute_rope2d_freqs_grid()`에서
어떤 dim 인덱스가 u축(width) 고주파 pair인지 확인

```python
# 분석 스크립트 작성 위치: infinity/probe_rope.py
#
# 핵심: RoPE cache shape = (2, max_h, max_w, dim)
# - cache[0] = cos values, cache[1] = sin values
# - dim 앞 절반 = height freq, 뒷 절반 = width freq (또는 반대)
# → 정확한 분할 방식은 precompute_rope2d_freqs_grid() 코드에서 확인
#
# 고주파 u축 pair 조건: inv_freq[d] × max_w / (2π) ≥ 0.5 정도 되는 것들
```

분석해서 `probe_outputs/infinity_band_assignments.json` 에 저장.

---

### Step 5: Scale-Adaptive Spherical RoPE 구현

**구현 위치:** `infinity/spherical_rope_infinity.py` (새 파일)

```python
import torch, math, numpy as np

class SphericalRoPEInfinityConfig:
    replacement_type = "A"   # 현재는 Type A만
    alpha = 1.0              # 각도 보간 비율 (1.0 = full spherical)

class SphericalInfinityPosEmbed:
    """
    Scale-adaptive Spherical RoPE for Infinity.

    각 scale k에서 W_k 크기에 맞는 최적 circular 주파수를 계산:
      freq_sph[d] = round(inv_freq[d] × W_k / 2π) × 2π / W_k

    기존 precompute_rope2d_freqs_grid()를 대체.
    """
    def __init__(self, model, config, scale_schedule):
        # scale_schedule: [(h_1, w_1), (h_2, w_2), ..., (h_K, w_K)]
        # 각 scale별 별도 cache 사전 계산
        ...

class SphericalRoPEInfinityPatcher:
    """Context manager: Infinity transformer에 Spherical RoPE를 주입."""
    def __enter__(self):
        # model.rope_cache 또는 precompute 함수를 교체
        ...
    def __exit__(self, *_):
        # 원래 상태로 복원
        ...
```

**FLUX와의 차이점:**
- FLUX: 단일 W=90 → 하나의 k_d 테이블
- Infinity: scale마다 W_k 다름 → `cache_k` 딕셔너리 `{scale_idx: (2, h_k, w_k, dim)}`

---

### Step 6: ERP 평가 파이프라인

**평가 메트릭 (SphereFlow에서 사용한 것과 동일):**

```python
def seam_ssim(img, s=32): ...         # 좌우 32px 스트립 SSIM
def seam_cont(img): ...               # 내부 gradient 대비 seam gradient 비율
def cubemap_clip(img, prompt): ...    # ERP → 6 cubemap faces, CLIP per face
                                      # clip_equatorial (F/R/B/L mean)
                                      # clip_polar      (U/D mean)
```

**테스트 프롬프트:**
```python
PROMPTS = [
    "This is a panorama image. The photo shows a breathtaking snowy mountain summit at sunrise...",
    "This is a panorama image. The photo shows a modern city skyline at night...",
    "This is a panorama image. The photo shows a tranquil tropical beach at sunset...",
    "This is a panorama image. The photo shows the interior of a grand cathedral...",
    "This is a panorama image. The photo shows a quiet forest path in autumn...",
]
SEEDS = [0, 1234, 5536, 8650, 9902]
```

---

## 현재 SphereFlow 실험 상황 (참고용)

| 서버 GPU | 실험 | 상태 |
|---------|------|------|
| GPU 0 | lat_adapter_single_sph_a10_d32 | 훈련 중 (~6h) |
| GPU 1 | lat_adapter_all_sph_a10_d32 | 훈련 중 (~10h) |

완료되면:
```bash
cd /home/mmai5k_00/jw/mount/sphereflow
python eval_lora_sampling.py    # 새 조건 추가 후 재실행
python eval_cubemap_clip.py     # cubemap CLIP 평가
```

---

## 핵심 참고 파일들

| 파일 | 설명 |
|------|------|
| `infinity/infinity/models/basic.py` | RoPE 핵심 — `precompute_rope2d_freqs_grid()` |
| `infinity/infinity/models/infinity.py` | Transformer forward |
| `infinity/tools/run_infinity.py` | 정상 동작 확인용 inference 스크립트 |
| `sphereflow/spherical_rope.py` | FLUX용 Spherical RoPE 참고 구현 |
| `sphereflow/PROJECT_STATUS.md` | SphereFlow 전체 실험 현황 |

---

## 최우선 작업 순서 요약

1. `conda activate infinity` + `cd infinity`
2. `python tools/run_infinity.py` 로 베이스라인 ERP 이미지 생성 확인
3. `precompute_rope2d_freqs_grid()` 코드 분석 → 어떤 dim이 width 담당인지 확인
4. `probe_rope.py` 작성 → band_assignments 확인
5. `spherical_rope_infinity.py` 작성 → Scale-adaptive Spherical RoPE
6. seam_ssim / cubemap_clip 으로 효과 정량 평가
7. 2B 모델로 업스케일 후 최종 평가
