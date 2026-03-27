"""
Step 4: RoPE frequency 분석 스크립트

Infinity의 precompute_rope2d_freqs_grid()에서
어떤 dim 인덱스가 u축(width) 고주파 pair인지 확인하고
probe_outputs/infinity_band_assignments.json에 저장.

실행법:
  conda activate infinity
  cd /home/mmai6k_02/anaconda3/workspace/mnt/infinity
  python probe_rope.py
"""

import json
import math
import os

import torch

from infinity.utils.dynamic_resolution import dynamic_resolution_h_w

# ── 파라미터 (125M 모델 기준) ─────────────────────────────────────────
# infinity_layer12: embed_dim=768, num_heads=8 → head_dim=96
# (run_infinity.py: kwargs_model = dict(depth=12, embed_dim=768, num_heads=8, ...))
HEAD_DIM   = 96      # 125M: embed_dim=768, num_heads=8 → head_dim=96
BASE       = 10000.0
RATIO      = 1.0     # square image

# ── inv_freq 계산 ──────────────────────────────────────────────────────
half_dim  = HEAD_DIM // 2           # 32
qtr_dim   = half_dim // 2           # 16  (실제 inv_freq 개수)

inv_freq = 1.0 / (BASE ** (torch.arange(0, half_dim, 2, dtype=torch.float32) / half_dim))
# shape: [16]

print("=" * 60)
print(f"HEAD_DIM={HEAD_DIM} (embed=768, nheads=8), half_dim={half_dim}, inv_freq shape={inv_freq.shape}")
print()

# ── RoPE cache 구조 설명 ──────────────────────────────────────────────
# freqs_grid = cat([freqs_h_expanded, freqs_w_expanded], dim=-1)
#   → shape (max_h, max_w, half_dim=32)
#   → dim 0..15  = height freqs  (inv_freq[0..15])
#   → dim 16..31 = width  freqs  (inv_freq[0..15])
#
# cache = stack([cos, sin]) → (2, max_h, max_w, half_dim)

print("RoPE 구조:")
print(f"  freqs_grid[:, :, :half_dim//2] = height freqs (dim 0..{qtr_dim-1})")
print(f"  freqs_grid[:, :, half_dim//2:] = width  freqs (dim {qtr_dim}..{half_dim-1})")
print()

# ── 각 scale별 inv_freq와 W_k 분석 ────────────────────────────────────
scale_schedule = dynamic_resolution_h_w[RATIO]['1M']['scales']  # 13 scales
print(f"Scale schedule (ratio={RATIO}, 1M):")
print(f"  {[(t,h,w) for t,h,w in scale_schedule]}")
print()

band_assignments = {}
print(f"{'scale':>6}  {'W_k':>5}  {'d':>3}  {'inv_freq_d':>12}  {'k_d (W)':>8}  {'nyquist?':>9}  {'active_sph':>10}")
print("-" * 72)
for _, ph, pw in scale_schedule:
    scale_key = f"({ph},{pw})"
    dims_info = []
    for d_idx, freq in enumerate(inv_freq.tolist()):
        # width wavenumber
        k_d = max(1, round(freq * pw / (2 * math.pi)))
        nyquist = k_d <= pw / 2
        active  = k_d >= 1
        dims_info.append({
            "d_inv_freq_idx": d_idx,
            "dim_in_half_dim": qtr_dim + d_idx,   # width part starts at qtr_dim
            "inv_freq": freq,
            "pw": pw,
            "k_d": int(k_d),
            "within_nyquist": bool(nyquist),
            "sph_active": bool(active),
        })
        if d_idx < 4 or d_idx == qtr_dim - 1:  # show first 4 and last
            print(f"{scale_key:>6}  {pw:>5}  {d_idx:>3}  {freq:>12.6f}  {k_d:>8d}  {str(nyquist):>9}  {str(active):>10}")
    if (ph, pw) != (scale_schedule[-1][1], scale_schedule[-1][2]):
        print(f"{'...':>6}")
    band_assignments[scale_key] = dims_info

print()
print("=" * 60)
print("Width freq dims in rope cache (dim index within half_dim):")
print(f"  Width-freq range: dim {qtr_dim} .. {half_dim-1}")
print(f"  (height-freq range: dim 0 .. {qtr_dim-1})")

# ── 저장 ───────────────────────────────────────────────────────────────
os.makedirs("probe_outputs", exist_ok=True)
out = {
    "head_dim":  HEAD_DIM,
    "half_dim":  half_dim,
    "qtr_dim":   qtr_dim,
    "base":      BASE,
    "height_dim_range": [0, qtr_dim - 1],
    "width_dim_range":  [qtr_dim, half_dim - 1],
    "inv_freq": inv_freq.tolist(),
    "scale_band_assignments": band_assignments,
}
out_path = "probe_outputs/infinity_band_assignments.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved → {out_path}")

# ── k_d 요약: 어떤 scale에서 spherical 보정이 의미 있는가 ───────────────
print()
print("k_d > 0 인 (의미있는 보정) width dim 수 per scale:")
for _, ph, pw in scale_schedule:
    scale_key = f"({ph},{pw})"
    info = band_assignments[scale_key]
    active_count = sum(1 for x in info if x["k_d"] >= 1)
    max_kd = max(x["k_d"] for x in info)
    print(f"  scale {scale_key:>8}: {active_count}/{len(info)} dims active, max k_d={max_kd}")
