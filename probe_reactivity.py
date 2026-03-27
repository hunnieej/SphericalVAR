"""
RoPE 반응성 프로브: Infinity에서 Semantic PE vs Positional PE 분리 분석

목적:
  각 RoPE freq 차원(0..47)이
    (A) 프롬프트(semantic 내용)에 얼마나 반응하는지
    (B) 공간 위치(positional)에 얼마나 반응하는지
  를 측정 → FLUX처럼 두 종류 PE가 분리되어 있는지 확인.

방법:
  - apply_rotary_emb를 monkey-patch해서 pre-RoPE q 텐서 캡처
  - q shape (non-flash): (B=1, H=8, L=ph*pw, head_dim=96)
  - 가장 fine한 scale의 q를 (H, ph, pw, 96)으로 reshape
  - 각 dim pair d(0..47):
      semantic_std[d]   = std(q[:, center_pos, 2d]) across 8 prompts
      positional_std[d] = std(q[same_prompt, :, 2d]) across L positions

실행:
  cd /home/mmai6k_02/anaconda3/workspace/mnt/infinity
  PYTHONPATH=. conda run -n infinity python probe_reactivity.py

결과: probe_outputs/reactivity.json + reactivity.png
"""

import os, sys, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
from torch.cuda.amp import autocast

import infinity.models.basic as basic_module
from tools.run_infinity import gen_one_img, load_tokenizer, load_visual_tokenizer, load_transformer
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w

# ── 설정 ──────────────────────────────────────────────────────────────────────
PRETRAINED  = "pretrained"
PN          = "0.06M"
VAE_TYPE    = 16
OUT_DIR     = "probe_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# 프롬프트: semantic diversity가 큰 8개 (같은 seed)
PROMPTS = [
    "A breathtaking snowy mountain summit at sunrise.",
    "A modern city skyline at night with glittering lights.",
    "A tranquil tropical beach at sunset.",
    "A gothic cathedral interior with stained glass windows.",
    "A quiet forest path in autumn.",
    "A vast desert landscape under a scorching sun.",
    "An underwater coral reef with colorful fish.",
    "A futuristic city with neon lights and flying cars.",
]
SEED = 42

# ── Monkey-patch: apply_rotary_emb 캡처 ────────────────────────────────────────
_capture_buf = []   # list of (call_idx, scale_ind, q_clone)
_capture_on  = False
_call_idx    = 0    # layer 호출 순서

_orig_apply = basic_module.apply_rotary_emb

def _patched_apply(q, k, scale_schedule, rope2d_freqs_grid,
                   pad_to_multiplier, rope2d_normalized_by_hw, scale_ind):
    global _call_idx
    if _capture_on:
        # q: (B=1, H, L, head_dim) — non-flash path
        _capture_buf.append((_call_idx, scale_ind, q.detach().float().cpu()))
        _call_idx += 1
    return _orig_apply(q, k, scale_schedule, rope2d_freqs_grid,
                       pad_to_multiplier, rope2d_normalized_by_hw, scale_ind)

basic_module.apply_rotary_emb = _patched_apply

# ── 모델 로딩 ──────────────────────────────────────────────────────────────────
class VaeArgs:
    vae_type               = VAE_TYPE
    vae_path               = f"{PRETRAINED}/infinity_vae_d16.pth"
    apply_spatial_patchify = 0

class ModelArgs(VaeArgs):
    model_path                        = f"{PRETRAINED}/infinity_125M_256x256.pth"
    checkpoint_type                   = "torch"
    model_type                        = "infinity_layer12"
    rope2d_each_sa_layer              = 1
    rope2d_normalized_by_hw           = 2
    use_scale_schedule_embedding      = 0
    pn                                = PN
    use_bit_label                     = 1
    add_lvl_embeding_only_first_block = 0
    text_channels                     = 2048
    use_flex_attn                     = 0
    bf16                              = 1
    cache_dir                         = "/dev/shm"
    enable_model_cache                = 0

print("Loading T5...")
text_tokenizer, text_encoder = load_tokenizer("google/flan-t5-xl")

print("Loading VAE...")
vae = load_visual_tokenizer(VaeArgs())

print("Loading Infinity 125M...")
infinity = load_transformer(vae, ModelArgs())

scale_schedule = dynamic_resolution_h_w[1.0][PN]['scales']
scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
num_scales     = len(scale_schedule)
finest_ind     = num_scales - 1
finest_ph, finest_pw = scale_schedule[finest_ind][1], scale_schedule[finest_ind][2]
print(f"Scale schedule: {num_scales} scales, finest = {finest_ph}×{finest_pw}")

# head_dim / half_dim 정보
head_dim  = infinity.unregistered_blocks[0].sa.head_dim  # 96
half_dim  = head_dim // 2   # 48  (= qtr_dim_h + qtr_dim_w)
qtr_dim   = half_dim // 2   # 24  (per axis inv_freq count)
num_heads = infinity.unregistered_blocks[0].sa.num_heads  # 8
print(f"head_dim={head_dim}, half_dim={half_dim}, qtr_dim={qtr_dim}, num_heads={num_heads}")

# ── 추론 & 캡처 ────────────────────────────────────────────────────────────────
# (N_prompts, num_layers, ph, pw, head_dim) for finest scale
all_q = []   # per prompt: list[num_layers] of q tensor (H, ph, pw, head_dim)

for pi, prompt in enumerate(PROMPTS):
    print(f"\n[Prompt {pi+1}/{len(PROMPTS)}] {prompt[:60]}...")
    _capture_buf.clear()
    _capture_on  = True
    _call_idx    = 0

    with autocast(dtype=torch.bfloat16):
        with torch.no_grad():
            img = gen_one_img(
                infinity, vae, text_tokenizer, text_encoder,
                prompt,
                g_seed=SEED,
                gt_leak=0,
                gt_ls_Bl=None,
                cfg_list=3.0,
                tau_list=1.0,
                scale_schedule=scale_schedule,
                cfg_insertion_layer=[0],
                vae_type=VAE_TYPE,
                sampling_per_bits=1,
            )
    _capture_on = False

    print(f"  Captured {len(_capture_buf)} apply_rotary_emb calls")

    # finest scale call들만 추출 (scale_ind == finest_ind)
    finest_calls = [(ci, si, q) for ci, si, q in _capture_buf if si == finest_ind]
    print(f"  Finest scale calls: {len(finest_calls)}")

    if not finest_calls:
        print("  WARNING: no finest scale calls captured!")
        continue

    # 각 layer call에서 q: (1, H, L, head_dim) → (H, ph, pw, head_dim)
    layer_qs = []
    for ci, si, q in finest_calls:
        q_3d = q[0]  # (H, L, head_dim)
        L = q_3d.shape[1]
        if L != finest_ph * finest_pw:
            print(f"  Unexpected L={L} (expected {finest_ph*finest_pw}), skipping")
            continue
        q_4d = q_3d.reshape(num_heads, finest_ph, finest_pw, head_dim)
        layer_qs.append(q_4d)

    if layer_qs:
        all_q.append(layer_qs)

print(f"\nCollected {len(all_q)} prompts, {len(all_q[0]) if all_q else 0} layers each")

# ── 분석 ──────────────────────────────────────────────────────────────────────
if len(all_q) < 2:
    print("Not enough data for analysis. Exiting.")
    sys.exit(1)

N  = len(all_q)           # num_prompts
NL = len(all_q[0])        # num_layer_calls at finest scale

# q shape per sample: (NL, H, ph, pw, head_dim)
# Stack: (N, NL, H, ph, pw, head_dim)
try:
    Q = np.stack([np.stack([lq.numpy() for lq in prompt_qs], axis=0)
                  for prompt_qs in all_q], axis=0)
except Exception as e:
    # ragged → pad or skip
    min_nl = min(len(pq) for pq in all_q)
    Q = np.stack([np.stack([lq.numpy() for lq in prompt_qs[:min_nl]], axis=0)
                  for prompt_qs in all_q], axis=0)
    NL = min_nl

print(f"Q shape: {Q.shape}")  # (N, NL, H, ph, pw, head_dim)

# Average over layers and heads for cleaner signal
# Q_avg: (N, ph, pw, head_dim)
Q_avg = Q.mean(axis=(1, 2))

# ── (A) Semantic sensitivity: 같은 위치, 다른 프롬프트에서의 std ─────────────────
# For each spatial position (i,j) and dim d:
#   semantic_std(d, i, j) = std across N prompts of Q_avg[:, i, j, d]
# Then average over (i, j):
#   semantic_std(d) = mean over (i,j)

semantic_std = Q_avg.std(axis=0).mean(axis=(0, 1))   # (head_dim,)

# ── (B) Positional sensitivity: 같은 프롬프트, 다른 위치에서의 std ────────────────
# For each prompt p and dim d:
#   pos_std(d, p) = std across (i,j) of Q_avg[p, :, :, d]
# Then average over p:
#   pos_std(d) = mean over p

pos_std = Q_avg.std(axis=(1, 2)).mean(axis=0)        # (head_dim,)

# ── (C) Width-specific positional: column(j) 방향 std (width PE 분석) ──────────
# Width dims: pair d=24..47 → affects width position
# For each prompt, compute std along width dim (axis=2) averaged over height
col_std = Q_avg.std(axis=2).mean(axis=(0, 1))   # std along width, avg over prompts & height
row_std = Q_avg.std(axis=1).mean(axis=(0, 1))   # std along height, avg over prompts & width

# ── RoPE 위상 분석: 각 dim의 seam 불연속 크기 ──────────────────────────────────
# For mode 2 (star-style), finest scale positions:
# width dim d (d in 0..qtr_dim-1):
#   inv_freq = 1/(10000^(2d/half_dim))
#   position of column j: round(j * upw / pw)
#   phase at j: round(j * upw / pw) * inv_freq
# seam: difference between j=0 (pos=0) and j=pw-1 (pos=round((pw-1)*upw/pw))
_, uph, upw = scale_schedule[-1]  # finest scale dimensions
base = 10000.0
inv_freqs = 1.0 / (base ** (np.arange(0, half_dim, 2) / half_dim))  # (qtr_dim,)

seam_phase_diff = np.zeros(qtr_dim)   # for width dims
for d in range(qtr_dim):
    # pos of j=0: 0; pos of j=pw-1: round((pw-1)*upw/pw)
    pos_left  = 0
    pos_right = round((finest_pw - 1) * upw / finest_pw)
    phase_left  = pos_left  * inv_freqs[d]
    phase_right = pos_right * inv_freqs[d]
    # seam continuity error: |cos(phase_right) - cos(0)|
    seam_phase_diff[d] = abs(math.cos(phase_right) - math.cos(phase_left))

# Expected seam phase if circular (seam ≈ 0):
# k_d = round(inv_freq * upw / 2pi)
# sph_freq = k_d * 2pi / upw
# phase_at_upw = sph_freq * upw = k_d * 2pi → cos = 1, sin = 0
k_d_values = np.round(inv_freqs * upw / (2 * math.pi)).clip(min=1)
print(f"\nWidth dim k_d values: {k_d_values.astype(int)}")
print(f"Seam phase discontinuity (standard RoPE): {seam_phase_diff}")

# ── 출력 ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"{'dim':>4} {'type':>6} {'sem_std':>9} {'pos_std':>9} {'ratio_s':>9} {'col_std':>9} {'row_std':>9} {'seam_Δ':>8}")
print("-" * 70)

results = []
for d in range(head_dim):
    d_pair = d // 2     # which RoPE freq pair (0..47)
    is_height = d_pair < qtr_dim
    dim_type  = f"H{d_pair:02d}" if is_height else f"W{d_pair-qtr_dim:02d}"
    s_std = float(semantic_std[d])
    p_std = float(pos_std[d])
    ratio = s_std / (s_std + p_std + 1e-9)  # 1=semantic, 0=positional
    c_std = float(col_std[d])
    r_std = float(row_std[d])
    seam  = float(seam_phase_diff[d_pair - qtr_dim]) if not is_height else float('nan')

    results.append({
        "dim": d, "pair": d_pair, "type": dim_type,
        "semantic_std": round(s_std, 5),
        "positional_std": round(p_std, 5),
        "semantic_ratio": round(ratio, 4),
        "col_std": round(c_std, 5),
        "row_std": round(r_std, 5),
        "seam_phase_diff": round(seam, 4) if not np.isnan(seam) else None,
    })
    seam_str = f"{seam:8.4f}" if not np.isnan(seam) else "       -"
    print(f"{d:4d} {dim_type:>6} {s_std:9.5f} {p_std:9.5f} {ratio:9.4f} {c_std:9.5f} {r_std:9.5f} {seam_str}")

print("=" * 70)

# ── Per-pair summary (평균) ────────────────────────────────────────────────────
print("\n── Per RoPE freq pair summary ──────────────────────────────────────────")
print(f"{'pair':>5} {'type':>6} {'sem_std':>9} {'pos_std':>9} {'ratio':>8} {'seam_Δ':>8}")
pair_results = []
for dp in range(half_dim):
    r0, r1 = results[2*dp], results[2*dp+1]
    is_h  = dp < qtr_dim
    ptype = f"H{dp:02d}" if is_h else f"W{dp-qtr_dim:02d}"
    s = (r0['semantic_std']   + r1['semantic_std'])   / 2
    p = (r0['positional_std'] + r1['positional_std']) / 2
    ratio = s / (s + p + 1e-9)
    seam  = r0['seam_phase_diff']
    seam_str = f"{seam:8.4f}" if seam is not None else "       -"
    pair_results.append({"pair": dp, "type": ptype, "sem": s, "pos": p, "ratio": ratio, "seam": seam})
    print(f"{dp:5d} {ptype:>6} {s:9.5f} {p:9.5f} {ratio:8.4f} {seam_str}")

# ── 최종 결론 ──────────────────────────────────────────────────────────────────
height_pairs = [r for r in pair_results if r['type'].startswith('H')]
width_pairs  = [r for r in pair_results if r['type'].startswith('W')]

h_sem_avg = np.mean([r['sem'] for r in height_pairs])
h_pos_avg = np.mean([r['pos'] for r in height_pairs])
w_sem_avg = np.mean([r['sem'] for r in width_pairs])
w_pos_avg = np.mean([r['pos'] for r in width_pairs])

print("\n── 결론 ────────────────────────────────────────────────────────────────")
print(f"Height dims (H00-H{qtr_dim-1:02d}): sem_std={h_sem_avg:.5f}, pos_std={h_pos_avg:.5f}, ratio={h_sem_avg/(h_sem_avg+h_pos_avg+1e-9):.4f}")
print(f"Width  dims (W00-W{qtr_dim-1:02d}): sem_std={w_sem_avg:.5f}, pos_std={w_pos_avg:.5f}, ratio={w_sem_avg/(w_sem_avg+w_pos_avg+1e-9):.4f}")

# Top "semantic" width dims (semantic_ratio high)
w_sorted = sorted(width_pairs, key=lambda r: r['ratio'], reverse=True)
print("\nTop 5 most SEMANTIC width dims:")
for r in w_sorted[:5]:
    print(f"  {r['type']}: sem={r['sem']:.5f}, pos={r['pos']:.5f}, ratio={r['ratio']:.4f}, seam_Δ={r['seam']}")
print("\nTop 5 most POSITIONAL width dims:")
for r in w_sorted[-5:]:
    print(f"  {r['type']}: sem={r['sem']:.5f}, pos={r['pos']:.5f}, ratio={r['ratio']:.4f}, seam_Δ={r['seam']}")

# ── 시각화 ────────────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    pairs     = [r['pair']  for r in pair_results]
    sem_vals  = [r['sem']   for r in pair_results]
    pos_vals  = [r['pos']   for r in pair_results]
    ratios    = [r['ratio'] for r in pair_results]

    ax = axes[0, 0]
    ax.bar(pairs[:qtr_dim], sem_vals[:qtr_dim], label='height', color='steelblue', alpha=0.7)
    ax.bar([p + half_dim for p in range(qtr_dim)], sem_vals[qtr_dim:], label='width', color='tomato', alpha=0.7)
    ax.set_xticks(list(range(qtr_dim)) + [p+half_dim for p in range(qtr_dim)])
    ax.set_xticklabels([f'H{i}' for i in range(qtr_dim)] + [f'W{i}' for i in range(qtr_dim)], rotation=90, fontsize=7)
    ax.set_title('Semantic Std (across prompts)')
    ax.set_ylabel('std')
    ax.legend()

    ax = axes[0, 1]
    ax.bar(pairs[:qtr_dim], pos_vals[:qtr_dim], label='height', color='steelblue', alpha=0.7)
    ax.bar([p + half_dim for p in range(qtr_dim)], pos_vals[qtr_dim:], label='width', color='tomato', alpha=0.7)
    ax.set_xticks(list(range(qtr_dim)) + [p+half_dim for p in range(qtr_dim)])
    ax.set_xticklabels([f'H{i}' for i in range(qtr_dim)] + [f'W{i}' for i in range(qtr_dim)], rotation=90, fontsize=7)
    ax.set_title('Positional Std (across positions)')
    ax.set_ylabel('std')
    ax.legend()

    ax = axes[1, 0]
    ax.bar(pairs[:qtr_dim], ratios[:qtr_dim], label='height', color='steelblue', alpha=0.7)
    ax.bar([p + half_dim for p in range(qtr_dim)], ratios[qtr_dim:], label='width', color='tomato', alpha=0.7)
    ax.axhline(0.5, color='k', linestyle='--', alpha=0.5, label='50/50')
    ax.set_xticks(list(range(qtr_dim)) + [p+half_dim for p in range(qtr_dim)])
    ax.set_xticklabels([f'H{i}' for i in range(qtr_dim)] + [f'W{i}' for i in range(qtr_dim)], rotation=90, fontsize=7)
    ax.set_title('Semantic Ratio = sem/(sem+pos)  [1=semantic, 0=positional]')
    ax.set_ylim(0, 1)
    ax.legend()

    ax = axes[1, 1]
    seam_vals_w = [r['seam'] if r['seam'] is not None else 0.0 for r in width_pairs]
    ratios_w = [r['ratio'] for r in width_pairs]
    sc = ax.scatter(seam_vals_w, ratios_w, c=range(qtr_dim), cmap='viridis', s=60, zorder=3)
    plt.colorbar(sc, ax=ax, label='width dim idx')
    for i, r in enumerate(width_pairs):
        ax.annotate(r['type'], (seam_vals_w[i], ratios_w[i]), fontsize=7, ha='left', va='bottom')
    ax.set_xlabel('Seam Phase Discontinuity (standard RoPE)')
    ax.set_ylabel('Semantic Ratio')
    ax.set_title('Width dims: Seam Discontinuity vs Semantic Ratio')
    ax.axhline(0.5, color='k', linestyle='--', alpha=0.5)

    plt.tight_layout()
    save_path = f"{OUT_DIR}/reactivity.png"
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    print(f"\nPlot saved → {save_path}")
    plt.close()
except Exception as e:
    print(f"Plotting skipped: {e}")

# ── JSON 저장 ──────────────────────────────────────────────────────────────────
out = {
    "model": "infinity_125M",
    "num_prompts": N,
    "num_layer_calls": NL,
    "finest_scale": [finest_ph, finest_pw],
    "head_dim": head_dim,
    "half_dim": half_dim,
    "qtr_dim": qtr_dim,
    "per_pair": pair_results,
    "summary": {
        "height_sem_avg": float(h_sem_avg),
        "height_pos_avg": float(h_pos_avg),
        "width_sem_avg":  float(w_sem_avg),
        "width_pos_avg":  float(w_pos_avg),
    },
}
with open(f"{OUT_DIR}/reactivity.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"Results saved → {OUT_DIR}/reactivity.json")
