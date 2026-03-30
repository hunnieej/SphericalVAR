"""
Step 5: Scale-Adaptive Spherical RoPE for Infinity

Usage:
    from spherical_rope_infinity import SphericalRoPEInfinityPatcher

    patcher = SphericalRoPEInfinityPatcher(model, alpha=1.0)
    with patcher:
        img = gen_one_img(model, ...)   # spherical RoPE 적용

    # 또는 영구 적용:
    patcher.apply()

원리 (Type A):
    표준 RoPE의 u축(width) 주파수 pair에서
      k_d = max(1, round(inv_freq_d × W_k / 2π))
      φ_sph(j) = k_d × 2π × j / W_k
    → seam(j+W_k) 위상 = seam(j) 위상 + k_d×2π  (정수배 → 연속)
    → seam(Δj=W_k-1)의 위상차 ≈ 인접(Δj=1)의 위상차

Scale-adaptive:
    Infinity는 scale마다 W_k가 다름:
      coarse (W_k=1,2) → k_d≈0 → 보정 거의 없음 (올바른 동작)
      fine   (W_k=64)  → k_d 의미있는 값 → seam 보정 최대

References:
    - SphereFlow (FLUX용 Spherical RoPE): sphereflow/spherical_rope.py
    - INFINITY_SPHERICAL_ROPE_PLAN.md
"""

import math
import ast
from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from infinity.utils.dynamic_resolution import dynamic_resolution_h_w as _DYN_RES


# ─────────────────────────────────────────────────────────────────────────────
# 핵심 계산: 한 scale의 spherical width 주파수
# ─────────────────────────────────────────────────────────────────────────────


def _blend_freq_with_band(
    inv_freq: torch.Tensor, spherical_freq: torch.Tensor, ratio: float
) -> torch.Tensor:
    if ratio >= 1.0:
        return spherical_freq
    if ratio <= 0.0:
        return inv_freq
    qtr_dim = inv_freq.shape[0]
    band_len = max(1, int(round(qtr_dim * ratio)))
    band_start = max(0, qtr_dim - band_len)
    idx = torch.arange(qtr_dim, device=inv_freq.device)
    mask = (idx >= band_start).to(inv_freq.dtype)
    return mask * spherical_freq + (1.0 - mask) * inv_freq


def _compute_sph_freq_w(
    inv_freq: torch.Tensor, pw: int, alpha: float = 1.0, band_ratio: float = 1.0
) -> torch.Tensor:
    """
    Width 방향 spherical 주파수 계산.

    Args:
        inv_freq: [Q] 표준 inverse frequencies
        pw:       해당 scale의 width token 수
        alpha:    보간 비율 (1.0 = full spherical, 0.0 = standard)

    Returns:
        eff_freq: [Q] 보정된 주파수
    """
    k_d = (inv_freq * pw / (2.0 * math.pi)).round().clamp(min=1.0)
    sph_freq = k_d * (2.0 * math.pi / pw)
    eff_freq = alpha * sph_freq + (1.0 - alpha) * inv_freq
    return _blend_freq_with_band(inv_freq, eff_freq, band_ratio)


def _compute_sph_freq_h(
    inv_freq: torch.Tensor, ph: int, alpha: float = 0.0
) -> torch.Tensor:
    """
    Height 방향 주파수 (기본은 alpha=0 → 표준 유지).
    Pole 보정을 원한다면 alpha>0으로 설정.
    """
    if alpha == 0.0:
        return inv_freq
    k_d = (inv_freq * ph / (2.0 * math.pi)).round().clamp(min=1.0)
    sph_freq = k_d * (2.0 * math.pi / ph)
    return alpha * sph_freq + (1.0 - alpha) * inv_freq


# ─────────────────────────────────────────────────────────────────────────────
# Scale-adaptive RoPE cache 사전 계산
# ─────────────────────────────────────────────────────────────────────────────


def precompute_rope2d_freqs_grid_spherical(
    dim: int,
    dynamic_resolution_h_w: dict,
    rope2d_normalized_by_hw: int,
    pad_to_multiplier: int = 1,
    base: float = 10000.0,
    device: Optional[torch.device] = None,
    scaling_factor: float = 1.0,
    alpha_w: float = 1.0,
    alpha_h: float = 0.0,
    spherical_band_ratio: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    Spherical RoPE version of precompute_rope2d_freqs_grid().

    표준 구현과 같은 dict 구조를 반환하되,
    각 scale의 width 주파수를 circular freq으로 대체.

    Args:
        dim:                    head_dim (= embed_dim / num_heads)
        dynamic_resolution_h_w: 해상도 스케줄 딕셔너리
        rope2d_normalized_by_hw: 0=direct, 1=bilinear, 2=star-style
        pad_to_multiplier:      패딩 alignment
        base:                   RoPE base (default 10000)
        device:                 torch device
        scaling_factor:         위치 스케일링 (보통 1.0)
        alpha_w:                width 보정 강도 (1.0 = full spherical)
        alpha_h:                height 보정 강도 (0.0 = 표준 유지)

    Returns:
        rope2d_freqs_grid: {scale_schedule_key: (2,1,1,1,seq_len,half_head_dim)}
    """
    half_dim = dim // 2  # total freq dims per position (height + width)
    qtr_dim = half_dim // 2  # per-axis freq count

    inv_freq = 1.0 / (
        base
        ** (torch.arange(0, half_dim, 2, dtype=torch.float32).to(device) / half_dim)
    )  # [qtr_dim]

    rope2d_freqs_grid: Dict[str, torch.Tensor] = {}

    for h_div_w in dynamic_resolution_h_w:
        # 1M scale 기준으로 finest (uph, upw) 결정
        scale_schedule_1M = dynamic_resolution_h_w[h_div_w]["1M"]["scales"]
        _, uph, upw = scale_schedule_1M[-1]

        # ── Width spherical freq (mode별 처리) ─────────────────────────────
        # mode 2 (star-style): 위치는 표준과 동일하게 round(j*upw/pw) 유지.
        #   seam 조건: phase(upw) - phase(0) = k_d×2π
        #   → freq 기준은 upw (finest scale width)
        # mode 0 (direct): 위치는 0..pw-1, freq 기준은 pw (current scale)
        if rope2d_normalized_by_hw == 2:
            k_d_w = (inv_freq * upw / (2.0 * math.pi)).round().clamp(min=1.0)
            sph_freq_w = k_d_w * (2.0 * math.pi / upw)
            eff_freq_w_global = alpha_w * sph_freq_w + (1.0 - alpha_w) * inv_freq
            eff_freq_w_global = _blend_freq_with_band(
                inv_freq, eff_freq_w_global, spherical_band_ratio
            )
        # mode 1도 upw 기준으로 동일하게 처리
        if rope2d_normalized_by_hw == 1:
            k_d_w = (inv_freq * upw / (2.0 * math.pi)).round().clamp(min=1.0)
            sph_freq_w = k_d_w * (2.0 * math.pi / upw)
            eff_freq_w_global = alpha_w * sph_freq_w + (1.0 - alpha_w) * inv_freq

        rope_cache_list: List[torch.Tensor] = []

        for _, ph, pw in scale_schedule_1M:
            ph_mul_pw = ph * pw

            # ── Width positions & freq ──────────────────────────────────────
            if rope2d_normalized_by_hw == 2:
                # star-style: 표준과 동일한 위치 사용, freq만 spherical로 교체
                t_w = (
                    torch.arange(pw, device=device, dtype=torch.float32) * (upw / pw)
                ).round() / scaling_factor
                eff_freq_w = eff_freq_w_global
            elif rope2d_normalized_by_hw == 1:
                # bilinear: 연속 위치, freq는 upw 기준 spherical
                t_w = (
                    torch.arange(pw, device=device, dtype=torch.float32) * (upw / pw)
                ) / scaling_factor
                eff_freq_w = eff_freq_w_global
            else:  # mode 0: direct
                # 현재 scale pw 기준으로 spherical freq 계산
                t_w = (
                    torch.arange(pw, device=device, dtype=torch.float32)
                    / scaling_factor
                )
                eff_freq_w = _compute_sph_freq_w(
                    inv_freq, pw, alpha=alpha_w, band_ratio=spherical_band_ratio
                )
            freqs_w = torch.outer(t_w, eff_freq_w)  # [pw, qtr_dim]

            # ── Height positions & freq ─────────────────────────────────────
            if rope2d_normalized_by_hw == 2:
                t_h = (
                    torch.arange(ph, device=device, dtype=torch.float32) * (uph / ph)
                ).round() / scaling_factor
            elif rope2d_normalized_by_hw == 1:
                t_h = (
                    torch.arange(ph, device=device, dtype=torch.float32) * (uph / ph)
                ) / scaling_factor
            else:  # mode 0: direct
                t_h = (
                    torch.arange(ph, device=device, dtype=torch.float32)
                    / scaling_factor
                )

            eff_freq_h = _compute_sph_freq_h(inv_freq, ph, alpha=alpha_h)
            freqs_h = torch.outer(t_h, eff_freq_h)  # [ph, qtr_dim]

            # ── 2D freqs grid for this scale: [ph, pw, half_dim] ───────────
            freqs_grid = torch.cat(
                [
                    freqs_h[:, None, :].expand(-1, pw, -1),  # [ph, pw, qtr_dim]
                    freqs_w[None, :, :].expand(ph, -1, -1),  # [ph, pw, qtr_dim]
                ],
                dim=-1,
            )  # [ph, pw, half_dim]

            rope_cache = torch.stack(
                [torch.cos(freqs_grid), torch.sin(freqs_grid)], dim=0
            )  # [2, ph, pw, half_dim]

            rope_cache_list.append(rope_cache.reshape(2, ph_mul_pw, -1))

        cat_rope_cache = torch.cat(rope_cache_list, dim=1)  # (2, seq_len, half_dim)

        # Padding
        if cat_rope_cache.shape[1] % pad_to_multiplier:
            pad_len = pad_to_multiplier - (cat_rope_cache.shape[1] % pad_to_multiplier)
            pad = torch.zeros(2, pad_len, half_dim, device=device)
            cat_rope_cache = torch.cat([cat_rope_cache, pad], dim=1)

        cat_rope_cache = cat_rope_cache[
            :, None, None, None
        ]  # (2,1,1,1,seq_len,half_dim)

        # 모든 pn('0.06M','0.25M','1M') → 같은 key로 등록
        for pn in dynamic_resolution_h_w[h_div_w]:
            scale_schedule = dynamic_resolution_h_w[h_div_w][pn]["scales"]
            tmp_schedule = [(1, h, w) for _, h, w in scale_schedule]
            rope2d_freqs_grid[str(tuple(tmp_schedule))] = cat_rope_cache

    return rope2d_freqs_grid


# ─────────────────────────────────────────────────────────────────────────────
# Patcher: Context Manager & 영구 적용
# ─────────────────────────────────────────────────────────────────────────────


class SphericalRoPEInfinityPatcher:
    """
    Infinity 모델에 Scale-Adaptive Spherical RoPE를 주입하는 패처.

    사용법:
        patcher = SphericalRoPEInfinityPatcher(model, alpha_w=1.0)

        # (A) context manager — 임시 적용
        with patcher:
            img = gen_one_img(model, ...)

        # (B) 영구 적용
        patcher.apply()

        # (C) 원복
        patcher.restore()
    """

    def __init__(
        self,
        model,
        alpha_w: float = 1.0,
        alpha_h: float = 0.0,
        device: Optional[torch.device] = None,
        head_split_ratio: Optional[float] = None,
        spherical_band_ratio: float = 1.0,
        target_scales: Optional[Sequence[int]] = None,
    ):
        """
        Args:
            model:   Infinity 모델 인스턴스
            alpha_w: width 방향 spherical 보정 강도 (1.0 = full)
            alpha_h: height 방향 spherical 보정 강도 (0.0 = 표준 유지)
            device:  None이면 model.parameters()의 device 자동 감지
        """
        self.model = model
        self.alpha_w = alpha_w
        self.alpha_h = alpha_h
        self._orig_grid: Optional[dict] = None
        self._active_head_group_sizes: Optional[Tuple[int, int]] = None
        self.head_split_ratio = head_split_ratio
        self.spherical_band_ratio = spherical_band_ratio
        self.target_scales = (
            None
            if target_scales is None
            else tuple(sorted(set(int(x) for x in target_scales)))
        )

        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        self.device = device

        # 사전 계산
        self._sph_grid = self._build_sph_grid()

    def _build_sph_grid(self) -> dict:
        """Spherical RoPE cache를 미리 계산."""
        head_dim = self.model.C // self.model.num_heads
        rope2d_normalized_by_hw = self.model.rope2d_normalized_by_hw
        pad_to_multiplier = self.model.pad_to_multiplier

        print(
            f"[SphericalRoPE] head_dim={head_dim}, "
            f"rope2d_normalized_by_hw={rope2d_normalized_by_hw}, "
            f"alpha_w={self.alpha_w}, alpha_h={self.alpha_h}"
        )

        return precompute_rope2d_freqs_grid_spherical(
            dim=head_dim,
            dynamic_resolution_h_w=_DYN_RES,
            rope2d_normalized_by_hw=rope2d_normalized_by_hw,
            pad_to_multiplier=pad_to_multiplier,
            device=self.device,
            alpha_w=self.alpha_w,
            alpha_h=self.alpha_h,
            spherical_band_ratio=self.spherical_band_ratio,
        )

    def _iter_self_attn_modules(self):
        for block in getattr(self.model, "unregistered_blocks", []):
            attn = getattr(block, "attn", None)
            if attn is not None:
                yield attn
            sa = getattr(block, "sa", None)
            if sa is not None:
                yield sa

    def _apply_head_group_sizes(self, sizes: Optional[Tuple[int, int]]):
        for module in self._iter_self_attn_modules():
            module.set_rope_head_group_sizes(sizes)

    def _compute_head_group_sizes(self) -> Tuple[int, int]:
        total_heads = getattr(self.model, "num_heads", None)
        if total_heads is None:
            raise ValueError("Model does not expose num_heads for head split")
        ratio = float(self.head_split_ratio)
        ratio = max(0.0, min(1.0, ratio))
        structural = max(1, int(round(total_heads * ratio)))
        structural = min(structural, total_heads - 1)
        context = total_heads - structural
        return context, structural

    def _build_grouped_grid(self, base_grid: dict, structural_grid: dict) -> dict:
        combined = {}
        for key in base_grid:
            base = base_grid[key]
            struct = structural_grid[key].to(device=base.device, dtype=base.dtype)
            combined[key] = torch.stack((base, struct), dim=0)
        return combined

    def _get_scale_spans(self, schedule_key: str) -> List[Tuple[int, int]]:
        scale_schedule = ast.literal_eval(schedule_key)
        spans = []
        start = 0
        for pt, ph, pw in scale_schedule:
            length = pt * ph * pw
            spans.append((start, start + length))
            start += length
        return spans

    def _merge_scale_selected_grid(
        self, base_grid: dict, structural_grid: dict
    ) -> dict:
        if self.target_scales is None:
            return {
                key: structural_grid[key].to(
                    device=base_grid[key].device, dtype=base_grid[key].dtype
                )
                for key in base_grid
            }

        merged = {}
        for key in base_grid:
            base = base_grid[key]
            struct = structural_grid[key].to(device=base.device, dtype=base.dtype)
            mixed = base.clone()
            spans = self._get_scale_spans(key)
            for scale_idx in self.target_scales:
                if scale_idx < 0 or scale_idx >= len(spans):
                    continue
                start, end = spans[scale_idx]
                mixed[..., start:end, :] = struct[..., start:end, :]
            merged[key] = mixed
        return merged

    def apply(self):
        """Spherical RoPE를 모델에 영구 적용."""
        if self._orig_grid is None:
            self._orig_grid = self.model.rope2d_freqs_grid
        merged_sph_grid = self._merge_scale_selected_grid(
            self._orig_grid, self._sph_grid
        )
        if self.head_split_ratio is None:
            self._apply_head_group_sizes(None)
            self.model.rope2d_freqs_grid = merged_sph_grid
            self._active_head_group_sizes = None
        else:
            head_sizes = self._compute_head_group_sizes()
            combined_grid = self._build_grouped_grid(self._orig_grid, merged_sph_grid)
            self._apply_head_group_sizes(head_sizes)
            self.model.rope2d_freqs_grid = combined_grid
            self._active_head_group_sizes = head_sizes
        print("[SphericalRoPE] applied (permanent)")

    def restore(self):
        """원래 표준 RoPE로 복원."""
        self._apply_head_group_sizes(None)
        self._active_head_group_sizes = None
        if self._orig_grid is not None:
            self.model.rope2d_freqs_grid = self._orig_grid
            print("[SphericalRoPE] restored (standard)")
        else:
            print("[SphericalRoPE] nothing to restore")

    def __enter__(self):
        self.apply()
        return self

    def __exit__(self, *args):
        self.restore()


# ─────────────────────────────────────────────────────────────────────────────
# 빠른 검증 유틸리티
# ─────────────────────────────────────────────────────────────────────────────


def verify_seam_continuity(
    sph_grid: dict,
    scale_key: str,
    scale_idx: int,
    scale_schedule: list,
    head_dim: int = 96,
    base: float = 10000.0,
    alpha_w: float = 1.0,
    tol: float = 1e-4,
) -> dict:
    """
    Spherical RoPE가 실제로 seam-continuous한지 검증.

    seam continuity 조건:
      cos(φ_sph(j=pw)) ≈ 1  (φ(0)=0 에서 시작, pw 이동 후 2π×정수 차이)

    이는 φ_sph(pw) = k_d × 2π × pw / pw = k_d × 2π → cos = 1, sin = 0

    실제 cache는 j=0..pw-1만 저장 → 외삽으로 검증:
      φ(pw) = φ(pw-1) + φ(1) - φ(0) = φ(pw-1) + step
    등가로, cos(step × pw) ≈ 1 를 directly 계산.
    """
    _, ph, pw = scale_schedule[scale_idx]

    # inv_freq 재계산
    half_dim = head_dim // 2
    qtr_dim = half_dim // 2
    inv_freq = 1.0 / (
        base ** (torch.arange(0, half_dim, 2, dtype=torch.float32) / half_dim)
    )

    # mode 2: seam 기준은 upw (finest scale width)
    # mode 0: seam 기준은 pw
    from infinity.utils.dynamic_resolution import dynamic_resolution_h_w as _dyn

    upw = list(_dyn.values())[0]["1M"]["scales"][-1][
        2
    ]  # fallback; 실제론 h_div_w별 upw

    k_d = (inv_freq * upw / (2.0 * math.pi)).round().clamp(min=1.0)
    sph_freq = k_d * (2.0 * math.pi / upw)
    eff_freq = alpha_w * sph_freq + (1.0 - alpha_w) * inv_freq

    # seam: φ(upw) = eff_freq * upw  (mode 2 기준)
    phase_seam = eff_freq * upw  # shape [qtr_dim]
    cos_seam = torch.cos(phase_seam)  # should be ≈ 1 for spherical
    sin_seam = torch.sin(phase_seam)  # should be ≈ 0 for spherical

    seam_cos_err = (cos_seam - 1.0).abs().mean().item()
    seam_sin_err = sin_seam.abs().mean().item()

    passed = seam_cos_err < tol and seam_sin_err < tol

    return {
        "scale_idx": scale_idx,
        "pw": pw,
        "k_d_max": int(k_d.max().item()),
        "seam_cos_err": seam_cos_err,
        "seam_sin_err": seam_sin_err,
        "passed": passed,
    }


# ─────────────────────────────────────────────────────────────────────────────
# __main__: 독립 실행 테스트 (모델 불필요)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json, math, sys, os

    # flash_attn 없어도 동작하도록 — dynamic_resolution만 직접 import
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from infinity.utils.dynamic_resolution import dynamic_resolution_h_w

    print("=" * 60)
    print("Spherical RoPE for Infinity — 독립 검증")
    print("=" * 60)

    HEAD_DIM = 96  # infinity_layer12: embed_dim=768, num_heads=8 → 768/8=96
    PAD = 128

    sph_grid = precompute_rope2d_freqs_grid_spherical(
        dim=HEAD_DIM,
        dynamic_resolution_h_w=dynamic_resolution_h_w,
        rope2d_normalized_by_hw=2,  # star-style (기본값)
        pad_to_multiplier=PAD,
        device=torch.device("cpu"),
        alpha_w=1.0,
        alpha_h=0.0,
    )

    print(f"\n생성된 cache keys 수: {len(sph_grid)}")

    # ratio=1.0, 0.06M (256×256) 검증
    from infinity.utils.dynamic_resolution import dynamic_resolution_h_w

    RATIO = 1.0
    scale_schedule_raw = dynamic_resolution_h_w[RATIO]["0.06M"]["scales"]
    scale_schedule = [(1, h, w) for _, h, w in scale_schedule_raw]
    key = str(tuple(scale_schedule))

    print(f"\nRatio={RATIO}, pn=0.06M")
    print(f"scale_schedule: {scale_schedule}")
    print(f"cache shape: {sph_grid[key].shape}")

    # 각 scale의 seam continuity 검증
    print("\nSeam continuity 검증 (width):")
    print(f"  {'scale':>10}  {'k_d_max':>7}  {'cos_err':>10}  {'sin_err':>10}  result")
    all_passed = True
    for si, (_, ph, pw) in enumerate(scale_schedule):
        result = verify_seam_continuity(sph_grid, key, si, scale_schedule, HEAD_DIM)
        status = "✓ PASS" if result.get("passed") else "✗ FAIL"
        print(
            f"  ({ph:2d},{pw:2d}){' ':>5}  {result.get('k_d_max', '?'):>7}  "
            f"{result.get('seam_cos_err', 0):.2e}  "
            f"{result.get('seam_sin_err', 0):.2e}  {status}"
        )
        if not result.get("passed"):
            all_passed = False

    print(f"\n전체 결과: {'✓ ALL PASSED' if all_passed else '✗ SOME FAILED'}")

    # Standard RoPE와 차이 비교 (fine scale만)
    # flash_attn 없이 basic.py를 직접 임포트 (module cache hack)
    import importlib.util, sys as _sys

    _spec = importlib.util.spec_from_file_location(
        "_basic_direct",
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "infinity", "models", "basic.py"
        ),
    )
    # flash_attn stub 먼저 등록
    try:
        import flash_attn
    except (ImportError, OSError):
        import types

        _fa = types.ModuleType("flash_attn")
        _fa.flash_attn_func = None
        _fa.flash_attn_varlen_kvpacked_func = None
        _sys.modules["flash_attn"] = _fa
        _faops = types.ModuleType("flash_attn.ops")
        _sys.modules["flash_attn.ops"] = _faops
        _falayer = types.ModuleType("flash_attn.ops.layer_norm")
        _sys.modules["flash_attn.ops.layer_norm"] = _falayer
        _farms = types.ModuleType("flash_attn.ops.rms_norm")
        _sys.modules["flash_attn.ops.rms_norm"] = _farms
        _fafuse = types.ModuleType("flash_attn.ops.fused_dense")
        _sys.modules["flash_attn.ops.fused_dense"] = _fafuse

    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    precompute_rope2d_freqs_grid = _mod.precompute_rope2d_freqs_grid

    std_grid = precompute_rope2d_freqs_grid(
        dim=HEAD_DIM,
        dynamic_resolution_h_w=dynamic_resolution_h_w,
        rope2d_normalized_by_hw=2,
        pad_to_multiplier=PAD,
    )

    std_cache = std_grid[key][0, 0, 0, 0, :, :]  # cos part
    sph_cache = sph_grid[key][0, 0, 0, 0, :, :]

    print(f"\nStandard vs Spherical RoPE 차이 (L2 per scale):")
    qtr_dim = (HEAD_DIM // 2) // 2  # 96//2//2 = 24
    half_dim = HEAD_DIM // 2  # 96//2    = 48
    start = 0
    for si, (t, ph, pw) in enumerate(scale_schedule):
        end = start + t * ph * pw
        diff = (
            (
                std_cache[start:end, qtr_dim:half_dim]
                - sph_cache[start:end, qtr_dim:half_dim]
            )
            .norm()
            .item()
        )
        total = sph_cache[start:end, qtr_dim:half_dim].norm().item()
        print(f"  scale({ph:2d},{pw:2d}): L2 diff={diff:.4f}  (norm={total:.4f})")
        start = end
