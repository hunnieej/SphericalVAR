"""
Improved Spherical RoPE for Infinity (v2)

핵심 개선:
1. Scale-aware alpha: coarse scale은 표준 유지, fine scale만 spherical 보정
   → KV cache accumulation 시 frequency 연속성 보장

2. Mode 2 (star-style) base: upw 기준 global freq
   → 모든 scale에서 seam 조건 일관성 유지

원리:
- Coarse scale (1×1, 2×2): alpha=0 → standard RoPE (semantic stability)
- Mid scale: alpha=linear interpolation
- Fine scale (64×64): alpha=1.0 (full spherical)

이렇게 하면:
- coarse scale의 KV는 표준 freq basis → semantic consistency 유지
- fine scale의 새 KV는 spherical freq → seam continuity 개선
- 중간 scale에서 smooth transition → cross-scale mismatch 최소화

References:
- HACK (Qin et al. 2025): Contextual vs Structural heads
- spherical_rope_infinity.py v1
- INFINITY_SPHERICAL_ROPE_PLAN.md
"""

import math
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from infinity.utils.dynamic_resolution import dynamic_resolution_h_w as _DYN_RES


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


def _compute_sph_freq_w_adaptive(
    inv_freq: torch.Tensor,
    pw: int,
    upw: int,
    scale_idx: int,
    num_scales: int,
    alpha_schedule: str = "linear",
    band_ratio: float = 1.0,
) -> torch.Tensor:
    """
    Scale-aware spherical width frequency.

    coarse scale (scale_idx ≈ 0): alpha ≈ 0 → standard
    fine scale (scale_idx ≈ num_scales): alpha ≈ 1.0 → full spherical

    Args:
        inv_freq: [Q] standard inverse frequencies
        pw: current scale width
        upw: finest scale width
        scale_idx: current scale index (0 = coarsest)
        num_scales: total number of scales
        alpha_schedule: 'linear' | 'exp' | 'sigmoid'

    Returns:
        eff_freq: [Q] adaptive spherical frequency
    """
    # Alpha scheduling: 0 (coarse) → 1 (fine)
    if num_scales <= 1:
        alpha = 1.0
    else:
        t = float(scale_idx) / (num_scales - 1)  # [0, 1]

        if alpha_schedule == "linear":
            alpha = t
        elif alpha_schedule == "exp":
            # exp schedule: slower increase initially, faster at end
            alpha = (math.exp(t) - 1.0) / (math.e - 1.0)
        elif alpha_schedule == "sigmoid":
            # sigmoid: smooth S-curve
            alpha = 1.0 / (1.0 + math.exp(-10.0 * (t - 0.5)))
        else:
            alpha = t

    # Global spherical freq (upw basis)
    k_d = (inv_freq * upw / (2.0 * math.pi)).round().clamp(min=1.0)
    sph_freq = k_d * (2.0 * math.pi / upw)

    # Adaptive blend: start with standard, end with spherical
    eff_freq = alpha * sph_freq + (1.0 - alpha) * inv_freq

    return _blend_freq_with_band(inv_freq, eff_freq, band_ratio)


def precompute_rope2d_freqs_grid_spherical_v2(
    dim: int,
    dynamic_resolution_h_w: dict,
    rope2d_normalized_by_hw: int,
    pad_to_multiplier: int = 1,
    base: float = 10000.0,
    device: Optional[torch.device] = None,
    scaling_factor: float = 1.0,
    alpha_schedule: str = "linear",
    alpha_h: float = 0.0,
    spherical_band_ratio: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    Scale-aware Spherical RoPE version of precompute_rope2d_freqs_grid().

    All scales use upw (finest) as global freq reference,
    but alpha interpolates from 0 (coarse) to 1 (fine).

    Args:
        dim: head_dim
        dynamic_resolution_h_w: scale schedule
        rope2d_normalized_by_hw: 0=direct, 1=bilinear, 2=star-style
        pad_to_multiplier: padding alignment
        base: RoPE base
        device: torch device
        scaling_factor: position scaling
        alpha_schedule: 'linear' | 'exp' | 'sigmoid'
        alpha_h: height spherical strength (default 0 = standard)

    Returns:
        rope2d_freqs_grid: {scale_schedule_key: (2,1,1,1,seq_len,half_head_dim)}
    """
    half_dim = dim // 2
    qtr_dim = half_dim // 2

    inv_freq = 1.0 / (
        base
        ** (torch.arange(0, half_dim, 2, dtype=torch.float32).to(device) / half_dim)
    )  # [qtr_dim]

    rope2d_freqs_grid: Dict[str, torch.Tensor] = {}

    for h_div_w in dynamic_resolution_h_w:
        # Get finest scale (upw) from 1M resolution
        scale_schedule_1M = dynamic_resolution_h_w[h_div_w]["1M"]["scales"]
        _, uph, upw = scale_schedule_1M[-1]
        num_scales = len(scale_schedule_1M)

        rope_cache_list: List[torch.Tensor] = []

        for scale_idx, (_, ph, pw) in enumerate(scale_schedule_1M):
            ph_mul_pw = ph * pw

            # ── Adaptive spherical frequency ────────────────────────
            eff_freq_w = _compute_sph_freq_w_adaptive(
                inv_freq,
                pw,
                upw,
                scale_idx,
                num_scales,
                alpha_schedule=alpha_schedule,
                band_ratio=spherical_band_ratio,
            )

            # Height (use standard unless alpha_h > 0)
            if alpha_h > 0.0:
                k_d_h = (inv_freq * uph / (2.0 * math.pi)).round().clamp(min=1.0)
                sph_freq_h = k_d_h * (2.0 * math.pi / uph)
                eff_freq_h = alpha_h * sph_freq_h + (1.0 - alpha_h) * inv_freq
            else:
                eff_freq_h = inv_freq

            # ── Width positions (depending on mode) ──────────────────
            if rope2d_normalized_by_hw == 2:
                # star-style
                t_w = (
                    torch.arange(pw, device=device, dtype=torch.float32) * (upw / pw)
                ).round() / scaling_factor
            elif rope2d_normalized_by_hw == 1:
                # bilinear
                t_w = (
                    torch.arange(pw, device=device, dtype=torch.float32) * (upw / pw)
                ) / scaling_factor
            else:  # mode 0: direct
                t_w = (
                    torch.arange(pw, device=device, dtype=torch.float32)
                    / scaling_factor
                )

            freqs_w = torch.outer(t_w, eff_freq_w)  # [pw, qtr_dim]

            # ── Height positions (depending on mode) ────────────────
            if rope2d_normalized_by_hw == 2:
                t_h = (
                    torch.arange(ph, device=device, dtype=torch.float32) * (uph / ph)
                ).round() / scaling_factor
            elif rope2d_normalized_by_hw == 1:
                t_h = (
                    torch.arange(ph, device=device, dtype=torch.float32) * (uph / ph)
                ) / scaling_factor
            else:  # mode 0
                t_h = (
                    torch.arange(ph, device=device, dtype=torch.float32)
                    / scaling_factor
                )

            freqs_h = torch.outer(t_h, eff_freq_h)  # [ph, qtr_dim]

            # ── 2D freqs grid ──────────────────────────────────────
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

        # Register for all pn ('0.06M', '0.25M', '1M')
        for pn in dynamic_resolution_h_w[h_div_w]:
            scale_schedule = dynamic_resolution_h_w[h_div_w][pn]["scales"]
            tmp_schedule = [(1, h, w) for _, h, w in scale_schedule]
            rope2d_freqs_grid[str(tuple(tmp_schedule))] = cat_rope_cache

    return rope2d_freqs_grid


class ScaleAwareSphericalRoPEInfinityPatcher:
    """
    Improved Infinity Spherical RoPE with scale-aware alpha scheduling.

    Usage:
        patcher = ScaleAwareSphericalRoPEInfinityPatcher(
            model, alpha_schedule="linear"
        )
        with patcher:
            img = gen_one_img(model, ...)
    """

    def __init__(
        self,
        model,
        alpha_schedule: str = "linear",
        alpha_h: float = 0.0,
        device: Optional[torch.device] = None,
        head_split_ratio: Optional[float] = None,
        spherical_band_ratio: float = 1.0,
    ):
        """
        Args:
            model: Infinity model instance
            alpha_schedule: 'linear' (default) | 'exp' | 'sigmoid'
            alpha_h: height spherical strength
            device: torch device
        """
        self.model = model
        self.alpha_schedule = alpha_schedule
        self.alpha_h = alpha_h
        self._orig_grid: Optional[dict] = None
        self._active_head_group_sizes: Optional[Tuple[int, int]] = None
        self.head_split_ratio = head_split_ratio
        self.spherical_band_ratio = spherical_band_ratio

        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        self.device = device

        self._sph_grid = self._build_sph_grid()

    def _build_sph_grid(self) -> dict:
        """Build scale-aware spherical RoPE cache."""
        head_dim = self.model.C // self.model.num_heads
        rope2d_normalized_by_hw = self.model.rope2d_normalized_by_hw
        pad_to_multiplier = self.model.pad_to_multiplier

        print(
            f"[ScaleAwareSphericalRoPE] head_dim={head_dim}, "
            f"rope2d_normalized_by_hw={rope2d_normalized_by_hw}, "
            f"alpha_schedule={self.alpha_schedule}, alpha_h={self.alpha_h}"
        )

        return precompute_rope2d_freqs_grid_spherical_v2(
            dim=head_dim,
            dynamic_resolution_h_w=_DYN_RES,
            rope2d_normalized_by_hw=rope2d_normalized_by_hw,
            pad_to_multiplier=pad_to_multiplier,
            device=self.device,
            alpha_schedule=self.alpha_schedule,
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

    def apply(self):
        """Apply to model (permanent)."""
        if self._orig_grid is None:
            self._orig_grid = self.model.rope2d_freqs_grid
        if self.head_split_ratio is None:
            self._apply_head_group_sizes(None)
            self.model.rope2d_freqs_grid = self._sph_grid
            self._active_head_group_sizes = None
        else:
            head_sizes = self._compute_head_group_sizes()
            combined_grid = self._build_grouped_grid(self._orig_grid, self._sph_grid)
            self._apply_head_group_sizes(head_sizes)
            self.model.rope2d_freqs_grid = combined_grid
            self._active_head_group_sizes = head_sizes
        print("[ScaleAwareSphericalRoPE] applied (permanent)")

    def restore(self):
        """Restore original RoPE."""
        self._apply_head_group_sizes(None)
        self._active_head_group_sizes = None
        if self._orig_grid is not None:
            self.model.rope2d_freqs_grid = self._orig_grid
            print("[ScaleAwareSphericalRoPE] restored (standard)")

    def __enter__(self):
        self.apply()
        return self

    def __exit__(self, *args):
        self.restore()


if __name__ == "__main__":
    import json, sys, os

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from infinity.utils.dynamic_resolution import dynamic_resolution_h_w

    print("=" * 70)
    print("Scale-Aware Spherical RoPE for Infinity — Standalone Test")
    print("=" * 70)

    HEAD_DIM = 96
    PAD = 128

    # Test 3 alpha schedules
    for sched in ["linear", "exp", "sigmoid"]:
        print(f"\n✓ Testing alpha_schedule='{sched}'")

        sph_grid = precompute_rope2d_freqs_grid_spherical_v2(
            dim=HEAD_DIM,
            dynamic_resolution_h_w=dynamic_resolution_h_w,
            rope2d_normalized_by_hw=2,
            pad_to_multiplier=PAD,
            device=torch.device("cpu"),
            alpha_schedule=sched,
            alpha_h=0.0,
        )

        print(f"  - Cache keys: {len(sph_grid)}")

        # Check one key
        ratio = 1.0
        scale_schedule_raw = dynamic_resolution_h_w[ratio]["0.06M"]["scales"]
        scale_schedule = [(1, h, w) for _, h, w in scale_schedule_raw]
        key = str(tuple(scale_schedule))
        print(f"  - Key shape: {sph_grid[key].shape}")

    print("\n✅ Standalone test completed!")
