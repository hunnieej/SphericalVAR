from typing import Sequence

import torch
import torch.nn.functional as F


IMAGE_DOMAIN_MODES = {"nearest", "bilinear", "bicubic", "area"}
ALIGN_CORNERS_MODES = {"linear", "bilinear", "bicubic", "trilinear"}


def interpolate_latent(x: torch.Tensor, size: Sequence[int], mode: str) -> torch.Tensor:
    """Interpolate latent tensors, supporting 2D image-domain modes when T=1.

    Infinity stores image latents as 5D tensors shaped `(B, C, T, H, W)`. When
    `T == 1`, we can safely squeeze the temporal axis and use 2D interpolation
    modes such as `bilinear`, `bicubic`, or `nearest` for image-only ablations.
    """

    if (
        x.ndim == 5
        and len(size) == 3
        and x.shape[2] == 1
        and size[0] == 1
        and mode in IMAGE_DOMAIN_MODES
    ):
        x_2d = x.squeeze(2)
        kwargs = {}
        if mode in ALIGN_CORNERS_MODES:
            kwargs["align_corners"] = False
        out = F.interpolate(x_2d, size=size[1:], mode=mode, **kwargs)
        return out.unsqueeze(2)

    kwargs = {}
    if mode in ALIGN_CORNERS_MODES:
        kwargs["align_corners"] = False
    return F.interpolate(x, size=size, mode=mode, **kwargs)
