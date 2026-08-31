"""Frame transforms for checkpoints trained on transformed data.

DLPInference applies these automatically: any game listed in
``GAME_TRANSFORMS`` gets its transform applied to incoming frames before the
model sees them, so callers always pass raw emulator frames.

Boxing: the checkpoint is trained on recolored frames where every near-black
pixel is bright red - the black boxer was otherwise absorbed into the
background model and never detected. The transform is idempotent, so frames
that were already recolored upstream pass through unchanged.
"""

from __future__ import annotations

import numpy as np
import torch


def salient_downsample(x: torch.Tensor, factor: int) -> torch.Tensor:
    """Non-overlapping `factor`x`factor` decimation that keeps, per block, the
    pixel whose full RGB triplet deviates most (L2) from the block's own mean
    - not necessarily the brightest one. A block that's mostly one background
    color with a single differing pixel (a thin object's edge) always has
    that pixel as the outlier, whether the object is lighter or darker than
    what surrounds it - unlike a plain max-pool, which only ever keeps the
    brighter side and actively erases a dark object on a light background.

    Used as the ``resize_mode: "salient"`` downscale by both the training
    loader and DLPInference._preprocess (run on CPU in both so the argmax
    tie-breaking is identical and train/inference stay bit-exact).
    """
    b, c, h, w = x.shape
    oh, ow = h // factor, w // factor
    blocks = x.reshape(b, c, oh, factor, ow, factor)
    blocks = blocks.permute(0, 2, 4, 3, 5, 1).reshape(b, oh, ow, factor * factor, c)
    mean = blocks.mean(dim=3, keepdim=True)
    deviation = (blocks - mean).pow(2).sum(dim=-1)   # (B, oh, ow, f*f)
    idx = deviation.argmax(dim=-1, keepdim=True)     # (B, oh, ow, 1)
    idx = idx.unsqueeze(-1).expand(-1, -1, -1, 1, c)
    picked = blocks.gather(3, idx).squeeze(3)        # (B, oh, ow, C)
    return picked.permute(0, 3, 1, 2)                # (B, C, oh, ow)


def soft_salient_downsample(x: torch.Tensor, factor: int,
                            beta: float = 10.0) -> torch.Tensor:
    """Soft version of :func:`salient_downsample`: each block outputs the
    softmax(beta * deviation)-weighted average of its pixels instead of the
    hard argmax pick. beta=0 reduces to the plain box average, beta->inf to
    the hard salient pick. A lone strong outlier (thin object of either
    polarity) still receives ~all the weight and survives at full contrast,
    but uniform blocks, 50/50 tie blocks and weak texture fall back to the
    stable box average - removing the hard pick's edge flicker, tie
    artifacts and speckle amplification. Continuous in the pixel values, so
    temporally stable under sub-block motion.

    ``resize_mode: "softsal"`` in hparams; beta from ``softsal_beta``.
    """
    b, c, h, w = x.shape
    oh, ow = h // factor, w // factor
    blocks = x.reshape(b, c, oh, factor, ow, factor)
    blocks = blocks.permute(0, 2, 4, 3, 5, 1).reshape(b, oh, ow, factor * factor, c)
    mean = blocks.mean(dim=3, keepdim=True)
    deviation = (blocks - mean).pow(2).sum(dim=-1)   # (B, oh, ow, f*f)
    weight = torch.softmax(beta * deviation, dim=-1)
    out = (weight.unsqueeze(-1) * blocks).sum(dim=3)  # (B, oh, ow, C)
    return out.permute(0, 3, 1, 2)                    # (B, C, oh, ow)


def recolor_black(frames: np.ndarray, thresh: int = 30,
                  color=(220, 50, 50)) -> np.ndarray:
    """Replace every pixel with max(R,G,B) < thresh by ``color`` (RGB).

    Accepts a single (H, W, 3) frame or a (T, H, W, 3) batch. Matches the
    transform used to build data_boxing_red/ exactly; idempotent.
    """
    out = frames.copy()
    out[frames.max(axis=-1) < thresh] = color
    return out


# game name -> transform applied by DLPInference before the forward pass
GAME_TRANSFORMS = {
    "Boxing": recolor_black,
}
