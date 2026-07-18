#!/usr/bin/env python
"""
SatMAE++ with Wavelet-Aware Decoder (W2 ablation).

This script reuses the COMPLETE training pipeline from src/satmae_baseline.py
and changes only the decoder architecture.

Wavelet blocks are inserted at:
    56x56
    112x112

Each block:
    feature map
      -> Haar DWT
      -> separate LL low-frequency branch
      -> separate LH/HL/HH high-frequency branch
      -> Haar IDWT
      -> residual connection

No wavelet loss is used in this experiment.
The training loss remains exactly BCE + Dice from satmae_baseline.py.

Defaults inherited from satmae_baseline.py:
    epochs = 20
    patience = 5
    batch size = 1
    gradient accumulation = 4

Run:
    python src/satmae_wavelet_decoder.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


# ============================================================
# Import original baseline module
# ============================================================

SRC_DIR = Path(__file__).resolve().parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR),
    )

import satmae_baseline as baseline


# Keep a reference before monkey-patching the baseline module.
OriginalSatMAESegmenter = baseline.SatMAESegmenter


# ============================================================
# Haar DWT / IDWT
# ============================================================

def haar_dwt2d(
    x: torch.Tensor,
):
    """
    Differentiable one-level orthonormal Haar DWT.

    Input:
        [B, C, H, W]

    Returns:
        LL, LH, HL, HH
        each [B, C, H/2, W/2]
    """

    if x.ndim != 4:
        raise ValueError(
            f"Expected [B,C,H,W], got {tuple(x.shape)}"
        )

    height, width = x.shape[-2:]

    if height % 2 != 0 or width % 2 != 0:
        raise ValueError(
            "Wavelet blocks require even spatial dimensions. "
            f"Received {height}x{width}."
        )

    x00 = x[..., 0::2, 0::2]
    x01 = x[..., 0::2, 1::2]
    x10 = x[..., 1::2, 0::2]
    x11 = x[..., 1::2, 1::2]

    scale = 0.5

    ll = scale * (
        x00 + x01 + x10 + x11
    )

    lh = scale * (
        -x00 - x01 + x10 + x11
    )

    hl = scale * (
        -x00 + x01 - x10 + x11
    )

    hh = scale * (
        x00 - x01 - x10 + x11
    )

    return (
        ll,
        lh,
        hl,
        hh,
    )


def haar_idwt2d(
    ll: torch.Tensor,
    lh: torch.Tensor,
    hl: torch.Tensor,
    hh: torch.Tensor,
) -> torch.Tensor:
    """
    Exact inverse of haar_dwt2d.
    """

    if not (
        ll.shape
        == lh.shape
        == hl.shape
        == hh.shape
    ):
        raise ValueError(
            "LL, LH, HL and HH must have identical shapes."
        )

    x00 = 0.5 * (
        ll - lh - hl + hh
    )

    x01 = 0.5 * (
        ll - lh + hl - hh
    )

    x10 = 0.5 * (
        ll + lh - hl - hh
    )

    x11 = 0.5 * (
        ll + lh + hl + hh
    )

    batch, channels, height, width = (
        ll.shape
    )

    output = torch.empty(
        (
            batch,
            channels,
            height * 2,
            width * 2,
        ),
        dtype=ll.dtype,
        device=ll.device,
    )

    output[..., 0::2, 0::2] = x00
    output[..., 0::2, 1::2] = x01
    output[..., 1::2, 0::2] = x10
    output[..., 1::2, 1::2] = x11

    return output


# ============================================================
# Wavelet-aware feature block
# ============================================================

class WaveletAwareFeatureBlock(
    nn.Module
):
    """
    DWT
      -> LL: low-frequency branch
      -> LH/HL/HH: high-frequency branch
      -> IDWT
      -> residual connection

    This adapts the frequency-separated processing idea discussed
    from WaveDM / wavelet-based feature processing methods to a
    segmentation decoder.
    """

    def __init__(
        self,
        channels: int,
    ) -> None:

        super().__init__()

        self.channels = channels

        # Low-frequency branch:
        # learns global / structural information.
        self.low_branch = nn.Sequential(
            baseline.ConvBlock(
                channels,
                channels,
            ),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                channels
            ),
        )

        # High-frequency branch:
        # LH, HL and HH are concatenated.
        #
        # First grouped convolution lets the three frequency groups
        # be processed separately, then the 1x1 convolution allows
        # information exchange across bands.
        self.high_branch = nn.Sequential(
            nn.Conv2d(
                channels * 3,
                channels * 3,
                kernel_size=3,
                padding=1,
                groups=3,
                bias=False,
            ),
            nn.BatchNorm2d(
                channels * 3
            ),
            nn.GELU(),

            nn.Conv2d(
                channels * 3,
                channels * 3,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                channels * 3
            ),
            nn.GELU(),
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                channels
            ),
        )

        # Start close to identity for stable fine-tuning.
        self.residual_scale = (
            nn.Parameter(
                torch.tensor(
                    0.0,
                    dtype=torch.float32,
                )
            )
        )

        self.activation = nn.GELU()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        residual = x

        (
            ll,
            lh,
            hl,
            hh,
        ) = haar_dwt2d(
            x
        )

        # Process low frequencies independently.
        ll = self.low_branch(
            ll
        )

        # Process all high-frequency bands in a dedicated branch.
        high = torch.cat(
            (
                lh,
                hl,
                hh,
            ),
            dim=1,
        )

        high = self.high_branch(
            high
        )

        (
            lh,
            hl,
            hh,
        ) = torch.chunk(
            high,
            chunks=3,
            dim=1,
        )

        reconstructed = haar_idwt2d(
            ll,
            lh,
            hl,
            hh,
        )

        reconstructed = self.fusion(
            reconstructed
        )

        output = (
            residual
            + self.residual_scale
            * reconstructed
        )

        return self.activation(
            output
        )


# ============================================================
# SatMAE++ with Wavelet-Aware Decoder
# ============================================================

class SatMAEWaveletDecoder(
    nn.Module
):
    def __init__(
        self,
        satmae_root: Path,
        checkpoint_path: Path,
        image_size: int = 224,
        patch_size: int = 16,
        drop_path: float = 0.2,
    ) -> None:

        super().__init__()

        if (
            image_size
            % patch_size
            != 0
        ):
            raise ValueError(
                "image_size must be divisible by patch_size."
            )

        # Build the original model only to reuse the exact
        # SatMAE++ encoder/checkpoint loading logic.
        original = OriginalSatMAESegmenter(
            satmae_root=
                satmae_root,

            checkpoint_path=
                checkpoint_path,

            image_size=
                image_size,

            patch_size=
                patch_size,

            drop_path=
                drop_path,
        )

        self.encoder = original.encoder

        self.image_size = (
            image_size
        )

        self.patch_size = (
            patch_size
        )

        self.grid_size = (
            image_size
            // patch_size
        )

        encoder_dim = 1024

        # 14x14
        self.projection = (
            baseline.ConvBlock(
                encoder_dim,
                256,
            )
        )

        # 14 -> 28
        self.up1 = (
            baseline.UpsampleBlock(
                256,
                256,
            )
        )

        # 28 -> 56
        self.up2 = (
            baseline.UpsampleBlock(
                256,
                128,
            )
        )

        self.wavelet56 = (
            WaveletAwareFeatureBlock(
                128
            )
        )

        # 56 -> 112
        self.up3 = (
            baseline.UpsampleBlock(
                128,
                64,
            )
        )

        self.wavelet112 = (
            WaveletAwareFeatureBlock(
                64
            )
        )

        # 112 -> 224
        self.up4 = (
            baseline.UpsampleBlock(
                64,
                32,
            )
        )

        self.segmentation_head = (
            nn.Conv2d(
                32,
                1,
                kernel_size=1,
            )
        )

    def forward_encoder_features(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        batch_size = x.shape[0]

        x = self.encoder.patch_embed(
            x
        )

        cls_tokens = (
            self.encoder
            .cls_token
            .expand(
                batch_size,
                -1,
                -1,
            )
        )

        x = torch.cat(
            (
                cls_tokens,
                x,
            ),
            dim=1,
        )

        x = (
            x
            + self.encoder.pos_embed
        )

        x = self.encoder.pos_drop(
            x
        )

        for block in self.encoder.blocks:
            x = block(
                x
            )

        x = self.encoder.norm(
            x
        )

        # Remove CLS token.
        x = x[
            :,
            1:,
            :,
        ]

        x = x.transpose(
            1,
            2,
        ).contiguous()

        x = x.reshape(
            batch_size,
            x.shape[1],
            self.grid_size,
            self.grid_size,
        )

        return x

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        input_size = x.shape[
            -2:
        ]

        x = self.forward_encoder_features(
            x
        )

        # 14x14
        x = self.projection(
            x
        )

        # 28x28
        x = self.up1(
            x
        )

        # 56x56
        x = self.up2(
            x
        )

        x = self.wavelet56(
            x
        )

        # 112x112
        x = self.up3(
            x
        )

        x = self.wavelet112(
            x
        )

        # 224x224
        x = self.up4(
            x
        )

        if (
            x.shape[-2:]
            != input_size
        ):

            x = F.interpolate(
                x,
                size=
                    input_size,
                mode=
                    "bilinear",
                align_corners=
                    False,
            )

        return self.segmentation_head(
            x
        )


# ============================================================
# Run original SatMAE baseline training pipeline
# ============================================================

def main() -> None:

    # Replace only the model class used by baseline.main().
    baseline.SatMAESegmenter = (
        SatMAEWaveletDecoder
    )

    # Use a separate output directory unless explicitly provided.
    if "--output-dir" not in sys.argv:
        sys.argv.extend(
            [
                "--output-dir",
                "outputs/satmae_wavelet_decoder",
            ]
        )

    print(
        "Experiment: SatMAE++ + Wavelet-Aware Decoder"
    )
    print(
        "Wavelet blocks: 56x56 and 112x112"
    )
    print(
        "Loss: original BCE + Dice (no Wavelet Loss)"
    )

    baseline.main()


if __name__ == "__main__":
    main()
