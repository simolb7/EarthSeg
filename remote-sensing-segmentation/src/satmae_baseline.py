#!/usr/bin/env python
"""
SatMAE++ baseline for INRIA Building Footprint Segmentation.

This script uses the official SatMAE++ repository:
    external/satmae_pp/

and a pretrained FMoW-RGB SatMAE++ ViT-L checkpoint.

Expected project structure:
remote-sensing-segmentation/
├── src/
│   └── satmae_baseline.py
├── datasets/
│   └── inria_processed/
│       ├── metadata.csv
│       ├── train/
│       └── val/
├── external/
│   └── satmae_pp/
│       ├── models_vit.py
│       └── util/
└── checkpoints/
    └── satmae/
        └── checkpoint_satmae_rgb.pth

The default checkpoint path can be changed with:
    --pretrained-checkpoint PATH_TO_CHECKPOINT

Outputs:
outputs/satmae_baseline/
├── config.json
├── history.csv
├── best_metrics.json
├── best_model.pth
├── last_model.pth
├── val_predictions.csv
└── plots/
    ├── loss.png
    ├── metrics.png
    └── learning_rate.png

Default training settings:
- epochs: 20
- patience: 5
- input size: 224
- batch size: 1
- gradient accumulation: 4
  -> effective batch size = 4, comparable to the Swin-Unet baseline
- loss: BCE + Dice
- metrics: IoU, Dice, Precision, Recall, Accuracy

Example:
    python src/satmae_baseline.py

If your checkpoint has a different filename:
    python src/satmae_baseline.py --pretrained-checkpoint checkpoints/satmae/YOUR_CHECKPOINT.pth
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import sys
import time
from pathlib import Path

# Import torch before NumPy/Pandas to reduce possible OpenMP runtime conflicts
# on Windows environments.
import torch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm


# FMoW-RGB statistics used by the SatMAE++ codebase.
# The provided SatMAE++ RGB pretraining weights are documented as having been
# trained with BGR channel ordering, so this script converts INRIA RGB -> BGR
# before applying these statistics.
SATMAE_FMOW_MEAN = (
    0.4182007312774658,
    0.4214799106121063,
    0.3991275727748871,
)

SATMAE_FMOW_STD = (
    0.28774282336235046,
    0.27541765570640564,
    0.2764017581939697,
)


# ============================================================
# Arguments
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune a SatMAE++ ViT-L encoder for INRIA "
            "building footprint segmentation."
        )
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("datasets/inria_processed"),
    )

    parser.add_argument(
        "--satmae-root",
        type=Path,
        default=Path("external/satmae_pp"),
    )

    parser.add_argument(
        "--pretrained-checkpoint",
        type=Path,
        default=Path(
            "checkpoints/satmae/checkpoint_ViT-L_pretrain_fmow_rgb.pth"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/satmae_baseline"),
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
    )

    parser.add_argument(
        "--patch-size",
        type=int,
        default=16,
    )

    # ViT-L is much heavier than Swin-T. Batch 1 + accumulation 4 keeps
    # the effective batch size equal to 4.
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--accumulation-steps",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    # Lower LR for the pretrained encoder; higher LR for the new decoder.
    parser.add_argument(
        "--encoder-lr",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--decoder-lr",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--drop-path",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--bce-weight",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--dice-weight",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--pos-weight",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision.",
    )

    return parser.parse_args()


# ============================================================
# Utilities
# ============================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
        )


# ============================================================
# Dataset
# ============================================================

class InriaSatMAEDataset(Dataset):
    """
    INRIA dataset loader using the same processed splits as the Swin baseline.

    The original PNG patches are stored in RGB.
    SatMAE++ FMoW-RGB pretrained weights are used with BGR ordering here.
    """

    def __init__(
        self,
        root: Path,
        split: str,
        image_size: int = 224,
        augment: bool = False,
    ) -> None:

        self.root = root.resolve()
        self.split = split
        self.image_size = image_size
        self.augment = augment

        metadata_path = (
            self.root
            / "metadata.csv"
        )

        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"metadata.csv not found: {metadata_path}"
            )

        dataframe = pd.read_csv(
            metadata_path,
            keep_default_na=False,
        )

        self.dataframe = dataframe[
            dataframe["split"] == split
        ].reset_index(
            drop=True
        )

        if len(
            self.dataframe
        ) == 0:
            raise RuntimeError(
                f"No samples found for split='{split}'."
            )

        self.color_jitter = ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10,
            hue=0.02,
        )

    def __len__(self) -> int:
        return len(
            self.dataframe
        )

    def augment_pair(
        self,
        image: Image.Image,
        mask: Image.Image,
    ) -> tuple[
        Image.Image,
        Image.Image,
    ]:

        if random.random() < 0.5:
            image = ImageOps.mirror(
                image
            )
            mask = ImageOps.mirror(
                mask
            )

        if random.random() < 0.5:
            image = ImageOps.flip(
                image
            )
            mask = ImageOps.flip(
                mask
            )

        k = random.randint(
            0,
            3,
        )

        if k:
            angle = (
                90
                * k
            )

            image = image.rotate(
                angle,
                expand=False,
            )

            mask = mask.rotate(
                angle,
                expand=False,
            )

        image = self.color_jitter(
            image
        )

        return (
            image,
            mask,
        )

    def __getitem__(
        self,
        index: int,
    ) -> dict:

        row = self.dataframe.iloc[
            index
        ]

        image_path = (
            self.root
            / row[
                "image_path"
            ]
        )

        mask_path = (
            self.root
            / row[
                "mask_path"
            ]
        )

        if not image_path.is_file():
            raise FileNotFoundError(
                image_path
            )

        if not mask_path.is_file():
            raise FileNotFoundError(
                mask_path
            )

        with Image.open(
            image_path
        ) as image_file:

            image = (
                image_file
                .convert(
                    "RGB"
                )
            )

        with Image.open(
            mask_path
        ) as mask_file:

            mask = (
                mask_file
                .convert(
                    "L"
                )
            )

        if self.augment:

            image, mask = (
                self.augment_pair(
                    image,
                    mask,
                )
            )

        image = TF.resize(
            image,
            [
                self.image_size,
                self.image_size,
            ],
            interpolation=
                InterpolationMode.BICUBIC,
            antialias=True,
        )

        mask = TF.resize(
            mask,
            [
                self.image_size,
                self.image_size,
            ],
            interpolation=
                InterpolationMode.NEAREST,
        )

        # PIL RGB -> Tensor [R, G, B]
        image = TF.to_tensor(
            image
        )

        # SatMAE++ pretrained FMoW-RGB weights use BGR ordering.
        image = image[
            [2, 1, 0],
            :,
            :,
        ]

        image = TF.normalize(
            image,
            SATMAE_FMOW_MEAN,
            SATMAE_FMOW_STD,
        )

        mask = TF.pil_to_tensor(
            mask
        )

        mask = (
            mask
            > 127
        ).float()

        return {
            "image":
                image,

            "mask":
                mask,

            "patch_id":
                str(
                    row[
                        "patch_id"
                    ]
                ),
        }


# ============================================================
# SatMAE++ encoder and segmentation model
# ============================================================

def load_satmae_modules(
    satmae_root: Path,
):

    satmae_root = (
        satmae_root
        .resolve()
    )

    if not satmae_root.is_dir():
        raise FileNotFoundError(
            "SatMAE++ repository not found: "
            f"{satmae_root}"
        )

    if str(
        satmae_root
    ) not in sys.path:

        sys.path.insert(
            0,
            str(
                satmae_root
            ),
        )

    models_vit = (
        importlib
        .import_module(
            "models_vit"
        )
    )

    return models_vit


def load_pretrained_encoder_weights(
    encoder: nn.Module,
    checkpoint_path: Path,
) -> None:

    checkpoint_path = (
        checkpoint_path
        .resolve()
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "SatMAE++ checkpoint not found: "
            f"{checkpoint_path}\n"
            "Pass the correct path with "
            "--pretrained-checkpoint."
        )

    print(
        "Loading SatMAE++ checkpoint:",
        checkpoint_path,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(
        checkpoint,
        dict,
    ) and "model" in checkpoint:

        checkpoint_state = checkpoint[
            "model"
        ]

    elif isinstance(
        checkpoint,
        dict,
    ) and "state_dict" in checkpoint:

        checkpoint_state = checkpoint[
            "state_dict"
        ]

    elif isinstance(
        checkpoint,
        dict,
    ):

        checkpoint_state = checkpoint

    else:
        raise RuntimeError(
            "Unsupported checkpoint format."
        )

    encoder_state = (
        encoder.state_dict()
    )

    compatible_state = {}

    skipped_keys = []

    for key, value in checkpoint_state.items():

        # Handle possible DDP prefix.
        if key.startswith(
            "module."
        ):
            key = key[
                len(
                    "module."
                ):
            ]

        if (
            key
            in encoder_state
            and encoder_state[
                key
            ].shape
            == value.shape
        ):

            compatible_state[
                key
            ] = value

        else:

            skipped_keys.append(
                key
            )

    if not compatible_state:
        raise RuntimeError(
            "No compatible SatMAE++ encoder weights "
            "were found in the checkpoint."
        )

    message = (
        encoder
        .load_state_dict(
            compatible_state,
            strict=False,
        )
    )

    print(
        "Compatible pretrained keys loaded:",
        len(
            compatible_state
        ),
    )

    print(
        "Checkpoint keys skipped:",
        len(
            skipped_keys
        ),
    )

    print(
        "Missing encoder keys after loading:",
        len(
            message.missing_keys
        ),
    )

    print(
        "Unexpected encoder keys:",
        len(
            message.unexpected_keys
        ),
    )


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.GELU(),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.GELU(),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.block(
            x
        )


class UpsampleBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:

        super().__init__()

        self.block = nn.Sequential(

            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=2,
                stride=2,
            ),

            ConvBlock(
                out_channels,
                out_channels,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.block(
            x
        )


class SatMAESegmenter(nn.Module):
    """
    SatMAE++ ViT-L/16 encoder + convolutional segmentation decoder.

    The ViT-L encoder produces patch tokens on a 14x14 grid for a 224x224
    input. The decoder progressively upsamples:
        14 -> 28 -> 56 -> 112 -> 224
    """

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
                "image_size must be divisible "
                "by patch_size."
            )

        models_vit = (
            load_satmae_modules(
                satmae_root
            )
        )

        # Official SatMAE++ RGB configuration uses ViT-Large.
        self.encoder = (
            models_vit
            .vit_large_patch16(
                patch_size=
                    patch_size,

                img_size=
                    image_size,

                in_chans=
                    3,

                num_classes=
                    0,

                drop_path_rate=
                    drop_path,

                global_pool=
                    False,
            )
        )

        load_pretrained_encoder_weights(
            self.encoder,
            checkpoint_path,
        )

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

        # ViT-L embedding dimension in SatMAE++.
        encoder_dim = 1024

        self.projection = ConvBlock(
            encoder_dim,
            256,
        )

        self.up1 = UpsampleBlock(
            256,
            256,
        )

        self.up2 = UpsampleBlock(
            256,
            128,
        )

        self.up3 = UpsampleBlock(
            128,
            64,
        )

        self.up4 = UpsampleBlock(
            64,
            32,
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

        batch_size = (
            x.shape[0]
        )

        # Patch embedding:
        # [B, 3, 224, 224] -> [B, 196, 1024]
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
            + self.encoder
            .pos_embed
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

        # [B, N, C] -> [B, C, H, W]
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

        input_size = (
            x.shape[
                -2:
            ]
        )

        x = self.forward_encoder_features(
            x
        )

        x = self.projection(
            x
        )

        x = self.up1(
            x
        )

        x = self.up2(
            x
        )

        x = self.up3(
            x
        )

        x = self.up4(
            x
        )

        # Safety resize in case a different compatible image size is used.
        if x.shape[
            -2:
        ] != input_size:

            x = F.interpolate(
                x,
                size=
                    input_size,
                mode=
                    "bilinear",
                align_corners=
                    False,
            )

        return (
            self
            .segmentation_head(
                x
            )
        )


# ============================================================
# Loss
# ============================================================

class DiceLoss(nn.Module):
    def __init__(
        self,
        smooth: float = 1.0,
    ) -> None:

        super().__init__()

        self.smooth = (
            smooth
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        probabilities = (
            torch.sigmoid(
                logits
            )
        )

        probabilities = (
            probabilities
            .flatten(
                1
            )
        )

        targets = (
            targets
            .flatten(
                1
            )
        )

        intersection = (
            probabilities
            * targets
        ).sum(
            dim=1
        )

        denominator = (
            probabilities
            .sum(
                dim=1
            )
            + targets
            .sum(
                dim=1
            )
        )

        dice = (
            2.0
            * intersection
            + self.smooth
        ) / (
            denominator
            + self.smooth
        )

        return (
            1.0
            - dice.mean()
        )


class BCEDiceLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float,
        dice_weight: float,
        pos_weight: float,
    ) -> None:

        super().__init__()

        self.bce_weight = (
            bce_weight
        )

        self.dice_weight = (
            dice_weight
        )

        self.register_buffer(
            "pos_weight",
            torch.tensor(
                [
                    pos_weight
                ],
                dtype=
                    torch.float32,
            ),
        )

        self.dice = (
            DiceLoss()
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        bce = (
            F.binary_cross_entropy_with_logits(
                logits,
                targets,
                pos_weight=
                    self.pos_weight,
            )
        )

        dice = self.dice(
            logits,
            targets,
        )

        return (
            self.bce_weight
            * bce
            + self.dice_weight
            * dice
        )


# ============================================================
# Metrics
# ============================================================

class SegmentationMetrics:
    def __init__(
        self,
        threshold: float = 0.5,
        eps: float = 1e-7,
    ) -> None:

        self.threshold = (
            threshold
        )

        self.eps = (
            eps
        )

        self.reset()

    def reset(
        self,
    ) -> None:

        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    @torch.no_grad()
    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:

        predictions = (
            torch.sigmoid(
                logits
            )
            >= self.threshold
        )

        targets = (
            targets
            >= 0.5
        )

        self.tp += (
            torch.logical_and(
                predictions,
                targets,
            )
            .sum()
            .item()
        )

        self.fp += (
            torch.logical_and(
                predictions,
                ~targets,
            )
            .sum()
            .item()
        )

        self.fn += (
            torch.logical_and(
                ~predictions,
                targets,
            )
            .sum()
            .item()
        )

        self.tn += (
            torch.logical_and(
                ~predictions,
                ~targets,
            )
            .sum()
            .item()
        )

    def compute(
        self,
    ) -> dict[
        str,
        float,
    ]:

        iou = (
            self.tp
            / (
                self.tp
                + self.fp
                + self.fn
                + self.eps
            )
        )

        dice = (
            2
            * self.tp
            / (
                2
                * self.tp
                + self.fp
                + self.fn
                + self.eps
            )
        )

        precision = (
            self.tp
            / (
                self.tp
                + self.fp
                + self.eps
            )
        )

        recall = (
            self.tp
            / (
                self.tp
                + self.fn
                + self.eps
            )
        )

        accuracy = (
            self.tp
            + self.tn
        ) / (
            self.tp
            + self.fp
            + self.fn
            + self.tn
            + self.eps
        )

        return {
            "iou":
                float(
                    iou
                ),

            "dice":
                float(
                    dice
                ),

            "precision":
                float(
                    precision
                ),

            "recall":
                float(
                    recall
                ),

            "accuracy":
                float(
                    accuracy
                ),
        }


# ============================================================
# Training / Validation
# ============================================================

def run_train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    threshold: float,
    use_amp: bool,
    accumulation_steps: int,
    description: str,
) -> tuple[
    float,
    dict[
        str,
        float,
    ],
]:

    model.train()

    metrics = (
        SegmentationMetrics(
            threshold=
                threshold
        )
    )

    total_loss = 0.0
    total_samples = 0

    optimizer.zero_grad(
        set_to_none=True
    )

    progress = tqdm(
        loader,
        desc=
            description,
        leave=
            False,
    )

    for step, batch in enumerate(
        progress,
        start=1,
    ):

        images = (
            batch[
                "image"
            ]
            .to(
                device,
                non_blocking=True,
            )
        )

        masks = (
            batch[
                "mask"
            ]
            .to(
                device,
                non_blocking=True,
            )
        )

        batch_size = (
            images
            .shape[0]
        )

        with torch.autocast(
            device_type=
                device.type,
            enabled=
                use_amp,
        ):

            logits = model(
                images
            )

            raw_loss = criterion(
                logits,
                masks,
            )

            loss = (
                raw_loss
                / accumulation_steps
            )

        scaler.scale(
            loss
        ).backward()

        should_step = (
            step
            % accumulation_steps
            == 0
            or step
            == len(
                loader
            )
        )

        if should_step:

            scaler.step(
                optimizer
            )

            scaler.update()

            optimizer.zero_grad(
                set_to_none=True
            )

        total_loss += (
            raw_loss
            .item()
            * batch_size
        )

        total_samples += (
            batch_size
        )

        metrics.update(
            logits.detach(),
            masks,
        )

        progress.set_postfix(
            loss=
                f"{raw_loss.item():.4f}"
        )

    average_loss = (
        total_loss
        / max(
            total_samples,
            1,
        )
    )

    return (
        average_loss,
        metrics.compute(),
    )


@torch.no_grad()
def run_validation_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
    use_amp: bool,
    description: str,
) -> tuple[
    float,
    dict[
        str,
        float,
    ],
]:

    model.eval()

    metrics = (
        SegmentationMetrics(
            threshold=
                threshold
        )
    )

    total_loss = 0.0
    total_samples = 0

    progress = tqdm(
        loader,
        desc=
            description,
        leave=
            False,
    )

    for batch in progress:

        images = (
            batch[
                "image"
            ]
            .to(
                device,
                non_blocking=True,
            )
        )

        masks = (
            batch[
                "mask"
            ]
            .to(
                device,
                non_blocking=True,
            )
        )

        batch_size = (
            images
            .shape[0]
        )

        with torch.autocast(
            device_type=
                device.type,
            enabled=
                use_amp,
        ):

            logits = model(
                images
            )

            loss = criterion(
                logits,
                masks,
            )

        total_loss += (
            loss
            .item()
            * batch_size
        )

        total_samples += (
            batch_size
        )

        metrics.update(
            logits,
            masks,
        )

        progress.set_postfix(
            loss=
                f"{loss.item():.4f}"
        )

    average_loss = (
        total_loss
        / max(
            total_samples,
            1,
        )
    )

    return (
        average_loss,
        metrics.compute(),
    )


# ============================================================
# Per-patch validation metrics
# ============================================================

@torch.no_grad()
def save_validation_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    output_path: Path,
    use_amp: bool,
) -> None:

    model.eval()

    rows = []

    eps = 1e-7

    for batch in tqdm(
        loader,
        desc=
            "Validation predictions",
        leave=
            False,
    ):

        images = (
            batch[
                "image"
            ]
            .to(
                device
            )
        )

        targets = (
            batch[
                "mask"
            ]
            .to(
                device
            )
            >= 0.5
        )

        with torch.autocast(
            device_type=
                device.type,
            enabled=
                use_amp,
        ):

            logits = model(
                images
            )

        predictions = (
            torch.sigmoid(
                logits
            )
            >= threshold
        )

        for i in range(
            images
            .shape[0]
        ):

            prediction = (
                predictions[
                    i
                ]
                .flatten()
            )

            target = (
                targets[
                    i
                ]
                .flatten()
            )

            tp = (
                torch.logical_and(
                    prediction,
                    target,
                )
                .sum()
                .item()
            )

            fp = (
                torch.logical_and(
                    prediction,
                    ~target,
                )
                .sum()
                .item()
            )

            fn = (
                torch.logical_and(
                    ~prediction,
                    target,
                )
                .sum()
                .item()
            )

            tn = (
                torch.logical_and(
                    ~prediction,
                    ~target,
                )
                .sum()
                .item()
            )

            iou = (
                tp
                / (
                    tp
                    + fp
                    + fn
                    + eps
                )
            )

            dice = (
                2
                * tp
                / (
                    2
                    * tp
                    + fp
                    + fn
                    + eps
                )
            )

            precision = (
                tp
                / (
                    tp
                    + fp
                    + eps
                )
            )

            recall = (
                tp
                / (
                    tp
                    + fn
                    + eps
                )
            )

            accuracy = (
                tp
                + tn
            ) / (
                tp
                + fp
                + fn
                + tn
                + eps
            )

            rows.append(
                {
                    "patch_id":
                        batch[
                            "patch_id"
                        ][i],

                    "iou":
                        iou,

                    "dice":
                        dice,

                    "precision":
                        precision,

                    "recall":
                        recall,

                    "accuracy":
                        accuracy,

                    "predicted_building_ratio":
                        prediction
                        .float()
                        .mean()
                        .item(),

                    "target_building_ratio":
                        target
                        .float()
                        .mean()
                        .item(),
                }
            )

    pd.DataFrame(
        rows
    ).to_csv(
        output_path,
        index=False,
    )


# ============================================================
# Plots
# ============================================================

def save_plots(
    history: pd.DataFrame,
    output_dir: Path,
) -> None:

    plots_dir = (
        output_dir
        / "plots"
    )

    plots_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Loss plot.
    plt.figure(
        figsize=(
            8,
            5,
        )
    )

    plt.plot(
        history[
            "epoch"
        ],
        history[
            "train_loss"
        ],
        label=
            "Train",
    )

    plt.plot(
        history[
            "epoch"
        ],
        history[
            "val_loss"
        ],
        label=
            "Validation",
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "SatMAE++ training and validation loss"
    )

    plt.grid(
        alpha=
            0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        plots_dir
        / "loss.png",
        dpi=
            180,
    )

    plt.close()

    # Validation metrics plot.
    plt.figure(
        figsize=(
            8,
            5,
        )
    )

    plt.plot(
        history[
            "epoch"
        ],
        history[
            "val_iou"
        ],
        label=
            "IoU",
    )

    plt.plot(
        history[
            "epoch"
        ],
        history[
            "val_dice"
        ],
        label=
            "Dice",
    )

    plt.plot(
        history[
            "epoch"
        ],
        history[
            "val_precision"
        ],
        label=
            "Precision",
    )

    plt.plot(
        history[
            "epoch"
        ],
        history[
            "val_recall"
        ],
        label=
            "Recall",
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Score"
    )

    plt.ylim(
        0,
        1,
    )

    plt.title(
        "SatMAE++ validation metrics"
    )

    plt.grid(
        alpha=
            0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        plots_dir
        / "metrics.png",
        dpi=
            180,
    )

    plt.close()

    # Learning rates.
    plt.figure(
        figsize=(
            8,
            5,
        )
    )

    plt.plot(
        history[
            "epoch"
        ],
        history[
            "encoder_lr"
        ],
        label=
            "Encoder LR",
    )

    plt.plot(
        history[
            "epoch"
        ],
        history[
            "decoder_lr"
        ],
        label=
            "Decoder LR",
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Learning rate"
    )

    plt.yscale(
        "log"
    )

    plt.title(
        "SatMAE++ learning rate"
    )

    plt.grid(
        alpha=
            0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        plots_dir
        / "learning_rate.png",
        dpi=
            180,
    )

    plt.close()


# ============================================================
# Main
# ============================================================

def main() -> None:

    args = parse_args()

    seed_everything(
        args.seed
    )

    if (
        args.accumulation_steps
        < 1
    ):
        raise ValueError(
            "--accumulation-steps must be >= 1."
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    if (
        device.type
        == "cuda"
    ):

        print(
            "GPU:",
            torch.cuda
            .get_device_name(
                0
            ),
        )

    effective_batch_size = (
        args.batch_size
        * args.accumulation_steps
    )

    print(
        "Batch size:",
        args.batch_size,
    )

    print(
        "Gradient accumulation steps:",
        args.accumulation_steps,
    )

    print(
        "Effective batch size:",
        effective_batch_size,
    )

    print(
        "Maximum epochs:",
        args.epochs,
    )

    print(
        "Early stopping patience:",
        args.patience,
    )

    use_amp = (
        device.type
        == "cuda"
        and not args.no_amp
    )

    config_to_save = {
        key:
            str(
                value
            )
            if isinstance(
                value,
                Path,
            )
            else value

        for key, value
        in vars(
            args
        ).items()
    }

    config_to_save[
        "device"
    ] = str(
        device
    )

    config_to_save[
        "effective_batch_size"
    ] = (
        effective_batch_size
    )

    save_json(
        config_to_save,
        args.output_dir
        / "config.json",
    )

    train_dataset = (
        InriaSatMAEDataset(
            root=
                args.data_root,

            split=
                "train",

            image_size=
                args.image_size,

            augment=
                True,
        )
    )

    val_dataset = (
        InriaSatMAEDataset(
            root=
                args.data_root,

            split=
                "val",

            image_size=
                args.image_size,

            augment=
                False,
        )
    )

    train_loader = (
        DataLoader(
            train_dataset,

            batch_size=
                args.batch_size,

            shuffle=
                True,

            num_workers=
                args.num_workers,

            pin_memory=
                (
                    device.type
                    == "cuda"
                ),

            persistent_workers=
                (
                    args.num_workers
                    > 0
                ),
        )
    )

    val_loader = (
        DataLoader(
            val_dataset,

            batch_size=
                args.batch_size,

            shuffle=
                False,

            num_workers=
                args.num_workers,

            pin_memory=
                (
                    device.type
                    == "cuda"
                ),

            persistent_workers=
                (
                    args.num_workers
                    > 0
                ),
        )
    )

    print(
        "Train patches:",
        len(
            train_dataset
        ),
    )

    print(
        "Validation patches:",
        len(
            val_dataset
        ),
    )

    print(
        "Loading SatMAE++ ViT-L baseline..."
    )

    model = (
        SatMAESegmenter(
            satmae_root=
                args.satmae_root,

            checkpoint_path=
                args.pretrained_checkpoint,

            image_size=
                args.image_size,

            patch_size=
                args.patch_size,

            drop_path=
                args.drop_path,
        )
        .to(
            device
        )
    )

    criterion = (
        BCEDiceLoss(
            bce_weight=
                args.bce_weight,

            dice_weight=
                args.dice_weight,

            pos_weight=
                args.pos_weight,
        )
        .to(
            device
        )
    )

    encoder_parameters = (
        list(
            model.encoder.parameters()
        )
    )

    decoder_parameters = [
        parameter

        for name, parameter
        in model.named_parameters()

        if not name.startswith(
            "encoder."
        )
    ]

    optimizer = AdamW(
        [
            {
                "params":
                    encoder_parameters,

                "lr":
                    args.encoder_lr,
            },

            {
                "params":
                    decoder_parameters,

                "lr":
                    args.decoder_lr,
            },
        ],

        weight_decay=
            args.weight_decay,
    )

    scheduler = (
        CosineAnnealingLR(
            optimizer,
            T_max=
                args.epochs,
        )
    )

    scaler = (
        torch.amp.GradScaler(
            "cuda",
            enabled=
                use_amp,
        )
    )

    history = []

    best_iou = (
        -math.inf
    )

    best_epoch = (
        0
    )

    epochs_without_improvement = (
        0
    )

    for epoch in range(
        1,
        args.epochs
        + 1,
    ):

        epoch_start = (
            time.time()
        )

        train_loss, train_metrics = (
            run_train_epoch(
                model=
                    model,

                loader=
                    train_loader,

                criterion=
                    criterion,

                device=
                    device,

                optimizer=
                    optimizer,

                scaler=
                    scaler,

                threshold=
                    args.threshold,

                use_amp=
                    use_amp,

                accumulation_steps=
                    args.accumulation_steps,

                description=
                    f"Epoch {epoch} train",
            )
        )

        val_loss, val_metrics = (
            run_validation_epoch(
                model=
                    model,

                loader=
                    val_loader,

                criterion=
                    criterion,

                device=
                    device,

                threshold=
                    args.threshold,

                use_amp=
                    use_amp,

                description=
                    f"Epoch {epoch} val",
            )
        )

        encoder_lr = (
            optimizer
            .param_groups[
                0
            ][
                "lr"
            ]
        )

        decoder_lr = (
            optimizer
            .param_groups[
                1
            ][
                "lr"
            ]
        )

        epoch_seconds = (
            time.time()
            - epoch_start
        )

        row = {
            "epoch":
                epoch,

            "train_loss":
                train_loss,

            "train_iou":
                train_metrics[
                    "iou"
                ],

            "train_dice":
                train_metrics[
                    "dice"
                ],

            "train_precision":
                train_metrics[
                    "precision"
                ],

            "train_recall":
                train_metrics[
                    "recall"
                ],

            "train_accuracy":
                train_metrics[
                    "accuracy"
                ],

            "val_loss":
                val_loss,

            "val_iou":
                val_metrics[
                    "iou"
                ],

            "val_dice":
                val_metrics[
                    "dice"
                ],

            "val_precision":
                val_metrics[
                    "precision"
                ],

            "val_recall":
                val_metrics[
                    "recall"
                ],

            "val_accuracy":
                val_metrics[
                    "accuracy"
                ],

            "encoder_lr":
                encoder_lr,

            "decoder_lr":
                decoder_lr,

            "epoch_seconds":
                epoch_seconds,
        }

        history.append(
            row
        )

        history_dataframe = (
            pd.DataFrame(
                history
            )
        )

        history_dataframe.to_csv(
            args.output_dir
            / "history.csv",

            index=
                False,
        )

        checkpoint = {
            "epoch":
                epoch,

            "model_state_dict":
                model
                .state_dict(),

            "optimizer_state_dict":
                optimizer
                .state_dict(),

            "scheduler_state_dict":
                scheduler
                .state_dict(),

            "val_iou":
                val_metrics[
                    "iou"
                ],

            "config":
                config_to_save,
        }

        torch.save(
            checkpoint,
            args.output_dir
            / "last_model.pth",
        )

        improved = (
            val_metrics[
                "iou"
            ]
            > best_iou
        )

        if improved:

            best_iou = (
                val_metrics[
                    "iou"
                ]
            )

            best_epoch = (
                epoch
            )

            epochs_without_improvement = (
                0
            )

            torch.save(
                checkpoint,
                args.output_dir
                / "best_model.pth",
            )

            save_json(
                {
                    "epoch":
                        epoch,

                    "val_loss":
                        val_loss,

                    "val_iou":
                        val_metrics[
                            "iou"
                        ],

                    "val_dice":
                        val_metrics[
                            "dice"
                        ],

                    "val_precision":
                        val_metrics[
                            "precision"
                        ],

                    "val_recall":
                        val_metrics[
                            "recall"
                        ],

                    "val_accuracy":
                        val_metrics[
                            "accuracy"
                        ],
                },

                args.output_dir
                / "best_metrics.json",
            )

        else:

            epochs_without_improvement += (
                1
            )

        scheduler.step()

        save_plots(
            history_dataframe,
            args.output_dir,
        )

        print(
            f"\nEpoch "
            f"{epoch}/{args.epochs}"
        )

        print(
            f"Train | "
            f"Loss: {train_loss:.4f} | "
            f"IoU: {train_metrics['iou']:.4f} | "
            f"Dice: {train_metrics['dice']:.4f}"
        )

        print(
            f"Val   | "
            f"Loss: {val_loss:.4f} | "
            f"IoU: {val_metrics['iou']:.4f} | "
            f"Dice: {val_metrics['dice']:.4f} | "
            f"Precision: {val_metrics['precision']:.4f} | "
            f"Recall: {val_metrics['recall']:.4f}"
        )

        print(
            f"Best IoU: "
            f"{best_iou:.4f} "
            f"(epoch {best_epoch})"
        )

        if (
            args.patience
            > 0
            and epochs_without_improvement
            >= args.patience
        ):

            print(
                "Early stopping."
            )

            break

    print(
        "\nLoading best SatMAE++ model..."
    )

    best_checkpoint = (
        torch.load(
            args.output_dir
            / "best_model.pth",

            map_location=
                device,
        )
    )

    model.load_state_dict(
        best_checkpoint[
            "model_state_dict"
        ]
    )

    save_validation_predictions(
        model=
            model,

        loader=
            val_loader,

        device=
            device,

        threshold=
            args.threshold,

        output_path=
            args.output_dir
            / "val_predictions.csv",

        use_amp=
            use_amp,
    )

    print(
        "\nSatMAE++ baseline training completed."
    )

    print(
        "Results saved in:",
        args.output_dir
        .resolve(),
    )


if __name__ == "__main__":
    main()
