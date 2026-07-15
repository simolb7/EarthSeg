#!/usr/bin/env python
"""
Baseline Swin-Unet originale per INRIA Building Footprint Segmentation.

Questo script usa direttamente l'implementazione della repository:
    external/Swin-Unet/

e il checkpoint:
    external/Swin-Unet/pretrained_ckpt/swin_tiny_patch4_window7_224.pth

Struttura attesa:
remote-sensing-segmentation/
├── swinunet_baseline_original.py
├── datasets/
│   └── inria_processed/
│       ├── metadata.csv
│       ├── train/
│       └── val/
└── external/
    └── Swin-Unet/
        ├── config.py
        ├── configs/
        │   └── swin_tiny_patch4_window7_224_lite.yaml
        ├── networks/
        └── pretrained_ckpt/
            └── swin_tiny_patch4_window7_224.pth

Output:
outputs/swinunet_baseline/
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

Esempio:
python swinunet_baseline_original.py
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import importlib
from pathlib import Path
from types import SimpleNamespace

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


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ============================================================
# Arguments
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the original Swin-Unet on INRIA building segmentation."
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("datasets/inria_processed"),
    )

    parser.add_argument(
        "--swinunet-root",
        type=Path,
        default=Path("external/Swin-Unet"),
    )

    parser.add_argument(
        "--cfg",
        type=Path,
        default=Path(
            "external/Swin-Unet/configs/"
            "swin_tiny_patch4_window7_224_lite.yaml"
        ),
    )

    parser.add_argument(
        "--pretrained-checkpoint",
        type=Path,
        default=Path(
            "external/Swin-Unet/pretrained_ckpt/"
            "swin_tiny_patch4_window7_224.pth"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/swinunet_baseline"),
    )

    # La configurazione originale Swin-Unet usa 224x224.
    parser.add_argument("--image-size", type=int, default=224)

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--bce-weight", type=float, default=0.5)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--pos-weight", type=float, default=1.0)

    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable mixed precision training.",
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
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


# ============================================================
# Dataset
# ============================================================

class InriaDataset(Dataset):
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

        metadata_path = self.root / "metadata.csv"

        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"metadata.csv non trovato: {metadata_path}"
            )

        dataframe = pd.read_csv(
            metadata_path,
            keep_default_na=False,
        )

        self.dataframe = dataframe[
            dataframe["split"] == split
        ].reset_index(drop=True)

        if len(self.dataframe) == 0:
            raise RuntimeError(
                f"Nessun sample trovato per split='{split}'."
            )

        self.color_jitter = ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10,
            hue=0.02,
        )

    def __len__(self) -> int:
        return len(self.dataframe)

    def augment_pair(
        self,
        image: Image.Image,
        mask: Image.Image,
    ) -> tuple[Image.Image, Image.Image]:

        if random.random() < 0.5:
            image = ImageOps.mirror(image)
            mask = ImageOps.mirror(mask)

        if random.random() < 0.5:
            image = ImageOps.flip(image)
            mask = ImageOps.flip(mask)

        k = random.randint(0, 3)

        if k:
            angle = 90 * k
            image = image.rotate(angle, expand=False)
            mask = mask.rotate(angle, expand=False)

        image = self.color_jitter(image)

        return image, mask

    def __getitem__(self, index: int) -> dict:
        row = self.dataframe.iloc[index]

        image_path = self.root / row["image_path"]
        mask_path = self.root / row["mask_path"]

        if not image_path.is_file():
            raise FileNotFoundError(image_path)

        if not mask_path.is_file():
            raise FileNotFoundError(mask_path)

        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")

        with Image.open(mask_path) as mask_file:
            mask = mask_file.convert("L")

        if self.augment:
            image, mask = self.augment_pair(
                image,
                mask,
            )

        image = TF.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )

        mask = TF.resize(
            mask,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.NEAREST,
        )

        image = TF.to_tensor(image)

        # Il checkpoint Swin-T è pre-addestrato su ImageNet.
        image = TF.normalize(
            image,
            IMAGENET_MEAN,
            IMAGENET_STD,
        )

        mask = TF.pil_to_tensor(mask)
        mask = (mask > 127).float()

        return {
            "image": image,
            "mask": mask,
            "patch_id": str(row["patch_id"]),
        }


# ============================================================
# Original Swin-Unet
# ============================================================

def build_original_swinunet(
    swinunet_root: Path,
    cfg_path: Path,
    checkpoint_path: Path,
    image_size: int,
) -> nn.Module:

    swinunet_root = swinunet_root.resolve()
    cfg_path = cfg_path.resolve()
    checkpoint_path = checkpoint_path.resolve()

    if not swinunet_root.is_dir():
        raise FileNotFoundError(
            f"Repository Swin-Unet non trovata: {swinunet_root}"
        )

    if not cfg_path.is_file():
        raise FileNotFoundError(
            f"Config Swin-Unet non trovata: {cfg_path}"
        )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint Swin-T non trovato: {checkpoint_path}"
        )

    # Permette di importare config.py e networks/ dalla repository originale.
    sys.path.insert(
        0,
        str(swinunet_root),
    )

    config_module = importlib.import_module("config")
    vision_transformer_module = importlib.import_module("networks.vision_transformer")

    get_config = config_module.get_config
    SwinUnet = vision_transformer_module.SwinUnet

    # get_config() della repository originale si aspetta questi argomenti.
    config_args = SimpleNamespace(
        cfg=str(cfg_path),
        opts=None,
        batch_size=None,
        zip=False,
        cache_mode=None,
        resume=None,
        accumulation_steps=None,
        use_checkpoint=False,
        amp_opt_level=None,
        tag=None,
        eval=False,
        throughput=False,
    )

    config = get_config(config_args)

    # Modifichiamo solo i parametri necessari per INRIA.
    config.defrost()

    config.DATA.IMG_SIZE = image_size
    config.MODEL.PRETRAIN_CKPT = str(checkpoint_path)
    config.MODEL.NUM_CLASSES = 1

    config.freeze()

    model = SwinUnet(
        config,
        img_size=image_size,
        num_classes=1,
    )

    # Metodo originale della repository.
    # Carica il checkpoint Swin-T nell'encoder e copia i pesi
    # compatibili anche nel decoder Swin-Unet.
    model.load_from(config)

    return model


# ============================================================
# Loss
# ============================================================

class DiceLoss(nn.Module):
    def __init__(
        self,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        probabilities = torch.sigmoid(logits)

        probabilities = probabilities.flatten(1)
        targets = targets.flatten(1)

        intersection = (
            probabilities
            * targets
        ).sum(dim=1)

        denominator = (
            probabilities.sum(dim=1)
            + targets.sum(dim=1)
        )

        dice = (
            2.0 * intersection
            + self.smooth
        ) / (
            denominator
            + self.smooth
        )

        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float,
        dice_weight: float,
        pos_weight: float,
    ) -> None:

        super().__init__()

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        self.register_buffer(
            "pos_weight",
            torch.tensor(
                [pos_weight],
                dtype=torch.float32,
            ),
        )

        self.dice = DiceLoss()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
        )

        dice = self.dice(
            logits,
            targets,
        )

        return (
            self.bce_weight * bce
            + self.dice_weight * dice
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

        self.threshold = threshold
        self.eps = eps
        self.reset()

    def reset(self) -> None:
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
            torch.sigmoid(logits)
            >= self.threshold
        )

        targets = (
            targets >= 0.5
        )

        self.tp += torch.logical_and(
            predictions,
            targets,
        ).sum().item()

        self.fp += torch.logical_and(
            predictions,
            ~targets,
        ).sum().item()

        self.fn += torch.logical_and(
            ~predictions,
            targets,
        ).sum().item()

        self.tn += torch.logical_and(
            ~predictions,
            ~targets,
        ).sum().item()

    def compute(self) -> dict[str, float]:

        iou = self.tp / (
            self.tp
            + self.fp
            + self.fn
            + self.eps
        )

        dice = (
            2 * self.tp
        ) / (
            2 * self.tp
            + self.fp
            + self.fn
            + self.eps
        )

        precision = self.tp / (
            self.tp
            + self.fp
            + self.eps
        )

        recall = self.tp / (
            self.tp
            + self.fn
            + self.eps
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
            "iou": float(iou),
            "dice": float(dice),
            "precision": float(precision),
            "recall": float(recall),
            "accuracy": float(accuracy),
        }


# ============================================================
# Training / Validation
# ============================================================

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    threshold: float,
    use_amp: bool,
    description: str,
) -> tuple[float, dict[str, float]]:

    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    metrics = SegmentationMetrics(
        threshold=threshold
    )

    total_loss = 0.0
    total_samples = 0

    progress = tqdm(
        loader,
        desc=description,
        leave=False,
    )

    for batch in progress:

        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        masks = batch["mask"].to(
            device,
            non_blocking=True,
        )

        batch_size = images.shape[0]

        if is_training:
            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(
            is_training
        ):

            with torch.autocast(
                device_type=device.type,
                enabled=use_amp,
            ):

                logits = model(images)

                loss = criterion(
                    logits,
                    masks,
                )

            if is_training:

                scaler.scale(
                    loss
                ).backward()

                scaler.step(
                    optimizer
                )

                scaler.update()

        total_loss += (
            loss.item()
            * batch_size
        )

        total_samples += batch_size

        metrics.update(
            logits.detach(),
            masks,
        )

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
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
) -> None:

    model.eval()

    rows = []
    eps = 1e-7

    for batch in tqdm(
        loader,
        desc="Validation predictions",
        leave=False,
    ):

        images = batch[
            "image"
        ].to(device)

        targets = (
            batch[
                "mask"
            ].to(device)
            >= 0.5
        )

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
            images.shape[0]
        ):

            prediction = predictions[
                i
            ].flatten()

            target = targets[
                i
            ].flatten()

            tp = torch.logical_and(
                prediction,
                target,
            ).sum().item()

            fp = torch.logical_and(
                prediction,
                ~target,
            ).sum().item()

            fn = torch.logical_and(
                ~prediction,
                target,
            ).sum().item()

            tn = torch.logical_and(
                ~prediction,
                ~target,
            ).sum().item()

            iou = tp / (
                tp
                + fp
                + fn
                + eps
            )

            dice = (
                2 * tp
            ) / (
                2 * tp
                + fp
                + fn
                + eps
            )

            precision = tp / (
                tp
                + fp
                + eps
            )

            recall = tp / (
                tp
                + fn
                + eps
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
                        prediction.float()
                        .mean()
                        .item(),
                    "target_building_ratio":
                        target.float()
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

    # Loss
    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        history["epoch"],
        history["train_loss"],
        label="Train",
    )

    plt.plot(
        history["epoch"],
        history["val_loss"],
        label="Validation",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        plots_dir
        / "loss.png",
        dpi=180,
    )

    plt.close()

    # Metrics
    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        history["epoch"],
        history["val_iou"],
        label="IoU",
    )

    plt.plot(
        history["epoch"],
        history["val_dice"],
        label="Dice",
    )

    plt.plot(
        history["epoch"],
        history["val_precision"],
        label="Precision",
    )

    plt.plot(
        history["epoch"],
        history["val_recall"],
        label="Recall",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.title("Validation metrics")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        plots_dir
        / "metrics.png",
        dpi=180,
    )

    plt.close()

    # Learning rate
    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        history["epoch"],
        history["learning_rate"],
    )

    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.yscale("log")
    plt.title("Learning rate")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        plots_dir
        / "learning_rate.png",
        dpi=180,
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

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            ),
        )

    use_amp = (
        device.type == "cuda"
        and not args.no_amp
    )

    config_to_save = {
        key:
            str(value)
            if isinstance(
                value,
                Path,
            )
            else value
        for key, value
        in vars(args).items()
    }

    config_to_save[
        "device"
    ] = str(device)

    save_json(
        config_to_save,
        args.output_dir
        / "config.json",
    )

    train_dataset = InriaDataset(
        root=args.data_root,
        split="train",
        image_size=args.image_size,
        augment=True,
    )

    val_dataset = InriaDataset(
        root=args.data_root,
        split="val",
        image_size=args.image_size,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(
            device.type
            == "cuda"
        ),
        persistent_workers=(
            args.num_workers
            > 0
        ),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(
            device.type
            == "cuda"
        ),
        persistent_workers=(
            args.num_workers
            > 0
        ),
    )

    print(
        f"Train patches: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation patches: "
        f"{len(val_dataset)}"
    )

    print(
        "Loading original Swin-Unet..."
    )

    model = build_original_swinunet(
        swinunet_root=
            args.swinunet_root,
        cfg_path=
            args.cfg,
        checkpoint_path=
            args.pretrained_checkpoint,
        image_size=
            args.image_size,
    )

    model = model.to(
        device
    )

    criterion = BCEDiceLoss(
        bce_weight=
            args.bce_weight,
        dice_weight=
            args.dice_weight,
        pos_weight=
            args.pos_weight,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=
            args.learning_rate,
        weight_decay=
            args.weight_decay,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    history = []

    best_iou = -math.inf
    best_epoch = 0

    epochs_without_improvement = 0

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        epoch_start = (
            time.time()
        )

        train_loss, train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            threshold=
                args.threshold,
            use_amp=
                use_amp,
            description=
                f"Epoch {epoch} train",
        )

        val_loss, val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            scaler=scaler,
            threshold=
                args.threshold,
            use_amp=
                use_amp,
            description=
                f"Epoch {epoch} val",
        )

        learning_rate = (
            optimizer
            .param_groups[0][
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

            "learning_rate":
                learning_rate,

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
            index=False,
        )

        checkpoint = {
            "epoch":
                epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scheduler_state_dict":
                scheduler.state_dict(),

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

        if (
            val_metrics["iou"]
            > best_iou
        ):

            best_iou = (
                val_metrics[
                    "iou"
                ]
            )

            best_epoch = epoch

            epochs_without_improvement = 0

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

            epochs_without_improvement += 1

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
            f"IoU: "
            f"{train_metrics['iou']:.4f} | "
            f"Dice: "
            f"{train_metrics['dice']:.4f}"
        )

        print(
            f"Val   | "
            f"Loss: {val_loss:.4f} | "
            f"IoU: "
            f"{val_metrics['iou']:.4f} | "
            f"Dice: "
            f"{val_metrics['dice']:.4f} | "
            f"Precision: "
            f"{val_metrics['precision']:.4f} | "
            f"Recall: "
            f"{val_metrics['recall']:.4f}"
        )

        print(
            f"Best IoU: "
            f"{best_iou:.4f} "
            f"(epoch {best_epoch})"
        )

        if (
            args.patience > 0
            and
            epochs_without_improvement
            >= args.patience
        ):

            print(
                "Early stopping."
            )

            break

    print(
        "\nLoading best model..."
    )

    best_checkpoint = torch.load(
        args.output_dir
        / "best_model.pth",
        map_location=device,
    )

    model.load_state_dict(
        best_checkpoint[
            "model_state_dict"
        ]
    )

    save_validation_predictions(
        model=model,
        loader=val_loader,
        device=device,
        threshold=
            args.threshold,
        output_path=
            args.output_dir
            / "val_predictions.csv",
    )

    print(
        "\nTraining completed."
    )

    print(
        "Results saved in:",
        args.output_dir.resolve(),
    )


if __name__ == "__main__":
    main()
