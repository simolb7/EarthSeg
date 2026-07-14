"""
Train a Swin Transformer U-Net baseline for remote-sensing segmentation.

The script is intentionally dataset-agnostic so it can plug into most
preprocessed outputs:

1. Directory mode:
   data/
     train/images/*.(npy|npz|png|jpg|tif)
     train/masks/*.(npy|npz|png|jpg|tif)
     val/images/*
     val/masks/*

2. Manifest mode:
   CSV files with columns: image,mask
   paths may be absolute or relative to --data-root.

Examples:
   python src/swin_unet_baseline.py --data-root datasets/inria_processed --epochs 50
   python src/swin_unet_baseline.py --metadata-csv datasets/inria_processed/metadata.csv --data-root datasets/inria_processed
   python src/swin_unet_baseline.py --train-csv splits/train.csv --val-csv splits/val.csv --data-root datasets/inria_processed
   python src/swin_unet_baseline.py --in-channels 13 --num-classes 1 --pretrained
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

try:
    import tifffile
except ImportError:  # pragma: no cover - optional dependency
    tifffile = None

try:
    import timm
except ImportError as exc:  # pragma: no cover - clear runtime error
    raise ImportError(
        "This baseline needs timm. Install it with: pip install timm"
    ) from exc


IMAGE_EXTENSIONS = {".npy", ".npz", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Swin U-Net segmentation baseline")
    parser.add_argument("--data-root", type=Path, default=Path("datasets/inria_processed"))
    parser.add_argument("--metadata-csv", type=Path, default=None)
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--val-csv", type=Path, default=None)
    parser.add_argument("--train-image-dir", type=Path, default=None)
    parser.add_argument("--train-mask-dir", type=Path, default=None)
    parser.add_argument("--val-image-dir", type=Path, default=None)
    parser.add_argument("--val-mask-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/swin_unet_baseline"))
    parser.add_argument("--model-name", default="swin_tiny_patch4_window7_224")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA")
    parser.add_argument(
        "--positive-mask-values",
        default=None,
        help="Comma-separated mask pixel values treated as foreground in binary mode.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_path(path: str | Path, root: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else root / path


def read_manifest(csv_path: Path, data_root: Path, split: str | None = None) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        image_column = next((name for name in ("image", "image_path") if name in fieldnames), None)
        mask_column = next((name for name in ("mask", "mask_path", "label", "label_path") if name in fieldnames), None)
        if image_column is None or mask_column is None:
            raise ValueError(
                f"{csv_path} must contain image/mask columns. Supported names: "
                "image or image_path, and mask, mask_path, label, or label_path."
            )
        for row in reader:
            if split is not None and row.get("split") != split:
                continue
            if not row.get(mask_column):
                continue
            pairs.append((resolve_path(row[image_column], data_root), resolve_path(row[mask_column], data_root)))
    if not pairs:
        split_message = f" for split '{split}'" if split else ""
        raise ValueError(f"No image/mask rows found in {csv_path}{split_message}")
    return pairs


def list_pairs(image_dir: Path, mask_dir: Path) -> list[tuple[Path, Path]]:
    images = [p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    masks_by_stem = {p.stem: p for p in mask_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS}
    pairs = [(image, masks_by_stem[image.stem]) for image in sorted(images) if image.stem in masks_by_stem]
    if not pairs:
        raise ValueError(f"No image/mask pairs found in {image_dir} and {mask_dir}")
    return pairs


def load_array(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    if suffix == ".npz":
        data = np.load(path)
        key = "arr_0" if "arr_0" in data.files else data.files[0]
        return data[key]
    if suffix in {".tif", ".tiff"} and tifffile is not None:
        return tifffile.imread(path)
    return np.asarray(Image.open(path))


def to_channel_first(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return array[None, :, :]
    if array.ndim == 3:
        # Most remote-sensing tensors are HWC after image IO and CHW after numpy preprocessing.
        if array.shape[0] <= 32 and array.shape[1] > 32 and array.shape[2] > 32:
            return array
        return np.moveaxis(array, -1, 0)
    raise ValueError(f"Expected 2D or 3D array, got shape {array.shape}")


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    if image.max(initial=0) > 1.5:
        # Robust per-channel scaling works for uint8 RGB and many Sentinel-style scaled arrays.
        axes = (1, 2)
        lo = np.percentile(image, 1, axis=axes, keepdims=True)
        hi = np.percentile(image, 99, axis=axes, keepdims=True)
        image = (image - lo) / np.maximum(hi - lo, 1e-6)
    return np.clip(image, 0.0, 1.0)


def resize_tensor(tensor: torch.Tensor, size: int, mode: str) -> torch.Tensor:
    tensor = tensor.unsqueeze(0)
    align_corners = False if mode in {"bilinear", "bicubic"} else None
    tensor = F.interpolate(tensor, size=(size, size), mode=mode, align_corners=align_corners)
    return tensor.squeeze(0)


class SegmentationDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[tuple[Path, Path]],
        image_size: int,
        num_classes: int,
        positive_mask_values: set[int] | None = None,
        augment: bool = False,
    ) -> None:
        self.pairs = list(pairs)
        self.image_size = image_size
        self.num_classes = num_classes
        self.positive_mask_values = positive_mask_values
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.pairs[index]
        image = normalize_image(to_channel_first(load_array(image_path)))
        mask = load_array(mask_path)
        if mask.ndim == 3:
            mask = mask[..., 0] if mask.shape[-1] <= 4 else mask[0]

        image_tensor = torch.from_numpy(image).float()
        mask_tensor = torch.from_numpy(mask.astype(np.int64))

        image_tensor = resize_tensor(image_tensor, self.image_size, mode="bilinear")
        mask_tensor = resize_tensor(mask_tensor[None].float(), self.image_size, mode="nearest").squeeze(0).long()

        if self.num_classes == 1:
            if self.positive_mask_values:
                positive = torch.zeros_like(mask_tensor, dtype=torch.bool)
                for value in self.positive_mask_values:
                    positive |= mask_tensor == value
                mask_tensor = positive.float()
            else:
                mask_tensor = (mask_tensor > 0).float()

        if self.augment and random.random() < 0.5:
            image_tensor = torch.flip(image_tensor, dims=[2])
            mask_tensor = torch.flip(mask_tensor, dims=[1])
        if self.augment and random.random() < 0.5:
            image_tensor = torch.flip(image_tensor, dims=[1])
            mask_tensor = torch.flip(mask_tensor, dims=[0])

        return image_tensor, mask_tensor


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SwinUNet(nn.Module):
    def __init__(
        self,
        model_name: str,
        in_channels: int,
        num_classes: int,
        pretrained: bool,
        image_size: int,
    ) -> None:
        super().__init__()
        self.encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_channels,
            img_size=image_size,
            num_classes=0,
        )
        channels = [layer.dim for layer in self.encoder.layers]
        decoder_channels = [256, 128, 64, 32]

        self.center = ConvBlock(channels[-1], decoder_channels[0])
        self.up3 = ConvBlock(decoder_channels[0] + channels[-2], decoder_channels[1])
        self.up2 = ConvBlock(decoder_channels[1] + channels[-3], decoder_channels[2])
        self.up1 = ConvBlock(decoder_channels[2] + channels[-4], decoder_channels[3])
        self.head = nn.Conv2d(decoder_channels[3], num_classes, kernel_size=1)

    @staticmethod
    def _tokens_to_feature_map(tokens: torch.Tensor, resolution: tuple[int, int]) -> torch.Tensor:
        height, width = resolution
        batch_size, token_count, channels = tokens.shape
        if token_count != height * width:
            side = int(math.sqrt(token_count))
            if side * side != token_count:
                raise ValueError(
                    f"Cannot reshape {token_count} Swin tokens into resolution {resolution}."
                )
            height = width = side
        return tokens.transpose(1, 2).reshape(batch_size, channels, height, width).contiguous()

    def _encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.encoder.patch_embed(x)
        if self.encoder.absolute_pos_embed is not None:
            x = x + self.encoder.absolute_pos_embed
        x = self.encoder.pos_drop(x)

        features: list[torch.Tensor] = []
        for layer in self.encoder.layers:
            x = layer.blocks(x)
            features.append(self._tokens_to_feature_map(x, layer.input_resolution))
            if layer.downsample is not None:
                x = layer.downsample(x)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        features = self._encode(x)

        x = self.center(features[-1])
        x = F.interpolate(x, size=features[-2].shape[-2:], mode="bilinear", align_corners=False)
        x = self.up3(torch.cat([x, features[-2]], dim=1))
        x = F.interpolate(x, size=features[-3].shape[-2:], mode="bilinear", align_corners=False)
        x = self.up2(torch.cat([x, features[-3]], dim=1))
        x = F.interpolate(x, size=features[-4].shape[-2:], mode="bilinear", align_corners=False)
        x = self.up1(torch.cat([x, features[-4]], dim=1))
        x = self.head(x)
        return F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)


class DiceBCELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.unsqueeze(1)
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        intersection = (probs * targets).sum(dims)
        union = probs.sum(dims) + targets.sum(dims)
        dice = 1 - ((2 * intersection + 1.0) / (union + 1.0)).mean()
        return bce + dice


def dice_score_binary(logits: torch.Tensor, targets: torch.Tensor, threshold: float) -> float:
    preds = (torch.sigmoid(logits).squeeze(1) > threshold).float()
    targets = targets.float()
    intersection = (preds * targets).sum(dim=(1, 2))
    union = preds.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))
    return ((2 * intersection + 1.0) / (union + 1.0)).mean().item()


def mean_iou_multiclass(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    preds = logits.argmax(dim=1)
    ious: list[torch.Tensor] = []
    for cls in range(num_classes):
        pred_cls = preds == cls
        target_cls = targets == cls
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        if union > 0:
            ious.append(intersection / union)
    if not ious:
        return math.nan
    return torch.stack(ious).mean().item()


def build_pairs(args: argparse.Namespace) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    if args.train_csv and args.val_csv:
        return read_manifest(args.train_csv, args.data_root), read_manifest(args.val_csv, args.data_root)
    if args.metadata_csv:
        return (
            read_manifest(args.metadata_csv, args.data_root, split="train"),
            read_manifest(args.metadata_csv, args.data_root, split="val"),
        )

    metadata_csv = args.data_root / "metadata.csv"
    if metadata_csv.is_file():
        return (
            read_manifest(metadata_csv, args.data_root, split="train"),
            read_manifest(metadata_csv, args.data_root, split="val"),
        )

    train_image_dir = args.train_image_dir or args.data_root / "train" / "images"
    train_mask_dir = args.train_mask_dir or args.data_root / "train" / "masks"
    val_image_dir = args.val_image_dir or args.data_root / "val" / "images"
    val_mask_dir = args.val_mask_dir or args.data_root / "val" / "masks"
    return list_pairs(train_image_dir, train_mask_dir), list_pairs(val_image_dir, val_mask_dir)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    use_amp: bool,
) -> float:
    model.train()
    running_loss = 0.0
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    threshold: float,
    use_amp: bool,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_metric = 0.0
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, masks)
        metric = (
            dice_score_binary(logits, masks, threshold)
            if num_classes == 1
            else mean_iou_multiclass(logits, masks, num_classes)
        )
        running_loss += loss.item() * images.size(0)
        running_metric += metric * images.size(0)
    return running_loss / len(loader.dataset), running_metric / len(loader.dataset)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    positive_values = None
    if args.positive_mask_values:
        positive_values = {int(value.strip()) for value in args.positive_mask_values.split(",") if value.strip()}

    train_pairs, val_pairs = build_pairs(args)
    train_dataset = SegmentationDataset(
        train_pairs,
        image_size=args.image_size,
        num_classes=args.num_classes,
        positive_mask_values=positive_values,
        augment=True,
    )
    val_dataset = SegmentationDataset(
        val_pairs,
        image_size=args.image_size,
        num_classes=args.num_classes,
        positive_mask_values=positive_values,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SwinUNet(
        args.model_name,
        args.in_channels,
        args.num_classes,
        args.pretrained,
        args.image_size,
    ).to(device)
    criterion: nn.Module = DiceBCELoss() if args.num_classes == 1 else nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    use_amp = args.amp and device.type == "cuda"

    best_metric = -float("inf")
    metric_name = "dice" if args.num_classes == 1 else "miou"
    print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples")
    print(f"Device: {device} | Model: {args.model_name} | Metric: {metric_name}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, use_amp)
        val_loss, val_metric = validate(
            model,
            val_loader,
            criterion,
            device,
            args.num_classes,
            args.threshold,
            use_amp,
        )
        print(
            f"epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_{metric_name}={val_metric:.4f}"
        )

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            f"val_{metric_name}": val_metric,
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if val_metric > best_metric:
            best_metric = val_metric
            torch.save(checkpoint, args.output_dir / "best.pt")

    print(f"Best val_{metric_name}: {best_metric:.4f}")
    print(f"Checkpoints saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
