#!/usr/bin/env python
"""Unified evaluator for Swin-Unet and SatMAE variants.

Modes:
  dataset -> evaluate an entire split and save summary + per-image metrics
  single  -> evaluate one image and save report-ready visual outputs

Examples:
  python src/evaluate_models.py --model satmae --mode dataset
  python src/evaluate_models.py --model satmae_wavelet_loss --mode dataset
  python src/evaluate_models.py --model satmae_wavelet --mode single --patch-id PATCH_ID
  python src/evaluate_models.py --model satmae --mode single --image path/to/image.png --mask path/to/mask.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

SATMAE_MEAN = (0.4182007312774658, 0.4214799106121063, 0.3991275727748871)
SATMAE_STD = (0.28774282336235046, 0.27541765570640564, 0.2764017581939697)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_CHECKPOINTS = {
    "satmae": Path("remote-sensing-segmentation/outputs/satmae_baseline/best_model.pth"),
    "satmae_sam": Path("remote-sensing-segmentation/outputs/satmae_sam_pseudolabels/best_model.pth"),
    "satmae_wavelet_loss": Path("remote-sensing-segmentation/outputs/satmae_wavelet/best_model.pth"),
    "satmae_wavelet": Path("remote-sensing-segmentation/outputs/satmae_wavelet_decoder/best_model.pth"),
    "swin": Path("remote-sensing-segmentation/outputs/swinunet_baseline/best_model.pth"),
}


def parse_args():
    p = argparse.ArgumentParser(description="Unified segmentation evaluator")
    p.add_argument("--model", required=True, choices=["satmae", "satmae_sam", "satmae_wavelet_loss", "satmae_wavelet", "swin"])
    p.add_argument("--mode", default="dataset", choices=["dataset", "single"])
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--data-root", type=Path, default=Path("remote-sensing-segmentation/datasets/inria_processed"))
    p.add_argument("--metadata", type=Path, default=Path("remote-sensing-segmentation/datasets/inria_processed/metadata.csv"))
    p.add_argument("--split", default="val")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--output-dir", type=Path, default=None)

    # single-image options
    p.add_argument("--patch-id", default=None)
    p.add_argument("--image", type=Path, default=None)
    p.add_argument("--mask", type=Path, default=None)

    # SatMAE
    p.add_argument("--satmae-root", type=Path, default=Path("remote-sensing-segmentation/external/satmae_pp"))
    p.add_argument(
        "--satmae-pretrained",
        type=Path,
        default=Path("remote-sensing-segmentation/checkpoints/satmae/checkpoint_ViT-L_pretrain_fmow_rgb.pth"),
    )

    # Swin-Unet
    p.add_argument("--swin-root", type=Path, default=Path("remote-sensing-segmentation/external/Swin-Unet"))
    p.add_argument(
        "--swin-config",
        type=Path,
        default=Path("remote-sensing-segmentation/external/Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml"),
    )
    p.add_argument("--no-amp", action="store_true")
    return p.parse_args()


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
        return checkpoint
    raise RuntimeError("Could not find a model state_dict in checkpoint")


def build_satmae(args, wavelet=False):
    src_dir = Path(__file__).resolve().parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    if wavelet:
        from satmae_wavelet_decoder import SatMAEWaveletDecoder
        cls = SatMAEWaveletDecoder
    else:
        from satmae_baseline import SatMAESegmenter
        cls = SatMAESegmenter
    return cls(
        satmae_root=args.satmae_root.resolve(),
        checkpoint_path=args.satmae_pretrained.resolve(),
        image_size=args.image_size,
        patch_size=16,
        drop_path=0.2,
    )


def build_swin(args):
    root = args.swin_root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from config import get_config
    from networks.vision_transformer import SwinUnet

    cfg_args = SimpleNamespace(
        cfg=str(args.swin_config.resolve()), opts=None, batch_size=1,
        zip=False, cache_mode="part", resume=None, accumulation_steps=None,
        use_checkpoint=False, amp_opt_level="O1", tag=None,
        eval=False, throughput=False,
    )
    cfg = get_config(cfg_args)
    return SwinUnet(cfg, img_size=args.image_size, num_classes=1)


def load_model(args, device):
    checkpoint_path = (args.checkpoint or DEFAULT_CHECKPOINTS[args.model]).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if args.model in ("satmae", "satmae_sam", "satmae_wavelet_loss"):
        model = build_satmae(args, wavelet=False)
    elif args.model == "satmae_wavelet":
        model = build_satmae(args, wavelet=True)
    else:
        model = build_swin(args)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = extract_state_dict(ckpt)
    if state and all(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}

    result = model.load_state_dict(state, strict=False)
    if result.missing_keys:
        print("WARNING missing keys:", result.missing_keys[:10])
    if result.unexpected_keys:
        print("WARNING unexpected keys:", result.unexpected_keys[:10])

    return model.to(device).eval(), checkpoint_path


def load_rgb(path: Path):
    with Image.open(path) as im:
        return im.convert("RGB").copy()


def load_mask(path: Path):
    with Image.open(path) as im:
        return im.convert("L").copy()


def preprocess_image(image, model_name, size):
    image = TF.resize(image, [size, size], interpolation=InterpolationMode.BICUBIC, antialias=True)
    x = TF.to_tensor(image)
    if model_name.startswith("satmae"):
        x = x[[2, 1, 0], :, :]  # BGR for SatMAE FMoW-RGB
        x = TF.normalize(x, SATMAE_MEAN, SATMAE_STD)
    else:
        x = TF.normalize(x, IMAGENET_MEAN, IMAGENET_STD)
    return x.unsqueeze(0)


def preprocess_mask(mask, size):
    mask = TF.resize(mask, [size, size], interpolation=InterpolationMode.NEAREST)
    return (TF.pil_to_tensor(mask) > 127).numpy()[0].astype(np.uint8)


@torch.no_grad()
def predict(model, image_tensor, device, use_amp):
    image_tensor = image_tensor.to(device)
    with torch.autocast(device_type=device.type, enabled=use_amp):
        logits = model(image_tensor)
    return torch.sigmoid(logits).float().cpu().numpy()[0, 0]


def metrics(pred, target):
    p = pred.astype(bool)
    t = target.astype(bool)
    tp = np.logical_and(p, t).sum()
    fp = np.logical_and(p, ~t).sum()
    fn = np.logical_and(~p, t).sum()
    tn = np.logical_and(~p, ~t).sum()
    eps = 1e-7
    return {
        "iou": float(tp / (tp + fp + fn + eps)),
        "dice": float((2 * tp) / (2 * tp + fp + fn + eps)),
        "precision": float(tp / (tp + fp + eps)),
        "recall": float(tp / (tp + fn + eps)),
        "accuracy": float((tp + tn) / (tp + fp + fn + tn + eps)),
    }


class GlobalMetrics:
    def __init__(self):
        self.tp = self.fp = self.fn = self.tn = 0

    def update(self, pred, target):
        p = pred.astype(bool)
        t = target.astype(bool)
        self.tp += int(np.logical_and(p, t).sum())
        self.fp += int(np.logical_and(p, ~t).sum())
        self.fn += int(np.logical_and(~p, t).sum())
        self.tn += int(np.logical_and(~p, ~t).sum())

    def compute(self):
        eps = 1e-7
        return {
            "iou": float(self.tp / (self.tp + self.fp + self.fn + eps)),
            "dice": float(2 * self.tp / (2 * self.tp + self.fp + self.fn + eps)),
            "precision": float(self.tp / (self.tp + self.fp + eps)),
            "recall": float(self.tp / (self.tp + self.fn + eps)),
            "accuracy": float((self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn + eps)),
        }


def make_overlay(rgb, prediction):
    out = rgb.astype(np.float32).copy()
    mask = prediction.astype(bool)
    if mask.any():
        highlight = np.full_like(out, 255.0)
        out[mask] = 0.65 * out[mask] + 0.35 * highlight[mask]
    return np.clip(out, 0, 255).astype(np.uint8)


def resolve_single(args):
    if args.patch_id:
        df = pd.read_csv(args.metadata.resolve(), keep_default_na=False)
        rows = df[df["patch_id"].astype(str) == str(args.patch_id)]
        if rows.empty:
            raise ValueError(f"patch_id not found: {args.patch_id}")
        row = rows.iloc[0]
        image = args.data_root.resolve() / str(row["image_path"])
        mask = args.data_root.resolve() / str(row["mask_path"]) if str(row.get("mask_path", "")) else None
        return image, mask, str(args.patch_id)

    if args.image is None:
        raise ValueError("single mode requires --patch-id or --image")
    return args.image.resolve(), args.mask.resolve() if args.mask else None, args.image.stem


def save_single_outputs(image, probability, prediction, mask, out_dir, score_dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb = np.array(image)
    h, w = rgb.shape[:2]

    prob_img = Image.fromarray((probability * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BILINEAR)
    pred_img = Image.fromarray((prediction * 255).astype(np.uint8)).resize((w, h), Image.Resampling.NEAREST)
    pred_full = (np.array(pred_img) > 127).astype(np.uint8)
    overlay = make_overlay(rgb, pred_full)

    Image.fromarray(rgb).save(out_dir / "original.png")
    prob_img.save(out_dir / "probability.png")
    Image.fromarray(pred_full * 255).save(out_dir / "prediction.png")
    Image.fromarray(overlay).save(out_dir / "overlay.png")

    gt_full = None
    if mask is not None:
        gt_img = mask.resize((w, h), Image.Resampling.NEAREST)
        gt_full = (np.array(gt_img) > 127).astype(np.uint8)
        Image.fromarray(gt_full * 255).save(out_dir / "ground_truth.png")

    panels = 4 if gt_full is not None else 3
    fig, axes = plt.subplots(1, panels, figsize=(4.6 * panels, 4.6))
    axes = np.atleast_1d(axes)
    axes[0].imshow(rgb); axes[0].set_title("Original")

    if gt_full is not None:
        axes[1].imshow(gt_full, cmap="gray", vmin=0, vmax=1); axes[1].set_title("Ground Truth")
        axes[2].imshow(pred_full, cmap="gray", vmin=0, vmax=1); axes[2].set_title("Prediction")
        title = "Overlay"
        if score_dict:
            title += f"\nIoU={score_dict['iou']:.3f} | Dice={score_dict['dice']:.3f}"
        axes[3].imshow(overlay); axes[3].set_title(title)
    else:
        axes[1].imshow(pred_full, cmap="gray", vmin=0, vmax=1); axes[1].set_title("Prediction")
        axes[2].imshow(overlay); axes[2].set_title("Overlay")

    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "comparison.png", dpi=220, bbox_inches="tight")
    plt.close()


def evaluate_single(args, model, device, output_root, use_amp):
    image_path, mask_path, name = resolve_single(args)
    image = load_rgb(image_path)
    probability = predict(model, preprocess_image(image, args.model, args.image_size), device, use_amp)
    prediction = (probability >= args.threshold).astype(np.uint8)

    mask = None
    score_dict = None
    if mask_path is not None and mask_path.is_file():
        mask = load_mask(mask_path)
        target = preprocess_mask(mask, args.image_size)
        score_dict = metrics(prediction, target)

    out_dir = output_root / name
    save_single_outputs(image, probability, prediction, mask, out_dir, score_dict)
    if score_dict:
        save_json(score_dict, out_dir / "metrics.json")

    print("\nSingle-image evaluation complete")
    if score_dict:
        for k, v in score_dict.items():
            print(f"{k}: {v:.4f}")
    print("Saved to:", out_dir)


def evaluate_dataset(args, model, device, out_dir, use_amp):
    df = pd.read_csv(args.metadata.resolve(), keep_default_na=False)
    frame = df[df["split"] == args.split].copy()
    if frame.empty:
        raise RuntimeError(f"No samples found for split={args.split}")

    global_stats = GlobalMetrics()
    rows = []

    for _, row in tqdm(frame.iterrows(), total=len(frame), desc=f"Evaluating {args.model}"):
        image_path = args.data_root.resolve() / str(row["image_path"])
        mask_path = args.data_root.resolve() / str(row["mask_path"])
        image = load_rgb(image_path)
        mask = load_mask(mask_path)

        probability = predict(model, preprocess_image(image, args.model, args.image_size), device, use_amp)
        prediction = (probability >= args.threshold).astype(np.uint8)
        target = preprocess_mask(mask, args.image_size)

        score = metrics(prediction, target)
        global_stats.update(prediction, target)
        rows.append({"patch_id": str(row["patch_id"]), **score})

    per_image = pd.DataFrame(rows)
    per_image.to_csv(out_dir / "per_image_metrics.csv", index=False)
    global_result = global_stats.compute()
    summary = {
        "model": args.model,
        "split": args.split,
        "threshold": args.threshold,
        "num_images": int(len(frame)),
        "global": global_result,
        "per_image_mean": {
            metric: float(per_image[metric].mean())
            for metric in ["iou", "dice", "precision", "recall", "accuracy"]
        },
    }
    save_json(summary, out_dir / "metrics_summary.json")

    print("\nDataset evaluation complete")
    for k, v in global_result.items():
        print(f"{k}: {v:.4f}")
    print("Saved to:", out_dir)


def main():
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.metadata = args.metadata.resolve()
    args.satmae_root = args.satmae_root.resolve()
    args.satmae_pretrained = args.satmae_pretrained.resolve()
    args.swin_root = args.swin_root.resolve()
    args.swin_config = args.swin_config.resolve()

    out_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else (Path("outputs") / "evaluation" / args.model / args.mode).resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    model, checkpoint = load_model(args, device)
    print("Checkpoint:", checkpoint)

    if args.mode == "single":
        evaluate_single(args, model, device, out_dir, use_amp)
    else:
        evaluate_dataset(args, model, device, out_dir, use_amp)


if __name__ == "__main__":
    main()
