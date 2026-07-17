#!/usr/bin/env python
"""SatMAE++ + Haar Wavelet Loss for INRIA building segmentation.

Comparison target: src/satmae_baseline.py
Only methodological change: add high-frequency Haar wavelet loss.

Defaults:
  epochs=20, patience=5, batch_size=1, accumulation_steps=4
  total_loss = 0.5*BCE + 0.5*Dice + 0.1*WaveletHF

Run from project root:
  python src/satmae_wavelet_finetune.py
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, default=Path('datasets/inria_processed'))
    p.add_argument('--satmae-root', type=Path, default=Path('external/satmae_pp'))
    p.add_argument('--pretrained-checkpoint', type=Path,
                   default=Path('checkpoints/satmae/checkpoint_ViT-L_pretrain_fmow_rgb.pth'))
    p.add_argument('--output-dir', type=Path, default=Path('outputs/satmae_wavelet'))
    p.add_argument('--image-size', type=int, default=224)
    p.add_argument('--patch-size', type=int, default=16)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--accumulation-steps', type=int, default=4)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--encoder-lr', type=float, default=1e-5)
    p.add_argument('--decoder-lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=0.05)
    p.add_argument('--drop-path', type=float, default=0.2)
    p.add_argument('--bce-weight', type=float, default=0.5)
    p.add_argument('--dice-weight', type=float, default=0.5)
    p.add_argument('--wavelet-weight', type=float, default=0.1)
    p.add_argument('--wavelet-low-weight', type=float, default=0.0)
    p.add_argument('--pos-weight', type=float, default=1.0)
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no-amp', action='store_true')
    return p.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def import_baseline_components():
    src_dir = Path(__file__).resolve().parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    try:
        from satmae_baseline import InriaSatMAEDataset, SatMAESegmenter
    except ImportError as e:
        raise ImportError(
            'Cannot import InriaSatMAEDataset/SatMAESegmenter from src/satmae_baseline.py'
        ) from e
    return InriaSatMAEDataset, SatMAESegmenter


def haar_dwt2d(x):
    """One-level differentiable 2D Haar transform for [B,C,H,W]."""
    if x.ndim != 4:
        raise ValueError(f'Expected [B,C,H,W], got {tuple(x.shape)}')
    h, w = x.shape[-2:]
    if h % 2 or w % 2:
        x = x[..., :h - h % 2, :w - w % 2]

    x00 = x[..., 0::2, 0::2]
    x01 = x[..., 0::2, 1::2]
    x10 = x[..., 1::2, 0::2]
    x11 = x[..., 1::2, 1::2]
    s = 0.5

    ll = s * (x00 + x01 + x10 + x11)
    lh = s * (-x00 - x01 + x10 + x11)
    hl = s * (-x00 + x01 - x10 + x11)
    hh = s * (x00 - x01 - x10 + x11)
    return ll, lh, hl, hh


class CombinedWaveletLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, wavelet_weight=0.1,
                 low_weight=0.0, pos_weight=1.0, smooth=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.wavelet_weight = wavelet_weight
        self.low_weight = low_weight
        self.smooth = smooth
        self.register_buffer('pos_weight', torch.tensor([pos_weight], dtype=torch.float32))

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight
        )

        probs = torch.sigmoid(logits)
        pf = probs.flatten(1)
        tf = targets.flatten(1)
        inter = (pf * tf).sum(dim=1)
        dice = 1.0 - ((2.0 * inter + self.smooth) /
                      (pf.sum(dim=1) + tf.sum(dim=1) + self.smooth)).mean()

        pll, plh, phl, phh = haar_dwt2d(probs)
        tll, tlh, thl, thh = haar_dwt2d(targets)
        lh = F.l1_loss(plh, tlh)
        hl = F.l1_loss(phl, thl)
        hh = F.l1_loss(phh, thh)
        ll = F.l1_loss(pll, tll)
        hf = (lh + hl + hh) / 3.0
        wavelet = hf + self.low_weight * ll

        base = self.bce_weight * bce + self.dice_weight * dice
        total = base + self.wavelet_weight * wavelet
        parts = {
            'bce': bce.detach(), 'dice_loss': dice.detach(),
            'base_loss': base.detach(), 'wavelet_loss': wavelet.detach(),
            'wavelet_high': hf.detach(), 'wavelet_ll': ll.detach(),
            'wavelet_lh': lh.detach(), 'wavelet_hl': hl.detach(),
            'wavelet_hh': hh.detach(),
        }
        return total, parts


class Metrics:
    def __init__(self, threshold=0.5, eps=1e-7):
        self.threshold, self.eps = threshold, eps
        self.tp = self.fp = self.fn = self.tn = 0

    @torch.no_grad()
    def update(self, logits, targets):
        p = torch.sigmoid(logits) >= self.threshold
        t = targets >= 0.5
        self.tp += torch.logical_and(p, t).sum().item()
        self.fp += torch.logical_and(p, ~t).sum().item()
        self.fn += torch.logical_and(~p, t).sum().item()
        self.tn += torch.logical_and(~p, ~t).sum().item()

    def compute(self):
        e = self.eps
        return {
            'iou': self.tp / (self.tp + self.fp + self.fn + e),
            'dice': 2*self.tp / (2*self.tp + self.fp + self.fn + e),
            'precision': self.tp / (self.tp + self.fp + e),
            'recall': self.tp / (self.tp + self.fn + e),
            'accuracy': (self.tp+self.tn)/(self.tp+self.fp+self.fn+self.tn+e),
        }


def run_epoch(model, loader, criterion, device, threshold, use_amp, description,
              optimizer=None, scaler=None, accumulation_steps=1):
    training = optimizer is not None
    model.train(training)
    if training:
        optimizer.zero_grad(set_to_none=True)

    metrics = Metrics(threshold)
    total_loss = total_n = 0
    keys = ['base_loss', 'wavelet_loss', 'wavelet_lh', 'wavelet_hl', 'wavelet_hh']
    sums = {k: 0.0 for k in keys}

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for step, batch in enumerate(tqdm(loader, desc=description, leave=False), start=1):
            images = batch['image'].to(device, non_blocking=True)
            masks = batch['mask'].to(device, non_blocking=True)
            bs = images.shape[0]

            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                raw_loss, parts = criterion(logits, masks)
                loss = raw_loss / accumulation_steps

            if training:
                scaler.scale(loss).backward()
                if step % accumulation_steps == 0 or step == len(loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

            total_loss += raw_loss.item() * bs
            total_n += bs
            for k in keys:
                sums[k] += parts[k].item() * bs
            metrics.update(logits.detach(), masks)

    n = max(total_n, 1)
    return total_loss/n, metrics.compute(), {k: v/n for k, v in sums.items()}


@torch.no_grad()
def save_val_predictions(model, loader, device, threshold, use_amp, output_path):
    model.eval()
    rows, eps = [], 1e-7
    for batch in tqdm(loader, desc='Validation predictions', leave=False):
        images = batch['image'].to(device)
        targets = batch['mask'].to(device) >= 0.5
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
        preds = torch.sigmoid(logits) >= threshold
        for i in range(images.shape[0]):
            p, t = preds[i].flatten(), targets[i].flatten()
            tp = torch.logical_and(p,t).sum().item()
            fp = torch.logical_and(p,~t).sum().item()
            fn = torch.logical_and(~p,t).sum().item()
            tn = torch.logical_and(~p,~t).sum().item()
            rows.append({
                'patch_id': batch['patch_id'][i],
                'iou': tp/(tp+fp+fn+eps),
                'dice': 2*tp/(2*tp+fp+fn+eps),
                'precision': tp/(tp+fp+eps),
                'recall': tp/(tp+fn+eps),
                'accuracy': (tp+tn)/(tp+fp+fn+tn+eps),
            })
    pd.DataFrame(rows).to_csv(output_path, index=False)


def save_plots(history, output_dir):
    p = output_dir/'plots'
    p.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8,5))
    plt.plot(history.epoch, history.train_loss, label='Train')
    plt.plot(history.epoch, history.val_loss, label='Validation')
    plt.xlabel('Epoch'); plt.ylabel('Total loss'); plt.title('SatMAE + Haar Wavelet Loss')
    plt.grid(alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig(p/'loss.png', dpi=180); plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(history.epoch, history.train_wavelet_loss, label='Train wavelet')
    plt.plot(history.epoch, history.val_wavelet_loss, label='Validation wavelet')
    plt.xlabel('Epoch'); plt.ylabel('Wavelet loss'); plt.title('High-frequency Wavelet Loss')
    plt.grid(alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig(p/'wavelet_loss.png', dpi=180); plt.close()

    plt.figure(figsize=(8,5))
    for col, label in [('val_iou','IoU'),('val_dice','Dice'),('val_precision','Precision'),('val_recall','Recall')]:
        plt.plot(history.epoch, history[col], label=label)
    plt.xlabel('Epoch'); plt.ylabel('Score'); plt.ylim(0,1); plt.title('Validation metrics')
    plt.grid(alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig(p/'metrics.png', dpi=180); plt.close()


def main():
    args = parse_args()
    seed_all(args.seed)

    args.data_root = args.data_root.resolve()
    args.satmae_root = args.satmae_root.resolve()
    args.pretrained_checkpoint = args.pretrained_checkpoint.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    InriaSatMAEDataset, SatMAESegmenter = import_baseline_components()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    if device.type == 'cuda': print('GPU:', torch.cuda.get_device_name(0))
    print('Epochs:', args.epochs)
    print('Patience:', args.patience)
    print('Effective batch size:', args.batch_size * args.accumulation_steps)
    print('Wavelet weight:', args.wavelet_weight)

    use_amp = device.type == 'cuda' and not args.no_amp
    cfg = {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    save_json(cfg, args.output_dir/'config.json')

    train_ds = InriaSatMAEDataset(root=args.data_root, split='train',
                                   image_size=args.image_size, augment=True)
    val_ds = InriaSatMAEDataset(root=args.data_root, split='val',
                                 image_size=args.image_size, augment=False)
    print('Train patches:', len(train_ds))
    print('Validation patches:', len(val_ds))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=device.type=='cuda',
                              persistent_workers=args.num_workers>0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=device.type=='cuda',
                            persistent_workers=args.num_workers>0)

    model = SatMAESegmenter(satmae_root=args.satmae_root,
                            checkpoint_path=args.pretrained_checkpoint,
                            image_size=args.image_size, patch_size=args.patch_size,
                            drop_path=args.drop_path).to(device)

    criterion = CombinedWaveletLoss(
        args.bce_weight, args.dice_weight, args.wavelet_weight,
        args.wavelet_low_weight, args.pos_weight
    ).to(device)

    encoder_params = list(model.encoder.parameters())
    decoder_params = [p for n,p in model.named_parameters() if not n.startswith('encoder.')]
    optimizer = AdamW([
        {'params': encoder_params, 'lr': args.encoder_lr},
        {'params': decoder_params, 'lr': args.decoder_lr},
    ], weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    history = []
    best_iou = -math.inf
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, args.epochs+1):
        start = time.time()
        tr_loss, tr_m, tr_c = run_epoch(
            model, train_loader, criterion, device, args.threshold, use_amp,
            f'Epoch {epoch} train', optimizer, scaler, args.accumulation_steps)
        va_loss, va_m, va_c = run_epoch(
            model, val_loader, criterion, device, args.threshold, use_amp,
            f'Epoch {epoch} val')

        row = {
            'epoch': epoch,
            'train_loss': tr_loss, 'train_base_loss': tr_c['base_loss'],
            'train_wavelet_loss': tr_c['wavelet_loss'],
            'train_iou': tr_m['iou'], 'train_dice': tr_m['dice'],
            'train_precision': tr_m['precision'], 'train_recall': tr_m['recall'],
            'train_accuracy': tr_m['accuracy'],
            'val_loss': va_loss, 'val_base_loss': va_c['base_loss'],
            'val_wavelet_loss': va_c['wavelet_loss'],
            'val_iou': va_m['iou'], 'val_dice': va_m['dice'],
            'val_precision': va_m['precision'], 'val_recall': va_m['recall'],
            'val_accuracy': va_m['accuracy'],
            'encoder_lr': optimizer.param_groups[0]['lr'],
            'decoder_lr': optimizer.param_groups[1]['lr'],
            'epoch_seconds': time.time()-start,
        }
        history.append(row)
        hist = pd.DataFrame(history)
        hist.to_csv(args.output_dir/'history.csv', index=False)
        save_plots(hist, args.output_dir)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_iou': va_m['iou'],
            'config': cfg,
        }
        torch.save(checkpoint, args.output_dir/'last_model.pth')

        if va_m['iou'] > best_iou:
            best_iou, best_epoch, no_improve = va_m['iou'], epoch, 0
            torch.save(checkpoint, args.output_dir/'best_model.pth')
            save_json({
                'epoch': epoch, 'val_loss': va_loss,
                'val_base_loss': va_c['base_loss'],
                'val_wavelet_loss': va_c['wavelet_loss'],
                'val_iou': va_m['iou'], 'val_dice': va_m['dice'],
                'val_precision': va_m['precision'], 'val_recall': va_m['recall'],
                'val_accuracy': va_m['accuracy'],
            }, args.output_dir/'best_metrics.json')
        else:
            no_improve += 1

        scheduler.step()

        print(f'\nEpoch {epoch}/{args.epochs}')
        print(f"Train | Loss {tr_loss:.4f} | Base {tr_c['base_loss']:.4f} | Wavelet {tr_c['wavelet_loss']:.4f} | IoU {tr_m['iou']:.4f} | Dice {tr_m['dice']:.4f}")
        print(f"Val   | Loss {va_loss:.4f} | Base {va_c['base_loss']:.4f} | Wavelet {va_c['wavelet_loss']:.4f} | IoU {va_m['iou']:.4f} | Dice {va_m['dice']:.4f}")
        print(f'Best IoU: {best_iou:.4f} (epoch {best_epoch})')

        if args.patience > 0 and no_improve >= args.patience:
            print('Early stopping.')
            break

    best = torch.load(args.output_dir/'best_model.pth', map_location=device, weights_only=False)
    model.load_state_dict(best['model_state_dict'])
    save_val_predictions(model, val_loader, device, args.threshold, use_amp,
                         args.output_dir/'val_predictions.csv')
    print('\nCompleted. Results:', args.output_dir)


if __name__ == '__main__':
    main()
