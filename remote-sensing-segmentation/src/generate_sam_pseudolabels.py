#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from pathlib import Path

import torch
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

SATMAE_FMOW_MEAN = (0.4182007312774658, 0.4214799106121063, 0.3991275727748871)
SATMAE_FMOW_STD = (0.28774282336235046, 0.27541765570640564, 0.2764017581939697)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, default=Path('datasets/inria_processed'))
    p.add_argument('--satmae-root', type=Path, default=Path('external/satmae_pp'))
    p.add_argument('--satmae-pretrained-checkpoint', type=Path,
                   default=Path('checkpoints/satmae/checkpoint_ViT-L_pretrain_fmow_rgb.pth'))
    p.add_argument('--satmae-finetuned-checkpoint', type=Path,
                   default=Path('outputs/satmae_baseline/best_model.pth'))
    p.add_argument('--sam-checkpoint', type=Path,
                   default=Path('checkpoints/sam/sam_vit_b_01ec64.pth'))
    p.add_argument('--sam-model-type', choices=['vit_b', 'vit_l', 'vit_h'], default='vit_b')
    p.add_argument('--labeled-fraction', type=float, default=0.10)
    p.add_argument('--image-size', type=int, default=224)
    p.add_argument('--satmae-threshold', type=float, default=0.50)
    p.add_argument('--min-component-area', type=int, default=20)
    p.add_argument('--box-margin', type=int, default=8)
    p.add_argument('--min-sam-coarse-iou', type=float, default=0.10)
    p.add_argument('--sam-score-weight', type=float, default=0.25)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-pseudo-samples', type=int, default=None)
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--keep-coarse-masks', action='store_true')
    return p.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_train(metadata: pd.DataFrame, labeled_fraction: float, seed: int):
    train = metadata[metadata['split'] == 'train'].copy()
    group_col = 'source_image' if 'source_image' in train.columns else 'patch_id'
    groups = train[group_col].astype(str).unique().tolist()
    rng = random.Random(seed)
    rng.shuffle(groups)
    n_labeled = max(1, round(len(groups) * labeled_fraction))
    labeled_groups = set(groups[:n_labeled])
    train['label_type'] = np.where(
        train[group_col].astype(str).isin(labeled_groups),
        'ground_truth',
        'pseudo',
    )
    pseudo = train[train['label_type'] == 'pseudo'].copy()
    print('Grouping column:', group_col)
    print('Labeled groups:', n_labeled, '/', len(groups))
    print('Ground-truth patches:', (train['label_type'] == 'ground_truth').sum())
    print('Pseudo-label patches:', len(pseudo))
    return train, pseudo


class CoarseDataset(Dataset):
    def __init__(self, data_root: Path, rows: pd.DataFrame, image_size: int):
        self.data_root = data_root
        self.rows = rows.reset_index(drop=True)
        self.image_size = image_size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows.iloc[idx]
        path = self.data_root / row['image_path']
        with Image.open(path) as f:
            img = f.convert('RGB')
        img = TF.resize(img, [self.image_size, self.image_size],
                        interpolation=InterpolationMode.BICUBIC, antialias=True)
        x = TF.to_tensor(img)
        x = x[[2, 1, 0]]  # RGB -> BGR for SatMAE++ RGB checkpoint
        x = TF.normalize(x, SATMAE_FMOW_MEAN, SATMAE_FMOW_STD)
        return {'image': x, 'patch_id': str(row['patch_id'])}


def load_satmae_model(args, device):
    src_dir = Path(__file__).resolve().parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from satmae_baseline import SatMAESegmenter

    model = SatMAESegmenter(
        satmae_root=args.satmae_root,
        checkpoint_path=args.satmae_pretrained_checkpoint,
        image_size=args.image_size,
        patch_size=16,
        drop_path=0.2,
    )
    ckpt = torch.load(args.satmae_finetuned_checkpoint, map_location='cpu', weights_only=False)
    state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


@torch.no_grad()
def generate_coarse_masks(args, pseudo_rows, coarse_dir, device):
    coarse_dir.mkdir(parents=True, exist_ok=True)
    dataset = CoarseDataset(args.data_root, pseudo_rows, args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == 'cuda')
    model = load_satmae_model(args, device)
    use_amp = device.type == 'cuda'

    for batch in tqdm(loader, desc='SatMAE coarse masks'):
        images = batch['image'].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
        probs = torch.sigmoid(logits).float().cpu().numpy()
        for i, patch_id in enumerate(batch['patch_id']):
            out = coarse_dir / f'{patch_id}.png'
            if out.exists() and not args.overwrite:
                continue
            mask = (probs[i, 0] >= args.satmae_threshold).astype(np.uint8) * 255
            Image.fromarray(mask).save(out)

    del model, loader, dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def extract_components(mask_small, out_h, out_w, min_area, margin):
    binary = (mask_small > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    h, w = binary.shape
    sx, sy = out_w / w, out_h / h
    result = []

    for label_id in range(1, n):
        x, y, bw, bh, area = stats[label_id]
        if area < min_area:
            continue
        component = cv2.resize((labels == label_id).astype(np.uint8), (out_w, out_h),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
        x0 = max(0, int(round(x * sx)) - margin)
        y0 = max(0, int(round(y * sy)) - margin)
        x1 = min(out_w - 1, int(round((x + bw) * sx)) + margin)
        y1 = min(out_h - 1, int(round((y + bh) * sy)) + margin)
        if x1 > x0 and y1 > y0:
            result.append((np.array([x0, y0, x1, y1], dtype=np.float32), component))
    return result


def binary_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def load_sam(args, device):
    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError as exc:
        raise ImportError(
            'segment_anything non installato. Esegui: pip install -e external/segment-anything'
        ) from exc
    sam = sam_model_registry[args.sam_model_type](checkpoint=str(args.sam_checkpoint))
    sam.to(device=device).eval()
    return SamPredictor(sam)


@torch.no_grad()
def generate_sam_masks(args, pseudo_rows, coarse_dir, pseudo_dir, device):
    pseudo_dir.mkdir(parents=True, exist_ok=True)
    predictor = load_sam(args, device)
    report = []

    for _, row in tqdm(pseudo_rows.iterrows(), total=len(pseudo_rows), desc='SAM pseudo-labels'):
        patch_id = str(row['patch_id'])
        image_path = args.data_root / row['image_path']
        coarse_path = coarse_dir / f'{patch_id}.png'
        output_path = pseudo_dir / f'{patch_id}.png'

        if output_path.exists() and not args.overwrite:
            with Image.open(output_path) as f:
                existing = np.array(f.convert('L')) > 127
            report.append({
                'patch_id': patch_id,
                'pseudo_mask_path': output_path.relative_to(args.data_root).as_posix(),
                'num_components': np.nan,
                'accepted_components': np.nan,
                'building_ratio': float(existing.mean()),
                'status': 'existing',
            })
            continue

        with Image.open(image_path) as f:
            rgb = np.array(f.convert('RGB'))
        with Image.open(coarse_path) as f:
            coarse_small = np.array(f.convert('L'))

        h, w = rgb.shape[:2]
        components = extract_components(
            coarse_small, h, w, args.min_component_area, args.box_margin
        )
        pseudo = np.zeros((h, w), dtype=bool)
        accepted = 0

        if components:
            predictor.set_image(rgb)
            for box, coarse_component in components:
                masks, scores, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=box,
                    multimask_output=True,
                )
                candidates = []
                for j in range(len(masks)):
                    m = masks[j].astype(bool)
                    overlap = binary_iou(m, coarse_component)
                    combined = ((1.0 - args.sam_score_weight) * overlap
                                + args.sam_score_weight * float(scores[j]))
                    candidates.append((combined, overlap, m))
                candidates.sort(key=lambda x: x[0], reverse=True)
                _, overlap, best_mask = candidates[0]
                if overlap >= args.min_sam_coarse_iou:
                    pseudo |= best_mask
                    accepted += 1

        Image.fromarray(pseudo.astype(np.uint8) * 255).save(output_path)
        report.append({
            'patch_id': patch_id,
            'pseudo_mask_path': output_path.relative_to(args.data_root).as_posix(),
            'num_components': len(components),
            'accepted_components': accepted,
            'building_ratio': float(pseudo.mean()),
            'status': 'pseudo' if pseudo.any() else 'pseudo_empty',
        })

    return pd.DataFrame(report)


def build_output_metadata(metadata, train_split, pseudo_report, output_path):
    out = metadata.copy()
    out['label_type'] = ''
    out['training_mask_path'] = ''

    val_mask = out['split'] == 'val'
    out.loc[val_mask, 'label_type'] = 'validation'
    out.loc[val_mask, 'training_mask_path'] = out.loc[val_mask, 'mask_path']

    train_type = train_split.set_index('patch_id')['label_type'].to_dict()
    pseudo_paths = pseudo_report.set_index('patch_id')['pseudo_mask_path'].to_dict()

    for idx, row in out[out['split'] == 'train'].iterrows():
        patch_id = str(row['patch_id'])
        label_type = train_type.get(patch_id, '')
        out.at[idx, 'label_type'] = label_type
        if label_type == 'ground_truth':
            out.at[idx, 'training_mask_path'] = row['mask_path']
        elif label_type == 'pseudo':
            if patch_id not in pseudo_paths:
                raise RuntimeError(f'Pseudo-label mancante per {patch_id}')
            out.at[idx, 'training_mask_path'] = pseudo_paths[patch_id]

    out.to_csv(output_path, index=False)
    return out


def main():
    args = parse_args()
    seed_everything(args.seed)

    args.data_root = args.data_root.resolve()
    args.satmae_root = args.satmae_root.resolve()
    args.satmae_pretrained_checkpoint = args.satmae_pretrained_checkpoint.resolve()
    args.satmae_finetuned_checkpoint = args.satmae_finetuned_checkpoint.resolve()
    args.sam_checkpoint = args.sam_checkpoint.resolve()

    metadata_path = args.data_root / 'metadata.csv'
    output_metadata = args.data_root / 'metadata_semisupervised.csv'
    pseudo_root = args.data_root / 'pseudo'
    pseudo_dir = pseudo_root / 'masks'
    coarse_dir = pseudo_root / 'coarse_masks'
    report_path = pseudo_root / 'pseudo_label_report.csv'

    for description, path in [
        ('metadata', metadata_path),
        ('SatMAE pretrained checkpoint', args.satmae_pretrained_checkpoint),
        ('SatMAE fine-tuned checkpoint', args.satmae_finetuned_checkpoint),
        ('SAM checkpoint', args.sam_checkpoint),
    ]:
        if not path.is_file():
            raise FileNotFoundError(f'{description} not found: {path}')

    if not 0.0 < args.labeled_fraction < 1.0:
        raise ValueError('--labeled-fraction must be between 0 and 1.')

    pseudo_root.mkdir(parents=True, exist_ok=True)
    with (pseudo_root / 'pseudolabel_config.json').open('w', encoding='utf-8') as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, f, indent=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    if device.type == 'cuda':
        print('GPU:', torch.cuda.get_device_name(0))

    metadata = pd.read_csv(metadata_path, keep_default_na=False)
    train_split, pseudo_rows = split_train(metadata, args.labeled_fraction, args.seed)

    if args.max_pseudo_samples is not None:
        pseudo_rows = pseudo_rows.head(args.max_pseudo_samples).copy()
        keep_ids = set(pseudo_rows['patch_id'].astype(str))
        train_split = train_split[
            (train_split['label_type'] == 'ground_truth')
            | train_split['patch_id'].astype(str).isin(keep_ids)
        ].copy()
        print('Debug pseudo-label limit:', len(pseudo_rows))

    print('\nStage 1/2: SatMAE coarse building predictions')
    generate_coarse_masks(args, pseudo_rows, coarse_dir, device)

    print('\nStage 2/2: SAM box-prompt refinement')
    pseudo_report = generate_sam_masks(args, pseudo_rows, coarse_dir, pseudo_dir, device)
    pseudo_report.to_csv(report_path, index=False)

    final_metadata = build_output_metadata(
        metadata, train_split, pseudo_report, output_metadata
    )

    if not args.keep_coarse_masks:
        for p in coarse_dir.glob('*.png'):
            p.unlink()
        try:
            coarse_dir.rmdir()
        except OSError:
            pass

    train_final = final_metadata[final_metadata['split'] == 'train']
    print('\nCompleted.')
    print('Ground-truth train patches:', (train_final['label_type'] == 'ground_truth').sum())
    print('Pseudo-labeled train patches:', (train_final['label_type'] == 'pseudo').sum())
    print('Pseudo masks:', pseudo_dir)
    print('Report:', report_path)
    print('Semi-supervised metadata:', output_metadata)
    print("During training, read 'training_mask_path' instead of 'mask_path'.")


if __name__ == '__main__':
    main()
