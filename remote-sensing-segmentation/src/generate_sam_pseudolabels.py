#!/usr/bin/env python
"""Leak-free SatMAE -> SAM pseudo-label generation for INRIA.

Pipeline:
1) Split original TRAIN by source_image into 10% labeled / 90% unlabeled.
2) Train a NEW SatMAE teacher ONLY on the 10% labeled subset, starting from the
   original FMoW-RGB pretrained checkpoint.
3) Use that teacher only on the 90% unlabeled subset to create coarse masks.
4) Convert coarse connected components to box prompts and refine them with SAM.
5) Write metadata_semisupervised_leakfree.csv containing 10% GT + 90% pseudo.

Run from project root:
    python src/generate_sam_pseudolabels_leakfree.py
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
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

MEAN = (0.4182007312774658, 0.4214799106121063, 0.3991275727748871)
STD = (0.28774282336235046, 0.27541765570640564, 0.2764017581939697)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("datasets/inria_processed"))
    p.add_argument("--metadata", type=Path, default=None)
    p.add_argument("--satmae-root", type=Path, default=Path("external/satmae_pp"))
    p.add_argument("--sam-root", type=Path, default=Path("external/segment-anything"))
    p.add_argument("--satmae-pretrained-checkpoint", type=Path,
                   default=Path("checkpoints/satmae/checkpoint_ViT-L_pretrain_fmow_rgb.pth"))
    p.add_argument("--sam-checkpoint", type=Path,
                   default=Path("checkpoints/sam/sam_vit_b_01ec64.pth"))
    p.add_argument("--sam-model-type", default="vit_b", choices=("vit_b", "vit_l", "vit_h"))
    p.add_argument("--teacher-output-dir", type=Path,
                   default=Path("outputs/satmae_teacher_10pct_leakfree"))
    p.add_argument("--pseudo-output-dir", type=Path,
                   default=Path("datasets/inria_processed/pseudo_leakfree"))
    p.add_argument("--output-metadata", type=Path,
                   default=Path("datasets/inria_processed/metadata_semisupervised_leakfree.csv"))
    p.add_argument("--split-manifest", type=Path,
                   default=Path("datasets/inria_processed/leakfree_split_manifest.csv"))
    p.add_argument("--labeled-fraction", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--teacher-batch-size", type=int, default=1)
    p.add_argument("--accumulation-steps", type=int, default=4)
    p.add_argument("--teacher-epochs", type=int, default=20)
    p.add_argument("--teacher-patience", type=int, default=5)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--encoder-lr", type=float, default=1e-5)
    p.add_argument("--decoder-lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--drop-path", type=float, default=0.2)
    p.add_argument("--bce-weight", type=float, default=0.5)
    p.add_argument("--dice-weight", type=float, default=0.5)
    p.add_argument("--pos-weight", type=float, default=1.0)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--no-amp", action="store_true")

    p.add_argument("--pseudo-batch-size", type=int, default=1)
    p.add_argument("--satmae-threshold", type=float, default=0.50)
    p.add_argument("--min-component-area", type=int, default=20)
    p.add_argument("--box-margin", type=int, default=8)
    p.add_argument("--min-sam-coarse-iou", type=float, default=0.10)
    p.add_argument("--sam-score-weight", type=float, default=0.25)
    p.add_argument("--max-pseudo-samples", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def resolve(args):
    args.data_root = args.data_root.resolve()
    args.metadata = (args.data_root / "metadata.csv") if args.metadata is None else args.metadata.resolve()
    for name in ["satmae_root", "sam_root", "satmae_pretrained_checkpoint", "sam_checkpoint",
                 "teacher_output_dir", "pseudo_output_dir", "output_metadata", "split_manifest"]:
        setattr(args, name, getattr(args, name).resolve())


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {k: str(v) if isinstance(v, Path) else v for k, v in data.items()}
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")


def split_train(metadata, frac, seed):
    train = metadata[metadata["split"] == "train"].copy()
    group_col = "source_image" if "source_image" in train.columns else "patch_id"
    groups = train[group_col].astype(str).drop_duplicates().tolist()
    rng = random.Random(seed); rng.shuffle(groups)
    n = max(1, round(len(groups) * frac))
    labeled_groups = set(groups[:n])
    labeled = train[train[group_col].astype(str).isin(labeled_groups)].copy()
    unlabeled = train[~train[group_col].astype(str).isin(labeled_groups)].copy()
    overlap = set(labeled[group_col].astype(str)) & set(unlabeled[group_col].astype(str))
    if overlap:
        raise RuntimeError("Leakage: a source_image is present in both 10% and 90% subsets.")
    manifest = train[["patch_id", group_col]].copy()
    manifest["teacher_role"] = np.where(
        manifest[group_col].astype(str).isin(labeled_groups), "labeled_10pct", "unlabeled_90pct")
    print(f"Grouping column: {group_col}")
    print(f"Labeled groups: {n}/{len(groups)}")
    print(f"Labeled patches: {len(labeled)}")
    print(f"Unlabeled patches: {len(unlabeled)}")
    return labeled, unlabeled, manifest


class RowsDataset(Dataset):
    def __init__(self, root, rows, size=224, augment=False, with_masks=True):
        self.root = root; self.rows = rows.reset_index(drop=True); self.size = size
        self.augment = augment; self.with_masks = with_masks
        self.jitter = ColorJitter(brightness=.15, contrast=.15, saturation=.10, hue=.02)

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        row = self.rows.iloc[i]
        with Image.open(self.root / row["image_path"]) as f: image = f.convert("RGB")
        mask = None
        if self.with_masks:
            with Image.open(self.root / row["mask_path"]) as f: mask = f.convert("L")
            if self.augment:
                if random.random() < .5: image, mask = ImageOps.mirror(image), ImageOps.mirror(mask)
                if random.random() < .5: image, mask = ImageOps.flip(image), ImageOps.flip(mask)
                k = random.randint(0, 3)
                if k:
                    image, mask = image.rotate(90*k, expand=False), mask.rotate(90*k, expand=False)
                image = self.jitter(image)
        image = TF.resize(image, [self.size, self.size], interpolation=InterpolationMode.BICUBIC, antialias=True)
        image = TF.to_tensor(image)[[2, 1, 0], :, :]
        image = TF.normalize(image, MEAN, STD)
        out = {"image": image, "patch_id": str(row["patch_id"])}
        if mask is not None:
            mask = TF.resize(mask, [self.size, self.size], interpolation=InterpolationMode.NEAREST)
            out["mask"] = (TF.pil_to_tensor(mask) > 127).float()
        return out


def import_model():
    src = Path(__file__).resolve().parent
    if str(src) not in sys.path: sys.path.insert(0, str(src))
    from satmae_baseline import SatMAESegmenter
    return SatMAESegmenter


class BCEDice(nn.Module):
    def __init__(self, bw=.5, dw=.5, pos=1.0):
        super().__init__(); self.bw = bw; self.dw = dw
        self.register_buffer("pos_weight", torch.tensor([pos], dtype=torch.float32))

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight)
        p = torch.sigmoid(logits).flatten(1); t = targets.flatten(1)
        inter = (p*t).sum(1); dice = 1 - ((2*inter+1)/(p.sum(1)+t.sum(1)+1)).mean()
        return self.bw*bce + self.dw*dice


class Metrics:
    def __init__(self, thr=.5): self.thr = thr; self.tp=self.fp=self.fn=self.tn=0
    @torch.no_grad()
    def update(self, logits, targets):
        p = torch.sigmoid(logits) >= self.thr; t = targets >= .5
        self.tp += torch.logical_and(p,t).sum().item(); self.fp += torch.logical_and(p,~t).sum().item()
        self.fn += torch.logical_and(~p,t).sum().item(); self.tn += torch.logical_and(~p,~t).sum().item()
    def compute(self):
        e=1e-7; tp,fp,fn,tn=self.tp,self.fp,self.fn,self.tn
        return {"iou":tp/(tp+fp+fn+e), "dice":2*tp/(2*tp+fp+fn+e),
                "precision":tp/(tp+fp+e), "recall":tp/(tp+fn+e),
                "accuracy":(tp+tn)/(tp+fp+fn+tn+e)}


def run_epoch(model, loader, criterion, device, use_amp, threshold, optimizer=None, scaler=None, accum=1, desc=""):
    train = optimizer is not None
    model.train(train); metrics = Metrics(threshold); total=0.; n=0
    if train: optimizer.zero_grad(set_to_none=True)
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for step,batch in enumerate(tqdm(loader, desc=desc, leave=False),1):
            x=batch["image"].to(device,non_blocking=True); y=batch["mask"].to(device,non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits=model(x); raw=criterion(logits,y); loss=raw/accum
            if train:
                scaler.scale(loss).backward()
                if step%accum==0 or step==len(loader):
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            total += raw.item()*x.size(0); n += x.size(0); metrics.update(logits.detach(),y)
    return total/max(n,1), metrics.compute()


def train_teacher(args, labeled, val, device):
    out=args.teacher_output_dir
    if args.overwrite and out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    train_ds=RowsDataset(args.data_root,labeled,args.image_size,True,True)
    val_ds=RowsDataset(args.data_root,val,args.image_size,False,True)
    train_ld=DataLoader(train_ds,batch_size=args.teacher_batch_size,shuffle=True,num_workers=args.num_workers,
                        pin_memory=device.type=="cuda",persistent_workers=args.num_workers>0)
    val_ld=DataLoader(val_ds,batch_size=args.teacher_batch_size,shuffle=False,num_workers=args.num_workers,
                      pin_memory=device.type=="cuda",persistent_workers=args.num_workers>0)
    SatMAE=import_model()
    model=SatMAE(args.satmae_root,args.satmae_pretrained_checkpoint,args.image_size,args.patch_size,args.drop_path).to(device)
    crit=BCEDice(args.bce_weight,args.dice_weight,args.pos_weight).to(device)
    enc=list(model.encoder.parameters()); dec=[p for n,p in model.named_parameters() if not n.startswith("encoder.")]
    opt=AdamW([{"params":enc,"lr":args.encoder_lr},{"params":dec,"lr":args.decoder_lr}],weight_decay=args.weight_decay)
    sch=CosineAnnealingLR(opt,T_max=args.teacher_epochs)
    amp=device.type=="cuda" and not args.no_amp; scaler=torch.amp.GradScaler("cuda",enabled=amp)
    best=-math.inf; bad=0; hist=[]; best_path=out/"best_model.pth"
    for epoch in range(1,args.teacher_epochs+1):
        start=time.time()
        tl,tm=run_epoch(model,train_ld,crit,device,amp,args.threshold,opt,scaler,args.accumulation_steps,f"Teacher {epoch} train")
        vl,vm=run_epoch(model,val_ld,crit,device,amp,args.threshold,desc=f"Teacher {epoch} val")
        hist.append({"epoch":epoch,"train_loss":tl,"train_iou":tm["iou"],"train_dice":tm["dice"],
                     "val_loss":vl,"val_iou":vm["iou"],"val_dice":vm["dice"],"val_precision":vm["precision"],
                     "val_recall":vm["recall"],"val_accuracy":vm["accuracy"],"epoch_seconds":time.time()-start})
        pd.DataFrame(hist).to_csv(out/"history.csv",index=False)
        ck={"epoch":epoch,"model_state_dict":model.state_dict(),"val_iou":vm["iou"]}
        torch.save(ck,out/"last_model.pth")
        if vm["iou"]>best:
            best=vm["iou"]; bad=0; torch.save(ck,best_path)
            save_json({"epoch":epoch,**{f"val_{k}":v for k,v in vm.items()},"val_loss":vl},out/"best_metrics.json")
        else: bad+=1
        sch.step()
        print(f"Teacher {epoch}/{args.teacher_epochs}: val IoU={vm['iou']:.4f}, best={best:.4f}")
        if args.teacher_patience>0 and bad>=args.teacher_patience:
            print("Teacher early stopping."); break
    del model,opt,sch,scaler,train_ld,val_ld,train_ds,val_ds; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return best_path


def load_teacher(args, checkpoint, device):
    SatMAE=import_model()
    model=SatMAE(args.satmae_root,args.satmae_pretrained_checkpoint,args.image_size,args.patch_size,args.drop_path)
    ck=torch.load(checkpoint,map_location="cpu",weights_only=False)
    model.load_state_dict(ck["model_state_dict"],strict=True)
    return model.to(device).eval()


@torch.no_grad()
def coarse_masks(args, rows, teacher_ckpt, device):
    out=args.pseudo_output_dir/"coarse_masks"; out.mkdir(parents=True,exist_ok=True)
    ds=RowsDataset(args.data_root,rows,args.image_size,False,False)
    ld=DataLoader(ds,batch_size=args.pseudo_batch_size,shuffle=False,num_workers=args.num_workers,
                  pin_memory=device.type=="cuda",persistent_workers=args.num_workers>0)
    model=load_teacher(args,teacher_ckpt,device); amp=device.type=="cuda" and not args.no_amp
    for batch in tqdm(ld,desc="Leak-free SatMAE coarse masks"):
        x=batch["image"].to(device,non_blocking=True)
        with torch.autocast(device_type=device.type,enabled=amp): probs=torch.sigmoid(model(x)).float().cpu().numpy()
        for i,pid in enumerate(batch["patch_id"]):
            Image.fromarray(((probs[i,0]>=args.satmae_threshold).astype(np.uint8))*255).save(out/f"{pid}.png")
    del model,ld,ds; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return out


def components(coarse,h,w,min_area,margin):
    binary=(coarse>0).astype(np.uint8)
    n,labels,stats,_=cv2.connectedComponentsWithStats(binary,connectivity=8)
    sh,sw=binary.shape; sx=w/sw; sy=h/sh; out=[]
    for lab in range(1,n):
        if int(stats[lab,cv2.CC_STAT_AREA])<min_area: continue
        x=int(stats[lab,cv2.CC_STAT_LEFT]); y=int(stats[lab,cv2.CC_STAT_TOP])
        ww=int(stats[lab,cv2.CC_STAT_WIDTH]); hh=int(stats[lab,cv2.CC_STAT_HEIGHT])
        cm=cv2.resize((labels==lab).astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST).astype(bool)
        x0=max(0,int(round(x*sx))-margin); y0=max(0,int(round(y*sy))-margin)
        x1=min(w-1,int(round((x+ww)*sx))+margin); y1=min(h-1,int(round((y+hh)*sy))+margin)
        if x1>x0 and y1>y0: out.append({"box":np.array([x0,y0,x1,y1],np.float32),"coarse":cm})
    return out


def iou(a,b):
    inter=np.logical_and(a,b).sum(); union=np.logical_or(a,b).sum()
    return float(inter/union) if union else 0.0


def load_sam(args,device):
    if args.sam_root.is_dir() and str(args.sam_root) not in sys.path: sys.path.insert(0,str(args.sam_root))
    from segment_anything import SamPredictor, sam_model_registry
    sam=sam_model_registry[args.sam_model_type](checkpoint=str(args.sam_checkpoint)); sam.to(device=device); sam.eval()
    return SamPredictor(sam)


@torch.no_grad()
def sam_pseudolabels(args, rows, coarse_dir, device):
    masks_dir=args.pseudo_output_dir/"masks"; masks_dir.mkdir(parents=True,exist_ok=True)
    predictor=load_sam(args,device); report=[]
    for _,row in tqdm(rows.iterrows(),total=len(rows),desc="Leak-free SAM pseudo-labels"):
        pid=str(row["patch_id"])
        with Image.open(args.data_root/row["image_path"]) as f: image=np.array(f.convert("RGB"))
        with Image.open(coarse_dir/f"{pid}.png") as f: coarse=np.array(f.convert("L"))
        h,w=image.shape[:2]; comps=components(coarse,h,w,args.min_component_area,args.box_margin)
        final=np.zeros((h,w),dtype=bool); accepted=0
        if comps:
            predictor.set_image(image)
            for c in comps:
                masks,scores,_=predictor.predict(point_coords=None,point_labels=None,box=c["box"],multimask_output=True)
                best=None; best_value=-math.inf; best_overlap=0.
                for j in range(masks.shape[0]):
                    cand=masks[j].astype(bool); overlap=iou(cand,c["coarse"])
                    value=(1-args.sam_score_weight)*overlap + args.sam_score_weight*float(scores[j])
                    if value>best_value: best_value=value; best_overlap=overlap; best=cand
                if best is not None and best_overlap>=args.min_sam_coarse_iou:
                    final=np.logical_or(final,best); accepted+=1
        path=masks_dir/f"{pid}.png"; Image.fromarray(final.astype(np.uint8)*255).save(path)
        report.append({"patch_id":pid,"pseudo_mask_path":path.relative_to(args.data_root).as_posix(),
                       "num_components":len(comps),"accepted_components":accepted,"building_ratio":float(final.mean())})
    return pd.DataFrame(report)


def final_metadata(args, original, labeled, unlabeled, report):
    out=original.copy(); out["label_type"]=""; out["training_mask_path"]=""; out["teacher_seen_ground_truth"]=False
    labeled_ids=set(labeled["patch_id"].astype(str)); unlabeled_ids=set(unlabeled["patch_id"].astype(str))
    pseudo_map=report.set_index("patch_id")["pseudo_mask_path"].to_dict()
    for idx,row in out.iterrows():
        pid=str(row["patch_id"]); split=str(row["split"])
        if split=="train" and pid in labeled_ids:
            out.at[idx,"label_type"]="ground_truth"; out.at[idx,"training_mask_path"]=row["mask_path"]
            out.at[idx,"teacher_seen_ground_truth"]=True
        elif split=="train" and pid in unlabeled_ids:
            out.at[idx,"label_type"]="pseudo"; out.at[idx,"training_mask_path"]=pseudo_map[pid]
            out.at[idx,"teacher_seen_ground_truth"]=False
        elif split=="val":
            out.at[idx,"label_type"]="validation"; out.at[idx,"training_mask_path"]=row["mask_path"]
    if out.loc[out["label_type"]=="pseudo","teacher_seen_ground_truth"].any():
        raise RuntimeError("Leakage assertion failed.")
    args.output_metadata.parent.mkdir(parents=True,exist_ok=True); out.to_csv(args.output_metadata,index=False)
    return out


def main():
    args=parse_args(); resolve(args); seed_all(args.seed)
    if not args.metadata.is_file(): raise FileNotFoundError(args.metadata)
    if not args.satmae_pretrained_checkpoint.is_file(): raise FileNotFoundError(args.satmae_pretrained_checkpoint)
    if not args.sam_checkpoint.is_file(): raise FileNotFoundError(args.sam_checkpoint)
    if not 0<args.labeled_fraction<1: raise ValueError("labeled-fraction must be in (0,1)")
    if args.overwrite:
        if args.teacher_output_dir.exists(): shutil.rmtree(args.teacher_output_dir)
        if args.pseudo_output_dir.exists(): shutil.rmtree(args.pseudo_output_dir)
    args.pseudo_output_dir.mkdir(parents=True,exist_ok=True)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:",device)
    if device.type=="cuda": print("GPU:",torch.cuda.get_device_name(0))

    meta=pd.read_csv(args.metadata,keep_default_na=False)
    labeled,unlabeled,manifest=split_train(meta,args.labeled_fraction,args.seed)
    val=meta[meta["split"]=="val"].copy()
    if val.empty: raise RuntimeError("No validation rows found.")
    args.split_manifest.parent.mkdir(parents=True,exist_ok=True); manifest.to_csv(args.split_manifest,index=False)
    save_json(vars(args),args.pseudo_output_dir/"leakfree_config.json")

    run_unlabeled=unlabeled.copy()
    if args.max_pseudo_samples is not None:
        run_unlabeled=run_unlabeled.head(args.max_pseudo_samples).copy()
        print(f"DEBUG: generating only {len(run_unlabeled)} pseudo-labels; final metadata will not be written.")

    teacher_ckpt=train_teacher(args,labeled,val,device)
    coarse_dir=coarse_masks(args,run_unlabeled,teacher_ckpt,device)
    report=sam_pseudolabels(args,run_unlabeled,coarse_dir,device)
    report_path=args.pseudo_output_dir/"pseudo_label_report.csv"; report.to_csv(report_path,index=False)

    if args.max_pseudo_samples is None:
        final=final_metadata(args,meta,labeled,unlabeled,report)
        train=final[final["split"]=="train"]
        print("\nLeak-free pipeline completed.")
        print("GT train patches:",(train["label_type"]=="ground_truth").sum())
        print("Pseudo train patches:",(train["label_type"]=="pseudo").sum())
        print("Leakage check: PASSED")
        print("Teacher checkpoint:",teacher_ckpt)
        print("Final metadata:",args.output_metadata)
        print("Split manifest:",args.split_manifest)
        print("Pseudo-label report:",report_path)
    else:
        print("Debug run completed. Final semi-supervised metadata intentionally not created.")


if __name__ == "__main__":
    main()
