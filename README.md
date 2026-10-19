# Remote Sensing Image Segmentation with SatMAE++, SAM, and Wavelet-Based Strategies

**Authors**
- Simone La Bella — Student ID: 1995847
- Mattia Di Marco — Student ID: 2019367

## Overview

This project investigates deep-learning strategies for **building footprint segmentation from aerial imagery** using the **INRIA Aerial Image Labeling Dataset**.

The main goal is to evaluate the effectiveness of **remote-sensing-specific pretraining** through SatMAE++ and compare it with a Swin-Unet baseline. Starting from the SatMAE++ segmentation model, the project also explores:

- **Wavelet Loss**, to encourage consistency in high-frequency segmentation details;
- a **Wavelet-Aware Decoder**, which processes low- and high-frequency feature components separately;
- **SAM-based pseudo-labeling**, to study segmentation under limited ground-truth annotations.

The final models are evaluated using **IoU, Dice, Precision, Recall, and Accuracy**.

---

## Models

| Model | Description |
|---|---|
| **Swin-Unet** | Transformer-based segmentation baseline |
| **SatMAE++ Baseline** | ViT-Large encoder pretrained on FMoW-RGB + segmentation decoder |
| **SatMAE++ + Wavelet Loss** | Standard SatMAE++ architecture trained with an additional Haar high-frequency loss |
| **SatMAE++ + Wavelet-Aware Decoder** | SatMAE++ with Haar DWT/IDWT feature processing inside the decoder |
| **SatMAE++ + SAM Pseudo-Labels** | Semi-supervised model trained using 10% real labels and 90% SAM-refined pseudo-labels |

---

## Results

All models are evaluated on the same INRIA validation split using a prediction threshold of `0.5`.

| Model | IoU | Dice | Precision | Recall | Accuracy |
|---|---:|---:|---:|---:|---:|
| Swin-Unet | 0.685 | 0.813 | 0.800 | 0.827 | 0.891 |
| **SatMAE++ Baseline** | **0.732** | **0.845** | 0.809 | 0.885 | 0.907 |
| SatMAE++ + Wavelet Loss | 0.730 | 0.844 | 0.796 | 0.898 | 0.905 |
| SatMAE++ + Wavelet-Aware Decoder | 0.729 | 0.843 | **0.851** | 0.835 | **0.911** |
| SatMAE++ + SAM Pseudo-Labels | 0.685 | 0.813 | 0.722 | **0.930** | 0.877 |

The standard **SatMAE++ baseline achieves the best overall IoU and Dice**. The wavelet-based approaches mainly change the precision-recall trade-off, while the pseudo-label model obtains the highest recall using only a small fraction of real annotations.

> The SAM experiment is not directly comparable to the fully supervised models in terms of annotation budget: only **10% of the real training annotations** are used, while the remaining 90% are treated as unlabeled data.

---

## Environment

The experiments were developed using **Python 3.10** and PyTorch with CUDA support.

```bash
conda create -n remote-seg python=3.10
conda activate remote-seg
```

Install **PyTorch and Torchvision** according to your CUDA version:

[PyTorch Installation Guide](https://pytorch.org/get-started/locally/)

Install the remaining dependencies:

```bash
pip install numpy pandas pillow matplotlib tqdm rasterio opencv-python timm einops pyyaml yacs
```

### Main dependencies

| Package | Usage |
|---|---|
| `torch` | Model definition, training, losses, wavelet operations, inference |
| `torchvision` | Image transformations, normalization, augmentation |
| `numpy` | Numerical operations and mask processing |
| `pandas` | Metadata and evaluation results |
| `Pillow` | Image and mask loading |
| `matplotlib` | Training plots and qualitative evaluation |
| `tqdm` | Progress bars |
| `rasterio` | Reading original INRIA TIFF images |
| `opencv-python` | Connected-component extraction for pseudo-label generation |
| `timm` | Vision Transformer components |
| `einops` | Tensor rearrangement used by Swin-Unet |
| `PyYAML` | Swin-Unet configuration files |
| `yacs` | Swin-Unet configuration management |

No external wavelet library is required. The Haar DWT/IDWT operations used in the project are implemented directly with PyTorch tensor operations.

---

## External Repositories

Clone the required repositories inside `external/`:

```bash
mkdir external
cd external

git clone https://github.com/techmn/satmae_pp.git
git clone https://github.com/HuCaoFighting/Swin-Unet.git
git clone https://github.com/facebookresearch/segment-anything.git

cd ..
```

Required repositories:

- [SatMAE++](https://github.com/techmn/satmae_pp)
- [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet)
- [Segment Anything](https://github.com/facebookresearch/segment-anything)

SAMRS, WaveDM, and WaDiGAN-SR are methodological references and are **not required to execute the implemented pipeline**.

---

## Required Pretrained Checkpoints

### SatMAE++ ViT-Large — FMoW-RGB

[Download checkpoint](https://huggingface.co/mubashir04/checkpoint_ViT-L_pretrain_fmow_rgb/blob/main/checkpoint_ViT-L_pretrain_fmow_rgb.pth)

```text
checkpoints/
└── satmae/
    └── checkpoint_ViT-L_pretrain_fmow_rgb.pth
```

### Swin-Unet pretrained Swin-T

[Download checkpoint](https://drive.google.com/drive/folders/1UC3XOoezeum0uck4KBVGa8osahs6rKUY?usp=sharing)

```text
external/
└── Swin-Unet/
    └── pretrained_ckpt/
        └── swin_tiny_patch4_window7_224.pth
```

### Segment Anything ViT-B

[Download `sam_vit_b_01ec64.pth`](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

```text
checkpoints/
└── sam/
    └── sam_vit_b_01ec64.pth
```

---

## Dataset

The project uses the **INRIA Aerial Image Labeling Dataset**:

[INRIA Aerial Image Labeling Dataset](https://project.inria.fr/aerialimagelabeling/)

Store the raw dataset under:

```text
datasets/
└── INRIA-Aerial-Image-Labeling/
```

### Preprocessing

The preprocessing pipeline:

1. splits the annotated data at **source-image level**;
2. reserves **Vienna** for validation;
3. extracts aligned `512 × 512` image/mask patches;
4. converts masks to binary values;
5. generates PNG patches and `metadata.csv`.

Run:

```bash
python src/preprocessing/preprocess_inria.py \
    --dataset-root datasets/INRIA-Aerial-Image-Labeling \
    --output-root datasets/inria_processed \
    --tile-size 512 \
    --stride 512 \
    --val-city vienna
```

The processed dataset contains approximately:

- **14,400 training patches**
- **3,600 validation patches**

During training, patches are resized to `224 × 224`.

---

## Training

Run all commands from the project root.

### Swin-Unet baseline

```bash
python src/swinunet_baseline_original.py \
    --epochs 20 \
    --patience 5
```

Output: `outputs/swinunet_baseline/best_model.pth`

### SatMAE++ baseline

```bash
python src/satmae_baseline.py
```

Output: `outputs/satmae_baseline/best_model.pth`

### SatMAE++ + Wavelet Loss

```bash
python src/satmae_wavelet_finetune.py
```

Output: `outputs/satmae_wavelet/best_model.pth`

### SatMAE++ + Wavelet-Aware Decoder

```bash
python src/satmae_wavelet_decoder.py
```

Output: `outputs/satmae_wavelet_decoder/best_model.pth`

---

## Leak-Free SAM Pseudo-Label Pipeline

The semi-supervised experiment uses **10% labeled training data** and **90% treated as unlabeled**.

Generate pseudo-labels:

```bash
python src/generate_sam_pseudolabels_leakfree.py
```

Pipeline:

```text
10% ground truth
      ↓
SatMAE teacher
      ↓
Predictions on unseen 90%
      ↓
Connected components
      ↓
Bounding-box prompts
      ↓
SAM ViT-B
      ↓
Pseudo-labels
```

Generated metadata:

```text
datasets/inria_processed/
├── pseudo_leakfree/
│   └── masks/
├── leakfree_split_manifest.csv
└── metadata_semisupervised_leakfree.csv
```

Teacher checkpoint:

`outputs/satmae_teacher_10pct_leakfree/best_model.pth`

Train the final student:

```bash
python src/satmae_pseudolabel_finetune.py
```

Output:

`outputs/satmae_sam_pseudolabels/best_model.pth`

---

## Evaluation

### Full validation set

```bash
python src/evaluate_models.py --model swin --mode dataset
python src/evaluate_models.py --model satmae --mode dataset
python src/evaluate_models.py --model satmae_wavelet_loss --mode dataset
python src/evaluate_models.py --model satmae_wavelet --mode dataset
```

For the leak-free pseudo-label student:

```bash
python src/evaluate_models.py \
    --model satmae_sam \
    --mode dataset \
    --checkpoint outputs/satmae_sam_pseudolabels_leakfree/best_model.pth
```

The evaluator reports **IoU, Dice, Precision, Recall, and Accuracy**, and stores both global and per-image results.

### Single-image qualitative evaluation

```bash
python src/evaluate_models.py \
    --model satmae \
    --mode single \
    --patch-id PATCH_ID
```

This produces:

```text
original.png
ground_truth.png
prediction.png
probability.png
overlay.png
comparison.png
metrics.json
```

For a fair qualitative comparison, use the **same `PATCH_ID` for every model**.

---

## Main Output Structure

```text
outputs/
├── swinunet_baseline/
│   └── best_model.pth
├── satmae_baseline/
│   └── best_model.pth
├── satmae_wavelet/
│   └── best_model.pth
├── satmae_wavelet_decoder/
│   └── best_model.pth
├── satmae_teacher_10pct_leakfree/
│   └── best_model.pth
└── satmae_sam_pseudolabels/
    └── best_model.pth
```

---

## Reproduction Order

```text
1. Create the Python environment
2. Install dependencies
3. Clone SatMAE++, Swin-Unet, and Segment Anything
4. Download the three pretrained checkpoints
5. Download and preprocess INRIA
6. Train Swin-Unet
7. Train the SatMAE++ baseline
8. Train SatMAE++ + Wavelet Loss
9. Train SatMAE++ + Wavelet-Aware Decoder
10. Generate leak-free SAM pseudo-labels
11. Train the pseudo-label student
12. Evaluate all models on the same validation set
```

---

## References

- [SatMAE++](https://github.com/techmn/satmae_pp)
- [Segment Anything](https://github.com/facebookresearch/segment-anything)
- [SAMRS](https://github.com/ViTAE-Transformer/SAMRS)
- [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet)
- [WaveDM](https://github.com/Easquel/WaveDM)
- [WaDiGAN-SR](https://github.com/aloilor/WaDiGAN-SR)
- [INRIA Aerial Image Labeling Dataset](https://project.inria.fr/aerialimagelabeling/)

For the complete methodology, experimental setup, plots, qualitative examples, and discussion, refer to the accompanying project notebook.
