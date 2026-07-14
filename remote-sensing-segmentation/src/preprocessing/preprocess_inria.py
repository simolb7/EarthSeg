#!/usr/bin/env python
"""
Preprocess the INRIA Aerial Image Labeling dataset for building segmentation.

The script:
1. Detects the common INRIA folder layouts.
2. Splits annotated source images into train/validation by city.
3. Tiles 5000x5000 TIFF images into smaller PNG patches.
4. Converts ground-truth masks to binary values {0, 255}.
5. Creates metadata.csv with paths, source image, city, coordinates,
   dimensions and building coverage.

Raw files are never modified.

Example:
    python src/preprocessing/preprocess_inria.py \
        --dataset-root datasets/INRIA-Aerial-Image-Labeling \
        --output-root datasets/inria_processed \
        --tile-size 512 \
        --stride 512 \
        --val-city vienna
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio
from PIL import Image
from tqdm import tqdm


SUPPORTED_EXTENSIONS = {".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tile and split the INRIA building segmentation dataset."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help=(
            "Dataset directory. It may directly contain train/test, or contain "
            "data/train and data/test, or AerialImageDataset/train and test."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("datasets/inria_processed"),
        help="Destination directory for processed PNG patches.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=512,
        help="Square tile size in pixels. Default: 512.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=512,
        help="Tile stride. Use a value below tile-size for overlap.",
    )
    parser.add_argument(
        "--val-city",
        type=str,
        default="vienna",
        help=(
            "Annotated city reserved entirely for validation. "
            "Use 'none' to create no validation split."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite already existing output PNG files.",
    )
    return parser.parse_args()


def detect_data_root(dataset_root: Path) -> Path:
    candidates = [
        dataset_root,
        dataset_root / "data",
        dataset_root / "AerialImageDataset",
    ]

    for candidate in candidates:
        if (
            (candidate / "train" / "images").is_dir()
            and (candidate / "train" / "gt").is_dir()
            and (candidate / "test" / "images").is_dir()
        ):
            return candidate

    expected = "\n".join(
        f"  - {candidate / 'train' / 'images'}" for candidate in candidates
    )
    raise FileNotFoundError(
        "Could not detect the INRIA folder layout.\n"
        "Expected one of these image directories:\n"
        f"{expected}\n"
        "The corresponding train/gt and test/images folders must also exist."
    )


def list_tiffs(directory: Path) -> list[Path]:
    files = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(f"No TIFF images found in: {directory}")
    return files


def city_from_stem(stem: str) -> str:
    """Extract 'austin' from 'austin12' and 'tyrol-w' from 'tyrol-w12'."""
    match = re.match(r"^(.*?)(\d+)$", stem.lower())
    if match is None:
        return stem.lower()
    return match.group(1)


def tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    """
    Cover the complete axis.

    The final tile is aligned to the image border, so a 5000 px image with
    512 px tiles does not need zero padding.
    """
    if length <= tile_size:
        return [0]

    starts = list(range(0, length - tile_size + 1, stride))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def read_rgb(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        if src.count < 3:
            raise ValueError(f"Expected at least 3 image bands in {path}, got {src.count}.")
        array = src.read([1, 2, 3])

    array = np.moveaxis(array, 0, -1)

    if array.dtype != np.uint8:
        # INRIA RGB data is normally uint8. This fallback prevents accidental
        # clipping behaviour for another integer type.
        array = np.clip(array, 0, 255).astype(np.uint8)

    return array


def read_binary_mask(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        mask = src.read(1)

    return (mask > 0).astype(np.uint8) * 255


def crop_or_pad(
    array: np.ndarray,
    top: int,
    left: int,
    tile_size: int,
    is_mask: bool,
) -> np.ndarray:
    crop = array[top : top + tile_size, left : left + tile_size]

    height, width = crop.shape[:2]
    if height == tile_size and width == tile_size:
        return crop

    pad_height = tile_size - height
    pad_width = tile_size - width

    if array.ndim == 3:
        pad_widths = ((0, pad_height), (0, pad_width), (0, 0))
    else:
        pad_widths = ((0, pad_height), (0, pad_width))

    mode = "constant" if is_mask else "reflect"
    return np.pad(crop, pad_widths, mode=mode)


def save_png(array: np.ndarray, path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, compress_level=3)


def validate_pairs(image_paths: Iterable[Path], mask_dir: Path) -> None:
    missing_masks = [
        image_path.name
        for image_path in image_paths
        if not (mask_dir / image_path.name).is_file()
    ]
    if missing_masks:
        preview = ", ".join(missing_masks[:10])
        raise FileNotFoundError(
            f"Missing {len(missing_masks)} training masks. Examples: {preview}"
        )


def process_source_image(
    image_path: Path,
    mask_path: Path | None,
    split: str,
    output_root: Path,
    tile_size: int,
    stride: int,
    overwrite: bool,
) -> list[dict[str, object]]:
    image = read_rgb(image_path)
    mask = read_binary_mask(mask_path) if mask_path is not None else None

    if mask is not None and image.shape[:2] != mask.shape[:2]:
        raise ValueError(
            f"Image/mask size mismatch for {image_path.name}: "
            f"{image.shape[:2]} versus {mask.shape[:2]}"
        )

    height, width = image.shape[:2]
    rows: list[dict[str, object]] = []

    for top in tile_starts(height, tile_size, stride):
        for left in tile_starts(width, tile_size, stride):
            patch_id = f"{image_path.stem}_y{top:04d}_x{left:04d}"
            image_relative = Path(split) / "images" / f"{patch_id}.png"
            image_output = output_root / image_relative

            image_patch = crop_or_pad(
                image, top, left, tile_size, is_mask=False
            )
            save_png(image_patch, image_output, overwrite)

            mask_relative = ""
            building_ratio = ""

            if mask is not None:
                mask_relative_path = Path(split) / "masks" / f"{patch_id}.png"
                mask_output = output_root / mask_relative_path
                mask_patch = crop_or_pad(
                    mask, top, left, tile_size, is_mask=True
                )
                save_png(mask_patch, mask_output, overwrite)

                mask_relative = mask_relative_path.as_posix()
                building_ratio = float(np.mean(mask_patch > 0))

            rows.append(
                {
                    "patch_id": patch_id,
                    "split": split,
                    "city": city_from_stem(image_path.stem),
                    "source_image": image_path.name,
                    "top": top,
                    "left": left,
                    "tile_size": tile_size,
                    "source_height": height,
                    "source_width": width,
                    "building_ratio": building_ratio,
                    "image_path": image_relative.as_posix(),
                    "mask_path": mask_relative,
                }
            )

    return rows


def main() -> None:
    args = parse_args()

    if args.tile_size <= 0:
        raise ValueError("--tile-size must be greater than zero.")
    if args.stride <= 0:
        raise ValueError("--stride must be greater than zero.")
    if args.stride > args.tile_size:
        raise ValueError(
            "--stride cannot be larger than --tile-size, otherwise pixels are skipped."
        )

    data_root = detect_data_root(args.dataset_root.resolve())
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    train_image_dir = data_root / "train" / "images"
    train_mask_dir = data_root / "train" / "gt"
    test_image_dir = data_root / "test" / "images"

    annotated_images = list_tiffs(train_image_dir)
    official_test_images = list_tiffs(test_image_dir)
    validate_pairs(annotated_images, train_mask_dir)

    available_cities = sorted({city_from_stem(path.stem) for path in annotated_images})
    val_city = args.val_city.strip().lower()

    if val_city == "none":
        val_city = ""
    elif val_city not in available_cities:
        raise ValueError(
            f"Validation city '{val_city}' is not available. "
            f"Available annotated cities: {', '.join(available_cities)}"
        )

    split_counts = {"train": 0, "val": 0, "test": 0}
    metadata: list[dict[str, object]] = []

    print(f"Detected data root: {data_root}")
    print(f"Annotated cities: {', '.join(available_cities)}")
    print(f"Validation city: {val_city or 'none'}")
    print(
        f"Tiling configuration: {args.tile_size}x{args.tile_size}, "
        f"stride={args.stride}"
    )

    for image_path in tqdm(annotated_images, desc="Annotated source images"):
        city = city_from_stem(image_path.stem)
        split = "val" if city == val_city else "train"
        rows = process_source_image(
            image_path=image_path,
            mask_path=train_mask_dir / image_path.name,
            split=split,
            output_root=output_root,
            tile_size=args.tile_size,
            stride=args.stride,
            overwrite=args.overwrite,
        )
        metadata.extend(rows)
        split_counts[split] += len(rows)

    for image_path in tqdm(official_test_images, desc="Official test images"):
        rows = process_source_image(
            image_path=image_path,
            mask_path=None,
            split="test",
            output_root=output_root,
            tile_size=args.tile_size,
            stride=args.stride,
            overwrite=args.overwrite,
        )
        metadata.extend(rows)
        split_counts["test"] += len(rows)

    metadata_path = output_root / "metadata.csv"
    fieldnames = [
        "patch_id",
        "split",
        "city",
        "source_image",
        "top",
        "left",
        "tile_size",
        "source_height",
        "source_width",
        "building_ratio",
        "image_path",
        "mask_path",
    ]
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata)

    print("\nPreprocessing completed.")
    print(f"Output directory: {output_root}")
    print(f"Metadata: {metadata_path}")
    print(f"Train patches: {split_counts['train']}")
    print(f"Validation patches: {split_counts['val']}")
    print(f"Official test patches: {split_counts['test']}")


if __name__ == "__main__":
    main()
