"""
UCF Crime Dataset -> train / val / test  split  (image frames).

Kaggle link : https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset

Usage
-----
1) Download & extract the dataset under  data/ucf-crime-raw/
   Expected layout after extraction (may vary – the script searches
   for class-named sub-folders automatically):

       data/ucf-crime-raw/
           Abuse/
           Arrest/
           Arson/
           ...
           Vandalism/

   If there is an extra wrapper folder (e.g. data/ucf-crime-raw/UCF-Crime/...),
   just point RAW_DIR to the folder that directly contains class folders.

2) Run:
       python scripts/prepare_image_dataset.py

   Env-vars (optional):
       IMAGE_RAW_DIR   – path to extracted dataset  (default: data/ucf-crime-raw)
       IMAGE_DATA_DIR  – output path                (default: data/images)
       VAL_RATIO       – validation split ratio     (default: 0.15)
       TEST_RATIO      – test split ratio           (default: 0.10)
       MAX_PER_CLASS   – cap per class (0 = no cap) (default: 0)

The script copies (not moves) images so the original download stays intact.
"""

from __future__ import annotations

import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

_BASE = Path(__file__).resolve().parent.parent

EXPECTED_CLASSES = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary",
    "Explosion", "Fighting", "NormalVideos", "RoadAccidents",
    "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

RAW_DIR   = Path(os.environ.get("IMAGE_RAW_DIR",  str(_BASE / "data" / "ucf-crime-raw")))
OUT_DIR   = Path(os.environ.get("IMAGE_DATA_DIR",  str(_BASE / "data" / "images")))
VAL_RATIO  = float(os.environ.get("VAL_RATIO",  "0.15"))
TEST_RATIO = float(os.environ.get("TEST_RATIO", "0.10"))
MAX_PER_CLASS = int(os.environ.get("MAX_PER_CLASS", "0"))
SEED = 42


def _find_class_root(base: Path) -> Path:
    """Walk down at most 3 levels to find the folder that contains class dirs."""
    for depth in range(4):
        candidates = [base]
        for _ in range(depth):
            next_candidates = []
            for c in candidates:
                if c.is_dir():
                    next_candidates.extend(sorted(c.iterdir()))
            candidates = next_candidates

        for c in candidates:
            if c.is_dir() and c.name in EXPECTED_CLASSES:
                return c.parent
    return base


def _collect_images(class_dir: Path) -> list[Path]:
    """Recursively collect image files under *class_dir*."""
    images = []
    for root, _dirs, files in os.walk(class_dir):
        for f in files:
            if Path(f).suffix.lower() in IMAGE_EXTENSIONS:
                images.append(Path(root) / f)
    return sorted(images)


def main() -> None:
    random.seed(SEED)

    print(f"[INFO] Raw dataset dir : {RAW_DIR}")
    print(f"[INFO] Output dir      : {OUT_DIR}")
    print(f"[INFO] Val ratio       : {VAL_RATIO}")
    print(f"[INFO] Test ratio      : {TEST_RATIO}")

    if not RAW_DIR.exists():
        print(
            f"\n[ERROR] Raw dataset directory not found: {RAW_DIR}\n"
            f"  1. Download the UCF-Crime dataset from Kaggle.\n"
            f"  2. Extract it into {RAW_DIR}\n"
            f"  3. Re-run this script.\n"
        )
        return

    class_root = _find_class_root(RAW_DIR)
    print(f"[INFO] Class root      : {class_root}")

    found_classes = sorted(
        d.name for d in class_root.iterdir()
        if d.is_dir() and d.name in EXPECTED_CLASSES
    )

    if not found_classes:
        print(
            f"\n[ERROR] No expected class folders found under {class_root}.\n"
            f"  Expected folders: {EXPECTED_CLASSES}\n"
            f"  Found: {[d.name for d in class_root.iterdir() if d.is_dir()]}\n"
        )
        return

    print(f"[INFO] Found classes   : {found_classes}")

    stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for cls_name in found_classes:
        cls_dir = class_root / cls_name
        images = _collect_images(cls_dir)

        if not images:
            print(f"  [WARN] No images found for class '{cls_name}', skipping.")
            continue

        random.shuffle(images)

        if MAX_PER_CLASS > 0:
            images = images[:MAX_PER_CLASS]

        n = len(images)
        n_test = int(n * TEST_RATIO)
        n_val  = int(n * VAL_RATIO)
        n_train = n - n_val - n_test

        splits = {
            "train": images[:n_train],
            "val":   images[n_train : n_train + n_val],
            "test":  images[n_train + n_val :],
        }

        for split_name, split_images in splits.items():
            dest_dir = OUT_DIR / split_name / cls_name
            dest_dir.mkdir(parents=True, exist_ok=True)

            for img_path in split_images:
                dest_path = dest_dir / img_path.name
                # avoid overwriting if filenames collide across sub-dirs
                if dest_path.exists():
                    stem = img_path.stem
                    suffix = img_path.suffix
                    counter = 1
                    while dest_path.exists():
                        dest_path = dest_dir / f"{stem}_{counter}{suffix}"
                        counter += 1

                shutil.copy2(img_path, dest_path)

            stats[cls_name][split_name] = len(split_images)

        print(
            f"  {cls_name:16s} -> "
            f"train: {stats[cls_name]['train']:5d}  "
            f"val: {stats[cls_name]['val']:5d}  "
            f"test: {stats[cls_name]['test']:5d}  "
            f"(total: {n})"
        )

    total_train = sum(s["train"] for s in stats.values())
    total_val   = sum(s["val"]   for s in stats.values())
    total_test  = sum(s["test"]  for s in stats.values())
    print(f"\n[INFO] DONE  –  train: {total_train}  val: {total_val}  test: {total_test}")
    print(f"[INFO] Output directory: {OUT_DIR}")
    print(f"\n[INFO] Next step: run the training script:")
    print(f"       python scripts/train_image_model.py")


if __name__ == "__main__":
    main()
