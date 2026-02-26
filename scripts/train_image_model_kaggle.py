"""
Kaggle Notebook'ta UCF Crime Dataset ile görüntü modeli eğitimi.

- Veriyi KOPYALAMAZ: /kaggle/input read-only'dan path listesiyle okur (disk tasarrufu).
- Dataset: "UCF Crime Dataset" (odins0n/ucf-crime-dataset) notebook'a eklenmiş olmalı.
- GPU: Notebook Settings -> Accelerator -> GPU (P100) seç.

Kullanım (Kaggle'da tek hücrede):
  1) Sağ üstten Settings -> GPU aç.
  2) Bu dosyanın TÜM içeriğini bir Code hücresine yapıştır, Run.
  3) Bittikten sonra Output'tan best_emergency_model.pth ve image_class_names.json indir.
  4) Bu iki dosyayı kendi projende out_models/ içine koy.

Ortam değişkenleri (isteğe bağlı):
  KAGGLE_INPUT_PATH  - dataset kökü (varsayılan: /kaggle/input/ucf-crime-dataset)
  VAL_RATIO, TEST_RATIO, MAX_PER_CLASS - prepare_image_dataset ile aynı
  IMAGE_EPOCHS, IMAGE_BATCH_SIZE, IMAGE_LR - train_image_model ile aynı

Resume (kaldığın yerden devam):
  Kaggle'da "Pause" yok; run durunca oturum biter. Devam için checkpoint kullanılır.
  - Her epoch sonunda out_models/checkpoint_latest.pth kaydedilir.
  - Kesinti sonrası: Output'u "Save Version" ile kaydet, bu çıktıyı yeni bir Dataset yap.
  - Yeni notebook'da: UCF Crime + bu checkpoint dataset'ini input ekle.
  - RESUME_CHECKPOINT_DIR ile checkpoint klasörünü ver (örn. /kaggle/input/my-checkpoint/out_models).
  - Aynı scripti çalıştır; eğitim kaldığı epoch'tan devam eder.
"""

from __future__ import annotations

import copy
import json
import os
import random
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import models, transforms
from PIL import Image

# -------- Kaggle path'leri --------
_KAGGLE_INPUT = Path(os.environ.get("KAGGLE_INPUT_PATH", "/kaggle/input/ucf-crime-dataset"))
_KAGGLE_WORKING = Path(os.environ.get("KAGGLE_WORKING_PATH", "/kaggle/working"))

# Eğer Kaggle'da değilsek (lokal test) proje kökünü kullan
if not _KAGGLE_INPUT.exists():
    _BASE = Path(__file__).resolve().parent.parent
    _KAGGLE_INPUT = _BASE / "data" / "ucf-crime-raw"
    _KAGGLE_WORKING = _BASE

EXPECTED_CLASSES = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary",
    "Explosion", "Fighting", "NormalVideos", "RoadAccidents",
    "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism",
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
SEED = 42

VAL_RATIO = float(os.environ.get("VAL_RATIO", "0.15"))
TEST_RATIO = float(os.environ.get("TEST_RATIO", "0.10"))
MAX_PER_CLASS = int(os.environ.get("MAX_PER_CLASS", "0"))

DATA_DIR = os.environ.get("IMAGE_DATA_DIR", str(_KAGGLE_WORKING / "data" / "images"))
BATCH_SIZE = int(os.environ.get("IMAGE_BATCH_SIZE", "32"))
NUM_EPOCHS = int(os.environ.get("IMAGE_EPOCHS", "10"))
LEARNING_RATE = float(os.environ.get("IMAGE_LR", "0.001"))
WEIGHT_DECAY = float(os.environ.get("IMAGE_WEIGHT_DECAY", "1e-4"))
FROZEN_EPOCHS = int(os.environ.get("IMAGE_FROZEN_EPOCHS", "3"))
OUTPUT_DIR = _KAGGLE_WORKING / "out_models"
# Kaldığın yerden devam: örn. /kaggle/input/my-checkpoint/out_models
RESUME_CHECKPOINT_DIR = os.environ.get("RESUME_CHECKPOINT_DIR", "")

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# -------- Transforms (train_image_model ile aynı) --------
data_transforms = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    "val": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}


def _find_class_root(base: Path) -> Path:
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
    images = []
    for root, _dirs, files in os.walk(class_dir):
        for f in files:
            if Path(f).suffix.lower() in IMAGE_EXTENSIONS:
                images.append(Path(root) / f)
    return sorted(images)


def build_splits(raw_root: Path) -> tuple[list[tuple[str, int]], list[tuple[str, int]], list[tuple[str, int]], list[str]]:
    """Train/val/test için (path, label_idx) listeleri ve class_names döner. Dosya kopyalamaz."""
    random.seed(SEED)
    class_root = _find_class_root(raw_root)
    found_classes = sorted(
        d.name for d in class_root.iterdir()
        if d.is_dir() and d.name in EXPECTED_CLASSES
    )
    if not found_classes:
        raise FileNotFoundError(f"Hiç beklenen sınıf klasörü bulunamadı: {class_root}")

    class_to_idx = {c: i for i, c in enumerate(found_classes)}
    train_list: list[tuple[str, int]] = []
    val_list: list[tuple[str, int]] = []
    test_list: list[tuple[str, int]] = []

    for cls_name in found_classes:
        cls_dir = class_root / cls_name
        images = _collect_images(cls_dir)
        if not images:
            continue
        random.shuffle(images)
        if MAX_PER_CLASS > 0:
            images = images[:MAX_PER_CLASS]
        n = len(images)
        n_test = int(n * TEST_RATIO)
        n_val = int(n * VAL_RATIO)
        n_train = n - n_val - n_test
        idx = class_to_idx[cls_name]
        for i, p in enumerate(images):
            path_str = str(p.resolve())
            if i < n_train:
                train_list.append((path_str, idx))
            elif i < n_train + n_val:
                val_list.append((path_str, idx))
            else:
                test_list.append((path_str, idx))

    return train_list, val_list, test_list, found_classes


class PathListDataset(Dataset):
    """(path, label) listesinden resim yükleyen Dataset. Kopya yok."""
    def __init__(self, pairs: list[tuple[str, int]], transform=None):
        self.pairs = pairs
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int):
        path, label = self.pairs[i]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model.to(device)


def _set_requires_grad(model: nn.Module, requires_grad: bool, except_fc: bool = False) -> None:
    for name, param in model.named_parameters():
        if except_fc and (name == "fc.weight" or name == "fc.bias"):
            param.requires_grad = True
        else:
            param.requires_grad = requires_grad


def train_model(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: object,
    scheduler: Optional[object],
    dataloaders: dict,
    dataset_sizes: dict,
    num_epochs: int = 10,
    frozen_epochs: int = 3,
    start_epoch: int = 0,
    best_model_wts: Optional[dict] = None,
    best_acc: float = 0.0,
    class_names: Optional[list[str]] = None,
) -> nn.Module:
    if best_model_wts is None:
        best_model_wts = copy.deepcopy(model.state_dict())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = OUTPUT_DIR / "best_emergency_model.pth"
    checkpoint_path = OUTPUT_DIR / "checkpoint_latest.pth"

    for epoch in range(start_epoch, num_epochs):
        if epoch < frozen_epochs:
            _set_requires_grad(model, False, except_fc=True)
        else:
            _set_requires_grad(model, True)

        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = (running_corrects.double() / dataset_sizes[phase]).item()
            print(f"  {phase.upper():5s}  Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), str(save_path))
                print(f"         -> Best model saved (acc={best_acc:.4f})")

        if scheduler is not None:
            scheduler.step()

        # Her epoch sonunda checkpoint (resume için)
        try:
            ckpt = {
                "epoch": epoch,
                "best_acc": best_acc,
                "best_model_wts": best_model_wts,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler else None,
                "class_names": class_names or [],
            }
            torch.save(ckpt, str(checkpoint_path))
        except Exception:
            pass

    model.load_state_dict(best_model_wts)
    return model


def main() -> None:
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Raw data (Kaggle input): {_KAGGLE_INPUT}")
    print(f"[INFO] Output dir: {OUTPUT_DIR}")

    if not _KAGGLE_INPUT.exists():
        print(f"[ERROR] Dataset bulunamadı: {_KAGGLE_INPUT}")
        print("  Kaggle'da bu dataset'i notebook'a ekleyin: UCF Crime Dataset (odins0n/ucf-crime-dataset)")
        return

    resume_dir = Path(RESUME_CHECKPOINT_DIR) if RESUME_CHECKPOINT_DIR else None
    checkpoint_file = resume_dir / "checkpoint_latest.pth" if resume_dir else None
    do_resume = bool(resume_dir and checkpoint_file and checkpoint_file.exists())
    start_epoch = 0
    best_acc = 0.0
    best_model_wts = None

    print("\n[1/4] Veri taranıyor (dosya kopyalanmıyor)...")
    train_list, val_list, test_list, class_names = build_splits(_KAGGLE_INPUT)
    print(f"      Sınıflar: {class_names}")
    print(f"      Train: {len(train_list)}  Val: {len(val_list)}  Test: {len(test_list)}")

    train_ds = PathListDataset(train_list, transform=data_transforms["train"])
    val_ds = PathListDataset(val_list, transform=data_transforms["val"])
    dataloaders = {
        "train": torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2),
        "val": torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2),
    }
    dataset_sizes = {"train": len(train_ds), "val": len(val_ds)}

    from collections import Counter
    train_labels = [p[1] for p in train_list]
    class_counts = Counter(train_labels)
    total = sum(class_counts.values())
    num_classes = len(class_names)
    class_weights = torch.tensor(
        [total / (num_classes * class_counts.get(i, 1)) for i in range(num_classes)],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "image_class_names.json", "w", encoding="utf-8") as f:
        json.dump({"class_names": class_names}, f, indent=2)

    if do_resume:
        print("\n[2/4] Checkpoint'tan yükleniyor (kaldığın yerden devam)...")
        ckpt = torch.load(str(checkpoint_file), map_location=device, weights_only=False)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_acc = ckpt.get("best_acc", 0.0)
        best_model_wts = ckpt.get("best_model_wts") or ckpt.get("model")
        model = build_model(num_classes=num_classes)
        model.load_state_dict(ckpt["model"])
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
        if ckpt.get("optimizer"):
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except Exception:
                pass
        if ckpt.get("scheduler"):
            try:
                scheduler.load_state_dict(ckpt["scheduler"])
            except Exception:
                pass
        print(f"      Epoch {start_epoch} ile devam (best_acc={best_acc:.4f}).")
    else:
        print("\n[2/4] Model yükleniyor (ResNet50 pretrained)...")
        model = build_model(num_classes=num_classes)
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
        best_model_wts = None

    print("\n[3/4] Eğitim başlıyor...")
    train_model(
        model, criterion, optimizer, scheduler, dataloaders, dataset_sizes,
        num_epochs=NUM_EPOCHS,
        frozen_epochs=FROZEN_EPOCHS,
        start_epoch=start_epoch,
        best_model_wts=best_model_wts,
        best_acc=best_acc,
        class_names=class_names,
    )

    if test_list:
        print("\n[4/4] Test seti değerlendiriliyor...")
        test_ds = PathListDataset(test_list, transform=data_transforms["val"])
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        test_acc = correct / total if total else 0.0
        print(f"      Test accuracy: {test_acc:.4f} ({correct}/{total})")

    print(f"\n[INFO] Bitti. Model ve sınıf isimleri: {OUTPUT_DIR}")
    print("       Kaggle'da Output sekmesinden best_emergency_model.pth ve image_class_names.json indirip")
    print("       projende out_models/ klasörüne koyun.")


if __name__ == "__main__":
    main()
