from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms

_BASE = Path(__file__).resolve().parent.parent

DATA_DIR = os.environ.get("IMAGE_DATA_DIR", str(_BASE / "data" / "images"))
BATCH_SIZE = int(os.environ.get("IMAGE_BATCH_SIZE", "32"))
NUM_EPOCHS = int(os.environ.get("IMAGE_EPOCHS", "10"))
LEARNING_RATE = float(os.environ.get("IMAGE_LR", "0.001"))
WEIGHT_DECAY = float(os.environ.get("IMAGE_WEIGHT_DECAY", "1e-4"))
FROZEN_EPOCHS = int(os.environ.get("IMAGE_FROZEN_EPOCHS", "3"))
OUTPUT_DIR = _BASE / "out_models"

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"[INFO] Device selected: {device}")


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


DISPATCH_LOGIC = {
    "Abuse":          ["Police"],
    "Arrest":         ["Police", "Support Team"],
    "Arson":          ["Fire Department", "Police", "Ambulance"],
    "Assault":        ["Police", "Ambulance"],
    "Burglary":       ["Police"],
    "Explosion":      ["Disaster Response (AFAD)", "Fire Department", "Police", "Ambulance"],
    "Fighting":       ["Police"],
    "NormalVideos":   [],
    "RoadAccidents":  ["Police", "Ambulance", "Fire Department"],
    "Robbery":        ["Police"],
    "Shooting":       ["Police", "Special Forces", "Ambulance"],
    "Shoplifting":    ["Police"],
    "Stealing":       ["Police"],
    "Vandalism":      ["Police"],
}


def build_model(num_classes: int) -> nn.Module:
    print("[INFO] Loading pretrained ResNet50 …")
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
    optimizer: optim.Optimizer,
    scheduler: Optional[object],
    dataloaders: dict,
    dataset_sizes: dict,
    num_epochs: int = 10,
    frozen_epochs: int = 3,
) -> nn.Module:
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = OUTPUT_DIR / "best_emergency_model.pth"

    print(f"\n{'='*60}")
    print(f"  TRAINING STARTED  –  {num_epochs} epochs (backbone frozen for first {frozen_epochs})")
    print(f"{'='*60}")

    for epoch in range(num_epochs):
        if epoch < frozen_epochs:
            _set_requires_grad(model, False, except_fc=True)
        else:
            _set_requires_grad(model, True)

        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)

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
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"  {phase.upper():5s} ->  Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), str(save_path))
                print(f"         --> Best model saved! (acc={best_acc:.4f})")

        if scheduler is not None:
            scheduler.step()

    elapsed = time.time() - since
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE  –  {elapsed // 60:.0f}m {elapsed % 60:.0f}s")
    print(f"  Best validation accuracy: {best_acc:.4f}")
    print(f"  Checkpoint: {save_path}")
    print(f"{'='*60}")

    model.load_state_dict(best_model_wts)
    return model


def main() -> None:
    print(f"[INFO] Data directory : {DATA_DIR}")
    print(f"[INFO] Batch size     : {BATCH_SIZE}")
    print(f"[INFO] Epochs         : {NUM_EPOCHS}")
    print(f"[INFO] Learning rate  : {LEARNING_RATE}")

    print("\n[INFO] Scanning dataset …")
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
        for x in ["train", "val"]
    }
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x],
            batch_size=BATCH_SIZE,
            shuffle=(x == "train"),
            num_workers=2,
        )
        for x in ["train", "val"]
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
    class_names = image_datasets["train"].classes

    print(f"[INFO] Classes ({len(class_names)}): {class_names}")
    print(f"[INFO] Train: {dataset_sizes['train']}  |  Val: {dataset_sizes['val']}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    class_map_path = OUTPUT_DIR / "image_class_names.json"
    with open(class_map_path, "w", encoding="utf-8") as f:
        json.dump({"class_names": class_names}, f, indent=2)
    print(f"[INFO] Class names saved to {class_map_path}")

    from collections import Counter
    train_targets = [image_datasets["train"][i][1] for i in range(len(image_datasets["train"]))]
    class_counts = Counter(train_targets)
    total = sum(class_counts.values())
    num_classes = len(class_names)
    class_weights = torch.tensor(
        [total / (num_classes * class_counts.get(i, 1)) for i in range(num_classes)],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = build_model(num_classes=num_classes)

    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=0.9,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

    trained_model = train_model(
        model, criterion, optimizer, scheduler, dataloaders, dataset_sizes,
        num_epochs=NUM_EPOCHS,
        frozen_epochs=FROZEN_EPOCHS,
    )

    test_dir = os.path.join(DATA_DIR, "test")
    if os.path.isdir(test_dir):
        test_dataset = datasets.ImageFolder(test_dir, data_transforms["val"])
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
        )
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        test_acc = correct / total if total else 0.0
        print(f"\n[INFO] Test set accuracy: {test_acc:.4f} ({correct}/{total})")
    else:
        print("\n[INFO] No data/images/test folder – skipping test set evaluation.")

    print("\n--- DEMO: Sample Dispatch Scenario ---")
    sample_prediction = "Arson"
    required_units = DISPATCH_LOGIC.get(sample_prediction, [])
    print(f"  Detected event  : {sample_prediction}")
    print(f"  Dispatch units  : {', '.join(required_units)}")


if __name__ == "__main__":
    main()
