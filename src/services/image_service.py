from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_BASE = Path(__file__).resolve().parent.parent.parent
_MODEL_PATH = _BASE / "out_models" / "best_emergency_model.pth"
_CLASS_NAMES_PATH = _BASE / "out_models" / "image_class_names.json"

_CLASS_NAMES_FALLBACK: List[str] = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting",
    "NormalVideos", "RoadAccidents", "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism",
]

IMAGE_CLASS_TO_CATEGORY: Dict[str, str] = {
    "Abuse":          "crime",
    "Arrest":         "crime",
    "Arson":          "fire",
    "Assault":        "crime",
    "Burglary":       "crime",
    "Explosion":      "fire",
    "Fighting":       "crime",
    "NormalVideos":   "other",
    "RoadAccidents":  "medical",
    "Robbery":        "crime",
    "Shooting":       "crime",
    "Shoplifting":    "crime",
    "Stealing":       "crime",
    "Vandalism":      "crime",
}

DISPATCH_LOGIC: Dict[str, List[str]] = {
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

_transforms = None


def _get_transforms():
    global _transforms
    if _transforms is not None:
        return _transforms

    from torchvision import transforms

    _transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return _transforms


class ImageModelService:

    def __init__(self) -> None:
        self._model = None
        self._loaded = False
        self._device = None
        self._class_names: List[str] = _CLASS_NAMES_FALLBACK.copy()

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self, model_path: Optional[str] = None) -> bool:
        model_path = Path(model_path) if model_path else _MODEL_PATH
        if not model_path.exists():
            logger.warning(
                "Image model not found at %s – image analysis will be unavailable.",
                model_path,
            )
            return False

        try:
            import torch
            import torch.nn as nn
            from torchvision import models

            if _CLASS_NAMES_PATH.exists():
                with open(_CLASS_NAMES_PATH, encoding="utf-8") as f:
                    data = json.load(f)
                self._class_names = data.get("class_names", _CLASS_NAMES_FALLBACK)
            else:
                self._class_names = _CLASS_NAMES_FALLBACK.copy()
                logger.warning(
                    "image_class_names.json not found, using fallback class list."
                )

            if torch.backends.mps.is_available():
                self._device = torch.device("mps")
            elif torch.cuda.is_available():
                self._device = torch.device("cuda")
            else:
                self._device = torch.device("cpu")

            model = models.resnet50(weights=None)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, len(self._class_names))

            state_dict = torch.load(
                str(model_path),
                map_location=self._device,
                weights_only=True,
            )
            model.load_state_dict(state_dict)
            model.to(self._device)
            model.eval()

            self._model = model
            self._loaded = True
            logger.info("Image model loaded from %s on %s", model_path, self._device)
            return True

        except Exception as exc:
            logger.error("Failed to load image model: %s", exc)
            return False

    def predict(self, image_bytes: bytes) -> Dict[str, Any]:
        if not self._loaded:
            raise RuntimeError("Image model not loaded – call .load() first.")

        import torch
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        transform = _get_transforms()
        tensor = transform(img).unsqueeze(0).to(self._device)

        class_names = self._class_names
        with torch.no_grad():
            outputs = self._model(tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            top3_probs, top3_indices = torch.topk(probs, k=min(3, len(class_names)))

        top3 = [
            {"class": class_names[idx.item()], "confidence": round(prob.item(), 4)}
            for prob, idx in zip(top3_probs, top3_indices)
        ]

        best_class = top3[0]["class"]
        best_conf = top3[0]["confidence"]

        return {
            "detected_class": best_class,
            "confidence": best_conf,
            "top3": top3,
            "dispatch_units": DISPATCH_LOGIC.get(best_class, []),
            "mapped_category": IMAGE_CLASS_TO_CATEGORY.get(best_class, "other"),
        }


def analyze_consistency(
    image_result: Dict[str, Any],
    text_category: Optional[str] = None,
    text_triage_level: Optional[str] = None,
) -> Dict[str, Any]:
    img_class = image_result.get("detected_class", "")
    img_category = image_result.get("mapped_category", "other")
    img_confidence = image_result.get("confidence", 0.0)

    risk_notes: List[str] = []
    consistency_score = 0.5  # default: uncertain
    is_consistent = True
    possible_fake = False

    if img_class == "NormalVideos":
        if text_triage_level in ("CRITICAL", "URGENT"):
            possible_fake = True
            is_consistent = False
            consistency_score = 0.1
            risk_notes.append(
                "Image appears normal/non-emergency, but the verbal report "
                "describes a serious emergency. Possible fabrication or the "
                "image does not reflect the actual scene."
            )
        else:
            is_consistent = True
            consistency_score = 0.9
            risk_notes.append("Image and text both suggest a non-emergency situation.")

    elif text_category and img_category == text_category:
        is_consistent = True
        consistency_score = 0.8 + (0.2 * img_confidence)
        risk_notes.append(
            f"Image classification ({img_class}) aligns with text category "
            f"({text_category}). Consistent report."
        )

    elif text_category and img_category != text_category:
        is_consistent = False
        consistency_score = max(0.1, 0.4 - (0.3 * img_confidence))
        risk_notes.append(
            f"Image suggests '{img_category}' ({img_class}), but text analysis "
            f"points to '{text_category}'. Possible discrepancy – may need "
            f"operator review."
        )
        if img_category == "crime" and text_category == "medical":
            risk_notes.append(
                "Image shows a crime scene while caller describes a medical issue. "
                "The scene may involve both."
            )
        elif img_category == "fire" and text_category == "crime":
            risk_notes.append(
                "Image shows fire/explosion but text describes a crime. "
                "Possible arson or combined incident."
            )

    else:
        consistency_score = 0.5
        risk_notes.append(
            "No text-based triage available for cross-reference. "
            "Image analysis is standalone."
        )

    if img_confidence < 0.4:
        risk_notes.append(
            f"Image classification confidence is low ({img_confidence:.0%}). "
            f"Result should be treated with caution."
        )

    detail = "CONSISTENT" if is_consistent else "INCONSISTENT"
    if possible_fake:
        detail = "POSSIBLY FAKE / UNRELATED IMAGE"

    return {
        "is_consistent": is_consistent,
        "consistency_score": round(consistency_score, 3),
        "consistency_detail": detail,
        "possible_fake": possible_fake,
        "risk_notes": risk_notes,
    }


def analyze_image(
    image_bytes: bytes,
    text_category: Optional[str] = None,
    text_triage_level: Optional[str] = None,
) -> Dict[str, Any]:
    svc = get_image_model_service()

    if not svc.is_loaded:
        return {
            "classification": None,
            "consistency": None,
            "summary": "Image analysis unavailable – model not loaded.",
            "available": False,
        }

    classification = svc.predict(image_bytes)

    consistency = analyze_consistency(
        image_result=classification,
        text_category=text_category,
        text_triage_level=text_triage_level,
    )

    cls_name = classification["detected_class"]
    cls_conf = classification["confidence"]
    cons_detail = consistency["consistency_detail"]

    if classification["detected_class"] == "NormalVideos":
        summary = (
            f"Image analysis: No emergency detected in image "
            f"(confidence: {cls_conf:.0%}). "
            f"Cross-reference: {cons_detail}."
        )
    else:
        units = ", ".join(classification["dispatch_units"]) or "None"
        summary = (
            f"Image analysis: {cls_name} detected (confidence: {cls_conf:.0%}). "
            f"Recommended units: {units}. "
            f"Cross-reference with text: {cons_detail}."
        )

    return {
        "classification": classification,
        "consistency": consistency,
        "summary": summary,
        "available": True,
    }


_image_model_service = ImageModelService()


def get_image_model_service() -> ImageModelService:
    return _image_model_service
