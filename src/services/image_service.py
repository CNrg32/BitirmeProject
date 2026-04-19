from __future__ import annotations

import io
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_USE_HALF = os.environ.get("IMAGE_USE_HALF", "").strip().lower() in ("1", "true", "yes")

_BASE = Path(__file__).resolve().parent.parent.parent

# Optional overrides (absolute path, or relative to project root):
#   IMAGE_MODEL_PATH, IMAGE_CLASS_NAMES_PATH
# Defaults: out_models/best_emergency_model.pth and out_models/image_class_names.json,
# with fallback to out_models/image_outputs/ when the root files are absent.


def _env_path(name: str) -> Optional[Path]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    p = Path(raw)
    if not p.is_absolute():
        p = _BASE / p
    return p


def _resolve_model_path() -> Path:
    env = _env_path("IMAGE_MODEL_PATH")
    if env is not None:
        return env
    root = _BASE / "out_models" / "best_emergency_model.pth"
    nested = _BASE / "out_models" / "image_outputs" / "best_emergency_model.pth"
    if nested.exists() and not root.exists():
        return nested
    return root


def _resolve_class_names_path() -> Path:
    env = _env_path("IMAGE_CLASS_NAMES_PATH")
    if env is not None:
        return env
    root = _BASE / "out_models" / "image_class_names.json"
    nested = _BASE / "out_models" / "image_outputs" / "image_class_names.json"
    if nested.exists() and not root.exists():
        return nested
    return root


def _parse_s3_uri(uri: str) -> Tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"Invalid S3 URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def _download_s3_file(uri: str, target_path: Path) -> bool:
    try:
        import boto3
    except ImportError:
        logger.warning(
            "boto3 is not installed; cannot download image model artifact from %s",
            uri,
        )
        return False

    bucket, key = _parse_s3_uri(uri)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading image model artifact from %s to %s", uri, target_path)
    boto3.client("s3").download_file(bucket, key, str(target_path))
    return True


def _derive_class_names_s3_uri(model_s3_uri: str) -> Optional[str]:
    try:
        bucket, key = _parse_s3_uri(model_s3_uri)
    except ValueError:
        return None
    prefix = key.rsplit("/", 1)[0] if "/" in key else ""
    class_key = f"{prefix}/image_class_names.json" if prefix else "image_class_names.json"
    return f"s3://{bucket}/{class_key}"


def _ensure_cloud_artifacts(model_path: Path, class_names_path: Path) -> None:
    model_s3_uri = os.environ.get("IMAGE_MODEL_S3_URI", "").strip()
    class_names_s3_uri = os.environ.get("IMAGE_CLASS_NAMES_S3_URI", "").strip()

    if model_s3_uri and not model_path.exists():
        try:
            _download_s3_file(model_s3_uri, model_path)
        except Exception as exc:
            logger.error("Image model download failed from %s: %s", model_s3_uri, exc)

    if not class_names_s3_uri and model_s3_uri:
        class_names_s3_uri = _derive_class_names_s3_uri(model_s3_uri) or ""

    if class_names_s3_uri and not class_names_path.exists():
        try:
            _download_s3_file(class_names_s3_uri, class_names_path)
        except Exception as exc:
            logger.warning(
                "Image class names download failed from %s: %s",
                class_names_s3_uri,
                exc,
            )

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

CRITICAL_IMAGE_CLASSES = {"Arson", "Explosion", "RoadAccidents", "Shooting"}
URGENT_IMAGE_CLASSES = {"Abuse", "Arrest", "Assault", "Fighting", "Robbery"}
NON_CRITICAL_IMAGE_CLASSES = {"NormalVideos", "Shoplifting", "Stealing", "Vandalism", "Burglary"}

CLASS_VISUAL_FLAGS: Dict[str, List[str]] = {
    "Arson": ["fire_visible", "possible_smoke"],
    "Explosion": ["explosion_or_collapse", "mass_casualty_risk"],
    "RoadAccidents": ["vehicle_crash", "possible_injury"],
    "Shooting": ["weapon_threat", "possible_casualty"],
    "Assault": ["violence_risk", "possible_injury"],
    "Fighting": ["violence_risk"],
    "Robbery": ["crime_in_progress"],
    "Abuse": ["violence_risk", "vulnerable_person_risk"],
    "Arrest": ["police_event"],
    "NormalVideos": ["no_visible_emergency"],
}

FALLBACK_OPTIONS = [
    {"label": "Tıbbi", "category": "medical"},
    {"label": "Polis", "category": "crime"},
    {"label": "İtfaiye", "category": "fire"},
    {"label": "Trafik/Kaza", "category": "medical"},
    {"label": "Diğer", "category": "other"},
]

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
        model_path = Path(model_path) if model_path else _resolve_model_path()
        class_names_path = _resolve_class_names_path()
        _ensure_cloud_artifacts(model_path, class_names_path)
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

            if class_names_path.exists():
                with open(class_names_path, encoding="utf-8") as f:
                    data = json.load(f)
                self._class_names = data.get("class_names", _CLASS_NAMES_FALLBACK)
            else:
                self._class_names = _CLASS_NAMES_FALLBACK.copy()
                logger.warning(
                    "image_class_names.json not found at %s, using fallback class list.",
                    class_names_path,
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
            if _USE_HALF and self._device.type in ("cuda", "mps"):
                model = model.half()
                logger.info("Image model using FP16 (half precision) on %s", self._device)
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
        if _USE_HALF and self._device.type in ("cuda", "mps"):
            tensor = tensor.half()

        class_names = self._class_names
        with torch.inference_mode():
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


def analyze_image_quality(image_bytes: bytes) -> Dict[str, Any]:
    try:
        from PIL import Image, ImageStat

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        width, height = img.size
        gray = img.convert("L")
        stat = ImageStat.Stat(gray)
        brightness = float(stat.mean[0])
        contrast = float(stat.stddev[0])

        reasons: List[str] = []
        if width < 160 or height < 160:
            reasons.append("too_small")
        if brightness < 25:
            reasons.append("too_dark")
        elif brightness > 235:
            reasons.append("too_bright")
        if contrast < 8:
            reasons.append("low_contrast_or_blurry")

        return {
            "usable": not reasons,
            "width": width,
            "height": height,
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
            "reasons": reasons,
        }
    except Exception as exc:
        return {"usable": False, "reasons": ["invalid_image"], "error": str(exc)}


def derive_visual_triage(classification: Dict[str, Any]) -> Dict[str, Any]:
    cls_name = str(classification.get("detected_class") or "")
    confidence = float(classification.get("confidence") or 0.0)
    category = classification.get("mapped_category") or IMAGE_CLASS_TO_CATEGORY.get(cls_name, "other")
    visual_flags = CLASS_VISUAL_FLAGS.get(cls_name, [])

    if confidence < 0.5:
        triage_level = "URGENT"
        action = "MANUAL_FALLBACK"
        reason = "low_confidence"
    elif cls_name in CRITICAL_IMAGE_CLASSES and confidence >= 0.7:
        triage_level = "CRITICAL"
        action = "EARLY_DISPATCH"
        reason = "critical_visual_signal"
    elif cls_name in URGENT_IMAGE_CLASSES or (cls_name in CRITICAL_IMAGE_CLASSES and confidence < 0.7):
        triage_level = "URGENT"
        action = "VERIFY_THEN_DISPATCH"
        reason = "urgent_visual_signal"
    elif cls_name in NON_CRITICAL_IMAGE_CLASSES:
        triage_level = "NON_URGENT"
        action = "TEXT_REQUIRED"
        reason = "no_or_low_visible_emergency"
    else:
        triage_level = "URGENT"
        action = "VERIFY_THEN_DISPATCH"
        reason = "unmapped_visual_signal"

    return {
        "category": category,
        "triage_level": triage_level,
        "visual_flags": visual_flags,
        "action": action,
        "reason": reason,
        "requires_manual_fallback": action == "MANUAL_FALLBACK",
        "fallback_options": FALLBACK_OPTIONS if action == "MANUAL_FALLBACK" else [],
    }


def analyze_image(
    image_bytes: bytes,
    text_category: Optional[str] = None,
    text_triage_level: Optional[str] = None,
) -> Dict[str, Any]:
    svc = get_image_model_service()
    quality = analyze_image_quality(image_bytes)
    if not quality.get("usable"):
        return {
            "classification": None,
            "consistency": None,
            "summary": "Image could not be analyzed reliably. Please send a clearer photo or describe the emergency.",
            "available": False,
            "image_quality": quality,
            "visual_triage": {
                "triage_level": "URGENT",
                "category": "other",
                "action": "RECAPTURE_IMAGE",
                "reason": "image_quality_unusable",
                "requires_manual_fallback": True,
                "fallback_options": FALLBACK_OPTIONS,
                "visual_flags": [],
            },
        }

    if not svc.is_loaded:
        svc.load()

    if not svc.is_loaded:
        return {
            "classification": None,
            "consistency": None,
            "summary": "Image analysis unavailable – model not loaded.",
            "available": False,
            "image_quality": quality,
            "visual_triage": {
                "triage_level": "URGENT",
                "category": "other",
                "action": "MANUAL_FALLBACK",
                "reason": "model_not_loaded",
                "requires_manual_fallback": True,
                "fallback_options": FALLBACK_OPTIONS,
                "visual_flags": [],
            },
        }

    classification = svc.predict(image_bytes)
    visual_triage = derive_visual_triage(classification)

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
        "image_quality": quality,
        "visual_triage": visual_triage,
    }


_image_model_service = ImageModelService()


def get_image_model_service() -> ImageModelService:
    return _image_model_service
