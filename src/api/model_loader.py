from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

def apply_redflag_override(
    text_en: str,
    pred_label: str,
    meta: Optional[Dict[str, Any]] = None,
) -> Tuple[str, List[str]]:
    from mvp_rules import predict_mvp

    meta = meta or {}
    mvp = predict_mvp(
        text_en,
        deaths=meta.get("deaths", 0),
        potential_death=meta.get("potential_death", 0),
        false_alarm=meta.get("false_alarm", 0),
    )
    red_flags = mvp.get("red_flags") or []
    if red_flags and pred_label != "CRITICAL":
        return "CRITICAL", list(red_flags)
    return pred_label, list(red_flags)


def _text_with_meta(text_en: str, meta: Optional[Dict[str, Any]] = None) -> str:
    meta = meta or {}
    deaths = meta.get("deaths", 0) or 0
    potential_death = meta.get("potential_death", 0) or 0
    false_alarm = meta.get("false_alarm", 0) or 0
    try:
        deaths = float(deaths)
        potential_death = float(potential_death)
        false_alarm = float(false_alarm)
    except (TypeError, ValueError):
        deaths = potential_death = false_alarm = 0
    return f"{text_en} [deaths:{deaths} potential_death:{potential_death} false_alarm:{false_alarm}]"


logger = logging.getLogger(__name__)

_BASE = Path(__file__).resolve().parent.parent.parent
_MODEL_DIR = _BASE / "out_models"

_DISTILBERT_MAX_LEN = 256


class DistilBertTriageWrapper:
    """sklearn pipeline arayuzunu taklit eden wrapper (API uyumlulugu icin)."""

    def __init__(self, model_path, labels, label2id, id2label):
        self.model_path = model_path
        self.labels = labels
        self.label2id = label2id
        self.id2label = id2label
        self._model = None
        self._tokenizer = None
        self._device = None

    def _ensure_loaded(self):
        if self._model is None:
            from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
            import torch

            self._tokenizer = DistilBertTokenizer.from_pretrained(self.model_path)
            self._model = DistilBertForSequenceClassification.from_pretrained(self.model_path)
            if torch.backends.mps.is_available():
                self._device = torch.device("mps")
            elif torch.cuda.is_available():
                self._device = torch.device("cuda")
            else:
                self._device = torch.device("cpu")
            self._model.to(self._device)
            self._model.eval()

    @property
    def classes_(self):
        import numpy as np

        return np.array(self.labels)

    def predict(self, texts):
        import numpy as np

        proba = self.predict_proba(texts)
        return [self.labels[i] for i in np.argmax(proba, axis=1)]

    def predict_proba(self, texts):
        import torch

        self._ensure_loaded()
        if isinstance(texts, str):
            texts = [texts]
        encodings = self._tokenizer(
            list(texts),
            max_length=_DISTILBERT_MAX_LEN,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encodings["input_ids"].to(self._device)
        attention_mask = encodings["attention_mask"].to(self._device)

        with torch.no_grad():
            outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
            proba = torch.softmax(outputs.logits, dim=-1)
        return proba.cpu().numpy()


class TriageModelService:

    def __init__(self) -> None:
        self.pipeline = None
        self.tfidf = None
        self.clf = None
        self.labels: List[str] = ["CRITICAL", "URGENT", "NON_URGENT"]
        self._loaded = False
        self._model_type = "unknown"

    def load(self, model_dir: Path | str | None = None) -> bool:
        model_dir = Path(model_dir) if model_dir else _MODEL_DIR
        pipeline_path = model_dir / "triage_pipeline.joblib"
        tfidf_path = model_dir / "tfidf.joblib"
        clf_path = model_dir / "triage_lr.joblib"
        labels_path = model_dir / "label_maps.json"

        model_type_hint = None
        if labels_path.exists():
            with open(labels_path, encoding="utf-8") as f:
                lmap = json.load(f)
            self.labels = lmap.get("triage_labels", self.labels)
            model_type_hint = lmap.get("model_type")

        if pipeline_path.exists():
            import joblib
            import sys
            _this_module = sys.modules[__name__]
            for alias in ("__main__", "__mp_main__", "train_distilbert"):
                mod = sys.modules.get(alias)
                if mod is not None and not hasattr(mod, "DistilBertTriageWrapper"):
                    mod.DistilBertTriageWrapper = DistilBertTriageWrapper
                elif mod is None:
                    import types
                    stub = types.ModuleType(alias)
                    stub.DistilBertTriageWrapper = DistilBertTriageWrapper
                    sys.modules[alias] = stub

            self.pipeline = joblib.load(pipeline_path)
            self._loaded = True
            self._model_type = model_type_hint or "tfidf_lr"
            logger.info("ML pipeline loaded from %s (type: %s)", pipeline_path, self._model_type)
            return True

        if tfidf_path.exists() and clf_path.exists():
            import joblib

            self.tfidf = joblib.load(tfidf_path)
            self.clf = joblib.load(clf_path)
            self._loaded = True
            self._model_type = "legacy"
            logger.info("ML model (legacy) loaded from %s", model_dir)
            return True

        logger.warning(
            "ML model files not found at %s – falling back to rule-based.",
            model_dir,
        )
        return False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def predict(
        self, text_en: str, meta: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, float]:
        if not self._loaded:
            raise RuntimeError("Model not loaded – call .load() first.")

        if self._model_type == "distilbert":
            proba = self.pipeline.predict_proba([text_en])[0]
            classes = self.pipeline.classes_
        elif self.pipeline is not None:
            X_input = _text_with_meta(text_en, meta)
            proba = self.pipeline.predict_proba([X_input])[0]
            classes = self.pipeline.classes_
        else:
            X_input = _text_with_meta(text_en, meta)
            X = self.tfidf.transform([X_input])
            proba = self.clf.predict_proba(X)[0]
            classes = self.clf.classes_

        import numpy as np

        idx = int(np.argmax(proba))
        label = classes[idx]
        confidence = float(proba[idx])
        return label, confidence


_model_service = TriageModelService()


def get_model_service() -> TriageModelService:
    return _model_service


def load_sentiment_model(model_dir: Path | str | None = None) -> bool:
    try:
        from services.sentiment_service import get_sentiment_service
        return get_sentiment_service().load(model_dir)
    except Exception as e:
        logger.warning("Sentiment model load failed: %s", e)
        return False
