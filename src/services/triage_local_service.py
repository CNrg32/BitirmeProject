"""
Yerel XLM-R triage servisi — OpenAI fine-tuned model yerine gecer.

Akis:
  1. LLMService.chat(task="triage") bu servisi cagirir.
  2. Servis, history'yi [USER]/[ASSISTANT] biciminde duz metne donusturur.
  3. XLM-R model 4 basli tahmin yapar (triage/category/red_flag/input_quality).
  4. Conservative karar katmani:
      - input_quality != 'meaningful' veya NON_URGENT regex floor matchi
        => force NON_URGENT (red_flag high hari�).
      - Sentiment panic skoru dusuk (<0.3) iken CRITICAL => URGENT'a duser.
      - Sentiment panic skoru yuksek (>=0.75) + CRITICAL tahmini => esikler
        gevsetilerek CRITICAL commit edilebilir.
      - CRITICAL confidence esigi (varsayilan 0.70) altindaysa URGENT'a duser.
      - URGENT confidence esigi (varsayilan 0.55) altindaysa NON_URGENT'a duser.
      - Ilk MIN_TURNS_FOR_CRITICAL (varsayilan 2) kullanici turuna kadar
        CRITICAL'a cikilmaz (red flag head de 0.5 altindaysa).
      - mvp_rules override orchestrator tarafinda zaten asamaya giriyor.
  5. LLMService'in bekledigi JSON sozlugunu dondurur.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_DIR = _PROJECT_ROOT / "out_models" / "triage_xlmr"

DEFAULT_CRITICAL_THRESHOLD = 0.70
DEFAULT_URGENT_THRESHOLD = 0.55
DEFAULT_MIN_TURNS_FOR_CRITICAL = 2
DEFAULT_PANIC_HIGH = 0.75
DEFAULT_PANIC_LOW = 0.30
DEFAULT_CRITICAL_RELAX_BONUS = 0.10  # yuksek panik -> esigi 0.70'ten 0.60'a indir


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, "").strip() or default)
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, "").strip() or default)
    except ValueError:
        return default


def _pick_device():
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _flatten_history(history: List[Dict[str, str]], max_turns: int) -> str:
    trimmed = history[-max_turns:] if max_turns > 0 else history
    parts: list[str] = []
    for msg in trimmed:
        role = "ASSISTANT" if msg.get("role") == "assistant" else "USER"
        text = str(msg.get("text", "")).strip()
        if not text:
            continue
        parts.append(f"[{role}] {text}")
    return " ".join(parts) if parts else "[USER]"


def _count_user_turns(history: List[Dict[str, str]]) -> int:
    return sum(1 for m in history if m.get("role") != "assistant")


def _last_user_text(history: List[Dict[str, str]]) -> str:
    for msg in reversed(history):
        if msg.get("role") != "assistant":
            t = str(msg.get("text", "")).strip()
            if t:
                return t
    return ""


def _compute_panic_score(text: str) -> Optional[float]:
    """Hafif text-only panic skoru; sentiment servisi bulunursa kullan, yoksa None."""
    if not text:
        return None
    try:
        from services.sentiment_service import (  # type: ignore
            compute_text_panic_score,
            extract_text_features,
        )
    except Exception:
        return None
    try:
        feats = extract_text_features(text)
        return float(compute_text_panic_score(feats))
    except Exception as exc:
        logger.debug("panic score computation failed: %s", exc)
        return None


def _is_nonurgent_floor(text: str) -> bool:
    if not text:
        return False
    try:
        from mvp_rules import is_nonurgent_floor_match  # type: ignore
    except Exception:
        return False
    try:
        return bool(is_nonurgent_floor_match(text))
    except Exception:
        return False


class LocalTriageService:
    """Lazy-loaded XLM-R multitask triage service with conservative decision rules."""

    def __init__(self, model_dir: Optional[Path] = None) -> None:
        self.model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        self._model = None
        self._tokenizer = None
        self._device = None
        self._lock = Lock()
        self._max_len = 384
        self._triage_labels: list[str] = ["CRITICAL", "URGENT", "NON_URGENT"]
        self._category_labels: list[str] = ["medical", "fire", "crime", "other"]
        self._quality_labels: list[str] = ["meaningful", "gibberish", "out_of_scope"]
        self._has_quality_head: bool = True
        self._max_history_turns = _env_int("TRIAGE_MAX_HISTORY_TURNS", 32)
        self._critical_threshold = _env_float("TRIAGE_CRITICAL_THRESHOLD", DEFAULT_CRITICAL_THRESHOLD)
        self._urgent_threshold = _env_float("TRIAGE_URGENT_THRESHOLD", DEFAULT_URGENT_THRESHOLD)
        self._min_turns_for_critical = _env_int("TRIAGE_MIN_TURNS_CRITICAL", DEFAULT_MIN_TURNS_FOR_CRITICAL)
        self._redflag_threshold = _env_float("TRIAGE_REDFLAG_THRESHOLD", 0.5)
        self._panic_high = _env_float("TRIAGE_PANIC_HIGH", DEFAULT_PANIC_HIGH)
        self._panic_low = _env_float("TRIAGE_PANIC_LOW", DEFAULT_PANIC_LOW)
        self._critical_relax = _env_float("TRIAGE_CRITICAL_RELAX_BONUS", DEFAULT_CRITICAL_RELAX_BONUS)
        self._enable_sentiment_fusion = os.environ.get("TRIAGE_SENTIMENT_FUSION", "1") not in ("0", "false", "False")
        self._enable_nonurgent_floor = os.environ.get("TRIAGE_NONURGENT_FLOOR", "1") not in ("0", "false", "False")

    @property
    def is_available(self) -> bool:
        return (self.model_dir / "config.json").exists() and (self.model_dir / "model.pt").exists()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            from services.triage_model import XlmrMultiTaskTriage, load_tokenizer
            self._device = _pick_device()
            logger.info("Loading local triage model from %s (device=%s)", self.model_dir, self._device)
            self._tokenizer = load_tokenizer(self.model_dir)
            self._model = XlmrMultiTaskTriage.load(self.model_dir, device=self._device)
            import json
            try:
                with (self.model_dir / "config.json").open("r", encoding="utf-8") as f:
                    meta = json.load(f)
                self._max_len = int(meta.get("max_len", self._max_len))
                if isinstance(meta.get("triage_labels"), list):
                    self._triage_labels = list(meta["triage_labels"])
                if isinstance(meta.get("category_labels"), list):
                    self._category_labels = list(meta["category_labels"])
                if isinstance(meta.get("quality_labels"), list):
                    self._quality_labels = list(meta["quality_labels"])
                version = int(meta.get("model_version", 1))
                self._has_quality_head = version >= 2 or "n_quality" in meta
            except Exception as exc:
                logger.warning("Could not read triage config.json: %s", exc)

    def _predict_raw(self, text: str) -> tuple[list[float], list[float], float, list[float]]:
        import torch
        self._ensure_loaded()
        enc = self._tokenizer(
            text,
            max_length=self._max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self._device)
        attention_mask = enc["attention_mask"].to(self._device)
        with torch.no_grad():
            out = self._model(input_ids=input_ids, attention_mask=attention_mask)
            triage_prob = torch.softmax(out["logits_triage"], dim=-1)[0].cpu().tolist()
            cat_prob = torch.softmax(out["logits_category"], dim=-1)[0].cpu().tolist()
            rf_prob = torch.sigmoid(out["logits_redflag"])[0].cpu().item()
            if "logits_quality" in out and self._has_quality_head:
                q_prob = torch.softmax(out["logits_quality"], dim=-1)[0].cpu().tolist()
            else:
                q_prob = [1.0] + [0.0] * (len(self._quality_labels) - 1)
        return triage_prob, cat_prob, float(rf_prob), q_prob

    def _apply_conservative_rules(
        self,
        triage_probs: list[float],
        redflag_prob: float,
        user_turns: int,
        input_quality: str = "meaningful",
        panic_score: Optional[float] = None,
        nonurgent_floor: bool = False,
    ) -> tuple[str, float, Dict[str, Any]]:
        """Tum guardrails -> nihai triage label + debug info.

        Oncelik sirasi:
          1. Input quality != 'meaningful' (gibberish/out_of_scope) veya
             nonurgent_floor == True ve red flag dusuk
             -> FORCE NON_URGENT.
          2. Sentiment panic skoru >= panic_high VE CRITICAL tahmini
             -> CRITICAL esigini _critical_relax kadar indir.
          3. Sentiment panic skoru < panic_low VE top = CRITICAL
             -> URGENT'a downgrade (red_flag yuksek degilse).
          4. Normal confidence + min-turns guardrails.
        """
        label2idx = {l: i for i, l in enumerate(self._triage_labels)}
        crit_idx = label2idx.get("CRITICAL", 0)
        urg_idx = label2idx.get("URGENT", 1)
        non_idx = label2idx.get("NON_URGENT", 2)

        top_idx = int(max(range(len(triage_probs)), key=lambda i: triage_probs[i]))
        top_label = self._triage_labels[top_idx]
        top_prob = float(triage_probs[top_idx])

        rf_is_high = redflag_prob >= self._redflag_threshold
        reasons: list[str] = []

        # (1) NON_URGENT hard floor
        if (input_quality != "meaningful") and not rf_is_high:
            reasons.append(f"input_quality={input_quality}->NON_URGENT")
            return "NON_URGENT", float(max(triage_probs[non_idx], top_prob)), {"reasons": reasons}
        if nonurgent_floor and not rf_is_high:
            reasons.append("nonurgent_regex_floor->NON_URGENT")
            return "NON_URGENT", float(max(triage_probs[non_idx], top_prob)), {"reasons": reasons}

        # (2) / (3) Sentiment panic fusion: esigi dinamik belirle
        crit_thr = self._critical_threshold
        if panic_score is not None:
            if panic_score >= self._panic_high and top_label == "CRITICAL":
                crit_thr = max(0.5, self._critical_threshold - self._critical_relax)
                reasons.append(f"panic_high({panic_score:.2f})->relax_crit_thr={crit_thr:.2f}")
            elif panic_score < self._panic_low and top_label == "CRITICAL" and not rf_is_high:
                # CRITICAL ama metinde panik sinyali yok, red_flag da dusuk -> URGENT'a dusur.
                reasons.append(f"panic_low({panic_score:.2f})+no_rf->downgrade_to_URGENT")
                return "URGENT", float(max(triage_probs[crit_idx], triage_probs[urg_idx])), {"reasons": reasons}

        # (4) Normal gates
        critical_gate = (
            top_label == "CRITICAL"
            and triage_probs[crit_idx] >= crit_thr
            and (user_turns >= self._min_turns_for_critical or rf_is_high)
        )
        if critical_gate:
            reasons.append(f"critical_gate(conf={triage_probs[crit_idx]:.2f}>={crit_thr:.2f}, turns={user_turns}, rf={redflag_prob:.2f})")
            return "CRITICAL", float(triage_probs[crit_idx]), {"reasons": reasons}

        urgent_gate = (
            (top_label in ("CRITICAL", "URGENT"))
            and max(triage_probs[crit_idx], triage_probs[urg_idx]) >= self._urgent_threshold
        )
        if urgent_gate:
            reasons.append(f"urgent_gate(conf={max(triage_probs[crit_idx], triage_probs[urg_idx]):.2f})")
            return "URGENT", float(max(triage_probs[crit_idx], triage_probs[urg_idx])), {"reasons": reasons}

        reasons.append("default->NON_URGENT")
        return "NON_URGENT", float(max(top_prob, triage_probs[non_idx])), {"reasons": reasons}

    def predict_from_history(
        self,
        history: List[Dict[str, str]],
        language: str = "en",
    ) -> Dict[str, Any]:
        if not self.is_available:
            logger.error("Local triage model is not available at %s", self.model_dir)
            return self._empty_response()
        try:
            text = _flatten_history(history, max_turns=self._max_history_turns)
            triage_probs, cat_probs, rf_prob, q_probs = self._predict_raw(text)
            user_turns = _count_user_turns(history)

            q_idx = int(max(range(len(q_probs)), key=lambda i: q_probs[i]))
            input_quality = (self._quality_labels[q_idx]
                             if q_idx < len(self._quality_labels) else "meaningful")

            last_user = _last_user_text(history)
            panic_score: Optional[float] = None
            if self._enable_sentiment_fusion:
                panic_score = _compute_panic_score(last_user)

            nonurgent_floor = False
            if self._enable_nonurgent_floor:
                nonurgent_floor = _is_nonurgent_floor(last_user)

            triage_label, triage_conf, debug = self._apply_conservative_rules(
                triage_probs, rf_prob, user_turns,
                input_quality=input_quality,
                panic_score=panic_score,
                nonurgent_floor=nonurgent_floor,
            )

            cat_idx = int(max(range(len(cat_probs)), key=lambda i: cat_probs[i]))
            category = self._category_labels[cat_idx]

            red_flags: list[str] = ["critical_signal"] if rf_prob >= self._redflag_threshold else []

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "local_triage: label=%s conf=%.3f cat=%s rf=%.3f iq=%s panic=%s floor=%s reasons=%s",
                    triage_label, triage_conf, category, rf_prob, input_quality,
                    f"{panic_score:.2f}" if panic_score is not None else "n/a",
                    nonurgent_floor, debug.get("reasons"),
                )

            return {
                "response_text": "",
                "extracted_slots": {},
                "triage_level": triage_label,
                "category": category,
                "is_complete": False,
                "red_flags": red_flags,
                "dispatch_action": "none",
                "post_dispatch_collect": False,
                "legal_close": False,
                "confidence": float(triage_conf),
                "input_quality": input_quality,
                "is_witness": False,
                "_debug": {
                    "triage_probs": triage_probs,
                    "category_probs": cat_probs,
                    "red_flag_prob": rf_prob,
                    "quality_probs": q_probs,
                    "panic_score": panic_score,
                    "nonurgent_floor": nonurgent_floor,
                    "user_turns": user_turns,
                    "reasons": debug.get("reasons", []),
                },
            }
        except Exception as exc:
            logger.exception("Local triage inference failed: %s", exc)
            return self._empty_response()

    def _empty_response(self) -> Dict[str, Any]:
        return {
            "response_text": "",
            "extracted_slots": {},
            "triage_level": "URGENT",
            "category": "other",
            "is_complete": False,
            "red_flags": [],
            "dispatch_action": "none",
            "post_dispatch_collect": False,
            "legal_close": False,
            "confidence": None,
            "input_quality": "meaningful",
            "is_witness": False,
        }


_local_service: Optional[LocalTriageService] = None


def get_local_triage_service() -> LocalTriageService:
    global _local_service
    if _local_service is None:
        _local_service = LocalTriageService()
    return _local_service
