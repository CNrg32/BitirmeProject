"""
Final session report via OpenAI (GPT), while conversation stays on Groq.

Enable with:
  USE_OPENAI_FINAL_REPORT=true
  OPENAI_API_KEY=sk-...

Optional:
  OPENAI_FINAL_REPORT_MODEL=gpt-4.1-mini   (falls back to OPENAI_MODEL or project default)
  OPENAI_FINAL_REPORT_MAX_TOKENS=1200
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "gpt-4.1-mini-2025-04-14"


def _use_openai_final_report() -> bool:
    v = os.environ.get("USE_OPENAI_FINAL_REPORT", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _model_id() -> str:
    return (
        os.environ.get("OPENAI_FINAL_REPORT_MODEL", "").strip()
        or os.environ.get("OPENAI_MODEL", "").strip()
        or _DEFAULT_MODEL
    )


def _max_tokens() -> int:
    try:
        return int(os.environ.get("OPENAI_FINAL_REPORT_MAX_TOKENS", "1200").strip() or "1200")
    except ValueError:
        return 1200


def _language_name(code: str) -> str:
    c = (code or "en").split("-")[0].lower()
    names = {
        "tr": "Turkish",
        "en": "English",
        "de": "German",
        "fr": "French",
        "es": "Spanish",
        "ar": "Arabic",
    }
    return names.get(c, "English")


def _compact_image(image_analysis: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not image_analysis or not image_analysis.get("available"):
        return None
    out: Dict[str, Any] = {
        "summary": (image_analysis.get("summary") or "")[:2000],
    }
    cls = image_analysis.get("classification")
    if isinstance(cls, dict):
        out["scene"] = cls.get("detected_class")
        out["confidence"] = cls.get("confidence")
    cons = image_analysis.get("consistency")
    if isinstance(cons, dict):
        out["consistency"] = {
            "detail": (cons.get("consistency_detail") or "")[:500],
            "score": cons.get("consistency_score"),
            "possible_fake": cons.get("possible_fake"),
        }
    return out


def _payload_for_prompt(
    triage_result: Dict[str, Any],
    slots: Dict[str, Any],
    image_analysis: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    tr = {k: v for k, v in (triage_result or {}).items() if k != "slots"}
    return {
        "triage": tr,
        "collected_slots": dict(slots or {}),
        "image": _compact_image(image_analysis),
    }


def generate_final_report_openai(
    triage_result: Dict[str, Any],
    slots: Dict[str, Any],
    image_analysis: Optional[Dict[str, Any]] = None,
    language: str = "en",
) -> Optional[str]:
    """
    Returns a handoff-style final report string, or None to fall back to template compose_report.
    """
    if not _use_openai_final_report():
        return None
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not key:
        logger.warning("USE_OPENAI_FINAL_REPORT is set but OPENAI_API_KEY is missing; using template report.")
        return None

    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        logger.error("openai package not installed; cannot generate OpenAI final report. pip install openai")
        return None

    model = _model_id()
    lang_name = _language_name(language)
    data = _payload_for_prompt(triage_result, slots, image_analysis)
    user_content = (
        "Use ONLY the facts in this JSON. Do not invent patients, locations, or symptoms.\n"
        f"If a field is missing, say so briefly or omit it — do not guess.\n\n{json.dumps(data, ensure_ascii=False, default=str)}"
    )
    system = (
        "You write the final structured emergency handoff report for dispatch and records. "
        f"Write entirely in {lang_name}.\n"
        "Structure with clear sections:\n"
        "1) Triage (level, category) and confidence if present\n"
        "2) Key facts from slots (complaint, location, vitals, consciousness, etc.)\n"
        "3) Red flags if any\n"
        "4) Image / scene notes only if the JSON includes non-null image data\n"
        "5) Short safety / next-step bullets consistent with the triage level\n"
        "Use bullet points. Keep it concise and operational."
    )

    try:
        client = OpenAI(api_key=key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
            max_tokens=_max_tokens(),
        )
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            logger.warning("OpenAI final report returned empty; using template report.")
            return None
        logger.info("Final report generated with OpenAI (model=%s, lang=%s).", model, language)
        return text
    except Exception as exc:
        logger.exception("OpenAI final report failed: %s; using template report.", exc)
        return None
