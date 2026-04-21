"""
LLM service for emergency triage conversation.

Architecture (no single-provider fallback):
  - Groq: user-facing dialog (task=dialog, gibberish_check); triage fields from Groq are ignored by orchestrator.
  - OpenAI fine-tuned (OPENAI_FINE_TUNED_MODEL): triage each turn with full conversation (task=triage).

Requires in .env:
  GROQ_API_KEY="<...>"
  OPENAI_API_KEY="<...>"
  OPENAI_FINE_TUNED_MODEL="ft:..."   # triage — required for LLM mode; no base-model fallback

If any piece is missing, is_available is False and the orchestrator uses rule-based dialog.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from services.llm_prompt_config import build_system_prompt_with_few_shot
from services.triage_local_service import LocalTriageService, get_local_triage_service
from services.turn_trace import monotonic_start, trace_llm_response

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Language helpers
# ---------------------------------------------------------------------------

LANGUAGE_NAMES: Dict[str, str] = {
    "en": "English",
    "tr": "Turkish",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "ar": "Arabic",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "it": "Italian",
    "nl": "Dutch",
    "pl": "Polish",
    "uk": "Ukrainian",
    "hi": "Hindi",
}

# ---------------------------------------------------------------------------
# System prompt (shared across providers)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a professional emergency dispatcher assistant. Your role is to rapidly collect \
critical information, assess the emergency, and guide the caller calmly.

CONVERSATION FLOW — follow this priority order strictly:
1. If chief_complaint is unknown → ask what happened / what the emergency is FIRST.
2. If caller_name is unknown → ask for the caller's name.
3. If age is unknown → ask for the age of the person who needs help.
4. Then collect category-specific details ONE question at a time:
   - medical : consciousness, breathing, bleeding, duration_minutes, sex
   - fire    : fire_size, smoke_inhalation, trapped, injuries
   - crime   : assailant_present, weapon, injuries, victim_count
   - other   : describe situation

DO NOT ask for location — it is obtained automatically from the phone.

IMPORTANT LANGUAGE RULE:
- The session language is fixed for the whole conversation (set from the user's app preference or their first message).
- ALWAYS write response_text in that session language only. Do not follow mid-session language changes in user text.

OUTPUT FORMAT:
You MUST return ONLY a valid JSON object – no markdown, no prose, no code fences.
{
  "response_text": "<your reply to the user, in their language, max 3 sentences unless giving first-aid instructions>",
  "extracted_slots": {
    "<slot_key>": "<value>"
  },
  "triage_level": "<CRITICAL|URGENT|NON_URGENT>",
  "category": "<medical|fire|crime|other>",
  "is_complete": <true|false>,
  "red_flags": ["<critical life-threatening sign if present>"]
}

RULES:
- Ask ONLY ONE question per turn — the next most important unknown.
- Only include slots explicitly stated by the user. Never guess or hallucinate values.
- Set is_complete=true when you have chief_complaint + caller_name + age + at least
  2 category-specific details, OR when the situation is clearly CRITICAL and
  further questioning could delay dispatch.
- For CRITICAL situations (cardiac arrest, severe bleeding, fire with trapped people)
  set is_complete=true IMMEDIATELY and give first-aid or safety instructions.
- If red_flags is non-empty AND triage_level is CRITICAL, always set is_complete=true.
- You have a maximum of 8 conversation turns. Aim to complete within 6 turns.
- red_flags should only contain genuinely life-threatening signs.
- red_flags entries must be short noun phrases in the SAME language as the user's messages. Use correct grammar for that language.
- red_flags must never include casual/commercial adjectives or malformed mixed tokens (e.g. "ucuz", "promo", broken words).
- If the user asks non-emergency questions (hotel prices, travel, weather, finance, entertainment, etc.), do NOT answer that content.
    Instead, politely state this is an emergency line and ask them to describe an actual emergency.
- Do NOT change the category once it has been clearly established (e.g. don't switch
  from "fire" to "other" in a later turn).
"""


# ---------------------------------------------------------------------------
# JSON parser (shared)
# ---------------------------------------------------------------------------

_EMPTY_LLM_RESPONSE: Dict[str, Any] = {
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


def _parse_llm_json(raw: str) -> Dict[str, Any]:
    """Parse LLM JSON output, tolerating minor formatting issues."""
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        inner, in_block = [], False
        for line in lines:
            if line.startswith("```") and not in_block:
                in_block = True
                continue
            if in_block and line.strip().startswith("```"):
                break
            if in_block:
                inner.append(line)
        raw = "\n".join(inner)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Could not parse LLM JSON")
        return dict(_EMPTY_LLM_RESPONSE)

    result = dict(_EMPTY_LLM_RESPONSE)
    conf_raw = data.get("confidence")
    conf: Optional[float] = None
    if conf_raw is not None:
        try:
            conf = float(conf_raw)
        except (TypeError, ValueError):
            pass
    iq = str(data.get("input_quality") or "meaningful").strip().lower()
    if iq not in ("meaningful", "gibberish", "out_of_scope"):
        iq = "meaningful"
    result.update({
        "response_text": data.get("response_text", ""),
        "extracted_slots": {
            k: v for k, v in (data.get("extracted_slots") or {}).items()
            if v not in (None, "", "unknown", "N/A")
        },
        "triage_level": data.get("triage_level", "URGENT"),
        "category": data.get("category", "other"),
        "is_complete": bool(data.get("is_complete", False)),
        "red_flags": list(data.get("red_flags") or []),
        "dispatch_action": str(data.get("dispatch_action", "none") or "none").strip().lower(),
        "post_dispatch_collect": bool(data.get("post_dispatch_collect", False)),
        "legal_close": bool(data.get("legal_close", False)),
        "confidence": conf,
        "input_quality": iq,
        "is_witness": bool(data.get("is_witness", False)),
    })
    return result


# ---------------------------------------------------------------------------
# OpenAI — triage only (fine-tuned model required)
# ---------------------------------------------------------------------------

class _OpenAITriageProvider:
    """Triage classification only; model id must be OPENAI_FINE_TUNED_MODEL."""

    def __init__(self, api_key: str, model_id: str) -> None:
        self._client = None
        self.model = model_id.strip()
        self.fast_finetune = (
            self.model.startswith("ft:")
            and os.environ.get("OPENAI_FINE_TUNED_FAST", "true").strip().lower()
            not in ("0", "false", "no")
        )
        default_mt = "360" if self.fast_finetune else "512"
        self.max_tokens = int(os.environ.get("OPENAI_TRIAGE_MAX_TOKENS", os.environ.get("OPENAI_MAX_TOKENS", default_mt)))
        try:
            from openai import OpenAI  # type: ignore
            self._client = OpenAI(api_key=api_key)
            logger.info("OpenAI triage model initialised (model=%s)", self.model)
        except ImportError:
            logger.error("openai package not installed. Run: pip install openai")
        except Exception as exc:
            logger.error("OpenAI triage init failed: %s", exc)

    @property
    def is_ready(self) -> bool:
        return self._client is not None and bool(self.model)

    def chat(
        self,
        history: List[Dict[str, str]],
        language: str,
        task: Optional[str] = None,
        session_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.is_ready:
            return dict(_EMPTY_LLM_RESPONSE)

        lang_name = LANGUAGE_NAMES.get(language, "English")
        max_few_shot = 0 if self.fast_finetune else 5
        system = build_system_prompt_with_few_shot(
            SYSTEM_PROMPT,
            lang_name,
            max_few_shot=max_few_shot,
            task="triage",
        )
        messages = [{"role": "system", "content": system}]
        for msg in history:
            role = "assistant" if msg.get("role") == "assistant" else "user"
            messages.append({"role": role, "content": msg.get("text", "")})

        _MAX_RETRIES = 3
        _RETRY_DELAYS = [2, 5, 10]

        for attempt in range(_MAX_RETRIES):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"},
                )
                raw = response.choices[0].message.content or ""
                logger.debug("OpenAI triage raw response: %s", raw[:500])
                return _parse_llm_json(raw)
            except Exception as exc:
                exc_str = str(exc).lower()
                is_rate_limit = (
                    "rate_limit" in exc_str
                    or "429" in exc_str
                    or "rate limit" in exc_str
                    or type(exc).__name__ in ("RateLimitError", "APIStatusError")
                )
                if is_rate_limit and attempt < _MAX_RETRIES - 1:
                    wait = _RETRY_DELAYS[attempt]
                    logger.warning(
                        "OpenAI triage rate limit (attempt %d/%d). Retrying in %ds. Error: %s",
                        attempt + 1, _MAX_RETRIES, wait, exc,
                    )
                    time.sleep(wait)
                    continue
                logger.error("OpenAI triage failed (attempt %d/%d): %s", attempt + 1, _MAX_RETRIES, exc)
                return dict(_EMPTY_LLM_RESPONSE)


# ---------------------------------------------------------------------------
# Local XLM-R triage provider — yerine gecen sinificandirici
# ---------------------------------------------------------------------------

class _LocalTriageProvider:
    """Yerel XLM-RoBERTa multi-task modeli ile triage siniflandirma."""

    def __init__(self) -> None:
        self._service: LocalTriageService = get_local_triage_service()
        self.model = "local/triage_xlmr"

    @property
    def is_ready(self) -> bool:
        return self._service.is_available

    def chat(
        self,
        history: List[Dict[str, str]],
        language: str,
        task: Optional[str] = None,
        session_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.is_ready:
            return dict(_EMPTY_LLM_RESPONSE)
        try:
            return self._service.predict_from_history(history=history, language=language)
        except Exception as exc:
            logger.error("Local triage provider failed: %s", exc)
            return dict(_EMPTY_LLM_RESPONSE)


# ---------------------------------------------------------------------------
# Groq provider (dialog + non-triage tasks)
# ---------------------------------------------------------------------------

class _GroqProvider:
    DEFAULT_MODEL = "llama-3.3-70b-versatile"

    def __init__(self, api_key: str) -> None:
        self._client = None
        self.model = (
            os.environ.get("GROQ_FINE_TUNED_MODEL", "").strip()
            or os.environ.get("GROQ_MODEL", "").strip()
            or self.DEFAULT_MODEL
        )
        self.fast_path = os.environ.get("GROQ_FAST_PATH", "true").strip().lower() not in ("0", "false", "no")
        self.max_tokens = int(os.environ.get("GROQ_MAX_TOKENS", "512" if self.fast_path else "1024"))
        try:
            from groq import Groq  # type: ignore
            self._client = Groq(api_key=api_key)
            logger.info("Groq LLM initialised (model=%s)", self.model)
        except ImportError:
            logger.error("groq package not installed. Run: pip install groq")
        except Exception as exc:
            logger.error("Groq init failed: %s", exc)

    @property
    def is_ready(self) -> bool:
        return self._client is not None

    def chat(
        self,
        history: List[Dict[str, str]],
        language: str,
        task: Optional[str] = None,
        session_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.is_ready:
            return dict(_EMPTY_LLM_RESPONSE)

        lang_name = LANGUAGE_NAMES.get(language, "English")
        prompt_task = task or "dialog"
        if prompt_task == "triage_dialog":
            prompt_task = "dialog"
        max_few_shot = 0 if self.fast_path else 5
        system = build_system_prompt_with_few_shot(
            SYSTEM_PROMPT,
            lang_name,
            max_few_shot=max_few_shot,
            task=prompt_task,
        )

        messages = [{"role": "system", "content": system}]

        if session_context and prompt_task == "dialog":
            ctx_parts = []
            if session_context.get("initial_category"):
                ctx_parts.append(f"LOCKED CATEGORY: {session_context['initial_category']}")
            if session_context.get("initial_triage_level"):
                ctx_parts.append(f"LOCKED TRIAGE LEVEL: {session_context['initial_triage_level']}")
            dispatch_status = session_context.get("dispatch_status", "")
            if dispatch_status:
                ctx_parts.append(f"DISPATCH STATUS: {dispatch_status}")
            if dispatch_status in ("DISPATCHED", "SILENT_DISPATCHED"):
                ctx_parts.append(
                    "DISPATCH ACTIVE: Emergency services are already on the way. "
                    "Do NOT set is_complete=true yet. Continue collecting micro-location details "
                    "(building, floor, apartment, entrance, landmark, gate code) ONE question at a time. "
                    "Only set is_complete=true when you have no more useful questions OR the caller "
                    "says they cannot provide more info."
                )
            if session_context.get("witness_mode"):
                ctx_parts.append("WITNESS MODE: true — caller is a bystander, NOT the victim. Apply witness question rules.")
            exhausted = session_context.get("exhausted_slots") or []
            if exhausted:
                ctx_parts.append(f"EXHAUSTED SLOTS (do NOT ask again): {', '.join(exhausted)}")
            if ctx_parts:
                ctx_msg = "SESSION CONTEXT (backend state — treat as authoritative):\n" + "\n".join(ctx_parts)
                messages.append({"role": "system", "content": ctx_msg})
                logger.debug("Session context injected into LLM messages: %s", ctx_parts)

        for msg in history:
            role = "assistant" if msg.get("role") == "assistant" else "user"
            messages.append({"role": role, "content": msg.get("text", "")})

        _MAX_RETRIES = 3
        _RETRY_DELAYS = [2, 5, 10]

        for attempt in range(_MAX_RETRIES):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"},
                )
                raw = response.choices[0].message.content or ""
                logger.debug("Groq raw response: %s", raw[:500])
                return _parse_llm_json(raw)
            except Exception as exc:
                exc_str = str(exc).lower()
                is_rate_limit = (
                    "rate_limit" in exc_str
                    or "429" in exc_str
                    or "rate limit" in exc_str
                    or "tokens per" in exc_str
                    or "requests per" in exc_str
                    or type(exc).__name__ in ("RateLimitError", "APIStatusError")
                )
                if is_rate_limit and attempt < _MAX_RETRIES - 1:
                    wait = _RETRY_DELAYS[attempt]
                    logger.warning(
                        "Groq rate limit hit (attempt %d/%d). Retrying in %ds. Error: %s",
                        attempt + 1, _MAX_RETRIES, wait, exc,
                    )
                    time.sleep(wait)
                    continue
                logger.error("Groq chat failed (attempt %d/%d): %s", attempt + 1, _MAX_RETRIES, exc)
                return dict(_EMPTY_LLM_RESPONSE)


# ---------------------------------------------------------------------------
# Unified LLM Service
# ---------------------------------------------------------------------------

class LLMService:
    """Groq for dialog; triage via local XLM-R veya OpenAI fine-tuned (TRIAGE_BACKEND)."""

    def __init__(self) -> None:
        self._groq: Optional[_GroqProvider] = None
        self._openai_triage: Optional[_OpenAITriageProvider] = None
        self._local_triage: Optional[_LocalTriageProvider] = None
        self._triage_backend: str = "local"
        self._init()

    def _init(self) -> None:
        groq_key = (os.environ.get("GROQ_API_KEY") or "").strip()
        openai_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
        triage_model = (os.environ.get("OPENAI_FINE_TUNED_MODEL") or "").strip()
        # Backend secimi: env > local-varsa > openai-varsa
        requested_backend = (os.environ.get("TRIAGE_BACKEND") or "").strip().lower()

        if groq_key:
            p = _GroqProvider(api_key=groq_key)
            if p.is_ready:
                self._groq = p

        local_candidate = _LocalTriageProvider()
        if local_candidate.is_ready:
            self._local_triage = local_candidate

        if openai_key and triage_model:
            p = _OpenAITriageProvider(api_key=openai_key, model_id=triage_model)
            if p.is_ready:
                self._openai_triage = p
        elif openai_key and not triage_model:
            logger.warning(
                "OPENAI_API_KEY is set but OPENAI_FINE_TUNED_MODEL is empty."
            )

        if requested_backend in ("local", "openai"):
            self._triage_backend = requested_backend
        else:
            self._triage_backend = "local" if self._local_triage else "openai"

        if self._triage_backend == "local" and not self._local_triage:
            logger.warning(
                "TRIAGE_BACKEND=local requested but model artefacts missing at out_models/triage_xlmr; falling back to openai."
            )
            self._triage_backend = "openai"
        if self._triage_backend == "openai" and not self._openai_triage:
            if self._local_triage:
                logger.warning("OpenAI triage unavailable; falling back to local XLM-R.")
                self._triage_backend = "local"

        if not self._groq or not self._groq.is_ready:
            logger.warning("Groq is not configured or failed to initialise (GROQ_API_KEY).")
        logger.info("LLMService triage backend = %s", self._triage_backend)

    @property
    def is_available(self) -> bool:
        triage_ok = (
            (self._triage_backend == "local" and self._local_triage and self._local_triage.is_ready)
            or (self._triage_backend == "openai" and self._openai_triage and self._openai_triage.is_ready)
        )
        return bool(self._groq and self._groq.is_ready and triage_ok)

    @property
    def MODEL(self) -> str:
        if not self._groq or not self._groq.is_ready:
            return "none"
        if self._triage_backend == "local" and self._local_triage:
            return f"groq/{self._groq.model}(triage:local/xlmr)"
        triage = self._openai_triage.model if self._openai_triage and self._openai_triage.is_ready else "none"
        return f"groq/{self._groq.model}(triage:{triage})"

    _MAX_HISTORY_TURNS = 10
    # Triage sees more context than Groq dialog (multi-turn reassessment).
    _MAX_TRIAGE_HISTORY_TURNS = max(
        10,
        int(os.environ.get("OPENAI_TRIAGE_MAX_HISTORY_TURNS", "32").strip() or "32"),
    )

    def chat(
        self,
        history: List[Dict[str, str]],
        language: str = "en",
        task: Optional[str] = None,
        session_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        t = task or "dialog"
        if t == "triage":
            triage_hist = (
                history[-self._MAX_TRIAGE_HISTORY_TURNS:]
                if len(history) > self._MAX_TRIAGE_HISTORY_TURNS
                else history
            )
            if len(history) != len(triage_hist):
                logger.debug(
                    "Triage history trimmed from %d to %d messages.",
                    len(history),
                    len(triage_hist),
                )

            if self._triage_backend == "local":
                if not self._local_triage or not self._local_triage.is_ready:
                    logger.error("Triage call but local model unavailable.")
                    return dict(_EMPTY_LLM_RESPONSE)
                t0 = monotonic_start()
                out = self._local_triage.chat(triage_hist, language, task="triage", session_context=None)
                trace_llm_response(
                    provider="local_xlmr",
                    model=self._local_triage.model,
                    task="triage",
                    elapsed_s=time.monotonic() - t0,
                    parsed=out,
                )
                return out

            if not self._openai_triage or not self._openai_triage.is_ready:
                logger.error("Triage call but OpenAI fine-tuned triage model is not available.")
                return dict(_EMPTY_LLM_RESPONSE)
            t0 = monotonic_start()
            out = self._openai_triage.chat(triage_hist, language, task="triage", session_context=None)
            trace_llm_response(
                provider="openai_finetune",
                model=self._openai_triage.model,
                task="triage",
                elapsed_s=time.monotonic() - t0,
                parsed=out,
            )
            return out

        trimmed = history[-self._MAX_HISTORY_TURNS:] if len(history) > self._MAX_HISTORY_TURNS else history
        if len(history) != len(trimmed):
            logger.debug(
                "History trimmed from %d to %d messages for token efficiency.",
                len(history),
                len(trimmed),
            )

        if not self._groq or not self._groq.is_ready:
            return dict(_EMPTY_LLM_RESPONSE)

        t0 = monotonic_start()
        out = self._groq.chat(
            history=trimmed,
            language=language,
            task=t,
            session_context=session_context,
        )
        trace_llm_response(
            provider="groq",
            model=self._groq.model,
            task=t,
            elapsed_s=time.monotonic() - t0,
            parsed=out,
        )
        return out


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
