"""
LLM service for emergency triage conversation.

Provider selection:
    1. GROQ_API_KEY   → Groq (llama-3.3-70b)

Set the key in .env file:
    GROQ_API_KEY="<YOUR_GROQ_API_KEY>"
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from services.llm_prompt_config import build_system_prompt_with_few_shot

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
- Detect the language of the user's messages and ALWAYS reply in that same language.
- Do NOT translate or switch languages between turns.

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
            if line.startswith("```") and in_block:
                break
            if in_block:
                inner.append(line)
        raw = "\n".join(inner)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}") + 1
        if start != -1 and end > start:
            try:
                data = json.loads(raw[start:end])
            except json.JSONDecodeError:
                logger.warning("Could not parse LLM JSON")
                return dict(_EMPTY_LLM_RESPONSE)
        else:
            return dict(_EMPTY_LLM_RESPONSE)

    result = dict(_EMPTY_LLM_RESPONSE)
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
    })
    return result


# ---------------------------------------------------------------------------
# Groq provider
# ---------------------------------------------------------------------------

class _GroqProvider:
    DEFAULT_MODEL = "llama-3.3-70b-versatile"

    def __init__(self, api_key: str) -> None:
        self._client = None
        # If you have a Groq fine-tuned/LoRA model id, set GROQ_FINE_TUNED_MODEL.
        # Falls back to GROQ_MODEL and then project default.
        self.model = (
            os.environ.get("GROQ_FINE_TUNED_MODEL", "").strip()
            or os.environ.get("GROQ_MODEL", "").strip()
            or self.DEFAULT_MODEL
        )
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
        
        # FAZ 4: Use task-specific prompt (triage vs dialog)
        prompt_task = task or "dialog"  # Default to dialog if not specified
        system = build_system_prompt_with_few_shot(SYSTEM_PROMPT, lang_name, task=prompt_task)
        
        # FAZ 3: Inject session context for category locking (future: FAZ 4 will use this actively)
        if session_context and session_context.get("initial_category"):
            logger.debug("Session context injected: category=%s, dispatch_status=%s, task=%s",
                        session_context.get("initial_category"), 
                        session_context.get("dispatch_status"),
                        prompt_task)

        messages = [{"role": "system", "content": system}]

        # Inject session context as a system message so LLM can use it
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
        _RETRY_DELAYS = [2, 5, 10]  # seconds between retries

        for attempt in range(_MAX_RETRIES):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=1024,
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
    """Uses Groq when GROQ_API_KEY is available."""

    def __init__(self) -> None:
        self._provider = None
        self._provider_name = "none"
        self._init()

    def _init(self) -> None:
        groq_key = os.environ.get("GROQ_API_KEY")

        if groq_key:
            p = _GroqProvider(api_key=groq_key)
            if p.is_ready:
                self._provider = p
                self._provider_name = f"groq/{p.model}"
                return

        logger.warning(
            "No LLM provider found (GROQ_API_KEY). "
            "LLM disabled - falling back to rule-based dialog."
        )

    @property
    def is_available(self) -> bool:
        return self._provider is not None

    @property
    def MODEL(self) -> str:
        return self._provider_name

    # 5. Token limiti: son 10 mesajı gönder (yaklaşık 3-4k token)
    _MAX_HISTORY_TURNS = 10

    def chat(
        self,
        history: List[Dict[str, str]],
        language: str = "en",
        task: Optional[str] = None,
        session_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self._provider is None:
            return dict(_EMPTY_LLM_RESPONSE)
        trimmed = history[-self._MAX_HISTORY_TURNS:] if len(history) > self._MAX_HISTORY_TURNS else history
        if len(history) != len(trimmed):
            logger.debug("History trimmed from %d to %d messages for token efficiency.",
                         len(history), len(trimmed))
        return self._provider.chat(
            history=trimmed,
            language=language,
            task=task,
            session_context=session_context,
        )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
