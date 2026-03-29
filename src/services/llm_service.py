"""
LLM service for emergency triage conversation.

Provider selection (first match wins):
  1. GROQ_API_KEY   → Groq (llama-3.3-70b) – free tier, works in Turkey
  2. GEMINI_API_KEY → Google Gemini         – requires billing in Turkey

Set the key in .env file:
    GROQ_API_KEY=gsk_...
    # or
    GEMINI_API_KEY=AIza...
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
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
- Do NOT change the category once it has been clearly established (e.g. don't switch
  from "fire" to "other" in a later turn).
"""


def _build_local_prompt(history: List[Dict[str, str]], language: str) -> str:
    """Build training-compatible prompt for local fine-tuned seq2seq model."""
    lines: List[str] = []
    for msg in history:
        role = "ASSISTANT" if msg.get("role") == "assistant" else "USER"
        text = str(msg.get("text", "")).strip()
        if text:
            lines.append(f"{role}: {text}")

    convo = "\n".join(lines)
    return (
        "You are a professional emergency dispatcher assistant. "
        "Return ONLY a valid JSON object with keys: "
        "response_text, extracted_slots, triage_level, category, is_complete, red_flags.\n"
        f"Language: {language or 'en'}\n"
        "Conversation:\n"
        f"{convo}\n\n"
        "Generate the next assistant JSON output:"
    )

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
        self.model = os.environ.get("GROQ_MODEL", self.DEFAULT_MODEL)
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

    def chat(self, history: List[Dict[str, str]], language: str) -> Dict[str, Any]:
        if not self.is_ready:
            return dict(_EMPTY_LLM_RESPONSE)

        lang_name = LANGUAGE_NAMES.get(language, "English")
        system = build_system_prompt_with_few_shot(SYSTEM_PROMPT, lang_name)

        messages = [{"role": "system", "content": system}]
        for msg in history:
            role = "assistant" if msg.get("role") == "assistant" else "user"
            messages.append({"role": role, "content": msg.get("text", "")})

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
            logger.error("Groq chat failed: %s", exc)
            return dict(_EMPTY_LLM_RESPONSE)


# ---------------------------------------------------------------------------
# Gemini provider
# ---------------------------------------------------------------------------

class _GeminiProvider:
    DEFAULT_MODEL = "gemini-2.0-flash"

    def __init__(self, api_key: str) -> None:
        self._client = None
        self.model = os.environ.get("GEMINI_MODEL", self.DEFAULT_MODEL)
        try:
            from google import genai  # type: ignore
            self._client = genai.Client(api_key=api_key)
            logger.info("Gemini LLM initialised (model=%s)", self.model)
        except ImportError:
            logger.error("google-genai package not installed. Run: pip install google-genai")
        except Exception as exc:
            logger.error("Gemini init failed: %s", exc)

    @property
    def is_ready(self) -> bool:
        return self._client is not None

    def chat(self, history: List[Dict[str, str]], language: str) -> Dict[str, Any]:
        if not self.is_ready:
            return dict(_EMPTY_LLM_RESPONSE)


# ---------------------------------------------------------------------------
# Local fine-tuned provider (offline)
# ---------------------------------------------------------------------------

class _LocalFineTunedProvider:
    """Loads a local seq2seq fine-tuned model for JSON-style chatbot outputs."""

    def __init__(self, model_dir: str) -> None:
        self._model = None
        self._tokenizer = None
        self._device = None
        self.model_dir = model_dir
        self.model_name = f"local/{Path(model_dir).name}"

        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore
            import torch  # type: ignore

            self._tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

            if torch.cuda.is_available():
                self._device = torch.device("cuda")
            else:
                self._device = torch.device("cpu")

            self._model.to(self._device)
            self._model.eval()
            logger.info("Local chatbot model initialised (dir=%s)", model_dir)
        except Exception as exc:
            logger.error("Local model init failed (%s): %s", model_dir, exc)

    @property
    def is_ready(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def chat(self, history: List[Dict[str, str]], language: str) -> Dict[str, Any]:
        if not self.is_ready:
            return dict(_EMPTY_LLM_RESPONSE)

        try:
            prompt = _build_local_prompt(history=history, language=language)
            enc = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            input_ids = enc["input_ids"].to(self._device)
            attention_mask = enc["attention_mask"].to(self._device)

            outputs = self._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=256,
                do_sample=False,
            )
            raw = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.debug("Local model raw response: %s", raw[:500])
            return _parse_llm_json(raw)
        except Exception as exc:
            logger.error("Local model chat failed: %s", exc)
            return dict(_EMPTY_LLM_RESPONSE)

        lang_name = LANGUAGE_NAMES.get(language, "English")
        system = build_system_prompt_with_few_shot(SYSTEM_PROMPT, lang_name)

        contents = []
        for msg in history:
            role = "model" if msg.get("role") == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": msg.get("text", "")}]})

        try:
            from google import genai  # type: ignore
            from google.genai import types  # type: ignore

            response = self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    response_mime_type="application/json",
                    temperature=0.3,
                    max_output_tokens=1024,
                ),
            )
            raw = response.text or ""
            logger.debug("Gemini raw response: %s", raw[:500])
            return _parse_llm_json(raw)
        except Exception as exc:
            logger.error("Gemini chat failed: %s", exc)
            return dict(_EMPTY_LLM_RESPONSE)


# ---------------------------------------------------------------------------
# Unified LLM Service
# ---------------------------------------------------------------------------

class LLMService:
    """Auto-selects Groq or Gemini based on available API keys."""

    def __init__(self) -> None:
        self._provider = None
        self._provider_name = "none"
        self._init()

    def _init(self) -> None:
        local_model_dir = os.environ.get("LOCAL_CHATBOT_MODEL_DIR", "").strip()
        groq_key = os.environ.get("GROQ_API_KEY")
        gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

        if local_model_dir:
            p = _LocalFineTunedProvider(model_dir=local_model_dir)
            if p.is_ready:
                self._provider = p
                self._provider_name = p.model_name
                return

        if groq_key:
            p = _GroqProvider(api_key=groq_key)
            if p.is_ready:
                self._provider = p
                self._provider_name = f"groq/{p.model}"
                return

        if gemini_key:
            p = _GeminiProvider(api_key=gemini_key)
            if p.is_ready:
                self._provider = p
                self._provider_name = f"gemini/{p.model}"
                return

        logger.warning(
            "No LLM provider found (LOCAL_CHATBOT_MODEL_DIR, GROQ_API_KEY, GEMINI_API_KEY). "
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

    def chat(self, history: List[Dict[str, str]], language: str = "en") -> Dict[str, Any]:
        if self._provider is None:
            return dict(_EMPTY_LLM_RESPONSE)
        trimmed = history[-self._MAX_HISTORY_TURNS:] if len(history) > self._MAX_HISTORY_TURNS else history
        if len(history) != len(trimmed):
            logger.debug("History trimmed from %d to %d messages for token efficiency.",
                         len(history), len(trimmed))
        return self._provider.chat(history=trimmed, language=language)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
