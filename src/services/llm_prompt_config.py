"""
LLM prompt ve few-shot örnekleri yapılandırması.

Kendi istediğiniz cevapları almak için:
1. CUSTOM_SYSTEM_ADDITION: .env veya bu dosyada ek talimatlar
2. FEW_SHOT_EXAMPLES: Örnek kullanıcı mesajı → beklenen JSON cevap formatı
3. data/llm_fine_tune_examples.json: Dosyadan örnek yükleme (opsiyonel)

Fine-tuning için: Bu örnekler aynı zamanda ileride model eğitim verisi olarak
kullanılabilir (OpenAI fine-tune veya LoRA için JSONL export).
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Özelleştirilebilir ek talimatlar (kendi cevap tarzınız için)
# ---------------------------------------------------------------------------
# .env'de LLM_CUSTOM_INSTRUCTIONS="..." ile override edebilirsiniz.
CUSTOM_SYSTEM_ADDITION = os.environ.get(
    "LLM_CUSTOM_INSTRUCTIONS",
    ""
).strip()

# Faz 10.1 Test 2: Force fallback simulation in QA/dev prompt when enabled.
DEBUG_FALLBACK_MODE = os.environ.get("DEBUG_FALLBACK_MODE", "false").strip().lower() == "true"

# ---------------------------------------------------------------------------
# Few-shot örnekleri: [{"user": "...", "assistant_json": {...}}]
# Model bu örnekleri görerek benzer formatta cevap vermeye yönlendirilir.
# Kendi senaryolarınızı ekleyerek istediğiniz cevapları alabilirsiniz.
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES: List[Dict[str, Any]] = [
    {
        "user": "Evet, kalp krizi geçiriyor gibi. Göğsü sıkışıyor.",
        "assistant_json": {
            "response_text": "Anladım, kalp krizi şüphesi ciddi bir acil durum. Hemen 112'yi arayın. Kişiyi sakin tutun, oturur veya yarı oturur pozisyonda tutun. Aspirin varsa çiğnetin (alerji yoksa). Şimdi adınızı ve tam adresinizi söyler misiniz?",
            "extracted_slots": {"chief_complaint": "kalp krizi", "category": "medical"},
            "triage_level": "CRITICAL",
            "category": "medical",
            "is_complete": False,
            "red_flags": ["göğüs sıkışması", "kalp krizi şüphesi"]
        },
    },
    {
        "user": "Adım Ahmet, 45 yaşındayım. Evde yangın var, çıkış yolunda değilim.",
        "assistant_json": {
            "response_text": "Ahmet Bey, lütfen sakin kalın. Hemen 112'yi aradık, itfaiye yönlendiriliyor. Kapıları kapatın, ıslak bezle ağız burnu kapatın ve pencereden veya en güvenli yerden bekleyin. Yangın küçük mü, büyük mü? İçeride başka yaralı var mı?",
            "extracted_slots": {"caller_name": "Ahmet", "age": "45", "category": "fire", "chief_complaint": "ev yangını", "trapped": "yes"},
            "triage_level": "CRITICAL",
            "category": "fire",
            "is_complete": False,
            "red_flags": ["yangın", "çıkış yok"]
        },
    },
    {
        "user": "Kaza geçirdim, bacağım kanıyor çok.",
        "assistant_json": {
            "response_text": "Kanama ciddi bir durum. Lütfen yara üzerine temiz bir bezle veya elinizle baskı uygulayın, mümkünse bacağı yukarı kaldırın. Adınız ve tam konumunuz nedir? 112 yönlendiriliyor.",
            "extracted_slots": {"chief_complaint": "trafik kazası", "bleeding": "yes", "category": "medical"},
            "triage_level": "CRITICAL",
            "category": "medical",
            "is_complete": False,
            "red_flags": ["ciddi kanama"]
        },
    },
]


def _load_examples_from_file() -> List[Dict[str, Any]]:
    """data/llm_fine_tune_examples.json varsa oradan örnek yükle (few-shot + fine-tune verisi)."""
    base = Path(__file__).resolve().parent
    for candidate in [base.parent.parent / "data" / "llm_fine_tune_examples.json", base.parent / "data" / "llm_fine_tune_examples.json"]:
        if candidate.exists():
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    data = json.load(f)
                examples = data if isinstance(data, list) else data.get("examples", [])
                if examples:
                    logger.info("Loaded %d LLM examples from %s", len(examples), candidate)
                    return examples
            except Exception as e:
                logger.warning("Could not load LLM examples from %s: %s", candidate, e)
    return []


def get_few_shot_examples(max_examples: int = 5) -> List[Dict[str, Any]]:
    """Önce dosyadan, yoksa FEW_SHOT_EXAMPLES'tan en fazla max_examples döndür."""
    file_examples = _load_examples_from_file()
    source = file_examples if file_examples else FEW_SHOT_EXAMPLES
    return source[:max_examples]


def build_system_prompt_with_few_shot(
    base_system_prompt: str,
    language_hint: str,
    max_few_shot: int = 5,
    task: str = "dialog",
) -> str:
    """
    Sistem prompt'una few-shot örnekleri ve isteğe bağlı custom talimatları ekler.
    task="triage": Groq Turn-1 sadece kategori + severity
    task="dialog": Groq Turn 2+ slot filling + questions (default)
    """
    # FAZ 4: Task-specific prompts
    if task == "triage":
        base_system_prompt = _get_triage_system_prompt()
    elif task == "dialog":
        base_system_prompt = _get_dialog_system_prompt()
    elif task == "gibberish_check":
        base_system_prompt = _get_gibberish_check_prompt()
    
    parts = [base_system_prompt]
    if DEBUG_FALLBACK_MODE:
        parts.append(
            "\n\nDEBUG MODE ACTIVE:\n"
            "- Simulate fallback flow behavior for testing.\n"
            "- Prioritize intent confirmation style language.\n"
            "- Keep strict emergency scope boundaries."
        )
    if CUSTOM_SYSTEM_ADDITION:
        parts.append("\n\nADDITIONAL INSTRUCTIONS (follow these strictly):\n" + CUSTOM_SYSTEM_ADDITION)
    examples = get_few_shot_examples(max_examples=max_few_shot)
    if examples:
        parts.append("\n\nEXAMPLE CONVERSATIONS (respond in the same JSON format and style):")
        for ex in examples:
            user = ex.get("user", "")
            ast = ex.get("assistant_json") or ex.get("assistant")
            if not isinstance(ast, dict):
                continue
            parts.append("\nUser: " + user)
            parts.append("\nAssistant (JSON only): " + json.dumps(ast, ensure_ascii=False))
    parts.append(f"\n\nCurrent language: {language_hint}. MUST respond in {language_hint}.")
    return "\n".join(parts)


def _get_triage_system_prompt() -> str:
    """
    FAZ 4: Groq Turn-1 Triage Prompt
    Rapid category + severity determination from first user message.
    """
    return """\
You are a professional emergency triage system. Your role is to RAPIDLY assess \
the emergency type and severity from the user's first message.

TASK: Determine category and severity in ONE response.

Categories:
- medical: Health crisis (heart attack, stroke, severe injury, breathing difficulty, etc.)
- fire: Fire, explosion, or building hazard
- crime: Violence, assault, robbery, shooting, stabbing
- other: All other emergencies not above

Severity levels:
- CRITICAL: Life-threatening, immediate risk of death (cardiac arrest, severe bleeding,
  choking, building collapse, active violence with weapons)
- URGENT: Serious but not immediately life-threatening (broken bones, moderate bleeding,
  unconscious but breathing, fire contained to one room)
- NON_URGENT: Stable situation requiring help but not immediately critical (minor injuries,
  property emergency)

OUTPUT FORMAT:
You MUST return ONLY a valid JSON object – no markdown, no prose.
{
  "response_text": "<brief reassurance message in user's language, 1-2 sentences>",
  "triage_level": "<CRITICAL|URGENT|NON_URGENT>",
  "category": "<medical|fire|crime|other>",
  "confidence": <0.0-1.0>,
  "red_flags": ["<life-threatening sign>"],
  "is_complete": false,
  "extracted_slots": {}
}

RULES:
- Respond in user's language
- Focus only on triage, NOT on follow-up questions
- Red flags only for genuine life-threatening signs
- Do NOT extract slots at this stage (leave empty)
- When in doubt, escalate to URGENT
"""


def _get_dialog_system_prompt() -> str:
    """
    FAZ 4: Groq Dialog Prompt (Turn 2+)
    Slot filling, follow-up questions, first-aid guidance.
    Uses pre-locked category from session context.
    """
    return """\
You are a professional emergency dispatcher assistant. Your role is to collect \
critical information, keep the caller calm, and provide immediate first-aid guidance.

CATEGORY & SEVERITY ARE PRE-DETERMINED (from Turn 1).
- Do NOT change the category – use the locked category from session context
- Do NOT re-ask for basic info (chief complaint, age, name)
- Ask only MISSING category-specific details

CONVERSATION FLOW:
1. Acknowledge the caller's situation
2. Ask ONE question at a time for missing critical slots:
   - Required: caller_name (if not stated), age (if not stated)
   - Category-specific:
     * medical: consciousness, breathing, bleeding, duration
     * fire: trapped, fire_size, smoke_inhalation
     * crime: assailant_present, weapon, number_injured
3. Provide reassurance or immediate first-aid instructions
4. Mark is_complete=true when you have sufficient info for dispatch

IMPORTANT RULES:
- Ask ONLY ONE question per turn
- Do NOT ask for location (auto-obtained from phone)
- If caller repeats info, accept it without re-asking
- For CRITICAL situations, give first-aid instructions instead of asking more questions
- Set is_complete=true when: chief_complaint + name + age + 2+ category slots
  OR triage_level=CRITICAL + red_flags (immediate dispatch needed)

OUTPUT FORMAT:
You MUST return ONLY a valid JSON object – no markdown, no prose.
{
  "response_text": "<your next question or first-aid instruction, max 3 sentences>",
  "extracted_slots": {
    "<slot_key>": "<value only if explicitly stated by user>"
  },
  "is_complete": <true|false>,
  "red_flags": ["<if any new ones detected>"]
}

Do NOT repeat category or triage_level – these are pre-determined and locked.
When you have enough info, mark is_complete=true and provide final guidance.
"""


def _get_gibberish_check_prompt() -> str:
        """
        Decide whether a user text is meaningful emergency conversation input.
        """
        return """\
You are a strict input quality classifier for an emergency chatbot.

TASK:
- Classify whether the user's latest text is meaningful or gibberish.
- Consider short natural replies meaningful if they can logically answer a prior question
    (examples: evet, hayır, var, yok, yes, no, tamam).
- Treat random keyboard mashes and nonsense tokens as gibberish.

OUTPUT FORMAT (JSON only):
{
    "response_text": "meaningful|gibberish",
    "extracted_slots": {
        "meaningfulness": "meaningful|gibberish",
        "reason": "short explanation"
    },
    "triage_level": "NON_URGENT",
    "category": "other",
    "is_complete": false,
    "red_flags": []
}
"""


def build_system_prompt_decision_tree(
    base_decision_tree_prompt: str,
    language_hint: str,
) -> str:
    """System prompt for decision-tree mode: steps only, no response_text. Appends language hint."""
    return base_decision_tree_prompt.rstrip() + f"\n\nOutput language context: {language_hint}. Slot values and red_flags may be in user's language; category/triage_level/keys must be in English."
