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
) -> str:
    """
    Sistem prompt'una few-shot örnekleri ve isteğe bağlı custom talimatları ekler.
    Fine-tuning olmadan modeli istediğiniz cevap formatına yönlendirir.
    """
    parts = [base_system_prompt]
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


def build_system_prompt_decision_tree(
    base_decision_tree_prompt: str,
    language_hint: str,
) -> str:
    """System prompt for decision-tree mode: steps only, no response_text. Appends language hint."""
    return base_decision_tree_prompt.rstrip() + f"\n\nOutput language context: {language_hint}. Slot values and red_flags may be in user's language; category/triage_level/keys must be in English."
