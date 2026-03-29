from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


from services import llm_service


def test_parse_llm_json_extracts_json_block():
    raw = """```json
    {"response_text":"ok","extracted_slots":{"age":30},"triage_level":"URGENT","category":"medical","is_complete":false,"red_flags":[]}
    ```"""
    parsed = llm_service._parse_llm_json(raw)
    assert parsed["response_text"] == "ok"
    assert parsed["extracted_slots"]["age"] == 30
    assert parsed["category"] == "medical"


def test_build_local_prompt_includes_roles_and_language():
    history = [
        {"role": "user", "text": "Help me"},
        {"role": "assistant", "text": "What happened?"},
    ]
    prompt = llm_service._build_local_prompt(history, "tr")
    assert "Language: tr" in prompt
    assert "USER: Help me" in prompt
    assert "ASSISTANT: What happened?" in prompt


def test_llm_service_uses_local_provider_when_configured(monkeypatch):
    class FakeLocalProvider:
        def __init__(self, model_dir: str) -> None:
            self.model_name = f"local/{model_dir}"

        @property
        def is_ready(self):
            return True

        def chat(self, history, language):
            return {
                "response_text": "ok",
                "extracted_slots": {},
                "triage_level": "URGENT",
                "category": "other",
                "is_complete": False,
                "red_flags": [],
            }

    monkeypatch.setenv("LOCAL_CHATBOT_MODEL_DIR", "out_models/chatbot_finetuned")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(llm_service, "_LocalFineTunedProvider", FakeLocalProvider)

    svc = llm_service.LLMService()
    assert svc.is_available is True
    assert svc.MODEL.startswith("local/")
