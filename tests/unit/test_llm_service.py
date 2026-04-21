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


def test_parse_llm_json_invalid_returns_empty_template():
    parsed = llm_service._parse_llm_json("not json {")
    assert parsed["response_text"] == ""
    assert parsed["triage_level"] == "URGENT"


def test_local_triage_provider_not_ready_without_artifacts(tmp_path, monkeypatch):
    from services import triage_local_service as tls

    monkeypatch.setattr(tls, "DEFAULT_MODEL_DIR", tmp_path / "no-model")
    # Singleton cache'ini sifirla ki yeni (bos) model dir'i kullansin
    tls._local_service = None

    provider = llm_service._LocalTriageProvider()
    assert provider.is_ready is False
    out = provider.chat(history=[{"role": "user", "text": "x"}], language="tr")
    assert out["response_text"] == ""
    assert out["triage_level"] == "URGENT"


def test_llm_service_selects_local_when_artifacts_exist(tmp_path, monkeypatch):
    from services import triage_local_service as tls

    model_dir = tmp_path / "triage_xlmr"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"base_model_name":"xlm-roberta-base"}', encoding="utf-8")
    (model_dir / "model.pt").write_bytes(b"stub")

    monkeypatch.setattr(tls, "DEFAULT_MODEL_DIR", model_dir)
    tls._local_service = None
    monkeypatch.setenv("GROQ_API_KEY", "stub")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_FINE_TUNED_MODEL", raising=False)
    monkeypatch.delenv("TRIAGE_BACKEND", raising=False)

    # Groq init'ini stub'la — httpx cagrisi yapmasin
    class _FakeGroq:
        is_ready = True
        model = "stub-groq"

    monkeypatch.setattr(llm_service, "_GroqProvider", lambda api_key: _FakeGroq())

    svc = llm_service.LLMService()
    assert svc._triage_backend == "local"
    assert "triage:local/xlmr" in svc.MODEL
