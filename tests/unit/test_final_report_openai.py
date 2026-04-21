"""Unit tests for final_report_openai (optional GPT handoff report)."""
from services import final_report_openai as fr


def test_generate_final_report_skipped_when_flag_off(monkeypatch):
    monkeypatch.delenv("USE_OPENAI_FINAL_REPORT", raising=False)
    assert fr.generate_final_report_openai(
        triage_result={"triage_level": "URGENT", "category": "medical"},
        slots={"chief_complaint": "chest pain"},
    ) is None


def test_generate_final_report_skipped_without_api_key(monkeypatch):
    monkeypatch.setenv("USE_OPENAI_FINAL_REPORT", "true")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    out = fr.generate_final_report_openai(
        triage_result={"triage_level": "URGENT", "category": "medical"},
        slots={},
    )
    assert out is None
