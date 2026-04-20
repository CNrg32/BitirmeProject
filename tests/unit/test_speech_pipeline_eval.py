"""Smoke checks for ASR/TTS/translation runtime info and eval fixture presence."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


class TestRuntimeSnapshots:
    def test_asr_runtime_info_has_expected_keys(self):
        from services.asr_service import get_asr_runtime_info

        info = get_asr_runtime_info()
        for key in (
            "model_size",
            "device",
            "compute_type",
            "beam_size",
            "condition_on_previous_text",
            "vad_filter",
            "preprocess_16k_mono",
        ):
            assert key in info

    def test_translation_backend_name_is_str(self):
        from services.translation_service import get_translation_backend_name

        assert isinstance(get_translation_backend_name(), str)
        assert len(get_translation_backend_name()) >= 1

    def test_tts_runtime_info_has_expected_keys(self):
        from services.tts_service import get_tts_runtime_info

        info = get_tts_runtime_info()
        assert "edge_max_retries" in info
        assert "cache_max_entries" in info


class TestEvalFixture:
    def test_speech_pipeline_eval_json_loads(self):
        path = _ROOT / "tests" / "fixtures" / "speech_pipeline_eval.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "translation_pairs" in data
        assert len(data["translation_pairs"]) >= 1
