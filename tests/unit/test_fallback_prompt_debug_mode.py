from __future__ import annotations

import importlib
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def test_debug_fallback_mode_injects_prompt(monkeypatch):
    monkeypatch.setenv("DEBUG_FALLBACK_MODE", "true")

    import services.llm_prompt_config as cfg
    importlib.reload(cfg)

    prompt = cfg.build_system_prompt_with_few_shot(
        base_system_prompt="BASE",
        language_hint="English",
        task="dialog",
    )
    assert "DEBUG MODE ACTIVE" in prompt
    assert "fallback" in prompt.lower()
