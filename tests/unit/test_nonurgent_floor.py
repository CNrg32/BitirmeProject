"""mvp_rules.is_nonurgent_floor_match icin birim testler."""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import mvp_rules  # noqa: E402


def _reset_cache():
    mvp_rules._NONURGENT_RULES = None
    mvp_rules._NONURGENT_COMPILED_REGEXES = None


def test_nonurgent_floor_phrase_matches():
    _reset_cache()
    assert mvp_rules.is_nonurgent_floor_match("Pizza siparisi verebilir miyim?")


def test_nonurgent_floor_turkish_bbq_smell():
    _reset_cache()
    assert mvp_rules.is_nonurgent_floor_match(
        "Komsumdan mangal dumani kokusu geliyor, acil degil."
    )


def test_nonurgent_floor_gibberish_pattern():
    _reset_cache()
    assert mvp_rules.is_nonurgent_floor_match("asdfasdf")


def test_nonurgent_floor_short_message():
    _reset_cache()
    assert mvp_rules.is_nonurgent_floor_match("hi")


def test_nonurgent_floor_does_not_match_real_emergency():
    _reset_cache()
    assert not mvp_rules.is_nonurgent_floor_match(
        "Evim yaniyor, alevler salona ulasti, lutfen hemen gelin!"
    )
    assert not mvp_rules.is_nonurgent_floor_match(
        "Babam nefes almiyor, bilinci kapandi!"
    )


def test_nonurgent_floor_noise_complaint():
    _reset_cache()
    assert mvp_rules.is_nonurgent_floor_match(
        "Komsumdan gurultu sikayeti var, gece yarisi muzik calisiyor."
    )
