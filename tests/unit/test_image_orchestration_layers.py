from __future__ import annotations

from unittest.mock import patch


def _patched_store():
    from orchestrator.session import SessionStore

    store = SessionStore(ttl_seconds=3600)
    return store, patch("orchestrator.session.get_session_store", return_value=store), patch(
        "orchestrator.orchestrator.get_session_store", return_value=store
    )


def test_image_only_critical_dispatches():
    from orchestrator.orchestrator import handle_message

    store, patch_session, patch_orchestrator = _patched_store()
    session = store.create(language="tr")
    image_result = {
        "available": True,
        "classification": {
            "detected_class": "RoadAccidents",
            "confidence": 0.91,
            "top3": [],
            "dispatch_units": ["Police", "Ambulance"],
            "mapped_category": "medical",
        },
        "consistency": None,
        "summary": "Road accident detected.",
        "image_quality": {"usable": True},
        "visual_triage": {
            "category": "medical",
            "triage_level": "CRITICAL",
            "action": "EARLY_DISPATCH",
            "visual_flags": ["vehicle_crash", "possible_injury"],
        },
    }

    with patch_session, patch_orchestrator, patch("services.image_service.analyze_image", return_value=image_result):
        response = handle_message(session.session_id, image_bytes=b"image")

    assert response["dispatch_status"] == "DISPATCHED"
    assert response["triage_result"]["triage_level"] == "CRITICAL"
    assert response["image_analysis"]["visual_triage"]["action"] == "EARLY_DISPATCH"


def test_image_only_model_unavailable_goes_to_fallback_after_two_attempts():
    from orchestrator.orchestrator import handle_message

    store, patch_session, patch_orchestrator = _patched_store()
    session = store.create(language="tr")
    image_result = {
        "available": False,
        "classification": None,
        "consistency": None,
        "summary": "Image analysis unavailable - model not loaded.",
        "image_quality": {"usable": True},
        "visual_triage": {
            "category": "other",
            "triage_level": "URGENT",
            "action": "MANUAL_FALLBACK",
            "requires_manual_fallback": True,
            "fallback_options": [{"label": "Tıbbi", "category": "medical"}],
            "visual_flags": [],
        },
    }

    with patch_session, patch_orchestrator, patch("services.image_service.analyze_image", return_value=image_result):
        first = handle_message(session.session_id, image_bytes=b"image")
        second = handle_message(session.session_id, image_bytes=b"image")

    assert first["dispatch_status"] == "PENDING"
    assert second["dispatch_status"] == "FALLBACK_PENDING"
    assert "Tıbbi" in second["assistant_text"]


def test_post_dispatch_image_is_update_not_new_case():
    from orchestrator.orchestrator import handle_message

    store, patch_session, patch_orchestrator = _patched_store()
    session = store.create(language="en")
    session.is_complete = True
    session.dispatch_status = "DISPATCHED"
    session.dispatch_target = "medical"
    session.triage_result = {
        "category": "medical",
        "triage_level": "URGENT",
        "confidence": 0.8,
        "red_flags": [],
        "slots": {},
    }
    image_result = {
        "available": True,
        "classification": {
            "detected_class": "Arson",
            "confidence": 0.9,
            "top3": [],
            "dispatch_units": ["Fire Department"],
            "mapped_category": "fire",
        },
        "consistency": None,
        "summary": "Fire detected.",
        "image_quality": {"usable": True},
        "visual_triage": {
            "category": "fire",
            "triage_level": "CRITICAL",
            "action": "EARLY_DISPATCH",
            "visual_flags": ["fire_visible"],
        },
    }

    with patch_session, patch_orchestrator, patch("services.image_service.analyze_image", return_value=image_result):
        response = handle_message(session.session_id, image_bytes=b"image")

    assert response["assistant_text"].startswith("Update received")
    assert response["triage_result"]["triage_level"] == "CRITICAL"
    assert session.image_updates
