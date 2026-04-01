from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .session import Session

_SRC = Path(__file__).resolve().parent.parent

# Generic required first (chief complaint)
REQUIRED_SLOTS = [
    ("chief_complaint", "Can you describe what happened or what is wrong?", 1),
    ("age", "How old is the person who needs help?", 2),
    ("severity_1_10", "On a scale of 1-10, how severe is the situation?", 3),
]

OPTIONAL_SLOTS = [
    ("sex", "Is the person male or female?", 4),
    ("duration_minutes", "How long has this been going on (in minutes)?", 5),
    ("location_hint", "Where are you right now? (home, street, car, etc.)", 6),
]

EMERGENCY_QUESTIONS: Dict[str, List[Tuple[str, str]]] = {
    "medical": [
        ("breathing", "Is the person breathing right now?"),
        ("consciousness", "Are they conscious? Can they open their eyes or respond?"),
        ("duration_medical", "How long have they been in this state?"),
        ("bleeding", "Is there any bleeding? If yes, where and how much?"),
    ],
    "fire": [
        ("trapped", "Is anyone trapped inside the building?"),
        ("fire_size", "How large is the fire? One room or spreading?"),
        ("smoke_inhalation", "Is anyone having trouble breathing from smoke?"),
    ],
    "crime": [
        ("assailant_present", "Is the assailant still there?"),
        ("weapon", "Do you see any weapon?"),
        ("injuries", "Is anyone injured?"),
    ],
    "other": [
        ("location_hint", "Where are you right now? (home, street, car, etc.)"),
        ("duration_minutes", "How long has this been going on?"),
    ],
}

CONFIDENCE_THRESHOLD = 0.60
MAX_QUESTION_ROUNDS_NORMAL = 4
MAX_QUESTION_ROUNDS_MEDIUM_PANIC = 3
MAX_QUESTION_ROUNDS_HIGH_PANIC = 2


def _infer_category(session: Session) -> str:
    if session.triage_result and session.triage_result.get("category"):
        return session.triage_result["category"]
    try:
        from mvp_rules import infer_category as _infer, load_rules
        rules = load_rules(str(_SRC / "mvp_regex_dictionary.json"))
        return _infer(session.text_en_accumulated or "", rules)
    except Exception:
        return "other"


def _get_max_rounds(session: Session) -> int:
    sent = session.sentiment_result
    if not sent:
        return MAX_QUESTION_ROUNDS_NORMAL
    panic = (sent.get("panic_level") or "").lower()
    if panic == "high":
        return MAX_QUESTION_ROUNDS_HIGH_PANIC
    if panic == "medium":
        return MAX_QUESTION_ROUNDS_MEDIUM_PANIC
    return MAX_QUESTION_ROUNDS_NORMAL


def get_missing_required_slots(session: Session) -> List[Tuple[str, str]]:
    missing = []
    for key, question, _priority in REQUIRED_SLOTS:
        if key not in session.collected_slots and key not in session.asked_questions:
            missing.append((key, question))
    return missing


def get_missing_optional_slots(session: Session) -> List[Tuple[str, str]]:
    missing = []
    for key, question, _priority in OPTIONAL_SLOTS:
        if key not in session.collected_slots and key not in session.asked_questions:
            missing.append((key, question))
    return missing


def _get_next_category_question(session: Session) -> Optional[Tuple[str, str]]:
    category = _infer_category(session)
    questions = EMERGENCY_QUESTIONS.get(category, EMERGENCY_QUESTIONS["other"])
    for key, question in questions:
        if key not in session.asked_questions and key not in session.collected_slots:
            return (key, question)
    return None


def increment_slot_attempt(session: Session, slot_key: str) -> None:
    """
    FAZ 5: Increment attempt counter for a slot.
    Used to track 2-attempt rule: if a slot can't be filled after 2 tries, mark Unknown.
    """
    if slot_key not in session.slot_attempt_counts:
        session.slot_attempt_counts[slot_key] = 0
    session.slot_attempt_counts[slot_key] += 1


def get_slot_attempt_count(session: Session, slot_key: str) -> int:
    """Get the attempt count for a slot (default 0)."""
    return session.slot_attempt_counts.get(slot_key, 0)


def should_skip_slot_after_max_attempts(session: Session, slot_key: str, max_attempts: int = 2) -> bool:
    """
    FAZ 5: Check if a slot should be skipped due to max attempts exceeded.
    Returns True if attempt count >= max_attempts (default 2).
    """
    return get_slot_attempt_count(session, slot_key) >= max_attempts


def mark_slot_unknown_and_skip(session: Session, slot_key: str) -> None:
    """
    FAZ 5: After 2 failed attempts, mark slot as 'Unknown' and move to next.
    This prevents infinite loops of asking the same question.
    """
    session.collected_slots[slot_key] = "Unknown"
    # Also add to asked_questions to prevent re-asking
    session.asked_questions.add(slot_key)


def decide_next_action(session: Session) -> Dict[str, Any]:
    """
    Decide the next action for the dialog.
    
    FAZ 5: Implements 2-attempt rule:
    - Each slot gets max 2 attempts
    - After 2 failed attempts, mark as "Unknown" and move to next slot
    - Witness mode skips unnecessary/non-witness-related slots
    """
    assistant_turns = len([m for m in session.messages if m.get("role") == "assistant"])
    max_rounds = _get_max_rounds(session)

    if assistant_turns >= max_rounds:
        return {"action": "run_triage", "reason": "max_rounds_reached"}

    if "chief_complaint" not in session.collected_slots:
        text = (session.text_en_accumulated or "").strip()
        min_words = 3
        if text and len(text.split()) >= min_words:
            session.collected_slots["chief_complaint"] = text
        else:
            return {
                "action": "ask_question",
                "question_key": "chief_complaint",
                "question_en": "Can you describe what happened or what is wrong?",
                "reason": "missing_chief_complaint",
            }

    missing_req = get_missing_required_slots(session)
    if missing_req:
        key, question = missing_req[0]
        
        # FAZ 5: 2-Attempt Rule Check
        # If this slot already has 2+ attempts without being filled, mark Unknown and skip
        if should_skip_slot_after_max_attempts(session, key, max_attempts=2):
            mark_slot_unknown_and_skip(session, key)
            # Continue to next missing slot by recursing (or fallthrough to next check)
            return decide_next_action(session)
        
        return {
            "action": "ask_question",
            "question_key": key,
            "question_en": question,
            "reason": f"missing_required_slot:{key}",
        }

    category_question = _get_next_category_question(session)
    if category_question and assistant_turns < max_rounds - 1:
        key, question = category_question
        
        # FAZ 5: 2-Attempt Rule Check (same as required slots)
        if should_skip_slot_after_max_attempts(session, key, max_attempts=2):
            mark_slot_unknown_and_skip(session, key)
            return decide_next_action(session)
        
        return {
            "action": "ask_question",
            "question_key": key,
            "question_en": question,
            "reason": f"category_question:{key}",
        }

    if session.triage_result:
        conf = session.triage_result.get("confidence") or 1.0
        if conf < CONFIDENCE_THRESHOLD:
            missing_opt = get_missing_optional_slots(session)
            if missing_opt and assistant_turns < max_rounds - 1:
                key, question = missing_opt[0]
                
                # FAZ 5: 2-Attempt Rule Check (optional slots too)
                if should_skip_slot_after_max_attempts(session, key, max_attempts=2):
                    mark_slot_unknown_and_skip(session, key)
                    return decide_next_action(session)
                
                return {
                    "action": "ask_question",
                    "question_key": key,
                    "question_en": question,
                    "reason": f"low_confidence_optional:{key}",
                }

    if session.triage_result is None:
        return {"action": "run_triage", "reason": "all_required_slots_collected"}

    return {"action": "complete", "reason": "sufficient_info"}


def merge_slots(session: Session, new_slots: Dict[str, Any]) -> None:
    for k, v in new_slots.items():
        if v is not None and v != "" and v != []:
            session.collected_slots[k] = v
