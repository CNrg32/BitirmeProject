from __future__ import annotations

from typing import Any, Dict, List, Optional


_TRIAGE_EMOJI = {
    "CRITICAL":    "[!!!]",
    "URGENT":      "[!!]",
    "NON_URGENT":  "[!]",
}

_TRIAGE_HEADLINE = {
    "CRITICAL":   "CRITICAL EMERGENCY – Immediate Action Required",
    "URGENT":     "URGENT – Prompt Medical / Emergency Attention Needed",
    "NON_URGENT": "Non-Urgent – Situation Under Control",
}

_CATEGORY_UNIT = {
    "medical": "Emergency Medical Services (EMS / Ambulance)",
    "crime":   "Police / Law Enforcement",
    "fire":    "Fire Department",
    "other":   "General Emergency Services",
}

_ACTION_RESPONSES: Dict[tuple, Dict[str, Any]] = {
    ("medical", "CRITICAL"): {
        "dispatch": "Ambulance has been dispatched and is on the way.",
        "instructions": [
            "Stay calm and stay on the line.",
            "Keep the person's airway clear.",
            "If they are not breathing, perform chest compressions if you know how.",
            "Stay with the person until the ambulance arrives.",
        ],
    },
    ("medical", "URGENT"): {
        "dispatch": "Emergency medical help has been notified and is on the way.",
        "instructions": [
            "Stay calm.",
            "Keep the person comfortable and do not move them unless necessary.",
            "Stay on the line until help arrives.",
        ],
    },
    ("medical", "NON_URGENT"): {
        "dispatch": "Your situation has been noted. You may request non-urgent medical advice or visit a clinic.",
        "instructions": ["If symptoms worsen, call back or go to the nearest emergency room."],
    },
    ("crime", "CRITICAL"): {
        "dispatch": "Police have been dispatched and are on the way.",
        "instructions": [
            "Get to a safe place if you can.",
            "Avoid confronting the assailant.",
            "If possible, lock doors and stay quiet.",
            "Stay on the line until police arrive.",
        ],
    },
    ("crime", "URGENT"): {
        "dispatch": "Police have been notified and are responding.",
        "instructions": [
            "Stay in a safe location.",
            "Do not approach the suspect.",
            "Stay on the line for further instructions.",
        ],
    },
    ("crime", "NON_URGENT"): {
        "dispatch": "Your report has been logged. Police will follow up as appropriate.",
        "instructions": ["If the situation becomes urgent, call back immediately."],
    },
    ("fire", "CRITICAL"): {
        "dispatch": "Fire department has been dispatched and is on the way.",
        "instructions": [
            "Leave the building immediately.",
            "Do not use the elevator; use the stairs.",
            "If there is smoke, stay low to the ground.",
            "Close doors behind you as you leave but do not lock them.",
        ],
    },
    ("fire", "URGENT"): {
        "dispatch": "Fire services have been notified.",
        "instructions": [
            "Evacuate the area if it is safe to do so.",
            "Do not re-enter until it is declared safe.",
        ],
    },
    ("fire", "NON_URGENT"): {
        "dispatch": "Your report has been recorded. Fire services will advise if needed.",
        "instructions": ["If the situation worsens, call back."],
    },
    ("other", "CRITICAL"): {
        "dispatch": "Emergency services have been dispatched.",
        "instructions": [
            "Stay calm and stay on the line.",
            "Follow any instructions from the operator.",
        ],
    },
    ("other", "URGENT"): {
        "dispatch": "Help has been notified and is on the way.",
        "instructions": ["Stay on the line and follow instructions."],
    },
    ("other", "NON_URGENT"): {
        "dispatch": "Your call has been logged. Assistance will be arranged as needed.",
        "instructions": ["If the situation changes, call back."],
    },
}


def compose_report(
    triage_result: Dict[str, Any],
    slots: Dict[str, Any],
    image_analysis: Optional[Dict[str, Any]] = None,
    language: str = "en",
) -> str:
    level = triage_result.get("triage_level", "URGENT")
    category = triage_result.get("category", "other")
    confidence = triage_result.get("confidence")
    red_flags: List[str] = triage_result.get("red_flags", [])

    lines: List[str] = []

    action_key = (category, level)
    action = _ACTION_RESPONSES.get(action_key) or _ACTION_RESPONSES.get(
        (category, "URGENT")
    ) or _ACTION_RESPONSES.get(("other", level))
    if action:
        lines.append(action["dispatch"])
        for instr in action.get("instructions", []):
            lines.append(f"  • {instr}")
        lines.append("")

    marker = _TRIAGE_EMOJI.get(level, "")
    headline = _TRIAGE_HEADLINE.get(level, level)
    lines.append(f"{marker} {headline}")
    lines.append("")

    unit = _CATEGORY_UNIT.get(category, category)
    lines.append(f"Category   : {category.upper()}")
    lines.append(f"Dispatch to: {unit}")
    lines.append("")

    detail_lines = _format_slots(slots)
    if detail_lines:
        lines.append("--- Patient / Incident Details ---")
        lines.extend(detail_lines)
        lines.append("")

    if red_flags:
        lines.append("RED FLAGS:")
        for rf in red_flags:
            lines.append(f"  - {rf}")
        lines.append("")

    if confidence is not None:
        lines.append(f"Model confidence: {confidence:.0%}")

    if image_analysis and image_analysis.get("available"):
        lines.append("")
        lines.append("--- Image Analysis ---")

        classification = image_analysis.get("classification")
        if classification:
            lines.append(f"  Scene detected : {classification.get('detected_class', 'N/A')}")
            img_conf = classification.get("confidence", 0)
            lines.append(f"  Confidence     : {img_conf:.0%}")

            dispatch = classification.get("dispatch_units", [])
            if dispatch:
                lines.append(f"  Image dispatch : {', '.join(dispatch)}")

            top3 = classification.get("top3", [])
            if len(top3) > 1:
                alt = ", ".join(
                    f"{t['class']} ({t['confidence']:.0%})" for t in top3[1:]
                )
                lines.append(f"  Alternatives   : {alt}")

        consistency = image_analysis.get("consistency")
        if consistency:
            lines.append("")
            lines.append("  Cross-reference (Image vs. Text):")
            lines.append(f"    Status : {consistency.get('consistency_detail', 'N/A')}")
            lines.append(f"    Score  : {consistency.get('consistency_score', 0):.0%}")

            if consistency.get("possible_fake"):
                lines.append("    [WARNING] Image may not match the reported emergency!")

            for note in consistency.get("risk_notes", []):
                lines.append(f"    - {note}")

        summary = image_analysis.get("summary")
        if summary:
            lines.append("")
            lines.append(f"  Summary: {summary}")

        lines.append("")

    if level == "CRITICAL":
        lines.append("")
        lines.append(
            "** This situation has been classified as CRITICAL. "
            "Please contact emergency services immediately. **"
        )

    return "\n".join(lines)


def _format_slots(slots: Dict[str, Any]) -> List[str]:
    mapping = {
        "caller_name":      "Caller Name",
        "chief_complaint":  "Complaint",
        "age":              "Age",
        "sex":              "Sex",
        "severity_1_10":    "Severity (1-10)",
        "duration_minutes": "Duration (min)",
        "location_hint":    "Location",
    }
    lines = []
    for key, label in mapping.items():
        val = slots.get(key)
        if val is not None and val != "" and val != []:
            lines.append(f"  {label}: {val}")

    # GPS koordinatları (4. konum entegrasyonu)
    lat = slots.get("latitude")
    lon = slots.get("longitude")
    if lat is not None and lon is not None:
        lines.append(f"  GPS       : {lat:.6f}, {lon:.6f}")
        lines.append(f"  Maps      : https://maps.google.com/?q={lat:.6f},{lon:.6f}")

    return lines
