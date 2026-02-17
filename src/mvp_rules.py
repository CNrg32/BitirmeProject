
# Kural tabanli triaj siniflandirici ve slot cikarici
from __future__ import annotations
import json, re
from pathlib import Path
from typing import Any, Dict, List, Optional

_RULES = None
_DEFAULT_RULES_PATH = Path(__file__).resolve().parent / "mvp_regex_dictionary.json"


def load_rules(path: Optional[str] = None) -> Dict[str, Any]:
    global _RULES
    if _RULES is None:
        p = Path(path) if path else _DEFAULT_RULES_PATH
        _RULES = json.loads(p.read_text(encoding="utf-8"))
    return _RULES

def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()

def _has_any(text: str, kws: List[str]) -> bool:
    return any(kw in text for kw in kws)

def infer_category(text: str, rules: Dict[str, Any]) -> str:
    text = _norm(text)
    cats = rules["categories"]
    # Lower priority number = higher precedence
    ordered = sorted(cats.items(), key=lambda kv: kv[1].get("priority", 999))
    for name, spec in ordered:
        if _has_any(text, spec.get("keywords", [])):
            return name
    return "other"

def infer_triage(text: str, deaths: float = 0, potential_death: float = 0, false_alarm: float = 0, rules: Dict[str, Any] = None):
    rules = rules or load_rules()
    textn = _norm(text)

    tri = rules["triage"]
    crit = tri["critical"]
    nonu = tri["non_urgent"]

    if (deaths or 0) > 0 or (potential_death or 0) == 1 or _has_any(textn, crit.get("keywords", [])):
        return "CRITICAL"
    if (false_alarm or 0) == 1 or _has_any(textn, nonu.get("keywords", [])):
        return "NON_URGENT"
    if _has_any(textn, tri["urgent"].get("keywords", [])):
        return "URGENT"
    return tri.get("default", "URGENT")

def extract_slots(text: str, rules: Dict[str, Any] = None) -> Dict[str, Any]:
    rules = rules or load_rules()
    textn = _norm(text)
    slots = {}

    for slot_name, patterns in rules.get("slot_extraction_regex", {}).items():
        matches = []
        for pat in patterns:
            for m in re.finditer(pat, textn):
                if m.groups():
                    matches.append(m.group(1))
                else:
                    matches.append(m.group(0))
        if matches:
            # Basic postprocess
            if slot_name == "age":
                try:
                    slots["age"] = int(matches[0])
                except:
                    slots["age"] = matches[0]
            elif slot_name == "severity_1_10":
                try:
                    v = int(matches[0])
                    if 0 <= v <= 10:
                        slots["severity_1_10"] = v
                    else:
                        slots["severity_1_10"] = matches[0]
                except:
                    slots["severity_1_10"] = matches[0]
            elif slot_name == "duration_minutes":
                # Convert hours to minutes if pattern captured hours
                val = matches[0]
                # If original match came from hours pattern, it still captures number
                # We'll try to detect hours by searching 'hour' near it.
                # Simpler: if 'hour' in nearby text, multiply by 60.
                num = None
                try:
                    num = int(val)
                except:
                    pass
                if num is not None:
                    if "hour" in textn or "hr" in textn:
                        slots["duration_minutes"] = num * 60
                    else:
                        slots["duration_minutes"] = num
                else:
                    slots["duration_minutes"] = val
            elif slot_name == "red_flags":
                slots["red_flags"] = sorted(set(matches))
            else:
                slots[slot_name] = matches[0] if len(matches)==1 else matches
    return slots

def predict_mvp(text: str, deaths: float = 0, potential_death: float = 0, false_alarm: float = 0) -> Dict[str, Any]:
    rules = load_rules()
    category = infer_category(text, rules)
    triage = infer_triage(text, deaths=deaths, potential_death=potential_death, false_alarm=false_alarm, rules=rules)
    slots = extract_slots(text, rules)
    red_flags = slots.get("red_flags", [])
    return {
        "category": category,
        "triage_level": triage,
        "red_flags": red_flags,
        "slots": slots,
    }
