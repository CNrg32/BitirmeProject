#!/usr/bin/env python3
"""
Generate emergency triage few-shot/fine-tune examples.

Writes:
  data/llm_fine_tune_examples.json

Usage:
  python scripts/generate_llm_finetune_examples.py --count 180
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = ROOT / "data" / "llm_fine_tune_examples.json"


TR_NAMES = [
    "Ahmet",
    "Ayse",
    "Mehmet",
    "Fatma",
    "Can",
    "Elif",
    "Merve",
    "Kerem",
    "Yusuf",
    "Seda",
]

EN_NAMES = [
    "John",
    "Sarah",
    "Michael",
    "Emily",
    "David",
    "Anna",
]

MEDICAL_CASES = [
    ("Babam nefes alamiyor, morariyor.", "nefes darligi", ["nefes alamama"], "CRITICAL"),
    ("Annem bayildi, cevap vermiyor.", "bilinc kaybi", ["bilinc kapali"], "CRITICAL"),
    ("Gogsumde siddetli agri var, terliyorum.", "gogus agrisi", ["kalp krizi suphe"], "CRITICAL"),
    ("Trafik kazasi oldu, bacagi cok kaniyor.", "ciddi kanama", ["asiri kanama"], "CRITICAL"),
    ("Dusme sonrasi kolunu oynatamiyor.", "kirik suphe", [], "URGENT"),
    ("Atesi var ve nefesi hizli.", "ates ve solunum sikintisi", [], "URGENT"),
]

FIRE_CASES = [
    ("Evde mutfakta yangin cikti, duman cok yogun.", "ev yangini", ["yogun duman"], "CRITICAL"),
    ("Apartmanda yangin alarmi caliyor, merdiven dumanli.", "apartman yangini", ["kacis yolu riskli"], "CRITICAL"),
    ("Depoda kucuk capli yangin var.", "depo yangini", [], "URGENT"),
    ("Aracimdan duman cikiyor.", "arac yangin riski", [], "URGENT"),
]

CRIME_CASES = [
    ("Disarida silah sesi duydum, biri yere dustu.", "silahli saldiri", ["silah"], "CRITICAL"),
    ("Biri bicakla tehdit ediyor.", "bicakli tehdit", ["bicak"], "CRITICAL"),
    ("Evime zorla girmeye calisiyorlar.", "haneye tesebbus", [], "URGENT"),
    ("Sokakta kavga var, biri yarali.", "kavga ve yaralanma", ["yarali"], "URGENT"),
]

OTHER_CASES = [
    ("Asansorde mahsur kaldik, cocuk panik yapiyor.", "asansorde mahsur kalma", [], "URGENT"),
    ("Yolda buyuk bir elektrik diregi devrildi.", "altyapi tehlikesi", [], "URGENT"),
    ("Yasli komsumdan iki gundur haber alamiyorum.", "refah kontrolu", [], "NON_URGENT"),
    ("Kedi agaca cikti inemiyor.", "hayvan kurtarma talebi", [], "NON_URGENT"),
]

GIBBERISH = [
    "asdasd qweqwe",
    ".... ???",
    "klavyeeeyyy mmm",
    "hmm bilmiyom",
]


def _pick_age() -> str:
    return str(random.randint(18, 82))


def _medical_example() -> dict:
    text, complaint, red_flags, level = random.choice(MEDICAL_CASES)
    name = random.choice(TR_NAMES)
    age = _pick_age()
    user = f"Adim {name}, {age} yasindayim. {text}"
    assistant = {
        "response_text": "Anladim, tibbi acil durum olabilir. Hastayi guvenli bir pozisyonda tutun. Su an bilinci acik mi?",
        "extracted_slots": {
            "caller_name": name,
            "age": age,
            "chief_complaint": complaint,
            "category": "medical",
        },
        "triage_level": level,
        "category": "medical",
        "is_complete": False,
        "red_flags": red_flags,
    }
    return {"user": user, "assistant_json": assistant}


def _fire_example() -> dict:
    text, complaint, red_flags, level = random.choice(FIRE_CASES)
    name = random.choice(TR_NAMES)
    age = _pick_age()
    trapped = random.choice(["yes", "no"])
    user = f"Ben {name}. {age} yasindayim. {text}"
    assistant = {
        "response_text": "Sakin kalin, hemen guvenli cikisi degerlendirin. Dumandan uzak durun ve agiz-burnunuzu ortun. Iceride mahsur kalan var mi?",
        "extracted_slots": {
            "caller_name": name,
            "age": age,
            "chief_complaint": complaint,
            "category": "fire",
            "trapped": trapped,
        },
        "triage_level": level,
        "category": "fire",
        "is_complete": False,
        "red_flags": red_flags,
    }
    return {"user": user, "assistant_json": assistant}


def _crime_example() -> dict:
    text, complaint, red_flags, level = random.choice(CRIME_CASES)
    name = random.choice(TR_NAMES)
    age = _pick_age()
    user = f"Adim {name}, {age} yasindayim. {text}"
    assistant = {
        "response_text": "Guvenliginizi onceleyin, riskli alandan uzaklasin. Emniyet yonlendiriliyor. Saldirgan hala yakinlarda mi?",
        "extracted_slots": {
            "caller_name": name,
            "age": age,
            "chief_complaint": complaint,
            "category": "crime",
        },
        "triage_level": level,
        "category": "crime",
        "is_complete": False,
        "red_flags": red_flags,
    }
    return {"user": user, "assistant_json": assistant}


def _other_example() -> dict:
    text, complaint, red_flags, level = random.choice(OTHER_CASES)
    name = random.choice(TR_NAMES)
    age = _pick_age()
    user = f"Ben {name}, {age} yas. {text}"
    assistant = {
        "response_text": "Durumu anladim, ek bilgi toplayalim. Su anda fiziksel yaralanma veya duman/kanama gibi tehlike var mi?",
        "extracted_slots": {
            "caller_name": name,
            "age": age,
            "chief_complaint": complaint,
            "category": "other",
        },
        "triage_level": level,
        "category": "other",
        "is_complete": False,
        "red_flags": red_flags,
    }
    return {"user": user, "assistant_json": assistant}


def _english_example() -> dict:
    name = random.choice(EN_NAMES)
    age = _pick_age()
    user = f"My name is {name}, I am {age}. My father collapsed and is not responding."
    assistant = {
        "response_text": "Understood, this may be critical. Check if he is breathing and keep him on a safe surface. Is he breathing right now?",
        "extracted_slots": {
            "caller_name": name,
            "age": age,
            "chief_complaint": "collapsed and unresponsive",
            "category": "medical",
        },
        "triage_level": "CRITICAL",
        "category": "medical",
        "is_complete": False,
        "red_flags": ["unresponsive"],
    }
    return {"user": user, "assistant_json": assistant}


def _gibberish_example() -> dict:
    user = random.choice(GIBBERISH)
    assistant = {
        "response_text": "Sizi net anlayamadim. Acil durumu kisaca tek cumleyle yazar misiniz?",
        "extracted_slots": {},
        "triage_level": "URGENT",
        "category": "other",
        "is_complete": False,
        "red_flags": [],
    }
    return {"user": user, "assistant_json": assistant}


def generate_examples(count: int) -> list[dict]:
    random.seed(42)
    generators = (
        [_medical_example] * 4
        + [_fire_example] * 3
        + [_crime_example] * 3
        + [_other_example] * 2
        + [_english_example]
        + [_gibberish_example]
    )
    examples: list[dict] = []
    for _ in range(count):
        fn = random.choice(generators)
        examples.append(fn())
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=180, help="How many examples to generate.")
    args = parser.parse_args()

    examples = generate_examples(args.count)
    payload = {
        "description": "Auto-generated emergency triage few-shot/fine-tune examples.",
        "examples": examples,
    }
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"Wrote {len(examples)} examples to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
