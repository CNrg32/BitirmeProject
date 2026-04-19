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

TR_INTRO_PATTERNS = [
    "Adim {name}, {age} yasindayim.",
    "Ben {name}, {age} yas.",
    "Ismim {name}. Yasim {age}.",
    "{name} ben, {age} yasindayim.",
]

EN_INTRO_PATTERNS = [
    "My name is {name}, I am {age}.",
    "This is {name}, age {age}.",
    "I am {name}, {age} years old.",
]

LOCATION_SNIPPETS = [
    "Adresim merkez mahalle.",
    "Konumum tam net degil ama cadde uzerindeyiz.",
    "Ilce hastane civari.",
    "Sokak kalabalik, panik var.",
    "Evdeyiz, ikinci kattayiz.",
    "Yol kenarindayiz, trafik akiyor.",
]

URGENCY_SNIPPETS = [
    "Lutfen cabuk olun.",
    "Durum her dakika kotulesiyor.",
    "Ne yapacagimi bilmiyorum.",
    "Hemen yardim lazim.",
    "Cok korkuyoruz.",
]

NOISY_SUFFIXES = [
    "",
    "ses kesiliyor",
    "hatta kopma var",
    "arkadan bagiris geliyor",
    "cok gurultu var",
]

MEDICAL_CASES = [
    (
        ["Babam nefes alamiyor, morariyor.", "Nefes alamiyor ve dudaklari morardi.", "Hasta hizli kotulesiyor, nefesi kesiliyor."],
        "nefes darligi",
        ["nefes alamama"],
        "CRITICAL",
    ),
    (
        ["Annem bayildi, cevap vermiyor.", "Kisi baygin ve seslenince tepki vermiyor.", "Yere yigilmis halde, bilinci yok gibi."],
        "bilinc kaybi",
        ["bilinc kapali"],
        "CRITICAL",
    ),
    (
        ["Gogsumde siddetli agri var, terliyorum.", "Goguste baski var, soguk terleme basladi.", "Kalp bolgesinde siddetli agri hissediyorum."],
        "gogus agrisi",
        ["kalp krizi suphe"],
        "CRITICAL",
    ),
    (
        ["Trafik kazasi oldu, bacagi cok kaniyor.", "Kazadan sonra bacagindan cok fazla kan geliyor.", "Yaralinin kanamasi durmuyor."],
        "ciddi kanama",
        ["asiri kanama"],
        "CRITICAL",
    ),
    (
        ["Dusme sonrasi kolunu oynatamiyor.", "Merdivenden dustu, kolunda kirik olabilir.", "Kolunu oynattiginda siddetli agri oluyor."],
        "kirik suphe",
        [],
        "URGENT",
    ),
    (
        ["Atesi var ve nefesi hizli.", "Yuksek ates ve hizli solunum var.", "Hasta atesli ve nefes alisi zorlasiyor."],
        "ates ve solunum sikintisi",
        [],
        "URGENT",
    ),
]

FIRE_CASES = [
    (
        ["Evde mutfakta yangin cikti, duman cok yogun.", "Mutfakta alev var, goz gozu gormuyor.", "Evde yangin buyuyor, duman her yere yayildi."],
        "ev yangini",
        ["yogun duman"],
        "CRITICAL",
    ),
    (
        ["Apartmanda yangin alarmi caliyor, merdiven dumanli.", "Binada alarm var, cikis merdiveni duman dolu.", "Apartmanda duman var, insanlar inemiyor."],
        "apartman yangini",
        ["kacis yolu riskli"],
        "CRITICAL",
    ),
    (
        ["Depoda kucuk capli yangin var.", "Depo tarafinda alevlenme oldu.", "Depoda duman var, yangin yeni basladi."],
        "depo yangini",
        [],
        "URGENT",
    ),
    (
        ["Aracimdan duman cikiyor.", "Arabanin motorundan duman ve koku geliyor.", "Arac alev alacak gibi, kaputu acamiyorum."],
        "arac yangin riski",
        [],
        "URGENT",
    ),
]

CRIME_CASES = [
    (
        ["Disarida silah sesi duydum, biri yere dustu.", "Silah sesi geldi, bir kisi vuruldu gibi.", "Sokakta ates acildi, birisi yerde yatiyor."],
        "silahli saldiri",
        ["silah"],
        "CRITICAL",
    ),
    (
        ["Biri bicakla tehdit ediyor.", "Elinde bicakla ustumuze yuruyor.", "Bicakli biri apartman girisinde bagiriyor."],
        "bicakli tehdit",
        ["bicak"],
        "CRITICAL",
    ),
    (
        ["Evime zorla girmeye calisiyorlar.", "Kapiyi kirip iceri girmeye ugrasiyorlar.", "Binada daireme zorla giris denemesi var."],
        "haneye tesebbus",
        [],
        "URGENT",
    ),
    (
        ["Sokakta kavga var, biri yarali.", "Kavgada bir kisi yere dustu ve kan var.", "Toplu kavga cikti, yaralilar olabilir."],
        "kavga ve yaralanma",
        ["yarali"],
        "URGENT",
    ),
]

OTHER_CASES = [
    (
        ["Asansorde mahsur kaldik, cocuk panik yapiyor.", "Asansor arizalandi, iceride sikistik.", "Asansorde kaldik, nefes daralmasi basladi."],
        "asansorde mahsur kalma",
        [],
        "URGENT",
    ),
    (
        ["Yolda buyuk bir elektrik diregi devrildi.", "Elektrik hatti yere dusmus gorunuyor.", "Direk devrildi, kablolar yola sarkiyor."],
        "altyapi tehlikesi",
        [],
        "URGENT",
    ),
    (
        ["Yasli komsumdan iki gundur haber alamiyorum.", "Komsumun kapisi acilmiyor, endiseliyim.", "Yasli birinden uzun suredir ses yok."],
        "refah kontrolu",
        [],
        "NON_URGENT",
    ),
    (
        ["Kedi agaca cikti inemiyor.", "Sokak kedisi catida mahsur.", "Hayvan asagi inemiyor, yardim lazim."],
        "hayvan kurtarma talebi",
        [],
        "NON_URGENT",
    ),
]

ENGLISH_CASES = [
    ("My father collapsed and is not responding.", "collapsed and unresponsive", ["unresponsive"], "CRITICAL"),
    ("My mother cannot breathe and her lips turned blue.", "severe breathing distress", ["cannot breathe"], "CRITICAL"),
    ("There was a car crash and heavy bleeding.", "major bleeding after traffic crash", ["heavy bleeding"], "CRITICAL"),
    ("My neighbor has chest pain and is sweating.", "severe chest pain", ["possible cardiac event"], "CRITICAL"),
]

GIBBERISH = [
    "asdasd qweqwe",
    ".... ???",
    "klavyeeeyyy mmm",
    "hmm bilmiyom",
    "yardimmmm ama ne desem",
    "ses... cizirti... alo alo",
]

MEDICAL_RESPONSES = [
    "Anladim, tibbi acil durum olabilir. Hastayi guvenli bir pozisyonda tutun. Su an bilinci acik mi?",
    "Bu tablo acil degerlendirme gerektiriyor. Solunumunu kontrol edin ve hastayi yalniz birakmayin. Nefes alisi duzenli mi?",
    "Durum kritik olabilir. Hastayi duz zemine alin ve hava yolunu acik tutun. Bilinci var mi?",
    "Hemen yonlendiriyoruz. Kanama varsa baski uygulayin ve hastayi hareket ettirmeyin. Su an en belirgin belirti ne?",
]

FIRE_RESPONSES = [
    "Sakin kalin, hemen guvenli cikisi degerlendirin. Dumandan uzak durun ve agiz-burnunuzu ortun. Iceride mahsur kalan var mi?",
    "Yangin icin ekip yonlendiriyoruz. Elektrik ve dogalgazi kapatabiliyorsaniz kapatin, asansor kullanmayin. Alev nerede yogun?",
    "Once can guvenligi. Dumansiz alana gecin ve bina icinde kalmayin. Cikis yolu acik mi?",
    "Anladim, itfaiye yonlendiriliyor. Duman hizla artiyorsa egilerek ilerleyin ve kapilari kapali tutun. Yarali var mi?",
]

CRIME_RESPONSES = [
    "Guvenliginizi onceleyin, riskli alandan uzaklasin. Emniyet yonlendiriliyor. Saldirgan hala yakinlarda mi?",
    "Tehlikeli bolgeden uzaklasin ve gorus mesafesinde kalmayin. Polis yonlendiriyoruz. Supheliyi tarif edebilir misiniz?",
    "Kendinizi guvene alin, catismaya girmeyin. Ekipler yolda. Olay su anda devam ediyor mu?",
    "Anladim, bu guvenlik acili olabilir. Guvenli bir noktadan konumunuzu paylasin. Yarali kisi var mi?",
]

OTHER_RESPONSES = [
    "Durumu anladim, ek bilgi toplayalim. Su anda fiziksel yaralanma veya duman/kanama gibi tehlike var mi?",
    "Not aliyorum, uygun ekip yonlendirmesi icin detay lazim. Tam konumu ve olayin ne zamandir surdugunu yazar misiniz?",
    "Anladim, once guvenlik durumunu netlestirelim. Cevrede can guvenligini tehdit eden bir risk var mi?",
    "Talebinizi aldim. Duruma gore yonlendirme yapabilmem icin olay yerinde mahsur kalan biri var mi?",
]

ENGLISH_RESPONSES = [
    "Understood, this may be critical. Check if the patient is breathing and keep them on a safe surface. Is breathing present right now?",
    "This sounds urgent. Stay with the patient, clear the airway if possible, and avoid moving them unnecessarily. Are they conscious?",
    "Emergency support is being arranged. Apply pressure if there is bleeding and monitor breathing. Is the person responding?",
]

FILLER_PHRASES = [
    "",
    "lutfen yardim edin",
    "cok panik olduk",
    "acele eder misiniz",
    "ne olur hizli olun",
]


def _pick_age() -> str:
    return str(random.randint(18, 82))


def _compose_tr_user(name: str, age: str, situation: str) -> str:
    intro = random.choice(TR_INTRO_PATTERNS).format(name=name, age=age)
    parts = [intro, situation]
    if random.random() < 0.5:
        parts.append(random.choice(LOCATION_SNIPPETS))
    if random.random() < 0.45:
        parts.append(random.choice(URGENCY_SNIPPETS))
    suffix = random.choice(NOISY_SUFFIXES)
    if suffix:
        parts.append(suffix)
    filler = random.choice(FILLER_PHRASES)
    if filler:
        parts.append(filler)
    return " ".join(parts)


def _compose_en_user(name: str, age: str, situation: str) -> str:
    intro = random.choice(EN_INTRO_PATTERNS).format(name=name, age=age)
    tail = random.choice(
        [
            "Please send help quickly.",
            "I am scared and need immediate support.",
            "The condition is getting worse.",
            "",
        ]
    )
    return " ".join(piece for piece in [intro, situation, tail] if piece)


def _medical_example() -> dict:
    variations, complaint, red_flags, level = random.choice(MEDICAL_CASES)
    name = random.choice(TR_NAMES)
    age = _pick_age()
    user = _compose_tr_user(name, age, random.choice(variations))
    assistant = {
        "response_text": random.choice(MEDICAL_RESPONSES),
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
    variations, complaint, red_flags, level = random.choice(FIRE_CASES)
    name = random.choice(TR_NAMES)
    age = _pick_age()
    trapped = random.choice(["yes", "no"])
    user = _compose_tr_user(name, age, random.choice(variations))
    assistant = {
        "response_text": random.choice(FIRE_RESPONSES),
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
    variations, complaint, red_flags, level = random.choice(CRIME_CASES)
    name = random.choice(TR_NAMES)
    age = _pick_age()
    user = _compose_tr_user(name, age, random.choice(variations))
    assistant = {
        "response_text": random.choice(CRIME_RESPONSES),
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
    variations, complaint, red_flags, level = random.choice(OTHER_CASES)
    name = random.choice(TR_NAMES)
    age = _pick_age()
    user = _compose_tr_user(name, age, random.choice(variations))
    assistant = {
        "response_text": random.choice(OTHER_RESPONSES),
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
    situation, complaint, red_flags, level = random.choice(ENGLISH_CASES)
    name = random.choice(EN_NAMES)
    age = _pick_age()
    user = _compose_en_user(name, age, situation)
    assistant = {
        "response_text": random.choice(ENGLISH_RESPONSES),
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
