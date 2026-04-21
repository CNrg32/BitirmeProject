#!/usr/bin/env python3
"""
generate_nonsense_dataset.py
============================
Anlamsiz / mesguliyet / yanlis arama veri seti. Tum ornekler NON_URGENT
etiketli; kategori "other"; input_quality "gibberish" veya "out_of_scope"
olarak isaretlenir -> modelin 4. basi bunu tahmin edecek.

Cikti: data/labels/nonsense_v1.csv
Sema: case_id, text_en, source, label_category_gold, label_triage_gold,
      red_flags_gold, input_quality
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_CSV = PROJECT_ROOT / "data" / "labels" / "nonsense_v1.csv"

OUT_OF_SCOPE_TR = [
    "Merhaba, pizza siparisi verebilir miyim?",
    "Taksi cagirmak istiyorum, adres kaza mi?",
    "Internet faturami nasil odeyebilirim?",
    "Yarin hava nasil olacak Istanbul'da?",
    "Otel rezervasyonu icin mi buradasiniz?",
    "Tercuman hattini mi aradim, yanlis numara galiba.",
    "Eczane nobetcisi kim bu gece?",
    "Posta kurye hattini ariyorum sanirim.",
    "Kargom gelmemis, takip numarasi yok.",
    "Bankamatik nerde, en yakin sube?",
    "Saatim durdu, dogru saati ogrenebilir miyim?",
    "Hangi otobus gider merkez'e?",
    "Kargo firmasi sordugumda bana bu numarayi verdi.",
    "Doviz kuru ne anda dolar?",
    "Cocugumun dogum gununu nerde kutlayabilirim?",
    "Telefonu test etmek icin aradim, affedersiniz.",
    "Biraz dinleyebilir misiniz, yalniz konusmak istiyorum.",
    "Annemle tartistim ne yapayim, danisman mi var?",
    "Acil servis mi polis mi karar veremedim, yanlis arayan biriyim.",
    "Sinav sorulari hakkinda bilgi alabilir miyim?",
]

OUT_OF_SCOPE_EN = [
    "Hi, can I place a pizza order?",
    "I'd like to call a taxi, is this the right number?",
    "How can I pay my internet bill?",
    "What's the weather tomorrow in Istanbul?",
    "Is this reservation service?",
    "Dialed the translator hotline I think, wrong number.",
    "Which pharmacy is on duty tonight?",
    "I'm calling the postal courier line I guess.",
    "My parcel didn't arrive, no tracking number.",
    "Where's the nearest ATM?",
    "My watch stopped, can you tell me the correct time?",
    "Which bus goes downtown?",
    "The courier company gave me this number.",
    "What's the current USD exchange rate?",
    "Where can I celebrate my kid's birthday?",
    "Called to test my phone, sorry.",
    "Can you just listen, I want to talk alone.",
    "I argued with my mom, is there a counselor?",
    "Can't decide if this is police or ambulance, wrong caller probably.",
    "Can I get info about exam questions?",
]

GIBBERISH_TR = [
    "asdasd asdfg hjkl",
    "ne dedin anlamadim konus iyice",
    "blabla blabla bip bip",
    "haha ses test 1 2 3",
    "alooo kim var burda",
    "hmm ne",
    "ahkhsk sdjf akj",
    "ay ay ay hahaha",
    "pwp pwp pwp",
    "ofof off off ne oldu",
    "mmm mmm mmm",
    "lollolol acil servis hahaha dalga",
    "denendi bye bye",
    "heheh prankprank",
    "pupupu papapa",
]

GIBBERISH_EN = [
    "asdf asdf qwerty",
    "what did you say repeat clearly",
    "bla bla beep beep",
    "haha sound test 1 2 3",
    "hello hello anyone there",
    "hmm what",
    "jkshak djs ak",
    "ay ay hahaha",
    "pwp pwp pwp",
    "lol lol prank prank bye",
    "testing testing",
    "heheh prank call",
    "nothing really just checking",
    "uhh umm um",
    "mm mm mm",
]


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "case_id", "text_en", "source", "label_category_gold",
            "label_triage_gold", "red_flags_gold", "input_quality",
        ])
        w.writeheader()
        w.writerows(rows)


def _make_rows(tr: list[str], en: list[str], quality: str, prefix: str,
               target_n: int, rng: random.Random) -> list[dict]:
    rows: list[dict] = []
    pool = [("tr", t) for t in tr] + [("en", t) for t in en]
    i = 0
    while len(rows) < target_n:
        lang, base = pool[i % len(pool)]
        rows.append({
            "case_id": f"{prefix}_{lang}_{i:04d}",
            "text_en": base,
            "source": "nonsense_v1",
            "label_category_gold": "other",
            "label_triage_gold": "NON_URGENT",
            "red_flags_gold": 0,
            "input_quality": quality,
        })
        i += 1
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-of-scope", type=int, default=250)
    ap.add_argument("--gibberish", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = _make_rows(OUT_OF_SCOPE_TR, OUT_OF_SCOPE_EN, "out_of_scope", "oos", args.out_of_scope, rng)
    rows += _make_rows(GIBBERISH_TR, GIBBERISH_EN, "gibberish", "gib", args.gibberish, rng)
    rng.shuffle(rows)
    _write(OUT_CSV, rows)
    print(f"Wrote {len(rows)} rows to {OUT_CSV}")
    from collections import Counter
    print(Counter(r["input_quality"] for r in rows))


if __name__ == "__main__":
    main()
