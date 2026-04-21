#!/usr/bin/env python3
"""
generate_fire_dataset.py
========================
Itfaiye (fire) kategorisi icin sinir ornekleri ureten sentetik veri.

Mevcut datasette sadece 310 fire ornegi var ve NON_URGENT fire 33.
Bu script NON_URGENT + URGENT + ek CRITICAL uretir.

Cikti: data/labels/fire_synthetic_v1.csv
Sema: case_id, text_en, source, label_category_gold, label_triage_gold, red_flags_gold
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_CSV = PROJECT_ROOT / "data" / "labels" / "fire_synthetic_v1.csv"

NON_URGENT_TR = [
    "Komsumdan sigara kokusu geliyor, hafif duman gibi ama gorunmuyor. Acil degil.",
    "Evimizin balkonunda mangal dumani var. Komsuyu haberdar ettim. Alev yok.",
    "Aksam ocakta tavaya az biraz yag dustu, hemen kapak kapattim. Simdi iyi.",
    "Yangin detektoru kendi kendine calisti ama hicbir sey yok. Pil bitti galiba.",
    "Sokakta bir cop kutusunda hafif duman vardi, sondurdum su doktum.",
    "Soba dumani cok, bacayi bir yil oncesinden temizletmemistim. Tehlike yok simdi.",
    "Komsum odun sobasi kullaniyor, duman pencereme geliyor. Yangin yok.",
    "Arabamin egzozu biraz duman yapti, servise gitmeliyim mi diye soracaktim.",
    "Ocakta ekmek kizarttim biraz fazla yandi, kokusu kaldi. Kapi pencere actim.",
    "Elektrikli supurgem biraz kokulu calisiyor, normal mi diye sormak istedim.",
    "Mutfakta piyaz kizarttim yag siciradi, masa orortusu hafif simsekti. Temizliyorum simdi.",
    "Sigara kullanmiyorum ama apartmanda birinden koku geliyor. Yonetim mi?",
    "Kiraci tasinirken birkac balon patlatti, sesi yangin sandim. Yangin yok.",
    "Arkadasim sonucta bir mum yakti yanimda, akladi ama temizledim. Alev yok.",
    "Cocugum cakmak oyuncagi ile oynamis, cakmak yok ama kokusu bir an geldi.",
    "Carsafim utu yaparken az yandi biraz, atabilirim. Tehlike yok.",
    "Komsumun camasir kazani bozulmus, motor biraz kokulu. Ben aramadim zaten.",
    "Garajda eski kutu yakmaya calistim ama sondurdum, kontrol altinda.",
    "Klimam biraz cizirdadi, elektrik kesildi hemen kapandi. Yangin yok.",
    "Yangin alarmi yanlislikla calisiyor, hassas galiba. Yanginciyi aramali miyim?",
]

NON_URGENT_EN = [
    "There's light smoke from my neighbor's cigarette, can't see anything. Not urgent.",
    "BBQ smoke from our balcony, I warned the neighbors. No flames.",
    "A bit of oil dripped on the stove, I put the lid on. All fine now.",
    "Fire alarm went off on its own, nothing here. Probably dead battery.",
    "Small smoke from a trash can on the street, I poured water on it.",
    "Lots of smoke from the chimney, haven't cleaned it in a year. No danger now.",
    "My neighbor uses a wood stove, smoke reaches my window. No fire.",
    "My car exhaust made some smoke, wondering if I should see a mechanic.",
    "I burned some toast on the stove, smell lingers. Opened windows.",
    "My vacuum smells a bit when running, is that normal?",
    "Cooked fried food, oil splattered, tablecloth singed a bit. Cleaning.",
    "I don't smoke but someone in the building does, smell coming in.",
    "Tenant moving in set off some balloons, sounded like fire. No fire.",
    "Friend lit a candle next to me, blew it out. No flame now.",
    "Kid was playing with a lighter toy, no real lighter, just caught a smell.",
    "Ironing slightly scorched my sheet, I can toss it. No danger.",
    "Neighbor's washing machine motor smells a bit. I didn't call.",
    "I tried burning an old box in the garage but I put it out, under control.",
    "AC made a zapping sound, electricity cut off on its own. No fire.",
    "Fire alarm triggering by itself, seems oversensitive. Should I call?",
]

URGENT_TR = [
    "Apartmanimda duman kokusu yayildi ama alev gormedim. Kat komsumda olabilir.",
    "Mutfakta tavadaki yag alev aldi, sondurucu var kullaniyorum ama yardim cagrayim mi?",
    "Prizden kivilcim cikti, duman var hafif. Sigortayi kapattim ama endiseliyim.",
    "Cop kutusu yangini apartman holunde, kucuk ama yanimda sondurucu yok.",
    "Baca dumani cok yogun, evde panik var ama alev gormedik.",
    "Araba motorundan duman cikiyor, yanmiyor ama koku yogun.",
    "Elektrik panosu ugulduyor ve koku var, bir seyler yaniyor galiba.",
    "Camasir makinesi cigararken duman yapti, fisten cektim. Duman hala var.",
    "Komsumun evinden simdi alarm sesi geliyor, cevap vermedi kapiyi calisam.",
    "Alt katta mutfak yangini oldu duyuyoruz, belediye arandi mi bilmem.",
    "Evimden cikarken pirizde kicramagi gordum, simdi uzaktayim, birine haber verin.",
    "Elektrikli battaniyem kokmaya basladi, fisten cektim ama yatagim sicak.",
    "Yangin alarmi calisiyor ama net sebep yok, koridorda hafif duman hissediyorum.",
    "Bacadan cok duman cikiyor, komsularda panik. Alev henuz yok.",
    "Benzin istasyonunda duman gordum digerlerine soyle diyorum, yangin varsa buyuk olur.",
]

URGENT_EN = [
    "Smoke smell spread in my building, I didn't see flames. Could be on my floor.",
    "Grease fire on the stove, I'm using an extinguisher. Should I call for help?",
    "Sparks from an outlet, light smoke. Turned breaker off but I'm worried.",
    "Trash can fire in the lobby, small but I don't have an extinguisher.",
    "Heavy chimney smoke, household is panicking. No flames.",
    "Car engine is smoking, not burning but strong smell.",
    "Electrical panel humming with a burnt smell, something's heating up.",
    "Washing machine smoked while running, I unplugged it. Smoke still lingers.",
    "Neighbor's alarm is going off, no one answers the door.",
    "Kitchen fire downstairs, not sure if fire department was called.",
    "Left home and saw a spark at the outlet, can someone check?",
    "Electric blanket started smelling, unplugged it. Bed is warm though.",
    "Fire alarm ringing, cause unclear, I smell light smoke in hallway.",
    "Heavy smoke from chimney, neighbors panicking. No visible flames yet.",
    "Saw smoke at a gas station, warning others, could become big if it's a fire.",
]

CRITICAL_TR = [
    "Evimde yangin var! Alevler mutfaktan salona yayiliyor! Lutfen hemen gelin!",
    "Apartmanda buyuk yangin var! Dumanda nefes alamiyorum! Merdivenler kapali!",
    "Araba yaniyor, benzin deposu patlayabilir! Yakindan uzaklasin!",
    "Cocugum odada mahsur, alev kapinin onunde! Kurtarin lutfen!",
    "Dogalgaz sizintisi ve kivilcim gordum, patlama olabilir! Acele!",
    "Bina catisinda buyuk alev var, asagidaki daireler etkileniyor!",
    "Zayif duvarli depoda yangin, icinde kimyasal var!",
    "Benzin istasyonunda yangin basladi, cevreye yayilabilir!",
    "Endustriyel tesiste patlama oldu, yaralilar var!",
]

CRITICAL_EN = [
    "My house is on fire! Flames spreading from kitchen to living room! Please come!",
    "Big fire in the building! I can't breathe! Stairs are blocked!",
    "Car is burning, gas tank may explode! Keep away!",
    "My child is trapped in a room, flames at the door! Save them please!",
    "Gas leak and I saw a spark, it may explode! Hurry!",
    "Huge flames on the building roof, units below affected!",
    "Fire in a weak-walled warehouse, chemicals inside!",
    "Fire started at a gas station, could spread to the area!",
    "Explosion at an industrial facility, there are casualties!",
]


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "case_id", "text_en", "source", "label_category_gold",
            "label_triage_gold", "red_flags_gold",
        ])
        w.writeheader()
        w.writerows(rows)


def _make_rows(tr: list[str], en: list[str], triage: str, prefix: str,
               rf: int, target_n: int, rng: random.Random) -> list[dict]:
    rows: list[dict] = []
    pool = [("tr", t) for t in tr] + [("en", t) for t in en]
    i = 0
    while len(rows) < target_n:
        lang, base = pool[i % len(pool)]
        # Hafif varyasyon: bazilarina filler ekle
        text = base
        if rng.random() < 0.35:
            filler_tr = ["Acelem yok.", "Teskilata haber vermeliyim mi?", "Panik etmiyorum.",
                         "Durum stabil.", "Kontrol altinda sanirim."]
            filler_en = ["I'm not in a panic.", "Just informing.", "Stable for now.",
                         "Not urgent.", "Under control I think."]
            fillers = filler_tr if lang == "tr" else filler_en
            text = f"{base} {rng.choice(fillers)}"
        rows.append({
            "case_id": f"{prefix}_{lang}_{i:04d}",
            "text_en": text,
            "source": "fire_synthetic_v1",
            "label_category_gold": "fire",
            "label_triage_gold": triage,
            "red_flags_gold": rf,
        })
        i += 1
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--non-urgent", type=int, default=300)
    ap.add_argument("--urgent", type=int, default=200)
    ap.add_argument("--critical", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows: list[dict] = []
    rows += _make_rows(NON_URGENT_TR, NON_URGENT_EN, "NON_URGENT", "fire_nu", 0, args.non_urgent, rng)
    rows += _make_rows(URGENT_TR, URGENT_EN, "URGENT", "fire_ur", 0, args.urgent, rng)
    rows += _make_rows(CRITICAL_TR, CRITICAL_EN, "CRITICAL", "fire_cr", 1, args.critical, rng)
    rng.shuffle(rows)
    _write(OUT_CSV, rows)
    print(f"Wrote {len(rows)} rows to {OUT_CSV}")
    from collections import Counter
    print(Counter(r["label_triage_gold"] for r in rows))


if __name__ == "__main__":
    main()
