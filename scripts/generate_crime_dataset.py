#!/usr/bin/env python3
"""
generate_crime_dataset.py
=========================
Polis (crime) kategorisi icin sinir ornekleri — ozellikle gunluk
NON_URGENT durumlar (gurultu sikayeti, park ihlali, vs) ve URGENT belirsiz
durumlar (supheli kisi, kapida israrcilik).

Cikti: data/labels/crime_synthetic_v1.csv
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_CSV = PROJECT_ROOT / "data" / "labels" / "crime_synthetic_v1.csv"

NON_URGENT_TR = [
    "Komsumda yine gurultu var, saat 22'yi gecti. Gelin demiyorum ama uyari olsun.",
    "Sokagima yanlis park etmis bir araba var, birkac gundur duruyor.",
    "Binanin duvarina grafiti yapmislar, tutanak tutulmali mi?",
    "Kayip bir kopek buldum mahallede, kimsenin degil gibi. Ne yapmaliyim?",
    "Esya hirsizligi oldu evden, bir hafta oncesinden. Simdi belediyeye mi?",
    "Komsularim konustular biraz, sesli oldu ama fiziksel kavga yok.",
    "Disarda birkac genc aglaniyor ama tehditkar gibi degil, sadece gurultulu.",
    "Arabamin aynasini kirmisler park halindeyken, gece. Bildirmem lazim.",
    "Eve gelince kapi aralik buldum, icerde bir sey eksik degil ama tutunmali miyim?",
    "Apartmanda asansorde bilinmeyen biri vardi, cekiciydi. Ihbar edeyim mi?",
    "Kendi evimin penceresinden yogun musluk sesi duyuyorum, su faturasini kontrol etmeli miyim?",
    "Sokakta dilenciler var surekli, rahatsiz ediyor ama siddet yok.",
    "Komsumdan horoz sesi geliyor her sabah, kurallara aykiri galiba.",
    "Caddede bir araba saatlerce motor calistiriyor bos. Ihbarlamali miyim?",
    "Esya bilmeyen birine vermis komsum, geri almak istiyor ama zor durumda. Hukuki danismanlik mi?",
    "Park ucuz carpisma olmus, kimse yaralanmamis, belgelemek icin gerek var mi polis?",
    "Parkta birisi kopeklere kotu davraniyor, hayvan koruma mi arayayim?",
    "Evimizin onunde sokak kopegi dolaniyor, tehditkar degil. Belediyeyi mi?",
    "Kirlicam var, kiraci tasinmadi ve odeme yapmiyor. Dava acacam.",
    "Kayip cocuk mahallede, genclerden biri ama tehlikede degil, evinden kactim diye duyuyorum.",
]

NON_URGENT_EN = [
    "Noise complaint about my neighbor, past 10pm. Not saying come out, just a warning.",
    "A car is wrongly parked on my street, been there a few days.",
    "Graffiti on the building wall, should I file a report?",
    "Found a lost dog in the neighborhood, doesn't seem to belong to anyone.",
    "Theft happened a week ago at my house, should I call now or go to a station?",
    "Neighbors had a loud argument, no physical fight.",
    "Teens outside being loud but not threatening.",
    "Someone broke my parked car's mirror at night, I should report it.",
    "Found my door slightly open when I got home, nothing missing but should I be cautious?",
    "An unfamiliar person in the elevator, looked shady, should I report?",
    "Heavy water running sound from a window, should I check the water bill?",
    "Homeless people on the street making me uncomfortable, no violence though.",
    "Rooster noise every morning from a neighbor, might be against rules.",
    "A car engine running for hours unattended on the street, should I report?",
    "Neighbor gave belongings to wrong person, wants them back, legal advice?",
    "Small parking lot fender bender, no injuries, need police for documentation?",
    "Someone being cruel to dogs in the park, should I call animal welfare?",
    "Street dog hanging around our building, not threatening. Municipality?",
    "Tenant didn't move out and not paying rent, I'll sue.",
    "Lost youth in the neighborhood, allegedly ran away from home but not in danger.",
]

URGENT_TR = [
    "Kapimda israrla birisi kapi caliyor, tanimiyorum. Bekliyor gidemiyorum.",
    "Dukkanimin onunde supheli bir araba duruyor, iki kisi beklemekteler.",
    "Bir sey kaybetmedim ama eve donunce kapi acikti, simdi nereye bakayim?",
    "Stalking suphesi var, ayni adam beni takip ediyor son uc gundur.",
    "Maddi siddet olmadi ama evde kavga cikti, komsum ben arandim galiba.",
    "Esimden ayri yasiyorum, simdi evin onunde arabada bekliyor, hareket etmiyor.",
    "Oglum okuldan henuz donmedi, zamanindan bir saat gec. Endiseliyim.",
    "Alis veris merkezinde bir adam cocugu izlemeye devam ediyor, aile degil gibi.",
    "Bir kisi kavga etti benimle sokakta, siddetle tehdit etti ama darpe olmadi.",
    "Dukkanda hirsiz gibi birisi gordum ama kacti, takip etmeye mi devam edeyim?",
    "Benzin istasyonunda kavga var iki kisi, ayrisirmis ama taraflar hala bagirisiyor.",
    "Trafik kazasi oldu, yarali yok gibi ama araba kacti.",
    "Kapida ayakkabilar calinmis galiba, zorlama izi yok ama hatirlamiyorum ne giydigim.",
    "Cantasi icinde ilaclari vardi, calindi sanirim otobuste. Onemli ilaclar.",
]

URGENT_EN = [
    "Someone's knocking on my door persistently, I don't know them. They're waiting.",
    "Suspicious car outside my shop, two people sitting and waiting.",
    "Nothing is missing but my door was open when I got home, where should I check?",
    "Possible stalker, same man has been following me for three days.",
    "No physical violence but a fight at home, a neighbor may have called me.",
    "Separated from my spouse, he's now waiting in a car outside, not moving.",
    "My son isn't home from school yet, an hour late. I'm worried.",
    "A man at the mall keeps watching a child, doesn't look like family.",
    "A man threatened me violently on the street, no blows landed.",
    "Saw what looked like a thief in my shop, they ran, should I follow?",
    "Fight at a gas station, two people, separated but still yelling.",
    "Traffic accident, no injuries apparent but the other car fled.",
    "Shoes stolen at the door, no forced entry, I don't remember what I wore though.",
    "Bag had my meds in it, I think stolen on the bus. Important meds.",
]

CRITICAL_TR = [
    "Evimde silahli bir adam var, kendimi kilitledim banyoya! Acil!",
    "Esim beni dovmeye basladi, bicak aldi! Kacmam lazim!",
    "Bir arkadasim bicaklandi, kan var! Hemen gelin!",
    "Cocuk kaciriliyor parkta, adam cekti aldi! Hemen!",
    "Silah sesi duyuyorum sokakta, bir kisi yere yigildi gibi!",
    "Evde rehine aldilar, benimle birlikte ailem de burada!",
    "Cinayete tanik oldum az once, adam kacti!",
    "Aktif bir saldiri var alis veris merkezinde, insanlar kaciyor!",
    "Oglum kayboldu, 5 yasinda, son 2 saat oldu! Lutfen her yeri arayin!",
    "Yetiskin bir adam cocuga cinsel saldiri yapiyor parkta!",
]

CRITICAL_EN = [
    "Armed man in my house, I locked myself in the bathroom! Emergency!",
    "My husband started beating me, he grabbed a knife! I need to run!",
    "A friend got stabbed, there's blood! Come now!",
    "A child is being abducted in the park, a man grabbed them!",
    "I hear gunshots outside, someone dropped to the ground!",
    "Hostage situation at home, my family is here with me!",
    "I witnessed a murder just now, the man fled!",
    "Active attack at a shopping center, people are running!",
    "My 5-year-old son is missing, last seen 2 hours ago! Please search everywhere!",
    "An adult man is sexually assaulting a child in the park!",
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
        text = base
        if rng.random() < 0.35:
            filler_tr = ["Polis arasak mi?", "Hangi birime bildirmeli?", "Acil degil tabii.",
                         "Yanimdakilerle guvendeyim.", "Bilgi vermek istedim."]
            filler_en = ["Should we call police?", "Which unit should I report to?",
                         "Not urgent of course.", "I'm safe with others around.", "Just sharing info."]
            fillers = filler_tr if lang == "tr" else filler_en
            text = f"{base} {rng.choice(fillers)}"
        rows.append({
            "case_id": f"{prefix}_{lang}_{i:04d}",
            "text_en": text,
            "source": "crime_synthetic_v1",
            "label_category_gold": "crime",
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
    rows += _make_rows(NON_URGENT_TR, NON_URGENT_EN, "NON_URGENT", "crime_nu", 0, args.non_urgent, rng)
    rows += _make_rows(URGENT_TR, URGENT_EN, "URGENT", "crime_ur", 0, args.urgent, rng)
    rows += _make_rows(CRITICAL_TR, CRITICAL_EN, "CRITICAL", "crime_cr", 1, args.critical, rng)
    rng.shuffle(rows)
    _write(OUT_CSV, rows)
    print(f"Wrote {len(rows)} rows to {OUT_CSV}")
    from collections import Counter
    print(Counter(r["label_triage_gold"] for r in rows))


if __name__ == "__main__":
    main()
