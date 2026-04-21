#!/usr/bin/env python3
"""
generate_minor_complaints.py
============================
Hafif/kronik sikayet (NON_URGENT) ve belirsiz semptom (URGENT) sentetik veri
seti ureticisi. Hem Turkce hem Ingilizce template'ler kullanilir; amac
triage modelini "hemen CRITICAL'a atla" davranisindan uzaklastirmak.

Cikti:
    data/labels/minor_complaints_v1.csv   (~1500 satir, NON_URGENT, medical)
    data/labels/ambiguous_symptoms_v1.csv (~700 satir, URGENT, medical)

Semalar `synthetic_triage_cases_v2.csv` ile ayni: case_id, text_en, source,
label_category_gold, label_triage_gold, red_flags_gold.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "labels"
MINOR_OUT = DATA_DIR / "minor_complaints_v1.csv"
AMBIG_OUT = DATA_DIR / "ambiguous_symptoms_v1.csv"

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# NON_URGENT — minor / chronic complaints
# ---------------------------------------------------------------------------

MINOR_TEMPLATES_TR = [
    "Merhaba, birkac gundur basim agriyor. Bugun biraz daha kotu ama idare ediyorum.",
    "Hafif bir mide bulantim var. Sabahtan beri boyle. Acelem yok.",
    "Iki gundur hafif bir grip oldum galiba. Burnum akiyor, hafif oksuruk var.",
    "Parmagimi kesmistim, kucuk bir kesik. Kan durdu ama pansuman lazim mi diye soracaktim.",
    "Dun aksamdan beri bogazim hafif agriyor. Ates yok.",
    "Hafif bas donmesi var ama ayakta duruyorum. Tansiyonum dustu galiba.",
    "Birkac gundur uykusuzum. Genel halsizlik hissediyorum.",
    "Bilegimi hafif burktum galiba. Yurumekte zorluk yok ama agriyor.",
    "Disim zonkluyor, randevu almak icin arastiriyordum. Acil degil.",
    "Kronik migren hastasiyim, bugun hafif tutuldu. Haplarim var.",
    "Hamileligim 6. ay. Hafif bulanti var, normal mi diye soracaktim.",
    "Bel agrim var kronik, bugun biraz artti ama yatabiliyorum.",
    "Iki saattir hickirigim gecmiyor. Can sikici ama tehlikeli mi?",
    "Gozum hafif kasiniyor, alerji olabilir. Hapi var mi diye bakacaktim.",
    "Ayagimda kucuk bir sislik var, birkac gundur. Sanirim bocek isirdi.",
    "Hafif ateslendim galiba, 37.5 gozukuyor. Dinlenecegim.",
    "Kolum kasildi biraz, jimnastikte fazla zorlamisim. Agri hafif.",
    "Bebegim 8 aylik, hafif huysuzluk var. Dislenme olabilir.",
    "Yoruldum gun icinde, kalp atisim hafif hizli hissediyor. Stres olabilir.",
    "Ayak bilegim sabah tutuk, yash esnedim, gecer mi diye soracaktim.",
    "Hafif kabizlik var birkac gundur. Yemek duzenim degismisti.",
    "Kulaklarim ugulduyor bazen. Surekli degil. Kontrol gerekir mi?",
    "Dun aksam biraz fazla yedim, mide yanmasi var. Hafif.",
    "Adetim gecikti birkac gun, hafif karin agrim var. Muhtemelen normal.",
    "Yuzumde kucuk bir sivilce, bas parmagim kadar. Hafif kizarmis.",
    "Tirnagimin kenari iltihaplanmis galiba. Hafif agri var.",
    "Ilacimi unuttum sabah, simdi aldim. Bir sorun olur mu diye soracaktim.",
    "Bugun kendimi biraz halsiz hissediyorum. Ates yok, aci yok.",
    "Eklem agrim var kronik, romatizma. Bugun hafif tutuk ama idare ediyorum.",
    "Hafif bir karin agrim var, yemege bagli galiba. Gecicidir herhalde.",
    "Kucuk bir yanik var elimde, kibritten. Su altinda tuttum, kizarik.",
    "Uyku ilacimi bitirdim, reçete yenilemesi icin ne yapmaliyim?",
    "Cocugumun atesi 37.8, halsizlik yok. Suyu icmeye devam ediyor.",
    "Hafif bir kasinti var sirtimda, belki allerji. Tehlike yok gibi.",
    "Ses kisildi, bogazim hafif gicik. Konusabiliyorum, dinleniyorum.",
    "Oksuruyorum birkac gundur, kuru oksuruk. Nefes darligi yok.",
    "Balkondan dustu kedim, iyi gibi ama veterinere gitmeli miyim?",
    "Telefonumun sarji bitiyordu, sadece 112'yi test ediyorum, acil degil.",
    "Arkadasima ulasamiyorum birkac gundur, acil servisi mi aramaliyim?",
    "Kaybolmus bir kopek gordum sokakta, ne yapmaliyim?",
    "Elektrik faturam gelmedi bu ay, bu numaradan mi sorabilirim?",
    "Ambulans cagirmaya gerek var mi bilmiyorum, hafif mide agrim var.",
    "Kucuk cocugum dustu, bir tarafi morarmis ama agliyordu ve simdi oynuyor.",
    "Dudagim catlak, kanadi azicik. Vazelin sureyim mi?",
    "Sabah tansiyonum dustu, su ictim simdi iyiyim. Kontrol mu?",
]

MINOR_TEMPLATES_EN = [
    "Hi, I've had a mild headache for a couple of days. It's a bit worse today but manageable.",
    "I have some mild nausea since this morning. Not urgent, just wanted to ask.",
    "I think I'm coming down with a cold for the last two days. Runny nose, slight cough.",
    "Small cut on my finger, bleeding stopped. Just wondering about the dressing.",
    "My throat has been slightly sore since last night. No fever.",
    "I feel a bit dizzy but I can stand. Blood pressure may be low.",
    "Haven't slept well for a few nights, generally tired.",
    "I twisted my ankle lightly. Can walk but it aches.",
    "Tooth pain, I was planning to book an appointment. Not urgent.",
    "Chronic migraine patient, slight flare up today. Have my meds at home.",
    "6 months pregnant, some mild nausea, wanted to know if it's normal.",
    "Chronic lower back pain, slightly worse today but I can lie down.",
    "Hiccups for two hours, annoying but is it dangerous?",
    "Eye itching a bit, maybe allergies. Wondering about OTC options.",
    "Small bump on my ankle for a few days, probably an insect bite.",
    "Slight fever maybe, thermometer reads 37.5C. I'll rest.",
    "Slight muscle strain in my arm from gym. Mild ache.",
    "My baby is 8 months, a bit fussy today, probably teething.",
    "Feel tired, heart rate slightly up. Probably stress.",
    "Ankle stiffness in the morning, I exercised too hard yesterday.",
    "Mild constipation for a few days, diet changed recently.",
    "Occasional ringing in my ears, not constant. Should I worry?",
    "I ate too much last night, some heartburn. Mild.",
    "Period is late a few days, mild cramps. Probably normal.",
    "Small pimple on my face, slightly red. Nothing serious.",
    "Edge of my nail looks a bit inflamed, slight pain.",
    "Missed my medication this morning, took it now, any issue?",
    "Feeling a bit under the weather today. No fever, no pain.",
    "Chronic joint pain, rheumatism. A bit stiff today but manageable.",
    "Small belly ache, probably related to lunch.",
    "Minor burn on my hand from a match, I ran it under water.",
    "Ran out of my sleeping pills, how do I renew the prescription?",
    "My child has a fever of 37.8 but no weakness, drinking water.",
    "Itchy back, probably an allergy. Not alarming.",
    "Lost my voice a bit, slight tickly throat. Resting.",
    "Dry cough for a few days, no shortness of breath.",
    "My cat fell off the balcony, seems fine, should I go to the vet?",
    "My phone battery was dying, just testing 911, no emergency.",
    "Can't reach a friend for a couple of days, is this the number?",
    "Found a lost dog on the street, what should I do?",
    "My electricity bill didn't arrive, is this the right number?",
    "Not sure if I need an ambulance, just have mild stomach pain.",
    "Toddler fell, one side is bruised but he was crying and now plays.",
    "Lips chapped, tiny bleed. Should I apply balm?",
    "BP dropped this morning, I drank water and I'm fine, should I still check in?",
]

MINOR_FILLERS_TR = [
    "Acil degil, sadece dogru yerde miyim diye soruyorum.",
    "Hastaneye gitmem gerekir mi emin degilim.",
    "Evde ilacim var, ama emin olmak istedim.",
    "Doktorum tatilde, nereye gitmeliyim?",
    "Zaman kaybettirmek istemem, hizli bir bilgi yeterli.",
    "Gercekten acil degil, kafam karisik.",
    "Bu gece geceyarisi aridyorum, sabah gec olur diye.",
]

MINOR_FILLERS_EN = [
    "Not urgent, just checking if I'm at the right place.",
    "Not sure if I need to go to hospital.",
    "I have meds at home but wanted to confirm.",
    "My doctor is on vacation, where should I go?",
    "Don't want to waste your time, quick info is enough.",
    "Really not an emergency, I'm just confused.",
    "Calling at midnight because it'll be too late in the morning.",
]

# ---------------------------------------------------------------------------
# URGENT — ambiguous / moderate symptoms (could escalate but not critical yet)
# ---------------------------------------------------------------------------

AMBIG_TEMPLATES_TR = [
    "Atesim 38.5 derece, titremeye basladim. Bas agrim da var.",
    "Bas donmesi ve kusma var. Ayakta duramiyorum neredeyse.",
    "Kolumda kesik var, kanama hafif ama derin gibi. Dikis gerekebilir.",
    "Karnim cok agriyor, saga dogru. Birkac saattir.",
    "Cocugum dustu, kafasini carpti. Agladi, simdi sakin ama uyuklar gibi.",
    "Alerjik reaksiyon olabilir, yuzum sisiyor biraz. Nefes alabiliyorum.",
    "Gogusumde baski var, sol koluma vuruyor hafif. Panik yapiyorum.",
    "Yuksek ateste titreme basladi, hafif unutkanlik hissediyorum.",
    "Bir anda bas donmesi oldu, yere oturdum. Simdi biraz iyiyim ama tekrar olabilir.",
    "Elimi kestim, hala kanamakta. Bez saridim ama hala sizar.",
    "Sirtim kitlendi, hareket edemiyorum ama kan akisi var.",
    "Yandim, elim kizardi ve kabarik. Yuzuk cikmadi, sisik.",
    "Hamileyim, alt karnimda kramp var. Kanama yok ama endiseliyim.",
    "Yasli annem dustu, ayaga kalkamiyor ama konusuyor. Kalcasi agriyor.",
    "Cocugum yanlislikla ilac yutmus olabilir. Uyanik ama uykulu.",
    "Boyle bir agri ilk kez, sag gogsum altinda. Nefes alirken artiyor.",
    "Araba ile hafif bir kaza, boynum agriyor, yuruyebiliyorum.",
    "Diskimda kan gordum, miktar az gibi. Panikliyorum.",
    "Genc bir adam, yerde yatiyor, bilincli ama konusamiyor gibi.",
    "Astim krizim baslar gibi, inhalerim yok yaninda.",
]

AMBIG_TEMPLATES_EN = [
    "My fever is 38.5C, I'm shivering. Headache too.",
    "I have vertigo and vomiting. Can barely stand.",
    "Cut on my arm, bleeding is mild but looks deep. May need stitches.",
    "Severe abdominal pain on the right side, for a few hours.",
    "My kid fell and hit his head. Cried, now calm but drowsy.",
    "Possible allergic reaction, my face is swelling a bit. I can still breathe.",
    "Pressure in my chest radiating slightly to my left arm. I'm anxious.",
    "High fever with shivering, I feel slightly confused.",
    "Sudden dizziness, had to sit down. Better now but afraid it'll recur.",
    "Cut my hand, still bleeding. Wrapped it but soaking through.",
    "My back locked up, can't move but circulation looks ok.",
    "Burned my hand, red and blistered. Ring is stuck due to swelling.",
    "I'm pregnant, lower abdominal cramps. No bleeding but worried.",
    "My elderly mother fell, can't stand but can speak. Her hip hurts.",
    "My child may have swallowed medication by accident. Awake but drowsy.",
    "Never felt pain like this, under the right chest. Worse on breathing.",
    "Minor car accident, my neck hurts but I can walk.",
    "I saw blood in my stool, small amount. I'm anxious.",
    "A young man is lying on the ground, conscious but not talking clearly.",
    "I feel an asthma attack coming, my inhaler is not with me.",
]

AMBIG_FILLERS_TR = [
    "Ambulans mi cagirayim?",
    "Hastaneye nasil gidecegimi bilmiyorum.",
    "Oglumu goturmeli miyim acile?",
    "Bir sure gozlemlemeli miyim?",
    "Ilacim etkisini gosterir mi yoksa beklemeyelim mi?",
    "Evde yalniz kalmayi istemiyorum simdi.",
]

AMBIG_FILLERS_EN = [
    "Should I call an ambulance?",
    "I don't know how to get to the hospital.",
    "Should I take my son to the ER?",
    "Should I observe for a while?",
    "Will my medication work or should we not wait?",
    "I don't want to be alone at home right now.",
]


def _compose(templates, fillers, rng: random.Random) -> str:
    base = rng.choice(templates)
    if rng.random() < 0.45:
        return f"{base} {rng.choice(fillers)}"
    return base


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "text_en",
                "source",
                "label_category_gold",
                "label_triage_gold",
                "red_flags_gold",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def generate_minor(n: int, rng: random.Random) -> list[dict]:
    rows: list[dict] = []
    n_tr = int(n * 0.6)
    n_en = n - n_tr
    for i in range(n_tr):
        text = _compose(MINOR_TEMPLATES_TR, MINOR_FILLERS_TR, rng)
        rows.append(
            {
                "case_id": f"minor_tr_{i:04d}",
                "text_en": text,
                "source": "minor_complaints_v1",
                "label_category_gold": "medical",
                "label_triage_gold": "NON_URGENT",
                "red_flags_gold": 0,
            }
        )
    for i in range(n_en):
        text = _compose(MINOR_TEMPLATES_EN, MINOR_FILLERS_EN, rng)
        rows.append(
            {
                "case_id": f"minor_en_{i:04d}",
                "text_en": text,
                "source": "minor_complaints_v1",
                "label_category_gold": "medical",
                "label_triage_gold": "NON_URGENT",
                "red_flags_gold": 0,
            }
        )
    rng.shuffle(rows)
    return rows


def generate_ambiguous(n: int, rng: random.Random) -> list[dict]:
    rows: list[dict] = []
    n_tr = int(n * 0.6)
    n_en = n - n_tr
    for i in range(n_tr):
        text = _compose(AMBIG_TEMPLATES_TR, AMBIG_FILLERS_TR, rng)
        rows.append(
            {
                "case_id": f"ambig_tr_{i:04d}",
                "text_en": text,
                "source": "ambiguous_symptoms_v1",
                "label_category_gold": "medical",
                "label_triage_gold": "URGENT",
                "red_flags_gold": 0,
            }
        )
    for i in range(n_en):
        text = _compose(AMBIG_TEMPLATES_EN, AMBIG_FILLERS_EN, rng)
        rows.append(
            {
                "case_id": f"ambig_en_{i:04d}",
                "text_en": text,
                "source": "ambiguous_symptoms_v1",
                "label_category_gold": "medical",
                "label_triage_gold": "URGENT",
                "red_flags_gold": 0,
            }
        )
    rng.shuffle(rows)
    return rows


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate minor + ambiguous triage synthetic datasets.")
    p.add_argument("--minor-count", type=int, default=1500)
    p.add_argument("--ambig-count", type=int, default=700)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    rng = random.Random(args.seed)

    minor_rows = generate_minor(args.minor_count, rng)
    _write_csv(MINOR_OUT, minor_rows)
    print(f"Wrote {len(minor_rows)} rows to {MINOR_OUT}")

    ambig_rows = generate_ambiguous(args.ambig_count, rng)
    _write_csv(AMBIG_OUT, ambig_rows)
    print(f"Wrote {len(ambig_rows)} rows to {AMBIG_OUT}")


if __name__ == "__main__":
    main()
