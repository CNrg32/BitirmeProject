#!/usr/bin/env python3
"""
generate_multiturn_dialogs.py
=============================
Multi-turn dispatcher-user sentetik dialoglarindan turn-snapshot kayitlari
uretir. Her kullanici turunda o noktaya kadar birikmis history [USER]/[ASSISTANT]
formatinda tek metin olarak yazilir ve o turun gold etiketi atanir.

Bu sekilde model "dialog ilerledikce dogru etikete dogru evril" sinyalini
dogrudan ogreniyor.

Cikti: data/labels/multiturn_dialogs_v1.jsonl
Her satir:
  {
    "text": "[USER] ... [ASSISTANT] ... [USER] ...",
    "triage_level": "URGENT|CRITICAL|NON_URGENT",
    "category": "medical|fire|crime|other",
    "red_flag_present": 0|1,
    "input_quality": "meaningful|gibberish|out_of_scope",
    "source": "multiturn_<dialog_id>_t<turn_index>",
    "dialog_id": "<id>",
    "turn_index": <int>,
    "sample_weight": 2.0
  }
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "data" / "labels" / "multiturn_dialogs_v1.jsonl"


def _user(text, triage, cat, rf, quality="meaningful"):
    return {
        "role": "user",
        "text": text,
        "triage": triage,
        "category": cat,
        "rf": int(rf),
        "quality": quality,
    }


def _bot(text):
    return {"role": "assistant", "text": text}


DIALOGS = [
    # ---- Medical: atesli cocuk → konvulsiyon (TR)
    {
        "id": "md_001",
        "turns": [
            _user("Cocugum atesli galiba, biraz halsiz.", "URGENT", "medical", 0),
            _bot("Kac derece ates olcebildiniz mi, nefesi normal mi?"),
            _user("39 derece, nefes normal ama uykulu.", "URGENT", "medical", 0),
            _bot("Konvulsiyon (titreme) veya goz kaymasi var mi?"),
            _user("Evet, simdi titriyor ve gozleri kayiyor!", "CRITICAL", "medical", 1),
        ],
    },
    {
        "id": "md_002",
        "turns": [
            _user("Dun aksamdan beri basim agriyor.", "NON_URGENT", "medical", 0),
            _bot("Agri seviyesi 1-10 arasinda kac, baska belirti var mi?"),
            _user("3-4 seviyesinde, baska bir sey yok.", "NON_URGENT", "medical", 0),
            _bot("Ates, bulanti, gorme bozuklugu hic var mi?"),
            _user("Yok, sadece gerilim gibi.", "NON_URGENT", "medical", 0),
        ],
    },
    # ---- Medical: hafif → ciddi kotulasma
    {
        "id": "md_003",
        "turns": [
            _user("Babam gogsunu sikisiyor diyor.", "URGENT", "medical", 0),
            _bot("Agri kola veya cenesine vuruyor mu?"),
            _user("Sol koluna yayiliyor.", "URGENT", "medical", 0),
            _bot("Bilinci yerinde mi, terliyor mu?"),
            _user("Simdi bayildi, nefes almiyor gibi!", "CRITICAL", "medical", 1),
        ],
    },
    {
        "id": "md_004",
        "turns": [
            _user("Hi, I think I have a cold, sneezing a lot.", "NON_URGENT", "medical", 0),
            _bot("Do you have fever or shortness of breath?"),
            _user("No fever, breathing is fine.", "NON_URGENT", "medical", 0),
            _bot("Any chest pain or confusion?"),
            _user("None, just sneezing and sore throat.", "NON_URGENT", "medical", 0),
        ],
    },
    {
        "id": "md_005",
        "turns": [
            _user("My mother fell in the bathroom.", "URGENT", "medical", 0),
            _bot("Is she conscious and can she move?"),
            _user("She's awake but her hip hurts a lot.", "URGENT", "medical", 0),
            _bot("Is there bleeding or is she pale?"),
            _user("No blood but she's very pale and sweating.", "URGENT", "medical", 0),
            _bot("Has she lost consciousness since?"),
            _user("She just passed out!", "CRITICAL", "medical", 1),
        ],
    },
    # ---- Medical: ambigu tehsis
    {
        "id": "md_006",
        "turns": [
            _user("Kendimi halsiz hissediyorum.", "NON_URGENT", "medical", 0),
            _bot("Ne zamandir bu sekilde, baska belirti var mi?"),
            _user("Iki gundur. Hafif mide bulantisi da var.", "NON_URGENT", "medical", 0),
            _bot("Ates, bas donmesi, gogus agrisi hissediyor musunuz?"),
            _user("Hafif bas donmesi var ama stres olabilir.", "NON_URGENT", "medical", 0),
        ],
    },
    # ---- Fire: ambigu → kritik
    {
        "id": "fr_001",
        "turns": [
            _user("Apartman koridorunda duman kokusu var.", "URGENT", "fire", 0),
            _bot("Alev goruyor musunuz, binadaki diger komsulari uyardiniz mi?"),
            _user("Alev yok ama koku artiyor.", "URGENT", "fire", 0),
            _bot("Dumanin kaynagi hangi kattan geliyor gibi?"),
            _user("Alt kattan simdi alev cikti! Kapinin altindan kiziyor!", "CRITICAL", "fire", 1),
        ],
    },
    {
        "id": "fr_002",
        "turns": [
            _user("Komsumdan mangal kokusu geliyor.", "NON_URGENT", "fire", 0),
            _bot("Alev veya duman goruyor musunuz?"),
            _user("Hayir sadece koku, balkondan geliyor.", "NON_URGENT", "fire", 0),
            _bot("Teskilati aramak ister misiniz?"),
            _user("Yok, sadece bilgi vermek istedim.", "NON_URGENT", "fire", 0),
        ],
    },
    {
        "id": "fr_003",
        "turns": [
            _user("Ocakta tava tutustu!", "CRITICAL", "fire", 1),
            _bot("Sondurucunuz var mi veya tencere kapagini kapatabilir misiniz?"),
            _user("Kapak kapattim simdi, alev soner gibi.", "URGENT", "fire", 0),
            _bot("Duman devam ediyor mu, baska yeriyle yayildi mi?"),
            _user("Duman var ama kontrol altinda.", "URGENT", "fire", 0),
        ],
    },
    {
        "id": "fr_004",
        "turns": [
            _user("Electrical smell from an outlet.", "URGENT", "fire", 0),
            _bot("Any smoke or sparks visible?"),
            _user("No smoke yet, just smell.", "URGENT", "fire", 0),
            _bot("Did you turn the breaker off?"),
            _user("Just did. Now I see sparks!", "CRITICAL", "fire", 1),
        ],
    },
    # ---- Crime: gurultu → fiziksel kavga
    {
        "id": "cr_001",
        "turns": [
            _user("Komsumda gurultu var gece yarisi.", "NON_URGENT", "crime", 0),
            _bot("Kavga sesi mi, muzik mi?"),
            _user("Bagristiyorlar ama fiziksel kavga sanmam.", "URGENT", "crime", 0),
            _bot("Siddet sesi duyuyor musunuz?"),
            _user("Simdi bir kadin 'yardim' diye bagiriyor!", "CRITICAL", "crime", 1),
        ],
    },
    {
        "id": "cr_002",
        "turns": [
            _user("Sokakta supheli bir arac var.", "NON_URGENT", "crime", 0),
            _bot("Kac kisi var, saldirgan gozukuyorlar mi?"),
            _user("Iki kisi, yakinima gelmediler ama izliyorlar.", "URGENT", "crime", 0),
            _bot("Sizinle iletisime gecmeye calistilar mi?"),
            _user("Simdi arabadan indi ve bana dogru geliyor!", "URGENT", "crime", 0),
        ],
    },
    {
        "id": "cr_003",
        "turns": [
            _user("Evime donunce kapi acikti.", "URGENT", "crime", 0),
            _bot("Icerde kimse var gibi mi, herhangi bir ses duyuyor musunuz?"),
            _user("Ses yok ama icerde biri varmis gibi.", "URGENT", "crime", 0),
            _bot("Evin icinde bir yerde saklanip guvende kalabilir misiniz?"),
            _user("Simdi icerden bir kisi cikti, silahli!", "CRITICAL", "crime", 1),
        ],
    },
    {
        "id": "cr_004",
        "turns": [
            _user("Sokakta birkac genc bagirisiyor.", "NON_URGENT", "crime", 0),
            _bot("Size veya kimseye yonelik tehdit var mi?"),
            _user("Hayir, sadece muzik acik.", "NON_URGENT", "crime", 0),
            _bot("Sikayet etmek ister misiniz?"),
            _user("Evet, gurultu sikayeti olsun yeter.", "NON_URGENT", "crime", 0),
        ],
    },
    # ---- Out of scope / gibberish / yanlis arama
    {
        "id": "ns_001",
        "turns": [
            _user("Merhaba, pizza siparisi icin arayabilir miyim?", "NON_URGENT", "other", 0, "out_of_scope"),
            _bot("Burasi acil servis. Pizza icin yanlis numara. Baska bir acil durum var mi?"),
            _user("Yok, yanlis aradim ozur dilerim.", "NON_URGENT", "other", 0, "out_of_scope"),
        ],
    },
    {
        "id": "ns_002",
        "turns": [
            _user("asdfasdf deniyorum alo", "NON_URGENT", "other", 0, "gibberish"),
            _bot("Net bir talebiniz var mi?"),
            _user("yok yok pranktim kusura bakmayin", "NON_URGENT", "other", 0, "gibberish"),
        ],
    },
    {
        "id": "ns_003",
        "turns": [
            _user("Hava durumu icin arayabilir miyim?", "NON_URGENT", "other", 0, "out_of_scope"),
            _bot("Hayir, burasi acil servis hatti."),
            _user("Tamam ozur dilerim.", "NON_URGENT", "other", 0, "out_of_scope"),
        ],
    },
    # ---- Ambigu baslangic → net ciddi
    {
        "id": "md_007",
        "turns": [
            _user("Bir arkadasim kendini iyi hissetmiyor.", "URGENT", "medical", 0),
            _bot("Ne tur bir kotu hissetme var, bilinci acik mi?"),
            _user("Sac kokleri agriyor, zayif hissediyor dedi.", "URGENT", "medical", 0),
            _bot("Morarma, terleme, nefes darligi var mi?"),
            _user("Morardi ve simdi yere yigildi!", "CRITICAL", "medical", 1),
        ],
    },
    {
        "id": "md_008",
        "turns": [
            _user("Cocugum suyu gidisiyor, titriyor biraz.", "URGENT", "medical", 0),
            _bot("Ates olctunuz mu?"),
            _user("38.5 derece, uc gundur devam ediyor.", "URGENT", "medical", 0),
            _bot("Konvulsiyon, bilincte degisiklik var mi?"),
            _user("Hayir, sadece ates ve bitkinlik.", "URGENT", "medical", 0),
        ],
    },
    # ---- CRITICAL basindan net
    {
        "id": "md_009",
        "turns": [
            _user("Nefes alamiyor babam!", "CRITICAL", "medical", 1),
            _bot("Bilinci acik mi, gogus agrisi var mi?"),
            _user("Bilinci kapaniyor simdi!", "CRITICAL", "medical", 1),
            _bot("Kalp masaji biliyor musunuz? Adres?"),
            _user("Adres merkez mahalle 5 nolu ev.", "CRITICAL", "medical", 1),
        ],
    },
    {
        "id": "cr_005",
        "turns": [
            _user("Bicakli biri saldirdi bana!", "CRITICAL", "crime", 1),
            _bot("Su anda guvende misiniz, kac kisi var?"),
            _user("Bir kisi, bircak attim simdi kactim.", "CRITICAL", "crime", 1),
            _bot("Yara var mi, konumunuz nedir?"),
            _user("Kolum kanamakta, merkezde taksim'deyim.", "CRITICAL", "crime", 1),
        ],
    },
    # ---- NON_URGENT dialog boyunca kararli
    {
        "id": "cr_006",
        "turns": [
            _user("Sokakta eski kayip kopek arama ilani var, benim bulmus olabilir miyim?", "NON_URGENT", "crime", 0),
            _bot("Kopegin ismi veya sahibinin numarasini biliyor musunuz?"),
            _user("Hayir, sadece sahibini bulmak istiyorum.", "NON_URGENT", "crime", 0),
            _bot("Belediye hayvan biriminden yardim alabilirsiniz."),
            _user("Tesekkurler, deneyecegim.", "NON_URGENT", "crime", 0),
        ],
    },
    {
        "id": "md_010",
        "turns": [
            _user("Mild back pain, not sure what to do.", "NON_URGENT", "medical", 0),
            _bot("Can you move, or is it paralyzing?"),
            _user("I can walk, just stiff.", "NON_URGENT", "medical", 0),
            _bot("Fever or numbness anywhere?"),
            _user("None, just muscle pain.", "NON_URGENT", "medical", 0),
        ],
    },
    # ---- URGENT sabit
    {
        "id": "md_011",
        "turns": [
            _user("Alerjik reaksiyon olabilir, yuzum sisti.", "URGENT", "medical", 0),
            _bot("Nefes alabiliyor musunuz, bogaz sisi var mi?"),
            _user("Biraz zor nefes aliyorum ama mumkun.", "URGENT", "medical", 0),
            _bot("Epi-pen'iniz var mi, daha once alerjiniz oldu mu?"),
            _user("Yok, ilk kez oluyor.", "URGENT", "medical", 0),
        ],
    },
    {
        "id": "fr_005",
        "turns": [
            _user("Baca dumani cok yogun, komsular endise ediyor.", "URGENT", "fire", 0),
            _bot("Alev goruyor musunuz?"),
            _user("Alev yok ama duman normalden cok fazla.", "URGENT", "fire", 0),
            _bot("Itfaiyeyi aramak ister misiniz? Tedbiren gecmek en iyisi."),
            _user("Evet, ariyalim.", "URGENT", "fire", 0),
        ],
    },
]


def _flatten_turns(dialog: dict) -> list[dict]:
    rows: list[dict] = []
    history_parts: list[str] = []
    for turn_idx, turn in enumerate(dialog["turns"]):
        role = "USER" if turn["role"] == "user" else "ASSISTANT"
        history_parts.append(f"[{role}] {turn['text']}")
        if turn["role"] != "user":
            continue
        history_text = " ".join(history_parts)
        rows.append({
            "text": history_text,
            "triage_level": turn["triage"],
            "category": turn["category"],
            "red_flag_present": int(turn["rf"]),
            "input_quality": turn.get("quality", "meaningful"),
            "source": f"multiturn_{dialog['id']}_t{turn_idx}",
            "dialog_id": dialog["id"],
            "turn_index": turn_idx,
            "sample_weight": 2.0,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    all_rows: list[dict] = []
    for d in DIALOGS:
        all_rows.extend(_flatten_turns(d))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    tri = Counter(r["triage_level"] for r in all_rows)
    iq = Counter(r["input_quality"] for r in all_rows)
    print(f"Wrote {len(all_rows)} turn-snapshots from {len(DIALOGS)} dialogs to {out_path}")
    print("Triage:", tri)
    print("Input quality:", iq)


if __name__ == "__main__":
    main()
