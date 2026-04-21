#!/usr/bin/env python3
"""
recategorize_synthetic_tr.py
============================
synthetic_triage_cases_final_tr.csv'deki tum satirlari (~5000) `label_category`
yok oldugu icin merge sirasinda 'other' olarak dusuyor. Bu script Turkce
anahtar kelime kuralina dayanarak her satira medical / fire / crime / other
ataar ve yeni bir CSV uretir.

Cikti: data/labels/synthetic_triage_cases_final_tr_categorized.csv
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "data" / "labels" / "synthetic_triage_cases_final_tr.csv"
OUT = PROJECT_ROOT / "data" / "labels" / "synthetic_triage_cases_final_tr_categorized.csv"

FIRE_RE = re.compile(
    r"\b(yang[iı]n|duman|alev|patl(a|ad)|yanm(is|ış|iyor|ıyor)|"
    r"itfaiye|bac(a|ada)|kiv[iı]lc[iı]m|benzin|dogalgaz|doğalgaz|"
    r"sigara k(o|ö)kusu|yang[iı]n alarm|kazan patlad)\b",
    re.IGNORECASE,
)
CRIME_RE = re.compile(
    r"\b(sil(a|â)h|bicak|bıçak|vuruldu|hirsiz|hırsız|kap(i|ı)y(i|ı) kirdi|"
    r"polis|pol(i|ı)s|kacirild|kaçırıld|cinayet|saldir|saldır|"
    r"darp|dov(u|ü)l|dövül|kavga|rehine|stalk)\b",
    re.IGNORECASE,
)
MEDICAL_RE = re.compile(
    r"\b(nefes alamıyor|nefes alamiyor|baygin|baygın|kalp|kalbim|kalp atış|"
    r"ambulans|doktor|hastane|yarali|yaralı|kan var|kan kusuyor|ates|ateş|"
    r"bilinc(i|ı)n(i|ı)|bilinç|zehirl|ilac ictig|ilaç içtiğ|goz karar|göz karar|"
    r"bulant|kusma|nefes darl|astim|astım|tansiyon|seker|şeker|epilepsi|"
    r"fenalast|fenalaş|komada|koma|alerji|yigild|yığıld|yigilıyor|"
    r"morar(d|m)|nefes|panik atak|sara|sancı|sanci)\b",
    re.IGNORECASE,
)


def _infer_category(text: str) -> str:
    txt = (text or "").lower()
    fire_hit = bool(FIRE_RE.search(txt))
    crime_hit = bool(CRIME_RE.search(txt))
    med_hit = bool(MEDICAL_RE.search(txt))
    # Oncelik: fire > crime > medical (acil yangin en nadir ama en kritik)
    # Birlikte gelirse agir olan kategoriye at.
    if fire_hit:
        return "fire"
    if crime_hit:
        return "crime"
    if med_hit:
        return "medical"
    return "other"


def main() -> None:
    if not SRC.exists():
        print(f"Source not found: {SRC}")
        return
    df = pd.read_csv(SRC)
    df["text_en"] = df["text"].astype(str).fillna("")  # merge script 'text_en' bekliyor
    df["label_category_gold"] = df["text_en"].apply(_infer_category)
    df["label_triage_gold"] = df["label_triage"].astype(str)
    if "red_flags_gold" not in df.columns:
        df["red_flags_gold"] = 0
    df["source"] = "synthetic_tr_final_cat"

    keep = ["case_id", "text_en", "source", "label_category_gold", "label_triage_gold", "red_flags_gold"]
    out = df[keep].copy()
    out.to_csv(OUT, index=False)

    from collections import Counter
    cat_counts = Counter(out["label_category_gold"])
    tri_counts = Counter(out["label_triage_gold"])
    print(f"Wrote {len(out)} rows to {OUT}")
    print("Category distribution:", cat_counts)
    print("Triage distribution:  ", tri_counts)


if __name__ == "__main__":
    main()
