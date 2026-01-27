from pathlib import Path
import random
import csv

PROJECT = Path(r"C:\Users\halil melih\BitirmeProject\bitirme_project")
CHUNKS = PROJECT / "data" / "interim" / "chunks"
OUT = PROJECT / "data" / "labels" / "train_template.csv"

wav_files = sorted(CHUNKS.glob("*.wav"))
random.seed(42)

N = min(200, len(wav_files))
chosen = random.sample(wav_files, N)

OUT.parent.mkdir(parents=True, exist_ok=True)

with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["path", "text"])
    for p in chosen:
        rel = p.relative_to(PROJECT).as_posix()
        w.writerow([rel, ""])   # text boş, sen dolduracaksın

print("Wrote:", OUT, "rows:", N)
