#!/usr/bin/env python3
"""
generate_panic_synthetic_final_tr.py
====================================
Generates 1500 Turkish synthetic panic risk score training data.
Optimized for panic_risk_score formula: 0.70×text_panic_score + 0.30×risk_score

Data split:
- CRITICAL: 750 examples (0.65-0.85 target panic_risk_score)
- URGENT: 550 examples (0.40-0.65)
- MONITOR: 200 examples (0.10-0.40)
"""

import csv
import random
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "labels"
OUT_CSV = DATA_DIR / "synthetic_triage_cases_final_tr.csv"

# Random seed for reproducibility
random.seed(42)

# ============================================
# CRITICAL Templates (750 examples)
# 0.65-0.85 panic_risk_score range
# ============================================

CRITICAL_TEMPLATES = [
    "Kan var! {entity} {critical_action}! Lütfen {help}!",
    "{stretched_verb}! {entity} {dying}!",
    "{panic_word} {panic_word} {panic_word}! {emergency}!",
    "{entity} ağır yaralı! Kan var! Acil yardım!",
    "Silah var! {critical_action}! Çok tehlikeli!",
    "{stretched_panic}! Hava alamıyorum! Yardım edin!",
    "Doktor lazım! {entity} bilincini kaybetti!",
    "{entity} yere yığıldı! Baygın! Hemen ambulans gönderin!",
    "{entity} fenalaştı ve nefes almıyor! Çabuk gelin!",
    "Kalp atışım çok hızlı! {dying}! Yardım!",
    "Göğsüm çok kötü sıkışıyor! Nefes alamıyorum! Yardım edin!",
    "Önümde biri yere düştü ve baygın yatıyor! Hemen gelin!",
    "Çocuğum nefes alamıyor! Morardı! Ambulans lazım!",
    "{entity} kan kusuyor! Çok kötü durumda! Acil yardım!",
    "Baygın biri var! Tepki vermiyor! Hemen gelin!",
    "Nefes darlığı çok arttı! Yığıldım! Yardım edin!",
    "{emergency_phrase} {emergency_phrase} {emergency_phrase}!",
    "Yardım yardım yardım! Acil acil acil!",
    "Hemen hemen hemen! Çabuk çabuk!",
    "{panic_word} {panic_word} {panic_word} {panic_word}!",
    "Lütfen lütfen lütfen! Şimdi şimdi şimdi!",
    "Yaaaardım! Nefes alamıyorum!",
    "Aciiiiil! Ölüyorummmmm!",
    "Kaaaaan! Çoooook kan!",
    "Evde yangın var! Kurtarın! Çabuk!",
    "Trafik kazası! Birçok yaralı! Ambulans lazım!",
    "Çocuğum yanlış ilaç içti! Zehirlenmiş olabilir! Yardım!",
    "Biri vuruldu! Çok kanıyor! Ambulans gönderin!",
]

# ============================================
# URGENT Templates (550 examples)
# 0.40-0.65 panic_risk_score range
# ============================================

URGENT_TEMPLATES = [
    "Göğüs ağrısı var. Hava almakta zorluk çekiyorum.",
    "{entity} {injury}. Kan fışkırıyor. Hemen gelmesini istiyorum.",
    "Ateş var, {temp} derece. Nefes alırken {pain_phrase}.",
    "Düştüm. Kolum çok ağrıyor ve şişti.",
    "Önümde biri yere düştü. Kalkamıyor ve çok acıyor diyor.",
    "{entity} fenalaştı. Şu an oturuyor ama iyi görünmüyor.",
    "Bir kişi baygın gibi oldu ama şimdi gözünü açtı.",
    "Nefes darlığı var. Konuşurken zorlanıyor.",
    "Canım çok acıyor. Özellikle {body_part} tarafımda ağrı var.",
    "{entity} {moderate_injury}. Hareket etmekte {difficulty}.",
    "Başım çok acıyor. Gözüm kararıyor. Çok kötüyüm.",
    "Alerjik reaksiyon olabilir. Yüzüm şişti. {difficulty}.",
    "{entity} {trauma_injury}. Kanama azaldı ama ağrısı sürüyor.",
    "Kasığımda {pain_phrase}. Oturmakta zorlanıyorum.",
    "Biri yere düştü, başını çarptı ve sersemledi.",
    "Göğsümde baskı var. Nefes darlığı başladı.",
    "Ağrı ağrı ağrı. Durmuyor. Tıbbi yardım lazım.",
    "Nefes nefes kaldım. Yardım lazım.",
    "Kolumun üstüne düştüm, çok acıyor ama kanama yok.",
    "Zehirlenmiş olabilir. Kontrol istiyorum.",
]

# ============================================
# MONITOR Templates (200 examples)
# 0.10-0.40 panic_risk_score range
# ============================================

MONITOR_TEMPLATES = [
    "Başım ağrıyor. Normal ilaçla geçer sanırım.",
    "Kolumda hafif ağrı var. Kas krampı olabilir.",
    "Bugün biraz yorgunum. Uykusuzluktan olabilir.",
    "Midem biraz bozuk. Yemekten sonra başladı.",
    "Boğazımda {minor_symptom} var. Soğuk algınlığı gibi geliyor.",
    "Derimde kızarıklık var. Kaşıntılı ama ağrısız.",
    "Ufak çizik aldım. Yüzeysel gibi görünüyor.",
    "Bisikletten yavaşça düştüm. Hafif ağrı var ama iyiyim.",
    "Ayağımı biraz burktum. Üstüne basabiliyorum.",
    "Canım biraz acıyor ama dayanılmayacak gibi değil.",
    "Ne yaptığımı bilmiyorum demiyorum, sadece biraz halsizim.",
    "Hafif öksürük var. Dinlenince geçebilir.",
    "Genel kontrol istiyorum. Bir yıl oldu.",
    "Aşı ihtiyacım var mı? Tavsiye verir misin?",
    "Beslenme danışmanı önerir misin? Sağlık için.",
    "Spor yaparken nelere dikkat etmeliyim?",
    "Uyku düzenim kötü. İyileştirmek için tavsiye?",
]

# ============================================
# Variable Pools
# ============================================

ENTITIES = [
    "Çocuğum", "Eşim", "Babam", "Annem", "Kardeşim", "Arkadaşım",
    "Komşum", "Yaşlı bir adam", "Yaşlı bir kadın", "Hasta", "Bir adam", "Bir kadın"
]

PANIC_WORDS = ["Yardım", "Acil", "Kurtarın", "Hemen", "Lütfen", "Çabuk"]
STRETCHED_VERB = ["Yaaaardım", "Aciiiiil", "İmdaaat", "Kurtarrrrın"]
DYING_WORDS = ["ölüyorum", "ölüyor", "bilincini kaybetti", "nefes alamıyor", "bayıldı"]
CRITICAL_ACTIONS = ["kanıyor", "vuruldu", "bıçaklandı", "nefes almıyor", "yere düştü", "fenalaştı"]
HELP_WORDS = ["çabuk", "acil", "şimdi", "hemen", "gelin"]
EMERGENCY_PHRASE = ["Nefes alamıyorum", "Bilincini kaybetti", "Kalp atışı yok", "Baygın yatıyor"]
INJURY = ["kolunu kırdı", "başını çarptı", "yaralandı", "kanıyor", "ayağını burktu"]
PAIN_PHRASES = ["ağrı oluyor", "çok acıyor", "yanma oluyor", "batma hissi var"]
DIFFICULTY = ["zorluk çekiyor", "hareket edemiyor", "zor nefes alıyor", "ayağa kalkamıyor"]
TEMP = ["38", "39", "40", "41"]
TRAUMA = ["travma geçirdi", "sert düştü", "başını vurdu", "yaralandı"]
BODY_PARTS = ["göğsümde", "kolumda", "bacağımda", "başımda", "karnımda"]
MINOR_SYMPTOM = ["kaşıntı", "kızarıklık", "şişlik", "hafif yanma"]

MANUAL_CRITICAL_EXAMPLES = [
    "Önümde biri yere düştü, nefes almıyor, hemen ambulans gönderin.",
    "Babam aniden fenalaştı ve baygın yatıyor, tepki vermiyor.",
    "Çocuğum morardı, nefes alamıyor, çok acil yardım lazım.",
    "Bir adam yere yığıldı, kalp krizi geçiriyor olabilir.",
    "Annem bilincini kaybetti, nabzını alamıyoruz, çabuk gelin.",
    "Biri başından vuruldu, çok kan kaybediyor, hemen ambulans lazım.",
    "Yardim edin, annem nefes alamiyom diyo ve bayildi.",
    "Gogsum sikisiyo, nefes alamiyom, yere yigilicam.",
]

MANUAL_URGENT_EXAMPLES = [
    "Canım çok acıyor, göğsüm sıkışıyor ve nefes darlığım var.",
    "Önümde biri yere düştü, şimdi konuşuyor ama kalkamıyor.",
    "Arkadaşım fenalaştı, başı dönüyor ve ayakta duramıyor.",
    "Bir kişi baygın gibi oldu, kısa süre sonra kendine geldi ama kötü durumda.",
    "Merdivenden düştüm, başımı çarptım, çok ağrıyor.",
    "Göğüs ağrım var, nefes alırken daha da artıyor.",
    "Nefes alamiyom gibi hissediyorum, gogsum sikisiyo.",
    "Bir anda bayildi ama sonra gozunu acti, yine de kotu.",
]

MANUAL_MONITOR_EXAMPLES = [
    "Kolum biraz ağrıyor ama hareket ettirebiliyorum.",
    "Bisikletten yavaşça düştüm, hafif sızlıyor ama iyiyim.",
    "Boğazımda hafif yanma var, acil gibi durmuyor.",
    "Başım ağrıyor ama dinlenince geçecek gibi.",
    "Ayağımı biraz burktum, üzerine basabiliyorum.",
    "Biraz halsizim, ciddi bir durum olduğunu sanmıyorum.",
]

TYPO_REPLACEMENTS = [
    ("yardım", "yardim"),
    ("çok", "cok"),
    ("ağrıyor", "agriyor"),
    ("acıyor", "aciyo"),
    ("düştü", "dustu"),
    ("fenalaştı", "fenalasti"),
    ("baygın", "baygin"),
    ("nefes", "nefes"),
    ("göğüs", "gogus"),
    ("çabuk", "cabuk"),
]

STT_REPLACEMENTS = [
    ("yardım edin", "yardım edın"),
    ("nefes alamıyor", "nefes alamiyo"),
    ("nefes alamıyorum", "nefes alamiyom"),
    ("yere düştü", "yere dustu"),
    ("çok acıyor", "cok aciyo"),
    ("bayıldı", "bayildi"),
    ("baygın", "baygin"),
    ("baygın yatıyor", "baygin yatiyo"),
    ("göğsüm sıkışıyor", "gogsum sikisiyo"),
    ("göğsümde baskı", "gogsumde baski"),
    ("başını çarptı", "basini carpti"),
]

FILLER_PREFIXES = [
    "Alo, ",
    "Merhaba, ",
    "Lütfen, ",
    "Şey, ",
    "Bakar mısınız, ",
]


def _normalize_text(text: str) -> str:
    text = " ".join(text.split())
    return text.strip()


def _apply_typo_noise(text: str) -> str:
    mutated = text
    replacements = TYPO_REPLACEMENTS.copy()
    random.shuffle(replacements)
    for src, dst in replacements[: random.randint(1, 2)]:
        if src in mutated.lower():
            mutated = mutated.replace(src, dst)
            mutated = mutated.replace(src.capitalize(), dst.capitalize())
    return _normalize_text(mutated)


def _apply_stt_noise(text: str) -> str:
    mutated = text.lower()
    replacements = STT_REPLACEMENTS.copy()
    random.shuffle(replacements)
    for src, dst in replacements[: random.randint(1, 2)]:
        mutated = mutated.replace(src, dst)
    if random.random() < 0.35:
        mutated = mutated.replace(",", "")
    if random.random() < 0.35:
        mutated = mutated.replace(".", "")
    return _normalize_text(mutated)


def _apply_style_variation(text: str, label: str) -> str:
    varied = text
    if random.random() < 0.20:
        varied = random.choice(FILLER_PREFIXES) + varied[0].lower() + varied[1:]
    if label != "NON_URGENT" and random.random() < 0.30:
        varied = _apply_typo_noise(varied)
    if label != "NON_URGENT" and random.random() < 0.18:
        varied = _apply_stt_noise(varied)
    if label == "NON_URGENT" and random.random() < 0.15:
        varied = _apply_typo_noise(varied)
    return _normalize_text(varied)

def generate_critical_examples(count=750):
    """Generate CRITICAL examples (0.65-0.85 panic_risk_score)"""
    examples = []
    case_id = 1000

    for manual_text in MANUAL_CRITICAL_EXAMPLES[: min(len(MANUAL_CRITICAL_EXAMPLES), count)]:
        examples.append({
            "case_id": f"TR_CRIT_{case_id:05d}",
            "text": _apply_style_variation(manual_text, "CRITICAL"),
            "source": "synthetic_tr_final",
            "label_triage": "CRITICAL",
            "red_flags_gold": 1,
            "target_panic_score": round(random.uniform(0.74, 0.86), 2),
        })
        case_id += 1
    
    for _ in range(count - len(examples)):
        template = random.choice(CRITICAL_TEMPLATES)
        
        # Fill template
        example_text = template.format(
            entity=random.choice(ENTITIES),
            critical_action=random.choice(CRITICAL_ACTIONS),
            help=random.choice(HELP_WORDS),
            stretched_verb=random.choice(STRETCHED_VERB),
            dying=random.choice(DYING_WORDS),
            panic_word=random.choice(PANIC_WORDS),
            emergency=random.choice(EMERGENCY_PHRASE),
            stretched_panic=random.choice(STRETCHED_VERB),
            emergency_phrase=random.choice(EMERGENCY_PHRASE),
        )
        
        example_text = _apply_style_variation(example_text, "CRITICAL")

        examples.append({
            "case_id": f"TR_CRIT_{case_id:05d}",
            "text": example_text,
            "source": "synthetic_tr_final",
            "label_triage": "CRITICAL",
            "red_flags_gold": 1,
            "target_panic_score": round(random.uniform(0.65, 0.85), 2),
        })
        case_id += 1
    
    return examples

def generate_urgent_examples(count=550):
    """Generate URGENT examples (0.40-0.65 panic_risk_score)"""
    examples = []
    case_id = 2000

    for manual_text in MANUAL_URGENT_EXAMPLES[: min(len(MANUAL_URGENT_EXAMPLES), count)]:
        examples.append({
            "case_id": f"TR_URG_{case_id:05d}",
            "text": _apply_style_variation(manual_text, "URGENT"),
            "source": "synthetic_tr_final",
            "label_triage": "URGENT",
            "red_flags_gold": 0,
            "target_panic_score": round(random.uniform(0.46, 0.64), 2),
        })
        case_id += 1
    
    for _ in range(count - len(examples)):
        template = random.choice(URGENT_TEMPLATES)
        
        example_text = template.format(
            entity=random.choice(ENTITIES),
            injury=random.choice(INJURY),
            temp=random.choice(TEMP),
            pain_phrase=random.choice(PAIN_PHRASES),
            difficulty=random.choice(DIFFICULTY),
            moderate_injury=random.choice(TRAUMA),
            trauma_injury=random.choice(TRAUMA),
            body_part=random.choice(BODY_PARTS),
        )
        
        example_text = _apply_style_variation(example_text, "URGENT")

        examples.append({
            "case_id": f"TR_URG_{case_id:05d}",
            "text": example_text,
            "source": "synthetic_tr_final",
            "label_triage": "URGENT",
            "red_flags_gold": 0,
            "target_panic_score": round(random.uniform(0.40, 0.65), 2),
        })
        case_id += 1
    
    return examples

def generate_monitor_examples(count=200):
    """Generate MONITOR examples (0.10-0.40 panic_risk_score)"""
    examples = []
    case_id = 3000

    for manual_text in MANUAL_MONITOR_EXAMPLES[: min(len(MANUAL_MONITOR_EXAMPLES), count)]:
        examples.append({
            "case_id": f"TR_MON_{case_id:05d}",
            "text": _apply_style_variation(manual_text, "NON_URGENT"),
            "source": "synthetic_tr_final",
            "label_triage": "NON_URGENT",
            "red_flags_gold": 0,
            "target_panic_score": round(random.uniform(0.12, 0.30), 2),
        })
        case_id += 1
    
    for _ in range(count - len(examples)):
        template = random.choice(MONITOR_TEMPLATES)
        
        example_text = template.format(
            minor_symptom=random.choice(MINOR_SYMPTOM),
        )
        
        example_text = _apply_style_variation(example_text, "NON_URGENT")

        examples.append({
            "case_id": f"TR_MON_{case_id:05d}",
            "text": example_text,
            "source": "synthetic_tr_final",
            "label_triage": "NON_URGENT",
            "red_flags_gold": 0,
            "target_panic_score": round(random.uniform(0.10, 0.40), 2),
        })
        case_id += 1
    
    return examples

def main():
    print("=" * 70)
    print("GENERATING TURKISH PANIC RISK SCORE TRAINING DATA")
    print("=" * 70)
    
    # Generate all examples
    print("\n📍 Generating CRITICAL (750 examples)...")
    critical = generate_critical_examples(750)
    
    print("📍 Generating URGENT (550 examples)...")
    urgent = generate_urgent_examples(550)
    
    print("📍 Generating MONITOR (200 examples)...")
    monitor = generate_monitor_examples(200)
    
    # Combine
    all_examples = critical + urgent + monitor
    random.shuffle(all_examples)  # Shuffle for randomness
    
    # Write CSV
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    fieldnames = ["case_id", "text", "source", "label_triage", "red_flags_gold", "target_panic_score"]
    
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_examples)
    
    print(f"\n✅ SUCCESS: {OUT_CSV}")
    print(f"   Total examples: {len(all_examples)}")
    print(f"   CRITICAL: {len(critical)}")
    print(f"   URGENT: {len(urgent)}")
    print(f"   MONITOR: {len(monitor)}")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
