# Bulutta Görüntü Modeli Eğitimi

Büyük veri setini kendi bilgisayarında eğitmek yerine bulutta (Kaggle, Colab, AWS vb.) eğitmek için seçenekler ve adımlar.

---

## Seçenek Özeti

| Platform        | GPU              | Süre / Limit     | Maliyet   | Dataset konumu      |
|----------------|------------------|------------------|-----------|----------------------|
| **Kaggle**     | P100 (ücretsiz)  | ~30 saat/hafta   | Ücretsiz  | Zaten Kaggle'da ✅   |
| **Google Colab** | T4 (ücretsiz) | ~12 saat oturum  | Ücretsiz  | Drive veya yükleme   |
| **Colab Pro**  | Daha iyi GPU     | Daha uzun        | Aylık ücret | Drive / yükleme   |
| **AWS / GCP**  | İstediğin GPU    | Sınırsız         | Saatlik   | S3 / bucket         |

**Öneri:** Dataset zaten [Kaggle'da](https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset) olduğu için **Kaggle Notebooks** ile eğitim en pratik yol. Veriyi tekrar yüklemen gerekmez.

---

## 1. Kaggle ile Eğitim (Önerilen)

### Adım 1: Kaggle hesabı ve dataset

1. [kaggle.com](https://www.kaggle.com) hesabı aç (gerekirse).
2. [UCF Crime Dataset](https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset) sayfasına git.
3. Sağ üstten **"New Notebook"** de (veya **Code** → **New Notebook**).  
   Böylece bu dataset otomatik olarak notebook’a eklenir (`/kaggle/input/ucf-crime-dataset`).

### Adım 2: GPU açma

- Sağ üstte **Settings** (veya **Notebook options**).
- **Accelerator** → **GPU** (örn. **GPU P100**).
- **Save** / **Save Version** ile kaydet.

### Adım 3: Eğitim scriptini kullanma

Projedeki **Kaggle için hazır script** tek hücrede veya birkaç hücrede çalışacak şekilde:

**Seçenek A – Tek notebook hücresi:**  
`scripts/train_image_model_kaggle.py` dosyasının **tüm içeriğini** Kaggle notebook’taki tek bir **Code** hücresine yapıştırıp çalıştır.

**Seçenek B – Sadece gerekli kısımlar:**  
Aşağıdaki gibi notebook’ta:

```python
# 1) Gerekli kütüphaneler (Kaggle'da zaten var)
!pip install --quiet torch torchvision

# 2) Proje scriptini yükle (train_image_model_kaggle.py içeriğini buraya yapıştır)
# veya aşağıdaki komutla repo'dan al (Kaggle'da internet açık)
# !git clone https://github.com/KULLANICI/BitirmeProject.git
# %cd BitirmeProject
# %run scripts/train_image_model_kaggle.py
```

En basiti: **train_image_model_kaggle.py** içeriğini kopyalayıp tek hücrede çalıştırmak.

### Adım 4: Çıktıları indirme

- Eğitim bittikten sonra model ve sınıf isimleri **Output** altında olur:
  - `/kaggle/working/out_models/best_emergency_model.pth`
  - `/kaggle/working/out_models/image_class_names.json`
- Sağda **Output** sekmesi → **Save Version** (veya **Download**) ile bu dosyaları indir.
- İndirdiğin `best_emergency_model.pth` ve `image_class_names.json` dosyalarını kendi projende `out_models/` klasörüne koy. Uygulama buradan yükleyecektir.

### Pause / Kaldığın yerden devam (Resume)

Kaggle'da **Pause** butonu yok. Run'ı durdurursan oturum biter. Devam etmek için:

1. Script her epoch sonunda **checkpoint** kaydeder: `out_models/checkpoint_latest.pth`.
2. Run kesilince veya bitince sağ üstten **Save Version** (veya **Save & Run All**) yap; çıktılar (out_models/ dahil) kaydedilir.
3. Bu çıktıyı **yeni bir Dataset** olarak oluştur (ör. "image-training-checkpoint").
4. Yeni bir notebook aç: **Input** olarak hem **UCF Crime Dataset** hem de bu checkpoint dataset'ini ekle.
5. Scripti çalıştırmadan önce ortam değişkeni ver:  
   `RESUME_CHECKPOINT_DIR=/kaggle/input/image-training-checkpoint/out_models`  
   (Dataset adına göre path’i ayarla; içinde `checkpoint_latest.pth` olan klasör bu olmalı.)
6. Aynı scripti tekrar çalıştır; eğitim kaldığı epoch’tan devam eder.

### Kaggle sınırları

- Ücretsiz hesapta haftalık GPU süresi sınırlı (ör. 30 saat).
- Tek oturum ~9–12 saat sürebilir; uzun eğitimde epoch sayısını veya veri miktarını (ör. `MAX_PER_CLASS`) düşürerek deneyebilirsin.

---

## 2. Google Colab ile Eğitim

Dataset’i Colab’a taşıman gerekir (Drive veya yükleme).

1. **colab.research.google.com** → Yeni notebook.
2. **Runtime** → **Change runtime type** → **GPU** (T4).
3. Dataset’i birinden seç:
   - **Google Drive’a** zip yükle, Colab’da mount et:
     ```python
     from google.colab import drive
     drive.mount("/content/drive")
     # ZIP'i aç: /content/drive/MyDrive/ucf-crime-dataset.zip -> /content/data/ucf-crime-raw
     ```
   - Veya Kaggle API ile Colab’da indir:
     ```python
     !pip install kaggle
     # kaggle.json'ı yükle, config yap
     !kaggle datasets download -d odins0n/ucf-crime-dataset -p /content/
     !unzip /content/ucf-crime-dataset.zip -d /content/ucf-crime-raw
     ```
4. Proje scriptlerini Colab’a al (repo clone veya dosya yükleme).
5. Ortam değişkenlerini Colab’a göre ayarla:
   - `IMAGE_RAW_DIR=/content/ucf-crime-raw` (veya açtığın klasör)
   - `IMAGE_DATA_DIR=/content/images`
   - Önce `prepare_image_dataset.py` (veya eşdeğeri) ile train/val/test böl, sonra `train_image_model.py` çalıştır; `DATA_DIR=/content/images`, `OUTPUT_DIR`’i de Colab’da yazılabilir bir yola (örn. `/content/out_models`) ver.
6. Eğitim bitince `best_emergency_model.pth` ve `image_class_names.json`’ı Drive’a kaydet veya bilgisayarına indir, projedeki `out_models/` içine koy.

---

## 3. AWS (veya GCP) ile Eğitim

Daha büyük veri ve uzun eğitimler için:

1. **EC2 / GCE** üzerinde GPU’lu instance (örn. g4dn, T4) aç.
2. PyTorch, proje kodu ve dataset’i kur:
   - Dataset’i S3/GCS’e yükle, instance’a indir veya doğrudan mount et.
   - Repo’yu clone edip `scripts/` altındaki scriptleri kullan.
3. Ortam değişkenleriyle yolları ver:
   - `IMAGE_DATA_DIR`, `IMAGE_RAW_DIR`, `OUTPUT_DIR` vb.
4. Eğitimi `nohup` veya `screen` ile arka planda çalıştır; bitince modeli S3/GCS’e yükleyip kendi bilgisayarına indir.

Örnek (AWS):

```bash
export IMAGE_DATA_DIR=/home/ubuntu/data/images
export OUTPUT_DIR=/home/ubuntu/out_models
python scripts/train_image_model.py
```

---

## Hangi Dosyalar Nerede?

| Amaç                         | Dosya / Klasör |
|-----------------------------|----------------|
| Bulutta nasıl eğitilir      | Bu rehber: `docs/CLOUD_TRAINING.md` |
| Kaggle’da tek script        | `scripts/train_image_model_kaggle.py` |
| Lokal veri hazırlama        | `scripts/prepare_image_dataset.py` |
| Lokal eğitim                | `scripts/train_image_model.py` |
| Eğitilmiş model kullanımı   | `out_models/best_emergency_model.pth` + `image_class_names.json` |

Kaggle’da sadece **train_image_model_kaggle.py**’yi çalıştırman yeterli; veriyi kopyalamadan path listesiyle okuyarak disk tasarrufu yapar ve dataset zaten Kaggle’da olduğu için en hızlı başlangıç bu yoldur.
