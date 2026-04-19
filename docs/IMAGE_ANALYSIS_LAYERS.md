# Fotoğraf Analiz Katmanları

Bu akış, sohbet içindeki fotoğrafı yalnızca kart olarak göstermek yerine operasyonel triage sinyali olarak kullanır.

## 1. Görsel Giriş ve Filtreleme

- Fotoğraf `image_base64` ile `/session/message` endpoint'ine gelir.
- Backend temel kalite kontrolü yapar: boyut, parlaklık, kontrast/blur göstergesi.
- Görsel kullanılamazsa `visual_triage.action = RECAPTURE_IMAGE` döner.
- Kullanıcı 2 kez kullanılabilir görsel gönderemezse oturum `FALLBACK_PENDING` olur ve manuel kategori + kısa açıklama istenir.

## 2. ImageAnalyze ve Triage

- Model yüklüyse `best_emergency_model.pth` ile sınıflandırma yapılır.
- Sınıf, kategori ve dispatch birimleri üretilir.
- `visual_triage` alanı şu operasyonel kararı taşır:
  - `EARLY_DISPATCH`
  - `VERIFY_THEN_DISPATCH`
  - `TEXT_REQUIRED`
  - `MANUAL_FALLBACK`
  - `RECAPTURE_IMAGE`

## 3. Görsel-Metin Tutarlılığı

- Metin triage sonucu varsa görsel kategoriyle karşılaştırılır.
- Uyumsuzluk `consistency_detail`, `possible_fake` ve `risk_notes` alanlarına yazılır.
- Görsel normal ama metin critical/urgent ise görsel olayla ilgisiz kabul edilebilir; metin güvenlik açısından önceliğini korur.

## 4. Görsel Slotlar

Görselden rapora eklenen alanlar:

- `image_detected_class`
- `image_action`
- `image_quality`
- `visual_flags`
- `image_category`
- `image_triage_level`

## 5. Operasyonel Mantık

- `EARLY_DISPATCH`: Critical görsel sinyalde önce dispatch yapılır, sonra mikro-konum ve gözlem istenir.
- `VERIFY_THEN_DISPATCH`: Urgent görsel sinyalde kısa doğrulama sorusu sorulur.
- `TEXT_REQUIRED`: Görsel acil durum göstermiyorsa kullanıcıdan kısa açıklama istenir.
- `MANUAL_FALLBACK`: Model yok veya düşük güven varsa manuel kategori + açıklama istenir.

## 6. Dispatch Sonrası Güncelleme

Dispatch yapıldıktan sonra oturum tamamen kilitlenmez. Kullanıcı yeni metin veya fotoğraf gönderirse:

- Yeni vaka açılmaz.
- Bilgi `image_updates` listesine eklenir.
- Yeni görsel critical sinyal taşıyorsa mevcut triage `CRITICAL` seviyesine yükseltilir.
- Kullanıcıya bilginin ekiplere güncelleme olarak iletileceği söylenir.

## 7. Cloud Model

Model lokalde yoksa S3 üzerinden indirilebilir:

```bash
IMAGE_MODEL_S3_URI=s3://bucket/models/image/best_emergency_model.pth
IMAGE_CLASS_NAMES_S3_URI=s3://bucket/models/image/image_class_names.json
AWS_REGION=eu-central-1
```

Model yüklenmemişse fotoğraf akışı kapanmaz; sistem manuel fallback'e düşer.
