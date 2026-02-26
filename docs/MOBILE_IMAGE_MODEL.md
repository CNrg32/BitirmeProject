# Eğitilmiş Görüntü Modelini Mobil Uygulamada Kullanma

Kaggle’da eğittiğin model (`best_emergency_model.pth` + `image_class_names.json`) backend’de çalışır; mobil uygulama bu sonuçları API üzerinden alır. Model telefonda çalışmaz, sunucuda çalışır.

---

## 1. Backend’i Hazırlamak

1. **Model dosyalarını koy**
   - `best_emergency_model.pth` → proje kökünde `out_models/`
   - `image_class_names.json` → `out_models/`
   - Klasör yapısı:
     ```
     BitirmeProject/
     └── out_models/
         ├── best_emergency_model.pth
         └── image_class_names.json
     ```

2. **Backend’i çalıştır**
   ```bash
   cd BitirmeProject
   source venv/bin/activate   # veya Windows: venv\Scripts\activate
   uvicorn src.main:app --host 0.0.0.0 --port 8000
   ```
   - Uygulama açılışta `out_models/best_emergency_model.pth` dosyasını arar; bulursa görüntü analizi açılır.
   - `/health` cevabında `image_model_loaded: true` görürsen model yüklü demektir.

---

## 2. Mobil Uygulamada Nasıl Kullanılıyor?

### A) Sohbet içinde (mevcut akış)

Kullanıcı sohbet ekranında:

1. Fotoğraf ekler (kamera veya galeri).
2. Metin yazıp gönderir **veya** sadece fotoğraf gönderir.
3. İstek `session/message` ile backend’e gider (`image_base64` ile).
4. Backend görüntüyü eğitilmiş modelle analiz eder; cevapta `image_analysis` döner.
5. Mobil tarafta **ImageAnalysisCard** bu cevabı gösterir:
   - **Sahne:** `detected_class` (örn. Arson, RoadAccidents)
   - **Güven:** `confidence` (yüzde)
   - **Gönderilecek birimler:** `dispatch_units` (örn. Police, Ambulance)
   - Metinle uyumsuzsa: `consistency_detail` / `risk_notes` uyarısı

Bu akış için ekstra bir şey yapmana gerek yok; backend’de model dosyaları doğru yerdeyse çalışır.

### B) Sadece fotoğraf analizi (isteğe bağlı)

Tek bir fotoğrafı analiz ettirmek için `ApiService.analyzeImage()` kullanılabilir:

```dart
final api = context.read<ApiService>();
final bytes = await File('path/to/photo.jpg').readAsBytes();
final result = await api.analyzeImage(bytes);
// result: classification, consistency, summary, available
// result['classification']['detected_class'], ['confidence'], ['dispatch_units']
```

Bunu istersen ayrı bir “Fotoğraf analiz et” ekranında kullanabilirsin.

---

## 3. API Yanıt Formatı (Referans)

Backend’den gelen `image_analysis` (veya `/analyze-image` cevabı) örneği:

```json
{
  "classification": {
    "detected_class": "RoadAccidents",
    "confidence": 0.92,
    "top3": [
      {"class": "RoadAccidents", "confidence": 0.92},
      {"class": "NormalVideos", "confidence": 0.05},
      ...
    ],
    "dispatch_units": ["Police", "Ambulance", "Fire Department"],
    "mapped_category": "medical"
  },
  "consistency": {
    "is_consistent": true,
    "consistency_score": 0.95,
    "consistency_detail": "CONSISTENT",
    "possible_fake": false,
    "risk_notes": []
  },
  "summary": "Image analysis: RoadAccidents detected (92%). Recommended units: ...",
  "available": true
}
```

Mobil tarafta `ImageAnalysisCard` bu alanları kullanır (`detected_class`, `dispatch_units`, `consistency_detail`, `risk_notes`).

---

## 4. Mobil Tarafı Backend’e Bağlama

- **Emülatör:** `ApiService` içinde `_baseUrl = 'http://10.0.2.2:8000'` (Android emulator için localhost).
- **Fiziksel cihaz:** Bilgisayarın yerel IP’si kullanılmalı, örn. `http://192.168.1.100:8000`. `mobile/lib/services/api_service.dart` içinde `_baseUrl`’i buna göre değiştir veya build flavor / env ile ver.

---

## Özet

| Adım | Ne yapılır |
|------|------------|
| 1 | `out_models/` içine `best_emergency_model.pth` ve `image_class_names.json` koy. |
| 2 | Backend’i çalıştır; `/health` ile `image_model_loaded: true` kontrol et. |
| 3 | Mobil uygulamada sohbetten fotoğraf gönder; gelen cevaptaki `image_analysis` ImageAnalysisCard’da gösterilir. |
| 4 | İstersen `ApiService.analyzeImage()` ile sadece fotoğraf analizi ekranı yap. |

Model çıktıları bu akışla mobil uygulamada kullanılmış olur.
