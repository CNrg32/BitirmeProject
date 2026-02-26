# Web'de Uygulamayı Çalıştırma – Adım Adım

Bu rehberi sırayla uygula. İki şey çalışacak: **1) Backend (API)** ve **2) Web arayüzü (tarayıcı)**.

---

## Ön koşul

- Bilgisayarda **Flutter** kurulu olmalı. Kurulu değilse: https://docs.flutter.dev/get-started/install
- **Python** ve proje sanal ortamı (venv) hazır olmalı.

---

## Adım 1: Backend’i başlat

1. **Terminal** aç (Cursor içindeki Terminal veya Mac’te Terminal.app / Windows’ta CMD).
2. **Proje köküne** git:
   ```bash
   cd /Users/gultekinqwe/Documents/GitHub/BitirmeProject
   ```
3. Sanal ortamı **aktive et**:
   ```bash
   source venv/bin/activate
   ```
   (Windows’ta: `venv\Scripts\activate`)
4. Backend’i **başlat**:
   ```bash
   uvicorn src.main:app --host 127.0.0.1 --port 8000
   ```
5. Şunu görünce hazır: `Uvicorn running on http://127.0.0.1:8000`
6. **Bu terminali kapatma.** Backend açık kalsın.

---

## Adım 2: Web arayüzünü hazırla (ilk kez yapıyorsan)

1. **Yeni bir terminal** aç (ikinci pencere).
2. **mobile** klasörüne git:
   ```bash
   cd /Users/gultekinqwe/Documents/GitHub/BitirmeProject/mobile
   ```
3. Bağımlılıkları al:
   ```bash
   flutter pub get
   ```
4. Web’i etkinleştir (bir kez yeterli):
   ```bash
   flutter config --enable-web
   ```
5. Web platformunu ekle (bir kez yeterli):
   ```bash
   flutter create . --platforms=web
   ```
   Üzerine yazılsın mı diye sorarsa **y** yaz, Enter.

---

## Adım 3: Web uygulamasını çalıştır

1. Aynı terminalde (hâlâ `mobile` klasöründeyken):
   ```bash
   flutter run -d chrome
   ```
2. Bir süre sonra **Chrome otomatik açılır** ve uygulama yüklenir. O sekme = web uygulaması.
3. Chrome açılmazsa terminalde şuna benzer bir satır ara:
   ```text
   http://localhost:XXXXX
   ```
   Bu adresi tarayıcıda **kendin aç**.

**Alternatif (portu sen seçmek için):**
```bash
flutter run -d web-server --web-port=8080
```
Sonra tarayıcıda **http://localhost:8080** adresine git.

---

## Adım 4: Kontrol

- **Web arayüzü:** Tarayıcıda açılan sayfa (localhost:XXXX veya 8080) = Emergency Assistant arayüzü. Buradan sohbet, ses, fotoğraf kullanırsın.
- **Backend:** `http://127.0.0.1:8000` adresinde API çalışıyor. Tarayıcıda `http://127.0.0.1:8000/docs` açarsan API dokümantasyonunu görürsün.

---

## Özet (kopyala-yapıştır)

**Terminal 1 – Backend (açık kalsın):**
```bash
cd /Users/gultekinqwe/Documents/GitHub/BitirmeProject
source venv/bin/activate
uvicorn src.main:app --host 127.0.0.1 --port 8000
```

**Terminal 2 – Web (ilk kez: pub get + create, sonra run):**
```bash
cd /Users/gultekinqwe/Documents/GitHub/BitirmeProject/mobile
flutter pub get
flutter run -d chrome
```

Tarayıcıda açılan sayfa = uygulama. Backend’i kapatırsan (Terminal 1’de Ctrl+C) uygulama API’ye ulaşamaz.

---

## Sık karşılaşılanlar

| Sorun | Çözüm |
|------|--------|
| Terminalde hiç çıktı yok | Komutları **sistem terminalinde** (Terminal.app / CMD) dene. |
| Chrome açılmıyor | `flutter run -d web-server --web-port=8080` yaz, sonra tarayıcıda **http://localhost:8080** aç. |
| “Connection refused” / API yanıt vermiyor | Önce **Terminal 1**’de backend’in çalıştığından emin ol (Adım 1). |
| `flutter: command not found` | Flutter kurulu değil veya PATH’te yok; kurulum rehberini uygula. |
