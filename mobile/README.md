# Emergency Assistant – Mobil Uygulama

Flutter ile yazılmış acil durum triaj asistanı. Backend API ile konuşur (ses, metin, görüntü analizi).

## Gereksinimler

- **Flutter SDK** (3.0+): https://docs.flutter.dev/get-started/install
- **Backend çalışıyor olmalı** (aynı bilgisayarda `uvicorn` ile)

---

## Bilgisayarda çalıştırma (önerilen)

Uygulama aynı bilgisayarda **masaüstü** veya **emülatör** üzerinde çalışır; API adresi `http://127.0.0.1:8000` (varsayılan).

### 1. Backend’i başlat

**İlk terminal** – proje kökünde (BitirmeProject/):

```bash
source venv/bin/activate   # Windows: venv\Scripts\activate
uvicorn src.main:app --host 127.0.0.1 --port 8000
```

Bu terminali kapatma; backend açık kalsın.

### 2. Masaüstü uygulamasını çalıştır (Windows / macOS / Linux)

**İkinci terminal** – masaüstü hedefini ekleyip uygulamayı çalıştır:

```bash
cd mobile
flutter pub get
```

**Masaüstü platformunu ilk kez ekliyorsan (bir kez yeterli):**

```bash
# macOS
flutter config --enable-macos-desktop
flutter create . --platforms=macos

# Windows
flutter config --enable-windows-desktop
flutter create . --platforms=windows

# Linux
flutter config --enable-linux-desktop
flutter create . --platforms=linux
```

**Çalıştır:**

```bash
# macOS
flutter run -d macos

# Windows
flutter run -d windows

# Linux
flutter run -d linux
```

Böylece uygulama bilgisayarda pencere olarak açılır ve backend’e `http://127.0.0.1:8000` üzerinden bağlanır.

### 3. Alternatif: Android emülatörde çalıştırma

Android Studio’da bir emülatör aç, sonra:

```bash
cd mobile
flutter pub get
flutter run
```

Emülatörde backend’e erişim için `lib/services/api_service.dart` içinde `_baseUrl` değerini `http://10.0.2.2:8000` yap (sadece emülatör için).

---

## Özet: Bilgisayarda iki adım

```bash
# Terminal 1 – Backend
cd /path/to/BitirmeProject
source venv/bin/activate
uvicorn src.main:app --host 127.0.0.1 --port 8000

# Terminal 2 – Uygulama (masaüstü)
cd /path/to/BitirmeProject/mobile
flutter pub get
flutter run -d macos    # veya windows / linux
```

## API adresi

- **Bilgisayarda (desktop):** `http://127.0.0.1:8000` (varsayılan; `api_service.dart` içinde tanımlı)
- **Android emülatör:** `http://10.0.2.2:8000` (emülatörden bilgisayara erişim için `_baseUrl`’i buna çevir)
- **Fiziksel telefon:** Bilgisayarın yerel IP’si (örn. `http://192.168.1.100:8000`); backend’i `--host 0.0.0.0` ile başlat
