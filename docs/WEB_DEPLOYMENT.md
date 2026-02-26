# Projeyi Web Sitesinde Açma

Uygulama iki parçadan oluşur: **Backend (FastAPI)** ve **Web arayüzü (Flutter Web)**. Önce bilgisayarda web olarak çalıştırma, sonra internette yayınlama adımları.

---

## A) Bilgisayarda web olarak çalıştırma (tarayıcıda)

Tarayıcıda `http://localhost:xxxx` açarak kullanırsın.

### 1. Backend’i başlat

Bir terminalde, proje kökünde:

```bash
source venv/bin/activate   # Windows: venv\Scripts\activate
uvicorn src.main:app --host 127.0.0.1 --port 8000
```

### 2. Flutter Web desteğini ekle (bir kez)

`mobile` klasöründe:

```bash
cd mobile
flutter pub get
flutter config --enable-web
flutter create . --platforms=web
```

### 3. Web uygulamasını çalıştır

Aynı `mobile` klasöründe:

```bash
flutter run -d chrome
```

veya

```bash
flutter run -d web-server --web-port=8080
```

Sonra tarayıcıda **http://localhost:8080** (veya çıktıda yazan port) açılır. Backend `http://127.0.0.1:8000` üzerinde çalışıyor olmalı; uygulama bu adrese istek atar (şu an `api_service.dart` içinde sabit).

**Özet:** Backend 8000’de, Flutter web 8080 (veya başka port) üzerinde; tarayıcıda web arayüzünü açıp kullanırsın.

---

## B) İnternette yayınlama (gerçek web sitesi)

Backend ve web arayüzünü internette açmak için ikisini de yayına alman gerekir.

### 1. Backend’i yayına al

Örnek servisler (ücretsiz katman mümkün):

- **Render:** https://render.com – Web Service, repo bağla, `uvicorn src.main:app --host 0.0.0.0 --port $PORT` ile başlat.
- **Railway:** https://railway.app – Proje ekle, GitHub repo seç, start command: `uvicorn src.main:app --host 0.0.0.0 --port $PORT`.
- **Fly.io:** https://fly.io – `fly launch` ve `fly deploy` ile.

Deploy sonrası backend’in adresi örneğin:  
`https://your-app.onrender.com` veya `https://your-app.railway.app`.

### 2. Web arayüzünü backend adresine bağla

Flutter web, backend’e bu yayındaki adresten istek atmalı. İki yol:

**Yol 1 – Build sırasında adres vermek (önerilen):**

```bash
cd mobile
flutter build web --dart-define=API_BASE_URL=https://your-backend.onrender.com
```

Kodda bu değişkeni kullanmak için `api_service.dart` içinde `_baseUrl`’i şöyle yapabilirsin:

```dart
// Build'de verilmezse localhost kullan
static final String _baseUrl = const String.fromEnvironment(
  'API_BASE_URL',
  defaultValue: 'http://127.0.0.1:8000',
);
```

**Yol 2 – Sabit adres:**  
`lib/services/api_service.dart` içinde `_baseUrl`’i doğrudan `https://your-backend.onrender.com` yapıp `flutter build web` alırsın.

### 3. Web çıktısını yayınla

Build’den sonra dosyalar `mobile/build/web/` altında oluşur. Bunları herhangi bir statik barındırıcıda yayınlayabilirsin:

- **Vercel:** `mobile/build/web` klasörünü sürükle-bırak veya GitHub repo’yu bağla, root’u `mobile` yap, build: `cd mobile && flutter build web`, output: `build/web`.
- **Netlify:** Aynı şekilde `build/web` çıktısını kullan.
- **Firebase Hosting:** `firebase init hosting` → `build/web` klasörünü seç, `firebase deploy`.
- **GitHub Pages:** `build/web` içeriğini `gh-pages` branch’e push et veya GitHub Actions ile otomatik deploy.

Deploy sonrası örneğin:  
`https://your-project.vercel.app` → bu adres “web sitesinde açtığın” adresin olur.

---

## Özet tablo

| Nerede çalışsın? | Backend | Web arayüzü |
|------------------|--------|-------------|
| **Sadece bilgisayar (tarayıcı)** | `uvicorn ... --port 8000` | `cd mobile && flutter run -d chrome` (veya `web-server`) |
| **İnternette (gerçek site)** | Render / Railway / Fly.io’da deploy | `flutter build web` → Vercel / Netlify / Firebase’e `build/web` deploy |

---

## CORS

Backend’te CORS zaten `allow_origins=["*"]` ile açık; tarayıcıdan farklı port veya farklı domain’den istek atıldığında engel olmaz. İnternette yayınlarken güvenlik için ileride `allow_origins` içine sadece kendi web siteni ekleyebilirsin.

---

## Hızlı komutlar (yerelde web)

```bash
# Terminal 1 – Backend
cd /path/to/BitirmeProject
source venv/bin/activate
uvicorn src.main:app --host 127.0.0.1 --port 8000

# Terminal 2 – Web arayüzü
cd /path/to/BitirmeProject/mobile
flutter run -d chrome
```

Tarayıcıda açılan sayfa, projeyi “web sitesinde açtığın” yer olur. İnternette açmak için B bölümündeki deploy adımlarını uygulaman yeterli.
