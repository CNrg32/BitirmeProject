/// Uygulama metinleri — acil durum senaryolarına uygun, net ve kısa.
class AppStrings {
  AppStrings._();

  // Genel
  static const String appName = 'Acil Yardım';
  static const String emergencyCall = '112 Ara';
  static const String call112 = '112\'yi Ara';
  static const String confirmCall112 = '112 acil hattını aramak istediğinize emin misiniz?';
  static const String yesCall = 'Evet, Ara';
  static const String cancel = 'İptal';

  // Başlangıç / Karşılama
  static const String welcomeTitle = 'Acil Yardım';
  static const String welcomeSubtitle =
      'Durumunuzu ses, yazı veya fotoğrafla anlatın. Size triyaj ve yönlendirme sunuyoruz.';
  static const String connecting = 'Acil servise bağlanılıyor…';
  static const String connectionFailed = 'Bağlantı kurulamadı';
  static const String retry = 'Tekrar dene';
  static const String startSession = 'Başla – Konuş veya yaz';
  static const String connectingLabel = 'Bağlanıyor…';

  // Dil
  static const String language = 'Dil';
  static const String autoDetect = 'Otomatik';

  // İzinler
  static const String permissionMic = 'Sesli anlatım için mikrofon gerekli';
  static const String permissionLocation = 'Konumunuz acil ekiplere iletilebilir';
  static const String permissionCamera = 'Fotoğraf eklemek için kamera/galeri kullanılır';
  static const String openSettings = 'Ayarlar';

  // Ana ekran
  static const String whatHappened = 'Ne oldu?';
  static const String describeOrChoose = 'Aşağıdan hızlı seçin veya başlayıp anlatın';
  static const String medical = 'Tıbbi';
  static const String fire = 'Yangın';
  static const String crime = 'Suç / Şiddet';
  static const String accident = 'Kaza';
  static const String other = 'Diğer';

  // Chat
  static const String sessionTitle = 'Acil Oturum';
  static const String stepDescribe = 'Durumu anlat';
  static const String stepAssessing = 'Değerlendiriliyor';
  static const String stepResult = 'Sonuç';
  static const String orTypeHere = 'Veya buraya yaz…';
  static const String attachPhoto = 'Galeriden fotoğraf ekle';
  static const String takePhotoAnalyze = 'Fotoğraf çek ve analiz et';
  static const String cameraRequired = 'Fotoğraf çekmek için kamera izni gerekli.';
  static const String cameraDeniedOpenSettings =
      'Kamera erişimi reddedildi. Lütfen Ayarlar\'dan etkinleştirin.';
  static const String recording = 'Kayıt';
  static const String tapToStopAndSend = 'Göndermek için dokun';
  static const String recordingSeconds = 'Kayıt: %s sn';
  static const String send = 'Gönder';
  static const String sessionComplete = 'Oturum tamamlandı';
  static const String viewReport = 'Raporu Görüntüle';
  static const String noConnection = 'İnternet bağlantısı yok';
  static const String noConnectionHint = 'Bağlantı gelene kadar bekleyin veya 112\'yi arayın';
  static const String errorOccurred = 'Bir hata oluştu';
  static const String call112Anyway = 'Yine de 112 Ara';
  static const String recordingTapToStop = 'Kayıt yapılıyor. Göndermek için kırmızı düğmeye dokunun.';
  static const String failedToPickImage = 'Fotoğraf seçilemedi';
  static const String couldNotStartRecording = 'Kayıt başlatılamadı';
  static const String micPermissionRequired = 'Ses kaydı için mikrofon izni gerekli. Lütfen tarayıcı penceresinde izin verin.';
  static const String micDeniedOpenSettings = 'Mikrofon erişimi reddedildi. Lütfen Ayarlar\'dan etkinleştirin.';
  static const String micRequired = 'Sesli anlatım için mikrofon gerekli.';
  static const String micNotGranted = 'Mikrofon izni verilmedi.';

  // Triage
  static const String triageCritical = 'KRİTİK';
  static const String triageUrgent = 'ACİL';
  static const String triageNonUrgent = 'ACİL DEĞİL';
  static const String category = 'Kategori';
  static const String redFlags = 'Uyarılar';

  // Rapor
  static const String emergencyReport = 'Acil Durum Raporu';
  static const String reportCopied = 'Rapor panoya kopyalandı';
  static const String coordinatesCopied = 'Koordinatlar kopyalandı';
  static const String copyReport = 'Raporu kopyala';
  static const String copyCoordinates = 'Koordinatları kopyala';

  // Test modu
  static const String testMode = 'Test modu';
  static const String testModeHint = 'Kayıt süresi ve tekrar dinleme sohbette görünür';

  // Görsel analiz
  static const String imageAnalysis = 'Görsel Analiz';
  static const String scene = 'Sahne';
  static const String dispatch = 'Sevk birimleri';

  static String formatRecordingSeconds(int seconds) =>
      recordingSeconds.replaceAll('%s', '$seconds');
}
