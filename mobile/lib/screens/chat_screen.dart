import 'dart:async';
import 'dart:convert';
import 'dart:io' if (dart.library.html) 'dart:io';
import 'dart:typed_data';

import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import 'package:geolocator/geolocator.dart';
import 'package:http/http.dart' as http_client;
import 'package:image_picker/image_picker.dart';
import 'package:path_provider/path_provider.dart';
import 'package:provider/provider.dart';
import 'package:audioplayers/audioplayers.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:record/record.dart';
import 'package:url_launcher/url_launcher.dart';

import '../core/app_strings.dart';
import '../core/app_theme.dart';
import '../models/chat_message.dart';
import '../screens/nearby_places_screen.dart';
import '../services/api_service.dart';
import '../widgets/chat_bubble.dart';
import '../widgets/image_analysis_card.dart';
import '../widgets/report_sheet.dart';
import '../widgets/triage_card.dart';

class ChatScreen extends StatefulWidget {
  final String sessionId;
  final String greeting;
  final String? greetingAudioUrl;
  final String? greetingAudioB64;
  final String language;
  final bool testMode;
  final String? initialMessage;

  const ChatScreen({
    super.key,
    required this.sessionId,
    required this.greeting,
    this.greetingAudioUrl,
    this.greetingAudioB64,
    required this.language,
    this.testMode = false,
    this.initialMessage,
  });

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final _textController = TextEditingController();
  final _scrollController = ScrollController();
  final _audioPlayer = AudioPlayer();
  final _recorder = AudioRecorder();
  final _imagePicker = ImagePicker();

  final List<ChatMessage> _messages = [];
  bool _isRecording = false;
  Timer? _inactivityTimer;
  static const Duration _inactivityTimeout = Duration(minutes: 3);
  bool _isSending = false;
  Map<String, dynamic>? _triageResult;
  Map<String, dynamic>? _imageAnalysis;
  String? _report;
  String? _dispatchStatus;
  String? _followupStatus;
  bool _isComplete = false;
  int _recordingSeconds = 0;
  Timer? _recordingTimer;
  String? _lastRecordingPath;
  final AudioPlayer _replayPlayer = AudioPlayer();

  /// Galeri önizlemesi — Web'de [Image.file] yok; baytlar [Image.memory] ile gösterilir.
  Uint8List? _pendingImageBytes;
  Position? _currentPosition;

  int? _playingMessageIndex;
  bool _initialMessageSent = false;
  bool _connectionError = false;
  bool _hardLockedByTimeout = false;

  @override
  void initState() {
    super.initState();
    _messages.add(ChatMessage(
      text: widget.greeting,
      isUser: false,
    ));

    if (widget.greetingAudioB64 != null &&
        widget.greetingAudioB64!.isNotEmpty) {
      _playingMessageIndex = 0;
      _playAudioBase64(widget.greetingAudioB64!);
    } else if (widget.greetingAudioUrl != null &&
        widget.greetingAudioUrl!.isNotEmpty) {
      _playingMessageIndex = 0;
      _playAudioUrl(widget.greetingAudioUrl!);
    } else {
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (widget.initialMessage != null &&
            widget.initialMessage!.trim().isNotEmpty) {
          _sendInitialMessage();
        } else {
          _startRecording();
        }
      });
    }

    _captureLocation();
    _restartInactivityTimer();

    _audioPlayer.onPlayerComplete.listen((_) {
      if (!mounted) return;
      final wasGreeting = _playingMessageIndex == 0;
      setState(() => _playingMessageIndex = null);
      if (wasGreeting) {
        if (widget.initialMessage != null &&
            widget.initialMessage!.trim().isNotEmpty &&
            !_initialMessageSent) {
          _sendInitialMessage();
        } else {
          _startRecording();
        }
      }
    });
  }

  bool _shouldLockSession() {
    if (_hardLockedByTimeout) return true;

    // Keep chat open after dispatch so caller can provide follow-up details,
    // but close when backend explicitly marks follow-up as finished.
    if (_dispatchStatus == 'DISPATCHED' ||
        _dispatchStatus == 'SILENT_DISPATCHED') {
      if (_followupStatus == 'no_dispatch_needed') {
        return true;
      }
      return false;
    }

    if (_dispatchStatus == 'CANCELLED') return true;
    return _isComplete;
  }

  void _restartInactivityTimer() {
    _inactivityTimer?.cancel();
    if (_shouldLockSession()) return;
    _inactivityTimer = Timer(_inactivityTimeout, _handleInactivityTimeout);
  }

  void _handleInactivityTimeout() {
    if (!mounted || _shouldLockSession()) return;

    if (_isSending || _isRecording) {
      _restartInactivityTimer();
      return;
    }

    final timeoutMsg = (widget.language == 'tr')
        ? '3 dakika boyunca yanıt alınamadı. Bu oturum güvenlik amacıyla sonlandırıldı. Yeni bir acil durum için lütfen yeniden başlatın.'
        : 'No response for 3 minutes. This session has been safely ended. Please start a new session for a new emergency.';

    setState(() {
      _dispatchStatus = 'CANCELLED';
      _followupStatus = 'no_dispatch_needed';
      _isComplete = true;
      _hardLockedByTimeout = true;
      _messages.add(ChatMessage(text: timeoutMsg, isUser: false));
    });
    _scrollToBottom();
  }

  bool _isTimeoutClosureText(String text) {
    final t = text.toLowerCase();
    return t.contains('3 dakika boyunca yanıt alınamadı') ||
      t.contains('3 dakika boyunca yanit alınamadı') ||
      t.contains('3 dakika boyunca yanit alinamadi') ||
        t.contains('3 dakika yanıt alamadığım için') ||
      t.contains('3 dakika yanit alamadigim icin') ||
        t.contains('no response for 3 minutes') ||
        t.contains('ended due to inactivity');
  }

    bool _isGeneralClosureText(String text) {
      final t = text.toLowerCase();
      return _isTimeoutClosureText(text) ||
      t.contains('anlamlı bir acil durum bilgisi alamadım') ||
      t.contains('anlamli bir acil durum bilgisi alamadim') ||
      t.contains('oturumu kapatıyorum') ||
      t.contains('oturumu kapatiyorum') ||
      t.contains('bu oturum tamamlandı') ||
      t.contains('bu oturum tamamlandi') ||
      t.contains('this session is already complete') ||
      t.contains('i am closing this session') ||
      t.contains('i could not get meaningful emergency details');
    }

    bool _responseRequestsFollowup(String text) {
      final t = text.toLowerCase();
      return t.contains('?') ||
      t.contains('daha fazla bilgi var mı') ||
      t.contains('daha fazla bilgi var mi') ||
      t.contains('ek bilgi') ||
      t.contains('any additional information') ||
      t.contains('any updates');
    }

  Future<void> _sendInitialMessage() async {
    if (_initialMessageSent) return;
    final text = widget.initialMessage!.trim();
    if (text.isEmpty) {
      _startRecording();
      return;
    }
    setState(() => _initialMessageSent = true);
    setState(() {
      _messages.add(ChatMessage(text: text, isUser: true));
      _isSending = true;
      _connectionError = false;
    });
    _restartInactivityTimer();
    _scrollToBottom();
    try {
      final api = context.read<ApiService>();
      final resp = await api.sendMessage(
        sessionId: widget.sessionId,
        text: text,
        latitude: _currentPosition?.latitude,
        longitude: _currentPosition?.longitude,
      );
      _handleResponse(resp);
    } catch (e) {
      setState(() {
        _connectionError = true;
        _isSending = false;
      });
      _addAssistant('${AppStrings.errorOccurred}: $e');
    } finally {
      if (mounted) setState(() => _isSending = false);
    }
  }

  @override
  void dispose() {
    _recordingTimer?.cancel();
    _inactivityTimer?.cancel();
    _textController.dispose();
    _scrollController.dispose();
    _audioPlayer.dispose();
    _replayPlayer.dispose();
    _recorder.dispose();
    super.dispose();
  }

  Future<void> _captureLocation() async {
    try {
      LocationPermission permission = await Geolocator.checkPermission();
      if (permission == LocationPermission.denied) {
        permission = await Geolocator.requestPermission();
      }
      if (permission == LocationPermission.deniedForever ||
          permission == LocationPermission.denied) {
        return;
      }
      final position = await Geolocator.getCurrentPosition(
        desiredAccuracy: LocationAccuracy.high,
      );
      if (mounted) {
        setState(() => _currentPosition = position);
      }
    } catch (_) {}
  }

  Future<void> _sendText() async {
    if (_shouldLockSession()) return;
    final text = _textController.text.trim();
    if (text.isEmpty || _isSending) return;

    _textController.clear();

    String? imageB64;
    final imgBytes = await _readAndClearPendingImage();
    if (imgBytes != null) {
      imageB64 = base64Encode(imgBytes);
    }

    setState(() {
      _messages.add(
        ChatMessage(text: text, isUser: true, imageBytes: imgBytes),
      );
      _isSending = true;
    });
    _restartInactivityTimer();
    _scrollToBottom();

    try {
      final api = context.read<ApiService>();
      final resp = await api.sendMessage(
        sessionId: widget.sessionId,
        text: text,
        imageBase64: imageB64,
        latitude: _currentPosition?.latitude,
        longitude: _currentPosition?.longitude,
      );
      _handleResponse(resp);
    } catch (e) {
      if (mounted) setState(() => _connectionError = true);
      _addAssistant('${AppStrings.errorOccurred}: $e');
    } finally {
      setState(() => _isSending = false);
    }
  }

  Future<void> _pickImage(ImageSource source) async {
    try {
      final picked = await _imagePicker.pickImage(
        source: source,
        maxWidth: 1024,
        maxHeight: 1024,
        imageQuality: 85,
      );
      if (picked == null) return;
      final bytes = await picked.readAsBytes();
      if (!mounted) return;
      setState(() => _pendingImageBytes = Uint8List.fromList(bytes));
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('${AppStrings.failedToPickImage}: $e')),
        );
      }
    }
  }

  Future<void> _sendImageOnly() async {
    if (_pendingImageBytes == null || _isSending) return;
    final imgBytes = _pendingImageBytes!;
    setState(() => _pendingImageBytes = null);
    await _sendImageBytes(imgBytes);
  }

  /// Sunucudaki görüntü modeli ile analiz: oturuma görsel gönderir, triyaj + görsel analiz döner.
  Future<void> _sendImageBytes(Uint8List imgBytes) async {
    if (_shouldLockSession()) return;
    if (_isSending) return;
    final imageB64 = base64Encode(imgBytes);

    setState(() {
      _messages.add(
        ChatMessage(
          text: '[Photo sent]',
          isUser: true,
          imageBytes: imgBytes,
        ),
      );
      _isSending = true;
    });
    _restartInactivityTimer();
    _scrollToBottom();

    try {
      final api = context.read<ApiService>();
      final resp = await api.sendMessage(
        sessionId: widget.sessionId,
        imageBase64: imageB64,
        latitude: _currentPosition?.latitude,
        longitude: _currentPosition?.longitude,
      );
      _handleResponse(resp);
    } catch (e) {
      setState(() => _connectionError = true);
      _addAssistant('${AppStrings.errorOccurred}: $e');
    } finally {
      if (mounted) setState(() => _isSending = false);
    }
  }

  /// Uygulama içi kamera ile çekim; önizleme olmadan doğrudan modele gönderilir.
  Future<void> _captureAndAnalyzePhoto() async {
    if (_isSending || _isRecording) return;

    if (!kIsWeb) {
      PermissionStatus status = await Permission.camera.status;
      if (!status.isGranted) {
        status = await Permission.camera.request();
      }
      if (!status.isGranted) {
        if (!mounted) return;
        final openSettings = status.isPermanentlyDenied;
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(
              openSettings
                  ? AppStrings.cameraDeniedOpenSettings
                  : AppStrings.cameraRequired,
            ),
            action: openSettings
                ? SnackBarAction(
                    label: AppStrings.openSettings,
                    onPressed: () => openAppSettings(),
                  )
                : null,
          ),
        );
        return;
      }
    }

    try {
      final picked = await _imagePicker.pickImage(
        source: ImageSource.camera,
        maxWidth: 1024,
        maxHeight: 1024,
        imageQuality: 85,
      );
      if (picked == null || !mounted) return;
      final bytes = await picked.readAsBytes();
      await _sendImageBytes(Uint8List.fromList(bytes));
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('${AppStrings.failedToPickImage}: $e')),
        );
      }
    }
  }

  Future<Uint8List?> _readAndClearPendingImage() async {
    if (_pendingImageBytes == null) return null;
    final bytes = _pendingImageBytes!;
    setState(() => _pendingImageBytes = null);
    return bytes;
  }

  Future<void> _toggleRecording() async {
    if (_shouldLockSession()) return;
    if (_isRecording) {
      await _stopAndSend();
    } else {
      await _startRecording();
    }
  }

  Future<void> _startRecording() async {
    if (_shouldLockSession()) return;
    if (!mounted) return;

    if (kIsWeb) {
      if (!await _recorder.hasPermission()) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text(AppStrings.micPermissionRequired)),
          );
        }
        return;
      }
    } else {
      PermissionStatus status = await Permission.microphone.status;
      if (!status.isGranted) {
        status = await Permission.microphone.request();
      }
      if (!status.isGranted) {
        if (mounted) {
          final openSettings = status.isPermanentlyDenied;
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text(
                openSettings
                    ? AppStrings.micDeniedOpenSettings
                    : AppStrings.micRequired,
              ),
              action: openSettings
                  ? SnackBarAction(
                      label: AppStrings.openSettings,
                      onPressed: () => openAppSettings(),
                    )
                  : null,
            ),
          );
        }
        return;
      }
      if (!await _recorder.hasPermission()) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text(AppStrings.micNotGranted)),
          );
        }
        return;
      }
    }

    late final RecordConfig config;
    late final String recordPath;

    if (kIsWeb) {
      config = const RecordConfig(encoder: AudioEncoder.opus);
      recordPath = '';
    } else {
      config = const RecordConfig(encoder: AudioEncoder.wav);
      final dir = await getTemporaryDirectory();
      recordPath =
          '${dir.path}/emergency_recording_${DateTime.now().millisecondsSinceEpoch}.wav';
    }

    try {
      await _recorder.start(config, path: recordPath);
      if (mounted) {
        setState(() {
          _isRecording = true;
          _recordingSeconds = 0;
        });
        _recordingTimer?.cancel();
        _recordingTimer = Timer.periodic(const Duration(seconds: 1), (_) {
          if (mounted) setState(() => _recordingSeconds++);
        });
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Row(
              children: [
                const Icon(Icons.mic, color: Colors.white, size: 20),
                const SizedBox(width: 8),
                Expanded(child: Text(AppStrings.recordingTapToStop)),
              ],
            ),
            duration: const Duration(seconds: 2),
            backgroundColor: Colors.red,
          ),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('${AppStrings.couldNotStartRecording}: $e')),
        );
      }
    }
  }

  Future<void> _stopAndSend() async {
    if (_shouldLockSession()) return;
    _recordingTimer?.cancel();
    final path = await _recorder.stop();
    setState(() {
      _isRecording = false;
      _isSending = true;
      _messages.add(ChatMessage(text: '...', isUser: true));
      if (widget.testMode && path != null) _lastRecordingPath = path;
    });
    _restartInactivityTimer();
    _scrollToBottom();

    if (path == null) {
      setState(() => _isSending = false);
      return;
    }

    try {
      Uint8List bytes;
      if (kIsWeb) {
        final response = await http_client.get(Uri.parse(path));
        bytes = response.bodyBytes;
      } else {
        bytes = await File(path).readAsBytes();
      }
      final b64 = base64Encode(bytes);
      final api = context.read<ApiService>();

      String transcriptText = '';
      try {
        final transcribeResp = await api.transcribeAudio(
          sessionId: widget.sessionId,
          audioBase64: b64,
        );
        transcriptText =
            (transcribeResp['transcript'] as String?)?.trim() ?? '';
      } catch (_) {}

      if (mounted) {
        final idx = _messages.lastIndexWhere(
          (m) => m.isUser && m.text == '...',
        );
        if (idx != -1) {
          setState(() {
            _messages[idx] = ChatMessage(
              text: transcriptText.isNotEmpty
                  ? transcriptText
                  : '[Voice message]',
              isUser: true,
              imageBytes: _messages[idx].imageBytes,
            );
          });
          _scrollToBottom();
        }
      }

      String? imageB64;
      if (_pendingImageBytes != null) {
        imageB64 = base64Encode(_pendingImageBytes!);
        setState(() => _pendingImageBytes = null);
      }

      final Map<String, dynamic> resp;
      if (transcriptText.isNotEmpty) {
        resp = await api.sendMessage(
          sessionId: widget.sessionId,
          text: transcriptText,
          imageBase64: imageB64,
          latitude: _currentPosition?.latitude,
          longitude: _currentPosition?.longitude,
        );
      } else {
        resp = await api.sendMessage(
          sessionId: widget.sessionId,
          audioBase64: b64,
          imageBase64: imageB64,
          latitude: _currentPosition?.latitude,
          longitude: _currentPosition?.longitude,
        );
      }
      _handleResponse(resp);
    } catch (e) {
      setState(() => _connectionError = true);
      _addAssistant('${AppStrings.errorOccurred}: $e');
    } finally {
      setState(() => _isSending = false);
    }
  }

  void _handleResponse(Map<String, dynamic> resp) {
    setState(() => _connectionError = false);
    final text = resp['assistant_text'] as String? ?? '';
    final audioB64 = resp['assistant_audio_b64'] as String?;
    final audioUrl = resp['assistant_audio_url'] as String?;

    _addAssistant(text, audioBase64: audioB64);

    final timedOut = _isTimeoutClosureText(text);
    final generalCloseText = _isGeneralClosureText(text);

    if (resp['triage_result'] != null) {
      setState(() {
        _triageResult = resp['triage_result'] as Map<String, dynamic>;
      });
    }
    if (resp['dispatch_status'] is String) {
      setState(() {
        _dispatchStatus = resp['dispatch_status'] as String;
      });
    }
    if (resp['followup_status'] is String) {
      setState(() {
        _followupStatus = resp['followup_status'] as String;
      });
    }
    if (resp['image_analysis'] != null) {
      setState(() {
        _imageAnalysis = resp['image_analysis'] as Map<String, dynamic>;
      });
    }
    if (resp['report'] != null) {
      setState(() => _report = resp['report'] as String);
    }
    if (resp['is_complete'] == true) {
      setState(() => _isComplete = true);
    }

    final followupStatus = resp['followup_status'] as String?;
    final backendClosed = followupStatus == 'no_dispatch_needed' ||
        (resp['dispatch_status'] as String?) == 'CANCELLED';
    final completed = resp['is_complete'] == true;

    // Lock the session when backend signals completion, when the report card
    // arrives, or any other closure condition. Do NOT check asksFollowup here —
    // the report text may contain '?' characters in map URLs which would
    // incorrectly prevent locking.
    if (timedOut || generalCloseText || backendClosed || completed) {
      setState(() {
        _isComplete = true;
        _followupStatus = 'no_dispatch_needed';
        if (timedOut) {
          _dispatchStatus = 'CANCELLED';
        }
        _hardLockedByTimeout = true;
      });
    }

    if (audioB64 != null && audioB64.isNotEmpty) {
      setState(() => _playingMessageIndex = _messages.length - 1);
      _playAudioBase64(audioB64);
    } else if (audioUrl != null && audioUrl.isNotEmpty) {
      setState(() => _playingMessageIndex = _messages.length - 1);
      _playAudioUrl(audioUrl);
    }

    _restartInactivityTimer();
  }

  void _addAssistant(String text, {String? audioBase64}) {
    setState(() {
      _messages.add(ChatMessage(
        text: text,
        isUser: false,
        audioBase64: audioBase64,
      ));
    });
    _scrollToBottom();
  }

  Future<void> _playAudioBase64(String b64) async {
    try {
      final bytes = base64Decode(b64);
      if (kIsWeb) {
        await _audioPlayer
            .play(BytesSource(Uint8List.fromList(bytes)));
      } else {
        final dir = await getTemporaryDirectory();
        final file = File(
            '${dir.path}/tts_response_${DateTime.now().millisecondsSinceEpoch}.mp3');
        await file.writeAsBytes(bytes);
        await _audioPlayer.play(DeviceFileSource(file.path));
      }
    } catch (_) {
      if (mounted) {
        setState(() => _playingMessageIndex = null);
      }
    }
  }

  Future<void> _playAudioUrl(String url) async {
    try {
      await _audioPlayer.play(UrlSource(url));
    } catch (_) {
      if (mounted) {
        setState(() => _playingMessageIndex = null);
      }
    }
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  Future<void> _cancelDispatch() async {
    final confirmed = await showDialog<bool>(
      context: context,
      barrierDismissible: false,
      builder: (ctx) => AlertDialog(
        title: const Text(AppStrings.cancelDispatchTitle),
        content: const Text(AppStrings.cancelDispatchConfirm),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: const Text(AppStrings.cancel),
          ),
          FilledButton(
            onPressed: () => Navigator.pop(ctx, true),
            style: FilledButton.styleFrom(
                backgroundColor: AppTheme.criticalRed),
            child: const Text(AppStrings.yesCancelDispatch),
          ),
        ],
      ),
    );
    if (confirmed != true || !mounted) return;
    setState(() {
      _dispatchStatus = 'CANCELLED';
      _isComplete = true;
      _followupStatus = 'no_dispatch_needed';
      _hardLockedByTimeout = true;
    });
    _addAssistant(
      widget.language == 'tr'
          ? AppStrings.dispatchCancelledMsg
          : AppStrings.dispatchCancelledMsgEn,
    );
  }

  Future<void> _callEmergency() async {
    final confirmed = await showDialog<bool>(
      context: context,
      barrierDismissible: false,
      builder: (ctx) => AlertDialog(
        title: const Text(AppStrings.emergencyCall),
        content: const Text(AppStrings.confirmCall112),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: const Text(AppStrings.cancel),
          ),
          FilledButton(
            onPressed: () => Navigator.pop(ctx, true),
            style: FilledButton.styleFrom(backgroundColor: AppTheme.criticalRed),
            child: const Text(AppStrings.yesCall),
          ),
        ],
      ),
    );
    if (confirmed != true || !mounted) return;
    final uri = Uri(scheme: 'tel', path: '112');
    if (await canLaunchUrl(uri)) {
      await launchUrl(uri);
    }
  }

  void _showReport() {
    if (_report == null) return;
    String? locText;
    if (_currentPosition != null) {
      locText =
          '${_currentPosition!.latitude.toStringAsFixed(5)}, ${_currentPosition!.longitude.toStringAsFixed(5)}';
    }

    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (_) => ReportSheet(
        report: _report!,
        triageResult: _triageResult,
        imageAnalysis: _imageAnalysis,
        locationText: locText,
      ),
    );
  }

  bool _shouldShowTriageCard() {
    if (_triageResult == null) return false;

    final hasConfirmedField = _triageResult!.containsKey('triage_confirmed');
    final triageConfirmed = _triageResult!['triage_confirmed'] == true;
    if (triageConfirmed) return true;
    if (_isComplete) return true;
    if (_dispatchStatus == 'DISPATCHED' ||
        _dispatchStatus == 'SILENT_DISPATCHED') {
      return true;
    }

    if (!hasConfirmedField) {
      final userTurnCount = _messages.where((m) => m.isUser).length;
      return userTurnCount >= 2;
    }

    return false;
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final isCritical = _triageResult?['triage_level'] == 'CRITICAL';
    final isSessionLocked = _shouldLockSession();

    final stepLabel = isSessionLocked
        ? AppStrings.stepResult
        : _triageResult != null
            ? AppStrings.stepAssessing
            : AppStrings.stepDescribe;

    return Scaffold(
      appBar: AppBar(
        title: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Text(AppStrings.sessionTitle),
            Text(
              stepLabel,
              style: theme.textTheme.bodySmall?.copyWith(
                color: Colors.white.withOpacity(0.9),
                fontSize: 12,
              ),
            ),
          ],
        ),
        backgroundColor: theme.colorScheme.primary,
        foregroundColor: Colors.white,
        actions: [
          if (_currentPosition != null)
            Tooltip(
              message:
                  'GPS: ${_currentPosition!.latitude.toStringAsFixed(4)}, '
                  '${_currentPosition!.longitude.toStringAsFixed(4)}',
              child: const Padding(
                padding: EdgeInsets.symmetric(horizontal: 4),
                child: Icon(Icons.location_on, size: 20),
              ),
            ),
          if (widget.testMode)
            Center(
              child: Container(
                margin: const EdgeInsets.only(right: 4),
                padding: const EdgeInsets.symmetric(
                    horizontal: 8, vertical: 4),
                decoration: BoxDecoration(
                  color: Colors.white24,
                  borderRadius: BorderRadius.circular(8),
                ),
                child: const Text('Test', style: TextStyle(fontSize: 12)),
              ),
            ),
          if (_report != null)
            IconButton(
              icon: const Icon(Icons.assignment),
              tooltip: AppStrings.viewReport,
              onPressed: _showReport,
            ),
          if (_isComplete)
            IconButton(
              icon: const Icon(Icons.check_circle),
              tooltip: AppStrings.sessionComplete,
              onPressed: () {},
            ),
        ],
      ),
      body: Column(
        children: [
          if (_connectionError) _buildErrorBanner(theme),
          if (_shouldShowTriageCard())
            TriageCard(
              result: _triageResult!,
              onCallEmergency: isCritical ? _callEmergency : null,
            ),
          if (_imageAnalysis != null)
            ImageAnalysisCard(analysis: _imageAnalysis!),
          Expanded(
            child: ListView.builder(
              controller: _scrollController,
              padding: const EdgeInsets.all(16),
              itemCount: _messages.length,
              itemBuilder: (_, i) => ChatBubble(
                message: _messages[i],
                isPlaying: _playingMessageIndex == i,
              ),
            ),
          ),
          if (_isRecording)
            Container(
              padding: const EdgeInsets.symmetric(vertical: 12),
              margin: const EdgeInsets.symmetric(horizontal: 12),
              decoration: BoxDecoration(
                color: Colors.red.withOpacity(0.2),
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: Colors.red.withOpacity(0.5), width: 1),
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(Icons.mic, color: Colors.red.shade700, size: 24),
                  const SizedBox(width: 10),
                  Text(
                    '${AppStrings.formatRecordingSeconds(_recordingSeconds)} — ${AppStrings.tapToStopAndSend}',
                    style: theme.textTheme.titleSmall?.copyWith(
                      color: Colors.red.shade800,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ],
              ),
            ),
          if (_pendingImageBytes != null) _buildImagePreview(theme),
          if (!isSessionLocked &&
              (_dispatchStatus == 'DISPATCHED' ||
                  _dispatchStatus == 'SILENT_DISPATCHED'))
            _buildDispatchCancelBanner(theme),
          if (isSessionLocked) _buildCompletedBar(theme),
          if (!isSessionLocked) _buildInputBar(theme),
        ],
      ),
    );
  }

  Widget _buildDispatchCancelBanner(ThemeData theme) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      color: Colors.orange.withOpacity(0.08),
      child: Row(
        children: [
          Icon(Icons.directions_run,
              color: Colors.orange.shade700, size: 20),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              AppStrings.dispatchedUnitsLabel,
              style: theme.textTheme.bodySmall?.copyWith(
                color: Colors.orange.shade800,
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
          OutlinedButton.icon(
            onPressed: _cancelDispatch,
            icon: const Icon(Icons.cancel_outlined, size: 16),
            label: const Text(AppStrings.cancelDispatch),
            style: OutlinedButton.styleFrom(
              foregroundColor: AppTheme.criticalRed,
              side: BorderSide(color: AppTheme.criticalRed),
              padding:
                  const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
              textStyle: const TextStyle(fontSize: 12),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildErrorBanner(ThemeData theme) {
    return Material(
      color: AppTheme.urgentAmber.withOpacity(0.15),
      child: SafeArea(
        bottom: false,
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
          child: Row(
            children: [
              Icon(Icons.warning_amber_rounded,
                  color: AppTheme.urgentAmber, size: 24),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      AppStrings.noConnection,
                      style: theme.textTheme.titleSmall?.copyWith(
                        fontWeight: FontWeight.w600,
                        color: theme.colorScheme.onSurface,
                      ),
                    ),
                    Text(
                      AppStrings.noConnectionHint,
                      style: theme.textTheme.bodySmall,
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 8),
              FilledButton(
                onPressed: _callEmergency,
                style: FilledButton.styleFrom(
                  backgroundColor: AppTheme.criticalRed,
                  padding: const EdgeInsets.symmetric(horizontal: 12),
                ),
                child: const Text(AppStrings.call112Anyway),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildImagePreview(ThemeData theme) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      color: theme.colorScheme.surfaceContainerHighest,
      child: Row(
        children: [
          ClipRRect(
            borderRadius: BorderRadius.circular(8),
            child: Image.memory(
              _pendingImageBytes!,
              width: 60,
              height: 60,
              fit: BoxFit.cover,
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Text('Fotoğraf eklendi',
                style: theme.textTheme.bodyMedium),
          ),
          IconButton(
            icon: const Icon(Icons.send),
            tooltip: AppStrings.send,
            onPressed: _isSending ? null : _sendImageOnly,
          ),
          IconButton(
            icon: const Icon(Icons.close),
            tooltip: AppStrings.cancel,
            onPressed: () => setState(() => _pendingImageBytes = null),
          ),
        ],
      ),
    );
  }

  Widget _buildCompletedBar(ThemeData theme) {
    final isCritical = _triageResult?['triage_level'] == 'CRITICAL';
    final isNonUrgent = _triageResult?['triage_level'] == 'NON_URGENT';
    final hasDispatchToCancel =
        _dispatchStatus == 'DISPATCHED' || _dispatchStatus == 'SILENT_DISPATCHED';
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: theme.colorScheme.surfaceContainerHighest,
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.08),
            blurRadius: 8,
            offset: const Offset(0, -2),
          ),
        ],
      ),
      child: SafeArea(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Row(
              children: [
                const Icon(Icons.check_circle, color: Colors.green),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    AppStrings.sessionComplete,
                    style: theme.textTheme.titleMedium
                        ?.copyWith(fontWeight: FontWeight.w600),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                if (_report != null)
                  OutlinedButton.icon(
                    onPressed: _showReport,
                    icon: const Icon(Icons.assignment),
                    label: const Text(AppStrings.viewReport),
                  ),
                if (isCritical)
                  FilledButton.icon(
                    onPressed: _callEmergency,
                    icon: const Icon(Icons.phone),
                    label: const Text(AppStrings.call112),
                    style: FilledButton.styleFrom(
                        backgroundColor: AppTheme.criticalRed),
                  ),
                if (hasDispatchToCancel)
                  OutlinedButton.icon(
                    onPressed: _cancelDispatch,
                    icon: const Icon(Icons.cancel_outlined),
                    label: const Text(AppStrings.cancelDispatch),
                    style: OutlinedButton.styleFrom(
                      foregroundColor: AppTheme.criticalRed,
                      side: BorderSide(color: AppTheme.criticalRed),
                    ),
                  ),
              ],
            ),
            if (isNonUrgent)
              Padding(
                padding: const EdgeInsets.only(top: 8),
                child: SizedBox(
                  width: double.infinity,
                  child: OutlinedButton.icon(
                    onPressed: () => Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (_) => const NearbyPlacesScreen(),
                      ),
                    ),
                    icon: const Icon(Icons.local_hospital_outlined),
                    label: const Text(
                        AppStrings.showNearbyFacilitiesSuggestion),
                  ),
                ),
              ),
            if (_currentPosition != null)
              Padding(
                padding: const EdgeInsets.only(top: 8),
                child: Row(
                  children: [
                    Icon(Icons.location_on,
                        size: 16, color: theme.colorScheme.outline),
                    const SizedBox(width: 4),
                    Text(
                      'GPS: ${_currentPosition!.latitude.toStringAsFixed(4)}, '
                      '${_currentPosition!.longitude.toStringAsFixed(4)}',
                      style: theme.textTheme.bodySmall
                          ?.copyWith(color: theme.colorScheme.outline),
                    ),
                  ],
                ),
              ),
          ],
        ),
      ),
    );
  }

  Future<void> _replayLastRecording() async {
    if (_lastRecordingPath == null) return;
    try {
      if (kIsWeb) {
        await _replayPlayer.play(UrlSource(_lastRecordingPath!));
      } else {
        final file = File(_lastRecordingPath!);
        if (!await file.exists()) return;
        await _replayPlayer.play(DeviceFileSource(_lastRecordingPath!));
      }
    } catch (_) {}
  }

  Widget _buildInputBar(ThemeData theme) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: theme.colorScheme.surface,
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.08),
            blurRadius: 8,
            offset: const Offset(0, -2),
          ),
        ],
      ),
      child: SafeArea(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            if (widget.testMode && _lastRecordingPath != null)
              Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: Row(
                  children: [
                    Text('Last recording: ',
                        style: theme.textTheme.bodySmall),
                    TextButton.icon(
                      onPressed: _replayLastRecording,
                      icon: const Icon(Icons.play_circle_outline,
                          size: 20),
                      label: const Text('Play'),
                    ),
                  ],
                ),
              ),
            Row(
              children: [
                GestureDetector(
                  onTap: _isSending ? null : _toggleRecording,
                  child: Container(
                    width: 48,
                    height: 48,
                    decoration: BoxDecoration(
                      color: _isRecording
                          ? Colors.red
                          : theme.colorScheme.primary,
                      shape: BoxShape.circle,
                      boxShadow: [
                        BoxShadow(
                          color: (_isRecording
                                  ? Colors.red
                                  : theme.colorScheme.primary)
                              .withOpacity(0.4),
                          blurRadius: 8,
                          offset: const Offset(0, 2),
                        ),
                      ],
                    ),
                    child: Icon(
                      _isRecording ? Icons.stop : Icons.mic,
                      color: Colors.white,
                      size: 24,
                    ),
                  ),
                ),
                const SizedBox(width: 8),
                IconButton.filledTonal(
                  onPressed: _isSending || _isRecording
                      ? null
                      : _captureAndAnalyzePhoto,
                  icon: const Icon(Icons.photo_camera),
                  tooltip: AppStrings.takePhotoAnalyze,
                ),
                const SizedBox(width: 4),
                IconButton(
                  onPressed: _isSending || _isRecording
                      ? null
                      : () => _pickImage(ImageSource.gallery),
                  icon: Icon(
                    Icons.photo_library_outlined,
                    color: _pendingImageBytes != null
                        ? theme.colorScheme.primary
                        : null,
                  ),
                  tooltip: AppStrings.attachPhoto,
                ),
                const SizedBox(width: 4),
                Expanded(
                  child: TextField(
                    controller: _textController,
                    enabled: !_isSending && !_isRecording,
                    decoration: InputDecoration(
                      hintText: AppStrings.orTypeHere,
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(24),
                        borderSide: BorderSide.none,
                      ),
                      filled: true,
                      fillColor:
                          theme.colorScheme.surfaceContainerHighest,
                      contentPadding: const EdgeInsets.symmetric(
                        horizontal: 16,
                        vertical: 12,
                      ),
                    ),
                    textInputAction: TextInputAction.send,
                    onSubmitted: (_) => _sendText(),
                  ),
                ),
                const SizedBox(width: 8),
                IconButton.filled(
                  onPressed: _isSending ? null : _sendText,
                  icon: _isSending
                      ? const SizedBox(
                          width: 20,
                          height: 20,
                          child: CircularProgressIndicator(
                              strokeWidth: 2),
                        )
                      : const Icon(Icons.send),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
