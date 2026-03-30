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
import '../services/api_service.dart';
import '../widgets/chat_bubble.dart';
import '../widgets/triage_card.dart';
import '../widgets/image_analysis_card.dart';
import '../widgets/report_sheet.dart';

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
  bool _isSending = false;
  Map<String, dynamic>? _triageResult;
  Map<String, dynamic>? _imageAnalysis;
  String? _report;
  bool _isComplete = false;
  int _recordingSeconds = 0;
  Timer? _recordingTimer;
  String? _lastRecordingPath;
  final AudioPlayer _replayPlayer = AudioPlayer();

  File? _pendingImage;
  Position? _currentPosition;

  int? _playingMessageIndex;
  bool _initialMessageSent = false;
  bool _connectionError = false;

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
        if (widget.initialMessage != null && widget.initialMessage!.trim().isNotEmpty) {
          _sendInitialMessage();
        } else {
          _startRecording();
        }
      });
    }
    _captureLocation();

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
      setState(() => _pendingImage = File(picked.path));
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('${AppStrings.failedToPickImage}: $e')),
        );
      }
    }
  }

  Future<void> _sendImageOnly() async {
    if (_pendingImage == null || _isSending) return;
    final imgBytes = await _pendingImage!.readAsBytes();
    setState(() => _pendingImage = null);
    await _sendImageBytes(Uint8List.fromList(imgBytes));
  }

  /// Sunucudaki görüntü modeli ile analiz: oturuma görsel gönderir, triyaj + görsel analiz döner.
  Future<void> _sendImageBytes(Uint8List imgBytes) async {
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
    if (_pendingImage == null) return null;
    final bytes = await _pendingImage!.readAsBytes();
    setState(() => _pendingImage = null);
    return Uint8List.fromList(bytes);
  }

  Future<void> _toggleRecording() async {
    if (_isRecording) {
      await _stopAndSend();
    } else {
      await _startRecording();
    }
  }

  Future<void> _startRecording() async {
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
    _recordingTimer?.cancel();
    final path = await _recorder.stop();
    setState(() {
      _isRecording = false;
      _isSending = true;
      _messages.add(ChatMessage(text: '...', isUser: true));
      if (widget.testMode && path != null) _lastRecordingPath = path;
    });
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
      if (!kIsWeb && _pendingImage != null) {
        final imgBytes = await _pendingImage!.readAsBytes();
        imageB64 = base64Encode(imgBytes);
        setState(() => _pendingImage = null);
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

    if (resp['triage_result'] != null) {
      setState(() {
        _triageResult = resp['triage_result'] as Map<String, dynamic>;
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

    if (audioB64 != null && audioB64.isNotEmpty) {
      setState(() => _playingMessageIndex = _messages.length - 1);
      _playAudioBase64(audioB64);
    } else if (audioUrl != null && audioUrl.isNotEmpty) {
      setState(() => _playingMessageIndex = _messages.length - 1);
      _playAudioUrl(audioUrl);
    }
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

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final isCritical = _triageResult?['triage_level'] == 'CRITICAL';

    final stepLabel = _isComplete
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
          if (_triageResult != null)
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
          if (_pendingImage != null) _buildImagePreview(theme),
          if (_isComplete) _buildCompletedBar(theme),
          if (!_isComplete) _buildInputBar(theme),
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
            child: Image.file(
              _pendingImage!,
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
            onPressed: () => setState(() => _pendingImage = null),
          ),
        ],
      ),
    );
  }

  Widget _buildCompletedBar(ThemeData theme) {
    final isCritical = _triageResult?['triage_level'] == 'CRITICAL';
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
            Row(
              children: [
                if (_report != null)
                  Expanded(
                    child: OutlinedButton.icon(
                      onPressed: _showReport,
                      icon: const Icon(Icons.assignment),
                      label: const Text(AppStrings.viewReport),
                    ),
                  ),
                if (_report != null && isCritical)
                  const SizedBox(width: 12),
                if (isCritical)
                  Expanded(
                    child: FilledButton.icon(
                      onPressed: _callEmergency,
                      icon: const Icon(Icons.phone),
                      label: const Text(AppStrings.call112),
                      style: FilledButton.styleFrom(
                          backgroundColor: AppTheme.criticalRed),
                    ),
                  ),
              ],
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
                    color: _pendingImage != null
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
