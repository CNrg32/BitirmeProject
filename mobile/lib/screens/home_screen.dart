import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:url_launcher/url_launcher.dart';
import '../core/app_strings.dart';
import '../core/app_theme.dart';
import '../services/api_service.dart';
import 'chat_screen.dart';

const Map<String, String> kLanguages = {
  'auto': 'Otomatik',
  'tr': 'Türkçe',
  'en': 'English',
  'de': 'Deutsch',
  'fr': 'Français',
  'es': 'Español',
  'ar': 'العربية',
  'ru': 'Русский',
  'zh': '中文',
  'ja': '日本語',
  'ko': '한국어',
  'pt': 'Português',
  'it': 'Italiano',
  'nl': 'Nederlands',
  'hi': 'हिन्दी',
};

/// Acil türü etiketleri (opsiyonel ilk ipucu için)
const Map<String, String> kEmergencyTypeLabels = {
  'medical': AppStrings.medical,
  'fire': AppStrings.fire,
  'crime': AppStrings.crime,
  'accident': AppStrings.accident,
  'other': AppStrings.other,
};

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  bool _loading = false;
  bool _testMode = false;
  String _selectedLanguage = 'tr';
  String? _selectedType; // medical, fire, crime, accident, other

  Future<void> _call112() async {
    final uri = Uri(scheme: 'tel', path: '112');
    if (await canLaunchUrl(uri)) {
      await launchUrl(uri);
    }
  }

  String? _getInitialMessage() {
    if (_selectedType == null) return null;
    switch (_selectedType!) {
      case 'medical':
        return 'Tıbbi acil durum';
      case 'fire':
        return 'Yangın';
      case 'crime':
        return 'Suç veya şiddet';
      case 'accident':
        return 'Kaza';
      case 'other':
        return 'Diğer acil durum';
      default:
        return null;
    }
  }

  Future<void> _startSession() async {
    setState(() => _loading = true);
    try {
      final api = context.read<ApiService>();
      final lang = _selectedLanguage == 'auto' ? null : _selectedLanguage;
      final result = await api.startSession(lang);
      if (!mounted) return;
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (_) => ChatScreen(
            sessionId: result['session_id'] as String,
            greeting: result['greeting'] as String,
            greetingAudioUrl: result['greeting_audio_url'] as String?,
            greetingAudioB64: result['greeting_audio_b64'] as String?,
            language: _selectedLanguage == 'auto' ? 'tr' : _selectedLanguage,
            testMode: _testMode,
            initialMessage: _getInitialMessage(),
          ),
        ),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('${AppStrings.connectionFailed}: $e'),
          action: SnackBarAction(
            label: AppStrings.retry,
            onPressed: () => _startSession(),
          ),
        ),
      );
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final isDark = theme.brightness == Brightness.dark;

    return Scaffold(
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: isDark
                ? [
                    theme.colorScheme.primary.withOpacity(0.15),
                    theme.colorScheme.surface,
                  ]
                : [
                    AppTheme.primaryRed.withOpacity(0.06),
                    theme.scaffoldBackgroundColor,
                  ],
          ),
        ),
        child: SafeArea(
          child: CustomScrollView(
            slivers: [
              SliverToBoxAdapter(
                child: Padding(
                  padding: const EdgeInsets.fromLTRB(16, 12, 16, 8),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.end,
                    children: [
                      FilledButton.icon(
                        onPressed: _call112,
                        icon: const Icon(Icons.phone, size: 20),
                        label: Text(AppStrings.emergencyCall),
                        style: FilledButton.styleFrom(
                          backgroundColor: AppTheme.criticalRed,
                          foregroundColor: Colors.white,
                          padding: const EdgeInsets.symmetric(
                            horizontal: 16,
                            vertical: 12,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              SliverToBoxAdapter(
                child: Padding(
                  padding: const EdgeInsets.fromLTRB(24, 16, 24, 8),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        AppStrings.whatHappened,
                        style: theme.textTheme.titleLarge?.copyWith(
                          fontWeight: FontWeight.bold,
                          color: theme.colorScheme.onSurface,
                        ),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        AppStrings.describeOrChoose,
                        style: theme.textTheme.bodyMedium?.copyWith(
                          color: theme.colorScheme.onSurface.withOpacity(0.7),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              // Acil türü chip'leri
              SliverPadding(
                padding: const EdgeInsets.symmetric(horizontal: 20),
                sliver: SliverToBoxAdapter(
                  child: Wrap(
                    spacing: 10,
                    runSpacing: 10,
                    children: kEmergencyTypeLabels.entries.map((e) {
                      final selected = _selectedType == e.key;
                      return FilterChip(
                        selected: selected,
                        label: Text(e.value),
                        onSelected: (v) {
                          setState(() =>
                              _selectedType = v ? e.key : null);
                        },
                        selectedColor: theme.colorScheme.primaryContainer,
                        checkmarkColor: theme.colorScheme.primary,
                      );
                    }).toList(),
                  ),
                ),
              ),
              const SliverToBoxAdapter(child: SizedBox(height: 24)),
              // Dil
              SliverToBoxAdapter(
                child: Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 24),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        AppStrings.language,
                        style: theme.textTheme.labelLarge?.copyWith(
                          color: theme.colorScheme.onSurface.withOpacity(0.8),
                        ),
                      ),
                      const SizedBox(height: 8),
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 14),
                        decoration: BoxDecoration(
                          color: theme.colorScheme.surfaceContainerHighest,
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: DropdownButton<String>(
                          value: _selectedLanguage,
                          isExpanded: true,
                          underline: const SizedBox.shrink(),
                          icon: Icon(
                            Icons.language,
                            color: theme.colorScheme.primary,
                          ),
                          items: kLanguages.entries
                              .map((e) => DropdownMenuItem(
                                    value: e.key,
                                    child: Text(e.value),
                                  ))
                              .toList(),
                          onChanged: (v) {
                            if (v != null) setState(() => _selectedLanguage = v);
                          },
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              const SliverToBoxAdapter(child: SizedBox(height: 16)),
              // Test modu
              SliverToBoxAdapter(
                child: Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 24),
                  child: Row(
                    children: [
                      Icon(
                        Icons.science_outlined,
                        size: 20,
                        color: theme.colorScheme.outline,
                      ),
                      const SizedBox(width: 8),
                      Text(
                        AppStrings.testMode,
                        style: theme.textTheme.bodyMedium?.copyWith(
                          color: theme.colorScheme.outline,
                        ),
                      ),
                      const SizedBox(width: 8),
                      Switch(
                        value: _testMode,
                        onChanged: (v) => setState(() => _testMode = v),
                      ),
                    ],
                  ),
                ),
              ),
              if (_testMode)
                const SliverToBoxAdapter(
                  child: Padding(
                    padding: EdgeInsets.only(left: 52, top: 4),
                    child: Text(
                      'Kayıt süresi ve tekrar dinleme sohbette görünür',
                      style: TextStyle(fontSize: 12),
                    ),
                  ),
                ),
              const SliverToBoxAdapter(child: SizedBox(height: 28)),
              // Başla butonu
              SliverToBoxAdapter(
                child: Padding(
                  padding: const EdgeInsets.fromLTRB(24, 0, 24, 32),
                  child: SizedBox(
                    width: double.infinity,
                    height: 56,
                    child: FilledButton.icon(
                      onPressed: _loading ? null : _startSession,
                      icon: _loading
                          ? const SizedBox(
                              width: 22,
                              height: 22,
                              child: CircularProgressIndicator(
                                strokeWidth: 2,
                                color: Colors.white,
                              ),
                            )
                          : const Icon(Icons.mic, size: 24),
                      label: Text(
                        _loading
                            ? AppStrings.connectingLabel
                            : AppStrings.startSession,
                        style: const TextStyle(fontSize: 16),
                      ),
                      style: FilledButton.styleFrom(
                        backgroundColor: theme.colorScheme.primary,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(14),
                        ),
                      ),
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
