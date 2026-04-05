import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';
import '../core/app_strings.dart';
import '../core/app_theme.dart';
import 'home_screen.dart';
import 'nearby_places_screen.dart';

/// Uygulama açılış ekranı: 112 hızlı erişim, kısa bilgi, "Başla" ile ana ekrana geçiş.
class WelcomeScreen extends StatelessWidget {
  const WelcomeScreen({super.key});

  Future<void> _call112(BuildContext context) async {
    final uri = Uri(scheme: 'tel', path: '112');
    if (await canLaunchUrl(uri)) {
      await launchUrl(uri);
    } else {
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('112 araması bu cihazda başlatılamıyor.')),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final isDark = theme.brightness == Brightness.dark;

    return Scaffold(
      body: Container(
        width: double.infinity,
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: isDark
                ? [
                    theme.colorScheme.primary.withOpacity(0.2),
                    theme.colorScheme.surface,
                  ]
                : [
                    AppTheme.primaryRed.withOpacity(0.08),
                    theme.scaffoldBackgroundColor,
                  ],
          ),
        ),
        child: SafeArea(
          child: Column(
            children: [
              // Üstte her zaman 112 — acil durumda tek dokunuşla arama
              Padding(
                padding: const EdgeInsets.fromLTRB(16, 12, 16, 8),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.end,
                  children: [
                    IconButton(
                      tooltip: AppStrings.nearbyPageTitle,
                      onPressed: () {
                        Navigator.push(
                          context,
                          MaterialPageRoute(
                            builder: (_) => const NearbyPlacesScreen(),
                          ),
                        );
                      },
                      icon: const Icon(Icons.place, size: 28),
                    ),
                    const SizedBox(width: 8),
                    Semantics(
                      button: true,
                      label: AppStrings.emergencyCall,
                      child: FilledButton.icon(
                        onPressed: () => _call112(context),
                        icon: const Icon(Icons.phone, size: 22),
                        label: Text(
                          AppStrings.emergencyCall,
                          style: const TextStyle(
                            fontSize: 15,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                        style: FilledButton.styleFrom(
                          backgroundColor: AppTheme.criticalRed,
                          foregroundColor: Colors.white,
                          padding: const EdgeInsets.symmetric(
                            horizontal: 20,
                            vertical: 14,
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 24),
              // Logo ve başlık
              Icon(
                Icons.emergency,
                size: 72,
                color: theme.colorScheme.primary,
              ),
              const SizedBox(height: 16),
              Text(
                AppStrings.welcomeTitle,
                style: theme.textTheme.headlineSmall?.copyWith(
                  fontWeight: FontWeight.bold,
                  color: theme.colorScheme.primary,
                ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 12),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 28),
                child: Text(
                  AppStrings.welcomeSubtitle,
                  style: theme.textTheme.bodyMedium?.copyWith(
                    color: theme.colorScheme.onSurface.withOpacity(0.8),
                    height: 1.4,
                  ),
                  textAlign: TextAlign.center,
                ),
              ),
              const Spacer(),
              // Ana CTA: Başla
              Padding(
                padding: const EdgeInsets.fromLTRB(24, 16, 24, 32),
                child: Column(
                  children: [
                    SizedBox(
                      width: double.infinity,
                      height: 56,
                      child: FilledButton.icon(
                        onPressed: () {
                          Navigator.pushReplacement(
                            context,
                            MaterialPageRoute(
                              builder: (_) => const HomeScreen(),
                            ),
                          );
                        },
                        icon: const Icon(Icons.mic, size: 24),
                        label: Text(
                          AppStrings.startSession,
                          style: const TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                        style: FilledButton.styleFrom(
                          backgroundColor: theme.colorScheme.primary,
                          foregroundColor: Colors.white,
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(14),
                          ),
                        ),
                      ),
                    ),
                    const SizedBox(height: 12),
                    Text(
                      'Dil seçimi ve acil türü sonraki ekranda',
                      style: theme.textTheme.bodySmall?.copyWith(
                        color: theme.colorScheme.outline,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
