import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import '../core/app_strings.dart';

class ReportSheet extends StatelessWidget {
  final String report;
  final Map<String, dynamic>? triageResult;
  final Map<String, dynamic>? imageAnalysis;
  final String? locationText;

  const ReportSheet({
    super.key,
    required this.report,
    this.triageResult,
    this.imageAnalysis,
    this.locationText,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final level = triageResult?['triage_level'] as String? ?? '';
    final bgColor = _levelColor(level);

    return DraggableScrollableSheet(
      initialChildSize: 0.85,
      minChildSize: 0.5,
      maxChildSize: 0.95,
      builder: (_, scrollController) => Container(
        decoration: BoxDecoration(
          color: theme.colorScheme.surface,
          borderRadius:
              const BorderRadius.vertical(top: Radius.circular(20)),
        ),
        child: Column(
          children: [
            Container(
              margin: const EdgeInsets.only(top: 12),
              width: 40,
              height: 4,
              decoration: BoxDecoration(
                color: theme.colorScheme.outline.withOpacity(0.3),
                borderRadius: BorderRadius.circular(2),
              ),
            ),
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: bgColor.withOpacity(0.1),
                border: Border(
                  bottom: BorderSide(color: bgColor.withOpacity(0.3)),
                ),
              ),
              child: Row(
                children: [
                  Icon(Icons.assignment, color: bgColor),
                  const SizedBox(width: 8),
                  Text(
                    AppStrings.emergencyReport,
                    style: theme.textTheme.titleLarge?.copyWith(
                      fontWeight: FontWeight.bold,
                      color: bgColor,
                    ),
                  ),
                  const Spacer(),
                  IconButton(
                    icon: const Icon(Icons.copy),
                    tooltip: AppStrings.copyReport,
                    onPressed: () {
                      Clipboard.setData(ClipboardData(text: report));
                      ScaffoldMessenger.of(context).showSnackBar(
                        const SnackBar(
                          content: Text(AppStrings.reportCopied),
                        ),
                      );
                    },
                  ),
                ],
              ),
            ),
            Expanded(
              child: ListView(
                controller: scrollController,
                padding: const EdgeInsets.all(16),
                children: [
                  if (locationText != null && locationText!.isNotEmpty)
                    Container(
                      padding: const EdgeInsets.all(12),
                      margin: const EdgeInsets.only(bottom: 12),
                      decoration: BoxDecoration(
                        color: theme.colorScheme.tertiaryContainer,
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: Row(
                        children: [
                          Icon(Icons.location_on,
                              size: 18,
                              color:
                                  theme.colorScheme.onTertiaryContainer),
                          const SizedBox(width: 8),
                          Expanded(
                            child: Text(
                              'GPS: $locationText',
                              style: theme.textTheme.bodyMedium?.copyWith(
                                color: theme
                                    .colorScheme.onTertiaryContainer,
                                fontWeight: FontWeight.w500,
                              ),
                            ),
                          ),
                          IconButton(
                            icon: const Icon(Icons.copy, size: 16),
                            tooltip: AppStrings.copyCoordinates,
                            onPressed: () {
                              Clipboard.setData(
                                ClipboardData(text: locationText!),
                              );
                              ScaffoldMessenger.of(context).showSnackBar(
                                const SnackBar(
                                  content: Text(AppStrings.coordinatesCopied),
                                ),
                              );
                            },
                          ),
                        ],
                      ),
                    ),
                  SelectableText(
                    report,
                    style: theme.textTheme.bodyMedium?.copyWith(
                      height: 1.6,
                      fontSize: 14,
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Color _levelColor(String level) {
    switch (level) {
      case 'CRITICAL':
        return const Color(0xFFD32F2F);
      case 'URGENT':
        return const Color(0xFFF57C00);
      default:
        return const Color(0xFF388E3C);
    }
  }
}
