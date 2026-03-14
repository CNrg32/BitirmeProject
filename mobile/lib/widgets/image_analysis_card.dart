import 'package:flutter/material.dart';
import '../core/app_strings.dart';

class ImageAnalysisCard extends StatelessWidget {
  final Map<String, dynamic> analysis;

  const ImageAnalysisCard({super.key, required this.analysis});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final classification = analysis['classification'] as Map<String, dynamic>?;
    final consistency = analysis['consistency'] as Map<String, dynamic>?;
    final summary = analysis['summary'] as String?;

    if (classification == null && summary == null) {
      return const SizedBox.shrink();
    }

    final topClass = classification?['detected_class'] as String? ??
        classification?['top_class'] as String? ??
        'Bilinmiyor';
    final confidence = classification?['confidence'] as num?;
    final dispatchUnits =
        (classification?['dispatch_units'] as List?)?.cast<String>() ?? [];
    final isConsistent = consistency?['is_consistent'] as bool? ?? true;
    final consistencyDetail = consistency?['consistency_detail'] as String?;
    final riskNotes = (consistency?['risk_notes'] as List?)?.cast<String>() ?? [];

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: theme.colorScheme.secondaryContainer,
        border: Border(
          bottom: BorderSide(
            color: theme.colorScheme.outline.withOpacity(0.2),
          ),
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.image_search,
                  size: 18,
                  color: theme.colorScheme.onSecondaryContainer),
              const SizedBox(width: 6),
              Text(
                AppStrings.imageAnalysis,
                style: theme.textTheme.labelLarge?.copyWith(
                  color: theme.colorScheme.onSecondaryContainer,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
          const SizedBox(height: 6),
          Text(
            '${AppStrings.scene}: $topClass'
            '${confidence != null ? ' (${(confidence * 100).toStringAsFixed(0)}%)' : ''}',
            style: theme.textTheme.bodySmall?.copyWith(
              color: theme.colorScheme.onSecondaryContainer,
            ),
          ),
          if (dispatchUnits.isNotEmpty)
            Padding(
              padding: const EdgeInsets.only(top: 2),
              child: Text(
                '${AppStrings.dispatch}: ${dispatchUnits.join(", ")}',
                style: theme.textTheme.bodySmall?.copyWith(
                  color: theme.colorScheme.onSecondaryContainer,
                ),
              ),
            ),
          if (!isConsistent)
            Padding(
              padding: const EdgeInsets.only(top: 4),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Icon(Icons.warning_amber,
                      size: 14, color: Colors.orange),
                  const SizedBox(width: 4),
                  Expanded(
                    child: Text(
                      consistencyDetail ??
                          (riskNotes.isNotEmpty
                              ? riskNotes.first
                              : 'Image may not match the reported incident'),
                      style: theme.textTheme.bodySmall
                          ?.copyWith(color: Colors.orange[800]),
                    ),
                  ),
                ],
              ),
            ),
          if (summary != null && summary.isNotEmpty)
            Padding(
              padding: const EdgeInsets.only(top: 4),
              child: Text(
                summary,
                style: theme.textTheme.bodySmall?.copyWith(
                  color: theme.colorScheme.onSecondaryContainer
                      .withOpacity(0.8),
                ),
                maxLines: 2,
                overflow: TextOverflow.ellipsis,
              ),
            ),
        ],
      ),
    );
  }
}
