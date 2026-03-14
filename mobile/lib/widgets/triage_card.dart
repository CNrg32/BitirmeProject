import 'package:flutter/material.dart';
import '../core/app_strings.dart';
import '../core/app_theme.dart';

class TriageCard extends StatelessWidget {
  final Map<String, dynamic> result;
  final VoidCallback? onCallEmergency;

  const TriageCard({
    super.key,
    required this.result,
    this.onCallEmergency,
  });

  static String _levelLabel(String level) {
    switch (level) {
      case 'CRITICAL':
        return AppStrings.triageCritical;
      case 'URGENT':
        return AppStrings.triageUrgent;
      default:
        return AppStrings.triageNonUrgent;
    }
  }

  @override
  Widget build(BuildContext context) {
    final level = result['triage_level'] as String? ?? 'URGENT';
    final category = result['category'] as String? ?? 'other';
    final confidence = result['confidence'] as num?;
    final redFlags = (result['red_flags'] as List?)?.cast<String>() ?? [];

    final (bgColor, fgColor, icon) = _triageStyle(level);
    final levelLabel = _levelLabel(level);

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: bgColor,
        boxShadow: [
          BoxShadow(
            color: bgColor.withOpacity(0.4),
            blurRadius: 8,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(icon, color: fgColor, size: 28),
              const SizedBox(width: 10),
              Expanded(
                child: Text(
                  levelLabel,
                  style: TextStyle(
                    color: fgColor,
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
              if (confidence != null)
                Text(
                  '${(confidence * 100).toStringAsFixed(0)}%',
                  style: TextStyle(color: fgColor.withOpacity(0.9)),
                ),
              if (onCallEmergency != null) ...[
                const SizedBox(width: 10),
                SizedBox(
                  height: 36,
                  child: FilledButton.icon(
                    onPressed: onCallEmergency,
                    icon: const Icon(Icons.phone, size: 18),
                    label: const Text(AppStrings.call112,
                        style: TextStyle(fontSize: 13)),
                    style: FilledButton.styleFrom(
                      backgroundColor: Colors.white,
                      foregroundColor: AppTheme.criticalRed,
                      padding: const EdgeInsets.symmetric(horizontal: 14),
                    ),
                  ),
                ),
              ],
            ],
          ),
          const SizedBox(height: 8),
          Text(
            '${AppStrings.category}: ${category.toUpperCase()}',
            style: TextStyle(
              color: fgColor.withOpacity(0.95),
              fontSize: 14,
            ),
          ),
          if (redFlags.isNotEmpty) ...[
            const SizedBox(height: 6),
            Text(
              '${AppStrings.redFlags}: ${redFlags.join(", ")}',
              style: TextStyle(color: fgColor, fontSize: 13),
            ),
          ],
        ],
      ),
    );
  }

  (Color, Color, IconData) _triageStyle(String level) {
    switch (level) {
      case 'CRITICAL':
        return (const Color(0xFFD32F2F), Colors.white, Icons.warning_amber);
      case 'URGENT':
        return (
          const Color(0xFFF57C00),
          Colors.white,
          Icons.priority_high
        );
      default:
        return (
          const Color(0xFF388E3C),
          Colors.white,
          Icons.check_circle
        );
    }
  }
}
