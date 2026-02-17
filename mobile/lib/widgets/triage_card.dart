import 'package:flutter/material.dart';

class TriageCard extends StatelessWidget {
  final Map<String, dynamic> result;
  final VoidCallback? onCallEmergency;

  const TriageCard({
    super.key,
    required this.result,
    this.onCallEmergency,
  });

  @override
  Widget build(BuildContext context) {
    final level = result['triage_level'] as String? ?? 'URGENT';
    final category = result['category'] as String? ?? 'other';
    final confidence = result['confidence'] as num?;
    final redFlags = (result['red_flags'] as List?)?.cast<String>() ?? [];

    final (bgColor, fgColor, icon) = _triageStyle(level);

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
              Icon(icon, color: fgColor, size: 24),
              const SizedBox(width: 8),
              Text(
                level,
                style: TextStyle(
                  color: fgColor,
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const Spacer(),
              if (confidence != null)
                Text(
                  '${(confidence * 100).toStringAsFixed(0)}%',
                  style: TextStyle(color: fgColor.withOpacity(0.8)),
                ),
              if (onCallEmergency != null) ...[
                const SizedBox(width: 8),
                SizedBox(
                  height: 32,
                  child: FilledButton.icon(
                    onPressed: onCallEmergency,
                    icon: const Icon(Icons.phone, size: 16),
                    label: const Text('Call 112',
                        style: TextStyle(fontSize: 12)),
                    style: FilledButton.styleFrom(
                      backgroundColor: Colors.white,
                      foregroundColor: bgColor,
                      padding:
                          const EdgeInsets.symmetric(horizontal: 12),
                    ),
                  ),
                ),
              ],
            ],
          ),
          const SizedBox(height: 4),
          Text(
            'Category: ${category.toUpperCase()}',
            style:
                TextStyle(color: fgColor.withOpacity(0.9), fontSize: 13),
          ),
          if (redFlags.isNotEmpty) ...[
            const SizedBox(height: 4),
            Text(
              'Red flags: ${redFlags.join(", ")}',
              style: TextStyle(color: fgColor, fontSize: 12),
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
