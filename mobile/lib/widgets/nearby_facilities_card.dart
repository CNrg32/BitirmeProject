import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';

import '../core/app_strings.dart';
import '../core/app_theme.dart';
import '../models/nearest_facility.dart';

class NearbyFacilitiesCard extends StatelessWidget {
  final List<NearestFacility> facilities;
  final NearbyFacilityType selectedType;
  final ValueChanged<NearbyFacilityType> onTypeChanged;
  final bool hasLocation;
  final VoidCallback? onRetry;
  final ValueChanged<NearestFacility>? onFacilityTap;

  const NearbyFacilitiesCard({
    super.key,
    required this.facilities,
    required this.selectedType,
    required this.onTypeChanged,
    required this.hasLocation,
    this.onRetry,
    this.onFacilityTap,
  });

  static List<NearestFacility> _visibleList(
    List<NearestFacility> facilities,
    NearbyFacilityType selectedType,
  ) {
    if (selectedType == NearbyFacilityType.all) {
      final copy = List<NearestFacility>.from(facilities);
      copy.sort((a, b) => a.distanceMeters.compareTo(b.distanceMeters));
      return copy;
    }
    return facilities
        .where((facility) => facility.type == selectedType)
        .toList(growable: false);
  }

  IconData _headerIcon() {
    switch (selectedType) {
      case NearbyFacilityType.hospital:
        return Icons.local_hospital;
      case NearbyFacilityType.police:
        return Icons.local_police;
      case NearbyFacilityType.all:
        return Icons.map;
    }
  }

  Color _headerColor(ThemeData theme) {
    switch (selectedType) {
      case NearbyFacilityType.hospital:
        return AppTheme.criticalRed;
      case NearbyFacilityType.police:
        return theme.colorScheme.primary;
      case NearbyFacilityType.all:
        return theme.colorScheme.secondary;
    }
  }

  String _emptyMessage() {
    switch (selectedType) {
      case NearbyFacilityType.hospital:
        return AppStrings.noNearbyHospitals;
      case NearbyFacilityType.police:
        return AppStrings.noNearbyPolice;
      case NearbyFacilityType.all:
        return AppStrings.noNearbyAny;
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final visibleFacilities = _visibleList(facilities, selectedType);

    return Container(
      width: double.infinity,
      margin: const EdgeInsets.fromLTRB(12, 8, 12, 0),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: theme.colorScheme.surface,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.06),
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
              Icon(
                _headerIcon(),
                color: _headerColor(theme),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  AppStrings.nearbyFacilities,
                  style: theme.textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Wrap(
            spacing: 8,
            children: [
              FilterChip(
                label: const Text(AppStrings.allFacilitiesOption),
                selected: selectedType == NearbyFacilityType.all,
                onSelected: (value) {
                  if (value) onTypeChanged(NearbyFacilityType.all);
                },
              ),
              FilterChip(
                label: const Text(AppStrings.hospitalOption),
                selected: selectedType == NearbyFacilityType.hospital,
                onSelected: (value) {
                  if (value) onTypeChanged(NearbyFacilityType.hospital);
                },
              ),
              FilterChip(
                label: const Text(AppStrings.policeOption),
                selected: selectedType == NearbyFacilityType.police,
                onSelected: (value) {
                  if (value) onTypeChanged(NearbyFacilityType.police);
                },
              ),
            ],
          ),
          const SizedBox(height: 12),
          if (!hasLocation)
            Text(
              AppStrings.locationRequiredForNearby,
              style: theme.textTheme.bodyMedium,
            )
          else if (visibleFacilities.isEmpty)
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Icon(
                      Icons.info_outline,
                      size: 18,
                      color: theme.colorScheme.secondary,
                    ),
                    const SizedBox(width: 6),
                    Expanded(
                      child: Text(
                        _emptyMessage(),
                        style: theme.textTheme.bodyMedium,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 10),
                if (onRetry != null)
                  OutlinedButton.icon(
                    onPressed: onRetry,
                    icon: const Icon(Icons.refresh, size: 18),
                    label: const Text(AppStrings.retrySearch),
                  ),
              ],
            )
          else
            ...visibleFacilities.map((facility) => Padding(
                  padding: const EdgeInsets.only(bottom: 10),
                  child: _FacilityTile(
                    facility: facility,
                    onHighlight: onFacilityTap,
                  ),
                )),
        ],
      ),
    );
  }
}

class _FacilityTile extends StatelessWidget {
  final NearestFacility facility;
  final ValueChanged<NearestFacility>? onHighlight;

  const _FacilityTile({
    required this.facility,
    this.onHighlight,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    final content = Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: theme.colorScheme.surfaceContainerHighest.withOpacity(0.45),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(
          color: theme.colorScheme.outline.withOpacity(0.12),
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Icon(
                facility.type == NearbyFacilityType.police
                    ? Icons.local_police
                    : Icons.local_hospital,
                size: 20,
                color: facility.type == NearbyFacilityType.police
                    ? theme.colorScheme.primary
                    : AppTheme.criticalRed,
              ),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  facility.name,
                  style: theme.textTheme.titleSmall?.copyWith(
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 4),
          Text(
            '${_distanceLabel(facility.distanceMeters)} • ${facility.address}',
            style: theme.textTheme.bodySmall,
          ),
          if (facility.etaMinutes != null)
            Padding(
              padding: const EdgeInsets.only(top: 4),
              child: Text(
                '${AppStrings.estimatedArrival}: ${facility.etaMinutes} dk',
                style: theme.textTheme.bodySmall,
              ),
            ),
          const SizedBox(height: 8),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [
              OutlinedButton.icon(
                onPressed: () => _openDirections(facility),
                icon: const Icon(Icons.directions),
                label: const Text(AppStrings.getDirections),
              ),
              if (facility.phone != null && facility.phone!.trim().isNotEmpty)
                OutlinedButton.icon(
                  onPressed: () => _callPhone(facility.phone!),
                  icon: const Icon(Icons.phone),
                  label: const Text(AppStrings.callFacility),
                ),
            ],
          ),
        ],
      ),
    );

    if (onHighlight == null) {
      return content;
    }

    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: () => onHighlight!(facility),
        borderRadius: BorderRadius.circular(14),
        child: content,
      ),
    );
  }

  String _distanceLabel(double distanceMeters) {
    if (distanceMeters >= 1000) {
      return '${(distanceMeters / 1000).toStringAsFixed(1)} km';
    }
    return '${distanceMeters.toStringAsFixed(0)} m';
  }

  Future<void> _openDirections(NearestFacility facility) async {
    final uri = Uri.parse(
      'https://www.openstreetmap.org/directions?to=${facility.latitude}%2C${facility.longitude}',
    );
    await launchUrl(uri, mode: LaunchMode.externalApplication);
  }

  Future<void> _callPhone(String phone) async {
    final uri = Uri(scheme: 'tel', path: phone);
    await launchUrl(uri);
  }
}