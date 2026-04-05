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

  const NearbyFacilitiesCard({
    super.key,
    required this.facilities,
    required this.selectedType,
    required this.onTypeChanged,
    required this.hasLocation,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final visibleFacilities = facilities
        .where((facility) => facility.type == selectedType)
        .toList(growable: false);

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
                selectedType == NearbyFacilityType.hospital
                    ? Icons.local_hospital
                    : Icons.local_police,
                color: selectedType == NearbyFacilityType.hospital
                    ? AppTheme.criticalRed
                    : theme.colorScheme.primary,
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
              ChoiceChip(
                label: const Text(AppStrings.hospitalOption),
                selected: selectedType == NearbyFacilityType.hospital,
                onSelected: (_) => onTypeChanged(NearbyFacilityType.hospital),
              ),
              ChoiceChip(
                label: const Text(AppStrings.policeOption),
                selected: selectedType == NearbyFacilityType.police,
                onSelected: (_) => onTypeChanged(NearbyFacilityType.police),
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
            Text(
              selectedType == NearbyFacilityType.hospital
                  ? AppStrings.noNearbyHospitals
                  : AppStrings.noNearbyPolice,
              style: theme.textTheme.bodyMedium,
            )
          else
            ...visibleFacilities.map((facility) => Padding(
                  padding: const EdgeInsets.only(bottom: 10),
                  child: _FacilityTile(facility: facility),
                )),
        ],
      ),
    );
  }
}

class _FacilityTile extends StatelessWidget {
  final NearestFacility facility;

  const _FacilityTile({required this.facility});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Container(
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
          Text(
            facility.name,
            style: theme.textTheme.titleSmall?.copyWith(
              fontWeight: FontWeight.w700,
            ),
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