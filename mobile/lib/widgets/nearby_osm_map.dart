import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:latlong2/latlong.dart';

import '../core/app_theme.dart';
import '../models/nearest_facility.dart';

/// Ücretsiz OSM karoları ([flutter_map]) — Google Maps anahtarı gerekmez.
///
/// Not: [userPoint] henüz yoksa harita yine yüklenir ve [fallbackCenter]
/// etrafında gösterilir. Böylece "harita yüklenmiyor" yanılgısı oluşmaz.
class NearbyOsmMap extends StatelessWidget {
  final MapController mapController;
  final LatLng? userPoint;
  final LatLng fallbackCenter;
  final double initialZoom;
  final List<NearestFacility> facilities;

  const NearbyOsmMap({
    super.key,
    required this.mapController,
    required this.userPoint,
    required this.fallbackCenter,
    required this.facilities,
    this.initialZoom = 15,
  });

  static const String _tileUserAgent = 'emergency_assistant';

  @override
  Widget build(BuildContext context) {
    final markers = <Marker>[
      if (userPoint != null)
        Marker(
          point: userPoint!,
          width: 44,
          height: 44,
          alignment: Alignment.center,
          child: Tooltip(
            message: 'Konumunuz',
            child: Container(
              decoration: BoxDecoration(
                color: Theme.of(context).colorScheme.primary,
                shape: BoxShape.circle,
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withOpacity(0.25),
                    blurRadius: 4,
                    offset: const Offset(0, 2),
                  ),
                ],
                border: Border.all(color: Colors.white, width: 2),
              ),
              padding: const EdgeInsets.all(6),
              child: const Icon(
                Icons.person_pin_circle,
                color: Colors.white,
                size: 22,
              ),
            ),
          ),
        ),
      ...facilities.map(
        (f) => Marker(
          point: LatLng(f.latitude, f.longitude),
          width: 44,
          height: 44,
          alignment: Alignment.center,
          child: Tooltip(
            message: f.name,
            child: _FacilityPin(type: f.type),
          ),
        ),
      ),
    ];

    return FlutterMap(
      mapController: mapController,
      options: MapOptions(
        initialCenter: userPoint ?? fallbackCenter,
        initialZoom: initialZoom,
        maxZoom: 18,
        minZoom: 3,
      ),
      children: [
        TileLayer(
          urlTemplate: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
          userAgentPackageName: _tileUserAgent,
        ),
        MarkerLayer(markers: markers),
        const SimpleAttributionWidget(
          alignment: Alignment.bottomRight,
          source: Text('OpenStreetMap contributors'),
        ),
      ],
    );
  }
}

class _FacilityPin extends StatelessWidget {
  final NearbyFacilityType type;

  const _FacilityPin({required this.type});

  @override
  Widget build(BuildContext context) {
    final isPolice = type == NearbyFacilityType.police;
    final bg = isPolice ? const Color(0xFF1565C0) : AppTheme.criticalRed;
    final icon = isPolice ? Icons.local_police : Icons.local_hospital;

    return Container(
      width: 40,
      height: 40,
      decoration: BoxDecoration(
        color: bg,
        shape: BoxShape.circle,
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.2),
            blurRadius: 3,
            offset: const Offset(0, 2),
          ),
        ],
        border: Border.all(color: Colors.white, width: 2),
      ),
      child: Icon(icon, color: Colors.white, size: 22),
    );
  }
}
