import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:geolocator/geolocator.dart';
import 'package:latlong2/latlong.dart';
import 'package:provider/provider.dart';

import '../core/app_strings.dart';
import '../models/nearest_facility.dart';
import '../services/api_service.dart';
import '../widgets/nearby_facilities_card.dart';
import '../widgets/nearby_osm_map.dart';

class NearbyPlacesScreen extends StatefulWidget {
  const NearbyPlacesScreen({super.key});

  @override
  State<NearbyPlacesScreen> createState() => _NearbyPlacesScreenState();
}

class _NearbyPlacesScreenState extends State<NearbyPlacesScreen> {
  final MapController _mapController = MapController();

  bool _loading = false;
  Position? _position;
  List<NearestFacility> _facilities = const [];
  NearbyFacilityType _selectedType = NearbyFacilityType.all;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) => _fetchNearby());
  }

  @override
  void dispose() {
    _mapController.dispose();
    super.dispose();
  }

  void _scheduleFitMap() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted) return;
      _fitMapBounds();
    });
  }

  void _fitMapBounds() {
    if (_position == null) return;
    final user = LatLng(_position!.latitude, _position!.longitude);
    if (_facilities.isEmpty) {
      _mapController.move(user, 14);
      return;
    }
    final points = <LatLng>[
      user,
      ..._facilities.map((f) => LatLng(f.latitude, f.longitude)),
    ];
    final bounds = LatLngBounds.fromPoints(points);
    _mapController.fitCamera(
      CameraFit.bounds(
        bounds: bounds,
        padding: const EdgeInsets.fromLTRB(40, 40, 40, 56),
        maxZoom: 16,
        minZoom: 3,
      ),
    );
  }

  void _focusOnFacility(NearestFacility facility) {
    _mapController.move(
      LatLng(facility.latitude, facility.longitude),
      16,
    );
  }

  Future<void> _fetchNearby() async {
    setState(() => _loading = true);
    try {
      LocationPermission permission = await Geolocator.checkPermission();
      if (permission == LocationPermission.denied) {
        permission = await Geolocator.requestPermission();
      }
      if (permission == LocationPermission.denied ||
          permission == LocationPermission.deniedForever) {
        if (!mounted) return;
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text(AppStrings.locationRequiredForNearby)),
        );
        return;
      }

      final current = await Geolocator.getCurrentPosition(
        desiredAccuracy: LocationAccuracy.high,
      );
      if (!mounted) return;
      final api = context.read<ApiService>();
      final rawItems = await api.fetchNearbyPlaces(
        latitude: current.latitude,
        longitude: current.longitude,
        limitPerType: 10,
      );
      final facilities = rawItems
          .map((item) => NearestFacility.fromJson(item))
          .toList(growable: false);

      if (!mounted) return;
      setState(() {
        _position = current;
        _facilities = facilities;
      });
      _scheduleFitMap();
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('${AppStrings.errorOccurred}: $e')),
      );
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      appBar: AppBar(
        title: const Text(AppStrings.nearbyPageTitle),
      ),
      body: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 12, 16, 0),
            child: Text(
              AppStrings.nearbyIntro,
              style: theme.textTheme.bodyMedium,
            ),
          ),
          const SizedBox(height: 8),
          Expanded(
            flex: 5,
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 12),
              child: ClipRRect(
                borderRadius: BorderRadius.circular(12),
                child: Stack(
                  fit: StackFit.expand,
                  children: [
                    if (_position != null)
                      NearbyOsmMap(
                        mapController: _mapController,
                        userPoint:
                            LatLng(_position!.latitude, _position!.longitude),
                        facilities: _facilities,
                      )
                    else
                      ColoredBox(
                        color: theme.colorScheme.surfaceContainerHighest
                            .withOpacity(0.85),
                        child: Center(
                          child: _loading
                              ? const CircularProgressIndicator()
                              : Padding(
                                  padding: const EdgeInsets.all(24),
                                  child: Text(
                                    AppStrings.locationRequiredForNearby,
                                    textAlign: TextAlign.center,
                                    style: theme.textTheme.bodyLarge,
                                  ),
                                ),
                        ),
                      ),
                    if (_loading && _position != null)
                      Positioned(
                        top: 0,
                        left: 0,
                        right: 0,
                        child: LinearProgressIndicator(
                          backgroundColor:
                              theme.colorScheme.surface.withOpacity(0.3),
                          minHeight: 3,
                        ),
                      ),
                  ],
                ),
              ),
            ),
          ),
          const SizedBox(height: 8),
          Expanded(
            flex: 6,
            child: ListView(
              padding: const EdgeInsets.fromLTRB(16, 0, 16, 24),
              children: [
                FilledButton.icon(
                  onPressed: _loading ? null : _fetchNearby,
                  icon: _loading
                      ? const SizedBox(
                          width: 18,
                          height: 18,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                      : const Icon(Icons.refresh),
                  label: Text(
                    _loading ? AppStrings.connectingLabel : AppStrings.findNearby,
                  ),
                ),
                if (_position != null) ...[
                  const SizedBox(height: 8),
                  Text(
                    'GPS: ${_position!.latitude.toStringAsFixed(5)}, '
                    '${_position!.longitude.toStringAsFixed(5)}',
                    style: theme.textTheme.bodySmall?.copyWith(
                      color: theme.colorScheme.outline,
                    ),
                  ),
                ],
                const SizedBox(height: 8),
                NearbyFacilitiesCard(
                  facilities: _facilities,
                  selectedType: _selectedType,
                  hasLocation: _position != null,
                  onRetry: _loading ? null : _fetchNearby,
                  onFacilityTap: _position != null ? _focusOnFacility : null,
                  onTypeChanged: (type) {
                    setState(() => _selectedType = type);
                  },
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
