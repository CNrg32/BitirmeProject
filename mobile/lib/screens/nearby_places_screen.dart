import 'package:flutter/material.dart';
import 'package:geolocator/geolocator.dart';
import 'package:provider/provider.dart';

import '../core/app_strings.dart';
import '../models/nearest_facility.dart';
import '../services/api_service.dart';
import '../widgets/nearby_facilities_card.dart';

class NearbyPlacesScreen extends StatefulWidget {
  const NearbyPlacesScreen({super.key});

  @override
  State<NearbyPlacesScreen> createState() => _NearbyPlacesScreenState();
}

class _NearbyPlacesScreenState extends State<NearbyPlacesScreen> {
  bool _loading = false;
  Position? _position;
  List<NearestFacility> _facilities = const [];
  NearbyFacilityType _selectedType = NearbyFacilityType.hospital;

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
      final api = context.read<ApiService>();
      final rawItems = await api.fetchNearbyPlaces(
        latitude: current.latitude,
        longitude: current.longitude,
      );
      final facilities = rawItems
          .map((item) => NearestFacility.fromJson(item))
          .toList(growable: false);

      if (!mounted) return;
      setState(() {
        _position = current;
        _facilities = facilities;
      });
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
    return Scaffold(
      appBar: AppBar(
        title: const Text(AppStrings.nearbyPageTitle),
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          Text(
            AppStrings.nearbyIntro,
            style: Theme.of(context).textTheme.bodyMedium,
          ),
          const SizedBox(height: 12),
          FilledButton.icon(
            onPressed: _loading ? null : _fetchNearby,
            icon: _loading
                ? const SizedBox(
                    width: 18,
                    height: 18,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                : const Icon(Icons.my_location),
            label: Text(_loading ? AppStrings.connectingLabel : AppStrings.findNearby),
          ),
          const SizedBox(height: 12),
          if (_position != null)
            Text(
              'GPS: ${_position!.latitude.toStringAsFixed(5)}, ${_position!.longitude.toStringAsFixed(5)}',
            ),
          const SizedBox(height: 8),
          NearbyFacilitiesCard(
            facilities: _facilities,
            selectedType: _selectedType,
            hasLocation: _position != null,
            onTypeChanged: (type) {
              setState(() => _selectedType = type);
            },
          ),
        ],
      ),
    );
  }
}