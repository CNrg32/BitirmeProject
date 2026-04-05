enum NearbyFacilityType { hospital, police }

class NearestFacility {
  final String id;
  final NearbyFacilityType type;
  final String name;
  final String address;
  final double distanceMeters;
  final double latitude;
  final double longitude;
  final String? phone;
  final int? etaMinutes;

  const NearestFacility({
    required this.id,
    required this.type,
    required this.name,
    required this.address,
    required this.distanceMeters,
    required this.latitude,
    required this.longitude,
    this.phone,
    this.etaMinutes,
  });

  factory NearestFacility.fromJson(Map<String, dynamic> json) {
    return NearestFacility(
      id: (json['id'] as String?) ?? '',
      type: _parseType(json['type'] as String?),
      name: (json['name'] as String?) ?? 'Bilinmeyen kurum',
      address: (json['address'] as String?) ?? 'Adres bilgisi yok',
      distanceMeters: (json['distance_meters'] as num?)?.toDouble() ?? 0,
      latitude: (json['latitude'] as num?)?.toDouble() ?? 0,
      longitude: (json['longitude'] as num?)?.toDouble() ?? 0,
      phone: json['phone'] as String?,
      etaMinutes: (json['eta_minutes'] as num?)?.toInt(),
    );
  }

  static NearbyFacilityType _parseType(String? rawType) {
    if (rawType == 'police') {
      return NearbyFacilityType.police;
    }
    return NearbyFacilityType.hospital;
  }
}