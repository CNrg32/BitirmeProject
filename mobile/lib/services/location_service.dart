import 'dart:async';
import 'dart:io' show Platform;

import 'package:flutter/foundation.dart' show kIsWeb, debugPrint;
import 'package:geolocator/geolocator.dart';

/// Result of a location fix attempt. Carries both the position (if any) and a
/// reason code so the UI can surface actionable feedback instead of silently
/// failing (which was the previous behaviour on iOS).
enum LocationFailureReason {
  none,
  serviceDisabled,
  permissionDenied,
  permissionDeniedForever,
  timeout,
  unknown,
}

class LocationFixResult {
  final Position? position;
  final LocationFailureReason reason;
  final Object? error;

  const LocationFixResult({
    required this.position,
    this.reason = LocationFailureReason.none,
    this.error,
  });

  bool get hasPosition => position != null;
}

/// Robust wrapper around `geolocator` that addresses the iOS-specific issues
/// seen in the field:
///
/// 1. `getCurrentPosition` can hang indefinitely on iPhone when GPS is cold
///    or indoors → we enforce explicit `timeLimit`s and fall back to lower
///    accuracy so the user still gets *some* fix.
/// 2. iOS reports service-level / permission-level failures differently than
///    Android; we check both before issuing a fix request.
/// 3. On Android we also try `getLastKnownPosition()` as a zero-latency fast
///    path (this API returns null on iOS).
class LocationService {
  LocationService._();

  static const Duration _highAccuracyTimeout = Duration(seconds: 8);
  static const Duration _mediumAccuracyTimeout = Duration(seconds: 5);

  /// Attempt to obtain a best-effort current position.
  ///
  /// Strategy:
  ///   1. Ensure location services are enabled.
  ///   2. Ensure permissions (request if needed).
  ///   3. On Android try `getLastKnownPosition()` for an instant fix cache.
  ///   4. Request a high-accuracy fix with a hard timeout.
  ///   5. On timeout retry with medium accuracy and a shorter timeout so the
  ///      UI does not stay frozen on iPhones that fail to lock GPS fast.
  static Future<LocationFixResult> getCurrentPosition() async {
    try {
      if (!kIsWeb) {
        final serviceEnabled = await Geolocator.isLocationServiceEnabled();
        if (!serviceEnabled) {
          return const LocationFixResult(
            position: null,
            reason: LocationFailureReason.serviceDisabled,
          );
        }
      }

      LocationPermission permission = await Geolocator.checkPermission();
      if (permission == LocationPermission.denied) {
        permission = await Geolocator.requestPermission();
      }
      if (permission == LocationPermission.deniedForever) {
        return const LocationFixResult(
          position: null,
          reason: LocationFailureReason.permissionDeniedForever,
        );
      }
      if (permission == LocationPermission.denied) {
        return const LocationFixResult(
          position: null,
          reason: LocationFailureReason.permissionDenied,
        );
      }

      // Instant fix if available (Android only; iOS always returns null).
      Position? lastKnown;
      if (!kIsWeb && Platform.isAndroid) {
        try {
          lastKnown = await Geolocator.getLastKnownPosition();
        } catch (_) {
          lastKnown = null;
        }
      }

      final fresh = await _requestFix(
        accuracy: LocationAccuracy.high,
        timeout: _highAccuracyTimeout,
      );
      if (fresh.hasPosition) return fresh;

      if (fresh.reason == LocationFailureReason.timeout) {
        // Downgrade accuracy for a second attempt – still accurate enough
        // for emergency dispatch to reach the caller.
        final fallback = await _requestFix(
          accuracy: LocationAccuracy.medium,
          timeout: _mediumAccuracyTimeout,
        );
        if (fallback.hasPosition) return fallback;

        if (lastKnown != null) {
          return LocationFixResult(position: lastKnown);
        }
        return fallback;
      }

      if (lastKnown != null) {
        return LocationFixResult(position: lastKnown);
      }
      return fresh;
    } catch (e) {
      debugPrint('LocationService error: $e');
      return LocationFixResult(
        position: null,
        reason: LocationFailureReason.unknown,
        error: e,
      );
    }
  }

  static Future<LocationFixResult> _requestFix({
    required LocationAccuracy accuracy,
    required Duration timeout,
  }) async {
    try {
      final position = await Geolocator.getCurrentPosition(
        desiredAccuracy: accuracy,
        timeLimit: timeout,
      );
      return LocationFixResult(position: position);
    } on TimeoutException catch (e) {
      return LocationFixResult(
        position: null,
        reason: LocationFailureReason.timeout,
        error: e,
      );
    } on LocationServiceDisabledException catch (e) {
      return LocationFixResult(
        position: null,
        reason: LocationFailureReason.serviceDisabled,
        error: e,
      );
    } on PermissionDeniedException catch (e) {
      return LocationFixResult(
        position: null,
        reason: LocationFailureReason.permissionDenied,
        error: e,
      );
    } catch (e) {
      debugPrint('LocationService._requestFix failed: $e');
      return LocationFixResult(
        position: null,
        reason: LocationFailureReason.unknown,
        error: e,
      );
    }
  }
}
