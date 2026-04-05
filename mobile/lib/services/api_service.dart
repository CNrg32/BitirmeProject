import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/foundation.dart' show kIsWeb, TargetPlatform, defaultTargetPlatform;
import 'package:http/http.dart' as http;

class ApiService {
  static String get _baseUrl {
    if (kIsWeb) {
      return 'http://localhost:8000';
    }
    if (defaultTargetPlatform == TargetPlatform.android) {
      return 'http://10.0.2.2:8000';
    }
    return 'http://localhost:8000';
  }

  Future<bool> healthCheck() async {
    try {
      final resp = await http.get(Uri.parse('$_baseUrl/health'));
      return resp.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  Future<Map<String, dynamic>> startSession(String? language) async {
    final body = <String, dynamic>{};
    if (language != null) body['language'] = language;
    final resp = await http.post(
      Uri.parse('$_baseUrl/session/start'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode(body),
    );
    if (resp.statusCode != 200) {
      throw Exception('Failed to start session: ${resp.body}');
    }
    return jsonDecode(resp.body);
  }

  Future<Map<String, dynamic>> sendMessage({
    required String sessionId,
    String? text,
    String? audioBase64,
    String? imageBase64,
    double? latitude,
    double? longitude,
  }) async {
    final body = <String, dynamic>{
      'session_id': sessionId,
    };
    if (text != null) body['text'] = text;
    if (audioBase64 != null) body['audio_base64'] = audioBase64;
    if (imageBase64 != null) body['image_base64'] = imageBase64;
    if (latitude != null) body['latitude'] = latitude;
    if (longitude != null) body['longitude'] = longitude;

    final resp = await http.post(
      Uri.parse('$_baseUrl/session/message'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode(body),
    );
    if (resp.statusCode != 200) {
      throw Exception('Message failed: ${resp.body}');
    }
    return jsonDecode(resp.body);
  }

  Future<List<Map<String, dynamic>>> fetchNearbyPlaces({
    required double latitude,
    required double longitude,
    String? preferredType,
    int limitPerType = 5,
  }) async {
    final body = <String, dynamic>{
      'latitude': latitude,
      'longitude': longitude,
      'limit_per_type': limitPerType,
    };
    if (preferredType != null) {
      body['preferred_type'] = preferredType;
    }

    final resp = await http.post(
      Uri.parse('$_baseUrl/nearby-places'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode(body),
    );
    if (resp.statusCode != 200) {
      throw Exception('Nearby places failed: ${resp.body}');
    }

    final decoded = jsonDecode(resp.body) as Map<String, dynamic>;
    final items = decoded['nearby_places'] as List? ?? const [];
    return items
        .whereType<Map>()
        .map((item) => Map<String, dynamic>.from(item.cast<Object?, Object?>()))
        .toList(growable: false);
  }

  Future<Map<String, dynamic>> transcribeAudio({
    required String sessionId,
    required String audioBase64,
  }) async {
    final resp = await http.post(
      Uri.parse('$_baseUrl/session/transcribe'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'session_id': sessionId,
        'audio_base64': audioBase64,
      }),
    );
    if (resp.statusCode != 200) {
      throw Exception('Transcribe failed: ${resp.body}');
    }
    return jsonDecode(resp.body);
  }

  Future<Map<String, dynamic>> predict(String textEn) async {
    final resp = await http.post(
      Uri.parse('$_baseUrl/predict'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'text_en': textEn,
        'meta': {'deaths': 0, 'potential_death': 0, 'false_alarm': 0},
        'slots': {},
      }),
    );
    if (resp.statusCode != 200) {
      throw Exception('Predict failed: ${resp.body}');
    }
    return jsonDecode(resp.body);
  }

  Future<Uint8List> tts(String text, String language) async {
    final resp = await http.post(
      Uri.parse('$_baseUrl/tts'),
      body: {'text': text, 'language': language},
    );
    if (resp.statusCode != 200) {
      throw Exception('TTS failed: ${resp.body}');
    }
    return resp.bodyBytes;
  }

  /// Eğitilmiş görüntü modeli ile sahne analizi (multipart image).
  /// Dönen map: classification, consistency, summary, available.
  Future<Map<String, dynamic>> analyzeImage(
    Uint8List imageBytes, {
    String? textCategory,
    String? textTriageLevel,
  }) async {
    final uri = Uri.parse('$_baseUrl/analyze-image');
    final request = http.MultipartRequest('POST', uri);
    request.files.add(http.MultipartFile.fromBytes(
      'image',
      imageBytes,
      filename: 'image.jpg',
    ));
    if (textCategory != null) {
      request.fields['text_category'] = textCategory;
    }
    if (textTriageLevel != null) {
      request.fields['text_triage_level'] = textTriageLevel;
    }
    final streamed = await request.send();
    final resp = await http.Response.fromStream(streamed);
    if (resp.statusCode != 200) {
      throw Exception('Image analysis failed: ${resp.body}');
    }
    return jsonDecode(resp.body) as Map<String, dynamic>;
  }
}
