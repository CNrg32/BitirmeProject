import 'dart:convert';
import 'dart:typed_data';
import 'package:http/http.dart' as http;

class ApiService {
  static const String _baseUrl = 'http://10.0.2.2:8000';

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
  }) async {
    final body = <String, dynamic>{
      'session_id': sessionId,
    };
    if (text != null) body['text'] = text;
    if (audioBase64 != null) body['audio_base64'] = audioBase64;
    if (imageBase64 != null) body['image_base64'] = imageBase64;

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
}
