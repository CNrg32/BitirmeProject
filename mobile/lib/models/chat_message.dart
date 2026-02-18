import 'dart:typed_data';

class ChatMessage {
  final String text;
  final bool isUser;
  final Uint8List? imageBytes;
  final String? audioBase64;

  ChatMessage({
    required this.text,
    required this.isUser,
    this.imageBytes,
    this.audioBase64,
  });
}
