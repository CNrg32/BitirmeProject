// Acil Yardım uygulaması widget testi.
// Ana ekranın (Welcome) açıldığını ve 112 butonunun göründüğünü doğrular.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';

import 'package:emergency_assistant/main.dart';
import 'package:emergency_assistant/services/api_service.dart';

void main() {
  testWidgets('Welcome screen shows app title and 112 button', (WidgetTester tester) async {
    await tester.pumpWidget(
      MultiProvider(
        providers: [Provider<ApiService>(create: (_) => ApiService())],
        child: const EmergencyAssistantApp(),
      ),
    );
    await tester.pumpAndSettle();

    expect(find.text('Acil Yardım'), findsWidgets);
    expect(find.text('112 Ara'), findsOneWidget);
  });
}
