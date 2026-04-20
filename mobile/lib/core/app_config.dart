/// Build-time flags via `--dart-define=KEY=value`.
///
/// Test flow (Welcome → Home with language & test mode):
/// `flutter run --dart-define=EMERGENCY_TEST_FLOW=true`
const bool kEmergencyTestFlow =
    bool.fromEnvironment('EMERGENCY_TEST_FLOW', defaultValue: false);
