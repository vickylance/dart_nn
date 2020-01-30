import 'package:dart_nn/dart_nn.dart';
import 'package:test/test.dart';

void main() {
  test('map function maps values in range correctly', () {
    expect(map(5, 0.0, 10.0, 0.0, 5), 2.5);
  });
}
