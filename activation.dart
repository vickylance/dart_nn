import 'dart:math';
import 'package:dart_numerics/dart_numerics.dart' as numerics;

class Activation {
  static double Sigmoid(double x, { bool derivative = false }) {
    if (!derivative) {
      return 1 / (1 + exp(-x));
    }
    return x * (1 - x);
  }

  static double DSigmoid(double x) {
    return 1 / (1 + exp(-1 * x));
  }

  static List<double> Relu(List<double> X, { bool derivative = false }) {
    if (!derivative) {
      return X.map((x) => (x < 0 ? 0.0 : x)).toList();
    }
    return X.map((x) => (x < 0 ? 0.0 : 1.0)).toList();
  }

  static double TanH(double x, { bool derivative = false }) {
    if (!derivative) {
      return numerics.tanh(x);
    }
    return 1 - pow(Activation.TanH(x), 2);
  }

  static double ArcTan(double x, { bool derivative = false }) {
    if (!derivative) {
      return atan(x);
    }
    return 1 / (pow(x, 2) + 1);
  }

  static double ArcSinH(double x, { bool derivative = false }) {
    if (!derivative) {
      return numerics.asinh(x);
    }
    return 1 / sqrt((pow(x, 2) + 1));
  }

  static double BentIdentity(double x, { bool derivative = false }) {
    if(!derivative) {
      return ((sqrt(pow(x, 2) + 1) - 1) / 2) + x;
    }
    return (x / (2 * sqrt(pow(x, 2) + 1))) + 1;
  }

  static double Gaussian(double x, { bool derivative = false }) {
    if(!derivative) {
      return exp(pow(-1 * x, 2));
    }
    return -2 * x * exp(pow(-1 * x, 2));
  }
}
