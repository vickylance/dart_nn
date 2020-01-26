import 'dart:math';
import 'package:dart_numerics/dart_numerics.dart' as numerics;

class Activation {
  Function func;
  Function dfunc;
  String name;

  Activation(String name, Function func, Function dfunc) {
    this.name = name;
    this.func = func;
    this.dfunc = dfunc;
  }
}

Activation Sigmoid = Activation(
  'Sigmoid',
  (double x) => 1 / (1 + exp(-1 * x)),
  (double y) => y * (1 - y),
);

Activation Relu = Activation(
  'Relu',
  (double x) => x < 0 ? 0.0 : x,
  (double y) => y < 0 ? 0.0 : 1.0,
);

Activation TanH = Activation(
  'TanH',
  (double x) => numerics.tanh(x),
  (double y) => 1 - pow(numerics.tanh(y), 2),
);

Activation ArcTan = Activation(
  'ArcTan',
  (double x) => atan(x),
  (double y) => 1 / (pow(y, 2) + 1),
);

Activation ArcSinH = Activation(
  'ArcSinH',
  (double x) => numerics.asinh(x),
  (double y) => 1 / sqrt((pow(y, 2) + 1)),
);

Activation BentIdentity = Activation(
  'BentIdentity',
  (double x) => ((sqrt(pow(x, 2) + 1) - 1) / 2) + x,
  (double y) => (y / (2 * sqrt(pow(y, 2) + 1))) + 1,
);

Activation Gaussian = Activation(
  'Gaussian',
  (double x) => exp(pow(-1 * x, 2)),
  (double y) => -2 * y * exp(pow(-1 * y, 2)),
);

var ActivationFunctions = {
  'Sigmoid': Sigmoid,
  'Relu': Relu,
  'TanH': TanH,
  'ArcTan': ArcTan,
  'ArcSinH': ArcSinH,
  'BentIdentity': BentIdentity,
  'Gaussian': Gaussian,
};