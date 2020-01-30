import 'dart:convert';
import 'package:dart_nn/src/activation.dart';

class Layer {
  int nodes;
  Activation activation_fn;

  Layer(int nodes, String activation_fn) {
    this.nodes = nodes;
    if (activation_fn != 'Input') {
      this.activation_fn = ActivationFunctions[activation_fn];
    }
  }

  // serialize
  Map<String, dynamic> toJson() {
    return {
      'nodes': nodes,
      'activation_fn':
          activation_fn?.name != null ? activation_fn.name : 'Input',
    };
  }

  // deserialize
  Layer.fromJson(Map<String, dynamic> json) {
    nodes = json['nodes'];
    activation_fn = json['activation_fn'] != 'Input'
        ? ActivationFunctions[json['activation_fn']]
        : null;
  }

  static String serialize(Layer layer) {
    return jsonEncode(layer);
  }

  static Layer deserialize(String jsonString) {
    Map layerMap = jsonDecode(jsonString);
    var result = Layer.fromJson(layerMap);
    return result;
  }
}
