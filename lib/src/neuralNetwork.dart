import 'dart:convert';
import 'package:dart_nn/src/matrix.dart';
import 'package:dart_nn/src/activation.dart';
import 'package:dart_nn/src/layer.dart';

class NeuralNetwork {
  Layer inputNodes;
  List<Layer> hiddenNodes;
  Layer outputNodes;
  List<Matrix> weights;
  List<Matrix> biases;
  List<Activation> activation_functions;
  double learning_rate;

  NeuralNetwork(int inputNodes, List<Layer> hiddenNodes, Layer outputNodes) {
    this.inputNodes = Layer(inputNodes, 'Input');
    this.hiddenNodes = hiddenNodes;
    this.outputNodes = outputNodes;
    weights = [];
    biases = [];
    activation_functions = [];

    var nodes = [this.inputNodes, ...hiddenNodes, outputNodes];

    for (var i = 0; i < nodes.length - 1; i++) {
      var weight = Matrix(nodes[i + 1].nodes, nodes[i].nodes);
      weight.randomize();
      weights.add(weight);
      var bias = Matrix(nodes[i + 1].nodes, 1);
      bias.randomize();
      biases.add(bias);
    }

    // only hidden and output layers can have activation functions
    var activation_layers = [...hiddenNodes, outputNodes];
    for (var i = 0; i < activation_layers.length; i++) {
      activation_functions.add(activation_layers[i].activation_fn);
    }

    setLearningRate();
  }

  List<double> predict(var input_array) {
    var output = Matrix.fromArray(input_array);
    for (var i = 0; i < weights.length; i++) {
      output = Matrix.dotProduct(weights[i], output);
      output.add(biases[i]);
      output.map(activation_functions[i].func);
    }

    var output_arr = Matrix.toArray(output);
    return output_arr;
  }

  void train(var inputs_array, var targets_array) {
    //  Steps:-
    //  1. Get input & target array convert to matrix
    //  2. Forward propagate
    //    2.1 Dot product previous matrix with hidden matrix
    //    2.2 Add hidden bias
    //    2.3 Apply activation function
    //    2.4 Repeat 2.1 -> 2.3 until you reach output matrix
    //  3. Calculate the output error
    //  4. Back propagate
    //    4.1 Apply derivate activation function to the last layer.
    //    4.2 Hadamard product by the errors
    //    4.3 Scalar product by the learning rate
    //    4.4 Transpose the last before layer
    //    4.5 Dot product of output of 4.3 with 4.4
    //    4.6 Add the last weights matrix with output of 4.5
    //    4.7 Add the last bias matrix with the output of 4.3

    // Convert arr to matrix object
    var inputs = Matrix.fromArray(inputs_array);
    var targets = Matrix.fromArray(targets_array);

    var forward_weights = [];

    // forward prop
    for (var i = 0; i < weights.length; i++) {
      if (i == 0) {
        var hidden = Matrix.dotProduct(weights[i], inputs);
        hidden.add(biases[i]);
        hidden.map(activation_functions[i].func);
        forward_weights.add(hidden);
      } else {
        var hidden = Matrix.dotProduct(weights[i], forward_weights[i - 1]);
        hidden.add(biases[i]);
        hidden.map(activation_functions[i].func);
        forward_weights.add(hidden);
      }
    }

    // Calculate the output error
    var errors = List.generate(weights.length, (_) => Matrix(1, 1).randomize());
    errors[errors.length - 1] =
        targets.subtract(forward_weights[forward_weights.length - 1]);

    var gradients = [];
    // backward prop
    for (var i = weights.length - 1; i >= 0; i--) {
      if (i != 0) {
        // Calculate the hidden->output gradients
        var gradient = Matrix.immutableMap(
            forward_weights[i], activation_functions[i].dfunc);
        gradient.multiply(errors[i], hadamard: true);
        gradient.multiply(learning_rate);
        gradients.add(gradient);

        // Calculate the hidden->output deltas
        var hidden_T = Matrix.transpose(forward_weights[i - 1]);
        var weight_delta = Matrix.dotProduct(gradient, hidden_T);
        // Update the hidden->output weights by deltas
        weights[i].add(weight_delta);
        biases[i].add(gradient);

        // Calculate the hidden layer error
        var who_t = Matrix.transpose(weights[i]);
        errors[i - 1] = Matrix.dotProduct(who_t, errors[i]);
      } else {
        // Calculate the hidden->output gradients
        var gradient = Matrix.immutableMap(
            forward_weights[i], activation_functions[i].dfunc);
        gradient.multiply(errors[i], hadamard: true);
        gradient.multiply(learning_rate);
        gradients.add(gradient);

        // Calculate the hidden->output deltas
        var hidden_T = Matrix.transpose(inputs);
        var weight_delta = Matrix.dotProduct(gradient, hidden_T);
        // Update the hidden->output weights by deltas
        weights[i].add(weight_delta);
        biases[i].add(gradient);
      }
    }
  }

  void setLearningRate({learning_rate = 0.1}) {
    this.learning_rate = learning_rate;
  }

  void setActivationFunction(List<Activation> activation_functions) {
    this.activation_functions = activation_functions;
  }

  NeuralNetwork clone() {
    var clone = NeuralNetwork(inputNodes.nodes, hiddenNodes, outputNodes);
    for (var i = 0; i < weights.length; i++) {
      clone.weights[i] = Matrix.clone(weights[i]);
    }
    for (var i = 0; i < biases.length; i++) {
      clone.biases[i] = Matrix.clone(biases[i]);
    }
    clone.setLearningRate(learning_rate: learning_rate);
    clone.setActivationFunction(activation_functions);
    return clone;
  }

  // serialize
  Map<String, dynamic> toJson() {
    return {
      'inputNodes': Layer.serialize(inputNodes),
      'hiddenNodes': List<Layer>.from(hiddenNodes)
          .map((hiddenNode) => Layer.serialize(hiddenNode))
          .toList(),
      'outputNodes': Layer.serialize(outputNodes),
      'weights': List<Matrix>.from(weights)
          .map((weight) => Matrix.serialize(weight))
          .toList(),
      'biases': List<Matrix>.from(biases)
          .map((bias) => Matrix.serialize(bias))
          .toList(),
      'learning_rate': learning_rate,
      'activation_functions': List<Activation>.from(activation_functions)
          .map((act_fn) => act_fn.name)
          .toList(),
    };
  }

  // deserialize
  NeuralNetwork.fromJson(Map<String, dynamic> json) {
    inputNodes = Layer.deserialize(json['inputNodes']);
    hiddenNodes = List<String>.from(json['hiddenNodes'])
        .map((layer) => Layer.deserialize(layer))
        .toList();
    outputNodes = Layer.deserialize(json['outputNodes']);
    weights = List<String>.from(json['weights'])
        .map((weight) => Matrix.deserialize(weight))
        .toList();
    biases = List<String>.from(json['biases'])
        .map((bias) => Matrix.deserialize(bias))
        .toList();
    setLearningRate(learning_rate: json['learning_rate']);
    setActivationFunction(List<String>.from(json['activation_functions'])
        .map((act_fn) => ActivationFunctions[act_fn])
        .toList());
  }

  static String serialize(NeuralNetwork nn) {
    return jsonEncode(nn);
  }

  static NeuralNetwork deserialize(String jsonString) {
    Map nnMap = jsonDecode(jsonString);
    var result = NeuralNetwork.fromJson(nnMap);
    return result;
  }
}
