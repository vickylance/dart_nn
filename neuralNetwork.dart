import 'dart:convert';

import './matrix.dart';
import './activation.dart';

class NeuralNetwork {
  int inputNodes;
  int hiddenNodes;
  int outputNodes;
  Matrix weights_ih;
  Matrix weights_ho;
  Matrix bias_h;
  Matrix bias_o;
  double learning_rate;
  Activation activation_function;

  NeuralNetwork(int inputNodes, int hiddenNodes, int outputNodes) {
    this.inputNodes = inputNodes;
    this.hiddenNodes = hiddenNodes;
    this.outputNodes = outputNodes;

    this.weights_ih = Matrix(this.hiddenNodes, this.inputNodes);
    this.weights_ho = Matrix(this.outputNodes, this.hiddenNodes);
    this.weights_ih.randomize();
    this.weights_ho.randomize();

    this.bias_h = Matrix(this.hiddenNodes, 1);
    this.bias_o = Matrix(this.outputNodes, 1);
    this.bias_h.randomize();
    this.bias_o.randomize();

    this.setLearningRate();
    this.setActivationFunction(Sigmoid);
  }

  List<double> predict(var input_array) {
    var inputs = Matrix.fromArray(input_array);
    var hidden = Matrix.dotProduct(weights_ih, inputs);
    hidden.add(bias_h);
    hidden.map(activation_function.func);

    var output = Matrix.dotProduct(weights_ho, hidden);
    output.add(bias_o);
    output.map(activation_function.func);

    var output_arr = Matrix.toArray(output);

    return output_arr;
  }

  train(var inputs_array, var targets) {
    var inputs = Matrix.fromArray(inputs_array);

    var hidden = Matrix.dotProduct(weights_ih, inputs);
    hidden.add(bias_h);
    hidden.map(activation_function.func);

    var outputs = Matrix.dotProduct(weights_ho, hidden);
    outputs.add(bias_o);
    outputs.map(activation_function.func);

    // Convert arr to matrix object
    targets = Matrix.fromArray(targets);

    // Calculate the output error
    var output_errors = targets.subtract(outputs);

    // Calculate the hidden->output gradients
    var gradients = Matrix.immutableMap(outputs, activation_function.dfunc);
    gradients.multiply(output_errors, hadamard: true);
    gradients.multiply(learning_rate);

    // Calculate the hidden->output deltas
    var hidden_T = Matrix.transpose(hidden);
    var weights_ho_deltas = Matrix.dotProduct(gradients, hidden_T);

    // Update the hidden->output weights by deltas
    weights_ho.add(weights_ho_deltas);
    bias_o.add(gradients);

    // Calculate the hidden layer error
    var who_t = Matrix.transpose(weights_ho);
    var hidden_errors = Matrix.dotProduct(who_t, output_errors);

    // Calculate the input->hidden gradients
    var hidden_gradient = Matrix.immutableMap(hidden, activation_function.dfunc);
    hidden_gradient.multiply(hidden_errors, hadamard: true);
    hidden_gradient.multiply(learning_rate);

    // Calculate the input->hidden deltas
    var inputs_T = Matrix.transpose(inputs);
    var weights_ih_deltas = Matrix.dotProduct(hidden_gradient, inputs_T);

    // Update the input->hidden weights by the deltas
    weights_ih.add(weights_ih_deltas);
    bias_h.add(hidden_gradient);
  }

  setLearningRate({learning_rate = 0.1}) {
    this.learning_rate = learning_rate;
  }

  setActivationFunction(activation_function) {
    this.activation_function = activation_function;
  }

  NeuralNetwork clone() {
    var clone = new NeuralNetwork(inputNodes, hiddenNodes, outputNodes);
    clone.weights_ih = Matrix.clone(weights_ih);
    clone.weights_ho = Matrix.clone(weights_ho);
    clone.bias_h = Matrix.clone(bias_h);
    clone.bias_o = Matrix.clone(bias_o);
    clone.setLearningRate(learning_rate: learning_rate);
    clone.setActivationFunction(activation_function);
    return clone;
  }

  // serialize
  Map<String, dynamic> toJson() {
    return {
      'inputNodes': inputNodes,
      'hiddenNodes': hiddenNodes,
      'outputNodes': outputNodes,
      'weights_ih': Matrix.serialize(weights_ih),
      'weights_ho': Matrix.serialize(weights_ho),
      'bias_h': Matrix.serialize(bias_h),
      'bias_o': Matrix.serialize(bias_o),
      'learning_rate': learning_rate,
      'activation_function': activation_function.name,
    };
  }

  // deserialize
  NeuralNetwork.fromJson(Map<String, dynamic> json) {
    this.inputNodes = json['inputNodes'];
    this.hiddenNodes = json['hiddenNodes'];
    this.outputNodes = json['outputNodes'];
    this.weights_ih = Matrix.deserialize(json['weights_ih']);
    this.weights_ho = Matrix.deserialize(json['weights_ho']);
    this.bias_h = Matrix.deserialize(json['bias_h']);
    this.bias_o = Matrix.deserialize(json['bias_o']);
    this.setLearningRate(learning_rate: json['learning_rate']);
    this.setActivationFunction(ActivationFunctions[json['activation_function']]);
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