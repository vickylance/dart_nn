import 'dart:convert';
import 'package:dart_nn/matrix.dart';
import 'package:dart_nn/activation.dart';

class NeuralNetwork {
  int inputNodes;
  int hiddenNodes;
  int outputNodes;
  Matrix weights_ih;
  Matrix weights_ho;
  Matrix bias_h;
  Matrix bias_o;
  double learning_rate;
  List<Activation> activation_functions;

  NeuralNetwork(int inputNodes, int hiddenNodes, int outputNodes) {
    this.inputNodes = inputNodes;
    this.hiddenNodes = hiddenNodes;
    this.outputNodes = outputNodes;

    weights_ih = Matrix(this.hiddenNodes, this.inputNodes);
    weights_ho = Matrix(this.outputNodes, this.hiddenNodes);
    weights_ih.randomize();
    weights_ho.randomize();

    bias_h = Matrix(this.hiddenNodes, 1);
    bias_o = Matrix(this.outputNodes, 1);
    bias_h.randomize();
    bias_o.randomize();

    setLearningRate();
    setActivationFunction([Relu, Sigmoid]);
  }

  List<double> predict(var input_array) {
    var inputs = Matrix.fromArray(input_array);
    var hidden = Matrix.dotProduct(weights_ih, inputs);
    hidden.add(bias_h);
    hidden.map(activation_functions[0].func);

    var output = Matrix.dotProduct(weights_ho, hidden);
    output.add(bias_o);
    output.map(activation_functions[1].func);

    var output_arr = Matrix.toArray(output);

    return output_arr;
  }

  void train(var inputs_array, var targets_array) {
    //  Steps:-
    //  1. Get input array convert to matrix
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

    var hidden = Matrix.dotProduct(weights_ih, inputs);
    hidden.add(bias_h);
    hidden.map(activation_functions[0].func);

    var outputs = Matrix.dotProduct(weights_ho, hidden);
    outputs.add(bias_o);
    outputs.map(activation_functions[1].func);

    // Calculate the output error
    var output_errors = targets.subtract(outputs);

    // Calculate the hidden->output gradients
    var gradients = Matrix.immutableMap(outputs, activation_functions[1].dfunc);
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
    var hidden_gradient =
        Matrix.immutableMap(hidden, activation_functions[0].dfunc);
    hidden_gradient.multiply(hidden_errors, hadamard: true);
    hidden_gradient.multiply(learning_rate);

    // Calculate the input->hidden deltas
    var inputs_T = Matrix.transpose(inputs);
    var weights_ih_deltas = Matrix.dotProduct(hidden_gradient, inputs_T);

    // Update the input->hidden weights by the deltas
    weights_ih.add(weights_ih_deltas);
    bias_h.add(hidden_gradient);
  }

  void setLearningRate({learning_rate = 0.1}) {
    this.learning_rate = learning_rate;
  }

  void setActivationFunction(activation_functions) {
    this.activation_functions = activation_functions;
  }

  NeuralNetwork clone() {
    var clone = NeuralNetwork(inputNodes, hiddenNodes, outputNodes);
    clone.weights_ih = Matrix.clone(weights_ih);
    clone.weights_ho = Matrix.clone(weights_ho);
    clone.bias_h = Matrix.clone(bias_h);
    clone.bias_o = Matrix.clone(bias_o);
    clone.setLearningRate(learning_rate: learning_rate);
    clone.setActivationFunction(activation_functions);
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
      'activation_functions': [
        activation_functions[0].name,
        activation_functions[1].name
      ],
    };
  }

  // deserialize
  NeuralNetwork.fromJson(Map<String, dynamic> json) {
    inputNodes = json['inputNodes'];
    hiddenNodes = json['hiddenNodes'];
    outputNodes = json['outputNodes'];
    weights_ih = Matrix.deserialize(json['weights_ih']);
    weights_ho = Matrix.deserialize(json['weights_ho']);
    bias_h = Matrix.deserialize(json['bias_h']);
    bias_o = Matrix.deserialize(json['bias_o']);
    setLearningRate(learning_rate: json['learning_rate']);
    setActivationFunction([
      ActivationFunctions[json['activation_functions'][0]],
      ActivationFunctions[json['activation_functions'][1]]
    ]);
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
