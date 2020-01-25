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

    this.learning_rate = 0.05;
  }

  List<double> feedForward(var inputs) {
    if(!(inputs is Matrix)) {
      inputs = Matrix.fromArray(inputs);
    }
    //for(int i = 0; i < inputs.rows; i++) {
    //}
    var hidden = Matrix.dotProduct(weights_ih, inputs);
    hidden.add(bias_h);
    hidden.map(Activation.Sigmoid);

    var output = Matrix.dotProduct(weights_ho, hidden);
    output.add(bias_o);
    output.map(Activation.Sigmoid);

    var output_arr = Matrix.toArray(output);

    return output_arr;
  }

  Matrix train(var inputs, var targets) {
    if(!(inputs is Matrix)) {
      inputs = Matrix.fromArray(inputs);
    }
    var hidden = Matrix.dotProduct(weights_ih, inputs);
    hidden.add(bias_h);
    hidden.map(Activation.Sigmoid);

    var outputs = Matrix.dotProduct(weights_ho, hidden);
    outputs.add(bias_o);
    outputs.map(Activation.Sigmoid);

    // Convert arr to matrix object
    targets = Matrix.fromArray(targets);

    // Calculate the output error
    var output_errors = targets.subtract(outputs);

    // Calculate the hidden->output gradients
    var gradients = Matrix.immutableMap(outputs, Activation.DSigmoid);
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
    print(who_t);
    var hidden_errors = Matrix.dotProduct(who_t, output_errors);

    // Calculate the input->hidden gradients
    var hidden_gradient = Matrix.immutableMap(hidden, Activation.DSigmoid);
    hidden_gradient.multiply(hidden_errors, hadamard: true);
    hidden_gradient.multiply(learning_rate);

    // Calculate the input->hidden deltas
    var inputs_T = Matrix.transpose(inputs);
    var weights_ih_deltas = Matrix.dotProduct(hidden_gradient, inputs_T);

    // Update the input->hidden weights by the deltas
    weights_ih.add(weights_ih_deltas);
    bias_h.add(hidden_gradient);
  }
}