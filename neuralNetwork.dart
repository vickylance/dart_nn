import './matrix.dart';

class NeuralNetwork {
  int inputNodes;
  int hiddenNodes;
  int outputNodes;
  Matrix weights_ih;
  Matrix weights_ho;

  NeuralNetwork(int inputNodes, int hiddenNodes, int outputNodes) {
    this.inputNodes = inputNodes;
    this.hiddenNodes = hiddenNodes;
    this.outputNodes = outputNodes;

    this.weights_ih = Matrix(this.hiddenNodes, this.inputNodes);
    this.weights_ih = this.weights_ih.randomize();
    this.weights_ho = Matrix(this.outputNodes, this.hiddenNodes);
    this.weights_ho = this.weights_ho.randomize();

    this.bias_h = Matrix(this.hiddenNodes, 1);
    this.bias_h = this.bias_h.randomize();
    this.bias_o = Matrix(this.outputNodes, 1);
    this.bias_o = this.bias_o.randomize();
  }
}