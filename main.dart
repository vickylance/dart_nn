import 'dart:math';
import './matrix.dart';

Random rnd = new Random();

double map(double x, double in_min, double in_max, double out_min, double out_max) {
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

class NeuralNetwork {
  int inputNodes;
  int hiddenNodes;
  int outputNodes;

  NeuralNetwork(int inputNodes, int hiddenNodes, int outputNodes) {
    this.inputNodes = inputNodes;
    this.hiddenNodes = hiddenNodes;
    this.outputNodes = outputNodes;
  }
}

class Perceptron {
  final weights = new List<double>(2);

  Perceptron() {
    for(int i = 0; i < weights.length; i++) {
      weights[i] = map(rnd.nextDouble(), 0, 1, -1, 1);
    }
    print(weights);
  }

  double weightedSum(List<double> inputs) {
    double sum = 0;
    for(int i = 0; i < inputs.length; i++) {
      sum += inputs[i]*weights[i];
    }
    return sum;
  }
}

void main() {
  //var p = new Perceptron();
  //print(p.weightedSum([0.8, 0.2]));
  // NeuralNetwork brain = NeuralNetwork(3,4,1);
  Matrix m1 = Matrix(3,4);
  Matrix m2 = Matrix(4,3);
  m1 = m1.randomize();
  m2 = m2.randomize();
  // print(m1);
  // print(m1.multiply(2));
  // print(m1);
  // print(m1.transpose());
  // print(m1);
  print(m1);
  print(m2);
  print(m1.multiply(m2));
  print(m1);
  print(m2);

  int x = 5;
  print(x);
  x = 10;
  print(x);
}
