import 'dart:math';
import './matrix.dart';
import './activation.dart';
import './neuralNetwork.dart';

void main() {
  var brain = NeuralNetwork(2,3,1);
  var train_data = [{
    "inputs": [1.0, 1.0],
    "outputs": [0.0],
  }, {
    "inputs": [0.0, 0.0],
    "outputs": [0.0],
  }, {
    "inputs": [0.0, 1.0],
    "outputs": [1.0],
  }, {
    "inputs": [1.0, 0.0],
    "outputs": [1.0],
  }];
  int epoch = 500000;

  Random rnd = Random();
  for(int i = 0; i < epoch; i++) {
    int d = rnd.nextInt(4);
    brain.train(train_data[d]["inputs"], train_data[d]["outputs"]);
  }
  for(int i = 0; i < train_data.length; i++) {
    print("Test: In: ${train_data[i]['inputs']} ${brain.predict(train_data[i]['inputs'])}");
  }
}
