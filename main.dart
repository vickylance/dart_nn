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
  int epoch = 50000;

  Random rnd = Random();
  for(int i = 0; i < epoch; i++) {
    int d = rnd.nextInt(4);
    brain.train(train_data[d]["inputs"], train_data[d]["outputs"]);
  }
  for(int i = 0; i < train_data.length; i++) {
    print("Test: In: ${train_data[i]['inputs']} ${brain.predict(train_data[i]['inputs'])}");
  }
  print('Clone brain method-----');
  var brain2 = brain.clone();
  for(int i = 0; i < train_data.length; i++) {
    print("Test: In: ${train_data[i]['inputs']} ${brain2.predict(train_data[i]['inputs'])}");
  }
  print('Loading brain from JSON method-----');
  var brain2serialized = NeuralNetwork.serialize(brain2);
  var brain3 = NeuralNetwork.deserialize(brain2serialized);
  for(int i = 0; i < train_data.length; i++) {
    print("Test: In: ${train_data[i]['inputs']} ${brain3.predict(train_data[i]['inputs'])}");
  }
}
