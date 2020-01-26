import 'dart:math';

import 'package:dart_nn/neuralNetwork.dart';

void main(List<String> arguments) {
  var brain = NeuralNetwork(2, 3, 1);

  // XOR training data
  var train_data = [
    {
      'inputs': [1.0, 1.0],
      'outputs': [0.0],
    },
    {
      'inputs': [0.0, 0.0],
      'outputs': [0.0],
    },
    {
      'inputs': [0.0, 1.0],
      'outputs': [1.0],
    },
    {
      'inputs': [1.0, 0.0],
      'outputs': [1.0],
    }
  ];
  var epoch = 50000;

  var rnd = Random();
  for (var i = 0; i < epoch; i++) {
    var d = rnd.nextInt(4);
    brain.train(train_data[d]['inputs'], train_data[d]['outputs']);
  }
  for (var i = 0; i < train_data.length; i++) {
    print(
        "Test: In: ${train_data[i]['inputs']} Out: ${brain.predict(train_data[i]['inputs'])}");
  }
  print('Clone brain method-----');
  var brain2 = brain.clone();
  for (var i = 0; i < train_data.length; i++) {
    print(
        "Test: In: ${train_data[i]['inputs']} ${brain2.predict(train_data[i]['inputs'])}");
  }
  print('Loading brain from JSON method-----');
  var brain2serialized = NeuralNetwork.serialize(brain2);
  var brain3 = NeuralNetwork.deserialize(brain2serialized);
  for (var i = 0; i < train_data.length; i++) {
    print(
        "Test: In: ${train_data[i]['inputs']} ${brain3.predict(train_data[i]['inputs'])}");
  }
}
