# dart_nn

[![Run on Repl.it](https://repl.it/badge/github/vickylance/dart_nn)](https://repl.it/github/vickylance/dart_nn) ![Pub Version](https://img.shields.io/pub/v/dart_nn)

A Simple Neural Network library written in dart.

Inspired from Toy Neural Network library by Coding Train.

## Usage

Create a neural network brain with any number of input, hidden and output perceptrons.

Note: Currently only 1 hidden layer is supported.

```dart
var brain = NeuralNetwork(2, 3, 1);
```

Pass in training data with inputs and outputs to the `train` function.
And run the loop for any arbitrary number of `epochs`.

```dart
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
```

You can then test the NeuralNetwork by passing in the test data to the `predict` function.

```dart
for (var i = 0; i < train_data.length; i++) {
  print("Test: In: ${train_data[i]['inputs']} Out: ${brain.predict(train_data[i]['inputs'])}");
}
```

You can clone the brain using the `clone` method.

```dart
var brain2 = brain.clone();
```

You can serialize the brain to save in a file. And later retrieve the brain using the `deserialize` method.

```dart
var brain2serialized = NeuralNetwork.serialize(brain2);
// You can save the `brain2serialized` string to any file.
var brain3 = NeuralNetwork.deserialize(brain2serialized);
```

## License

[license](https://github.com/dart-lang/stagehand/blob/master/LICENSE).
