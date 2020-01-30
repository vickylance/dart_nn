import 'dart:math';
import 'dart:io';
import 'dart:convert';
import 'package:csv/csv.dart';
import 'package:dart_nn/nn.dart';

List<double> petalType(String petal) {
  switch (petal) {
    case 'Iris-setosa':
      return [1.0, 0, 0];
    case 'Iris-versicolor':
      return [0, 1.0, 0];
    case 'Iris-virginica':
      return [0, 0, 1.0];
    default:
      return [0, 0, 0];
  }
}

List<double> roundPrediction(List<double> pred) {
  return pred.map((x) => x.roundToDouble()).toList();
}

bool compareResult(List<double> pred, List<double> actual) {
  var result = true;
  for (var i = 0; i < pred.length; i++) {
    if (pred[i] != actual[i]) {
      result = false;
    }
  }
  return result;
}

double accuracy(List<bool> results, {bool debug = false}) {
  var wrongs = 0;
  for (var i = 0; i < results.length; i++) {
    if (results[i] == false) {
      wrongs += 1;
    }
  }
  if (debug) {
    print(
        'Accuracy: ${((results.length - wrongs) / results.length) * 100}%, Wrongs: $wrongs, Correct: ${results.length - wrongs}, All: ${results.length}');
  }
  return (results.length - wrongs) / results.length;
}

void main(List<String> arguments) async {
  var rnd = Random();
  var brain = NeuralNetwork(
      4,
      [Layer(8, 'Sigmoid'), Layer(5, 'Sigmoid'), Layer(8, 'Sigmoid')],
      Layer(3, 'Sigmoid'));
  brain.setLearningRate(learning_rate: 0.001);
  var epoch = 1000000;

  var path = './example/iris/iris.data';
  var inputs = [];
  var outputs = [];
  await File(path)
      .openRead()
      .transform(utf8.decoder)
      .transform(LineSplitter())
      .forEach((l) {
    inputs.add(List<double>.from(CsvToListConverter()
        .convert(l)[0]
        .getRange(0, 4)
        .map((x) => x.toDouble())));
    outputs.add(CsvToListConverter().convert(l)[0][4]);
  });

  var train_test_split = 0.8;
  var train_inputs = [];
  var train_outputs = [];

  var total = inputs.length;
  while ((total - inputs.length) / total < train_test_split) {
    var d = rnd.nextInt(inputs.length);
    train_inputs.add(inputs.removeAt(d));
    train_outputs.add(outputs.removeAt(d));
  }
  var test_inputs = inputs;
  var test_outputs = outputs;

  print(train_inputs.length);
  print(train_outputs.length);
  print(test_inputs.length);
  print(test_outputs.length);

  for (var i = 0; i < epoch; i++) {
    var d = rnd.nextInt(train_inputs.length);
    brain.train(train_inputs[d], petalType(train_outputs[d]));
  }

  var results = [];
  for (var i = 0; i < test_inputs.length; i++) {
    var pred = roundPrediction(brain.predict(test_inputs[i]));
    var actual = petalType(test_outputs[i]);
    var result = compareResult(pred, actual);
    results.add(result);
    print(
        'Test: In: ${test_inputs[i]} Pred: ${pred} Actual: ${actual} Result: ${result}');
  }
  accuracy(results.cast<bool>(), debug: true);
}
