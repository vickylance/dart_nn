import 'dart:math';
import './utils.dart' as utils;

// Matrix Class
class Matrix {
  int rows;
  int cols;
  List<List<double>> matrix;
  Random rnd = Random();

  Matrix(int rows, int cols, { int rngSeed = null }) {
    this.rows = rows;
    this.cols = cols;
    if(rngSeed != null) {
      rnd = Random(rngSeed);
    }
    this.matrix = List.generate(rows, (_) => List(cols));
    this.ones();
  }

  static Matrix fromArray(List<double> arr) {
    Matrix result = Matrix(arr.length, 1);
    for(int i = 0; i < arr.length; i++) {
      result.matrix[i][0] = arr[i];
    }
    return result;
  }

  static List<double> toArray(Matrix mat) {
    List<double> result = [];
    for(int i = 0; i < mat.rows; i++) {
      for(int j = 0; j < mat.cols; j++) {
        result.add(mat.matrix[i][j]);
      }
    }
    return result;
  }

  static Matrix dotProduct(Matrix a, Matrix b) {
    // Dot product
    if(a.cols != b.rows) {
      throw ("The columns of A = ${a.cols} must match rows of B = ${b.rows}");
    }
    Matrix result = Matrix(a.rows, b.cols);
    for(int i = 0; i < result.rows; i++) {
      for(int j = 0; j < result.cols; j++) {
        var sum = 0.0;
        for(int k = 0; k < a.cols; k++) {
          sum += a.matrix[i][k] * b.matrix[k][j];
        }
        result.matrix[i][j] = sum;
      }
    }
    return result;
  }

  static Matrix clone(Matrix x) {
    Matrix result = Matrix(x.rows, x.cols);
    for(int i = 0; i < x.rows; i++) {
      for(int j = 0; j < x.cols; j++) {
        result.matrix[i][j] = x.matrix[i][j];
      }
    }
    return result;
  }

  static Matrix transpose(Matrix x) {
    Matrix result = Matrix(x.cols, x.rows);
    for(int i = 0; i < x.rows; i++) {
      for(int j = 0; j < x.cols; j++) {
        result.matrix[j][i] = x.matrix[i][j];
      }
    }
    return result;
  }

  static Matrix immutableMap(Matrix m, Function fn) {
    Matrix result = Matrix(m.rows, m.cols);
    for(int i = 0; i < m.rows; i++) {
      for(int j = 0; j < m.cols; j++) {
        result.matrix[i][j] = fn(m.matrix[i][j]);
      }
    }
    return result;
  }

  Matrix map(Function fn) {
    for(int i = 0; i < rows; i++) {
      for(int j = 0; j < cols; j++) {
        var val = matrix[i][j];
        matrix[i][j] = fn(val);
      }
    }
    return this;
  }

  Matrix multiply(var val, { bool hadamard = false }) {
    for(int i = 0; i < rows; i++) {
      for(int j = 0; j < cols; j++) {
        if(val is Matrix && hadamard) {
          // Hadamard product
          matrix[i][j] *= val.matrix[i][j];
        } else {
          // Scalar product
          matrix[i][j] *= val;
        }
      }
    }
    return this;
  }

  Matrix add(var val) {
    for(int i = 0; i < rows; i++) {
      for(int j = 0; j < cols; j++) {
        if(val is Matrix) {
          matrix[i][j] += val.matrix[i][j];
        } else {
          matrix[i][j] += val;
        }
      }
    }
    return this;
  }

  Matrix subtract(var val) {
    for(int i = 0; i < rows; i++) {
      for(int j = 0; j < cols; j++) {
        if(val is Matrix) {
          matrix[i][j] -= val.matrix[i][j];
        } else {
          matrix[i][j] -= val;
        }
      }
    }
    return this;
  }

  Matrix ones() {
    for(int i = 0; i < rows; i++) {
      for(int j = 0; j < cols; j++) {
        matrix[i][j] = 1.0;
      }
    }
    return this;
  }

  Matrix zeros() {
    for(int i = 0; i < rows; i++) {
      for(int j = 0; j < cols; j++) {
        matrix[i][j] = 0.0;
      }
    }
    return this;
  }

  Matrix randomize() {
    for(int i = 0; i < rows; i++) {
      for(int j = 0; j < cols; j++) {
        matrix[i][j] = utils.map(rnd.nextDouble(), 0, 1, -1, 1);
      }
    }
    return this;
  }

  String toString() {
    String m = "";
    for(int i = 0; i < rows; i++) {
      for(int j = 0; j < cols; j++) {
        m += matrix[i][j].toStringAsFixed(3) + ((j + 1 == cols) ? "" : ", ");
      }
      m += '\n';
    }
    return m;
  }
}