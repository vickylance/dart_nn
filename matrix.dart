import 'dart:math';

// Immutable Matrix Class
class Matrix {
  int rows;
  int cols;
  List<List<double>> matrix;
  Random rnd = new Random();

  Matrix(int rows, int cols, {int rngSeed = null}) {
    this.rows = rows;
    this.cols = cols;
    if(rngSeed != null) {
      rnd = new Random(rngSeed);
    }
    this.matrix = new List.generate(this.rows, (_) => new List(this.cols));

    for(int i = 0; i < rows; i++) {
      for(int j = 0; j < cols; j++) {
        matrix[i][j] = 0.0;
      }
    }
  }

  Matrix map(Function fn) {
    for(int i = 0; i < rows; i++) {
      for(int j = 0; j < cols; j++) {
        var val = matrix[i][j];
        matrix[i][j] = fn(val, i, j);
      }
    }
    return this;
  }

  Matrix clone() {
    Matrix result = new Matrix(rows, cols);
    for(int i = 0; i < rows; i++) {
      for(int j = 0; j < cols; j++) {
        result.matrix[i][j] = matrix[i][j];
      }
    }
    return result;
  }

  Matrix transpose() {
    Matrix result = new Matrix(cols, rows);
    for(int i = 0; i < rows; i++) {
      for(int j = 0; j < cols; j++) {
        result.matrix[j][i] = matrix[i][j];
      }
    }
    return result;
  }

  Matrix multiply(var val, {bool hadamard = false}) {
    if(val is Matrix && !hadamard) {
      // Dot product
      if(cols != val.rows) {
        throw ("The colums of A must match rows of B $cols, ${val.rows}");
      }
      Matrix result = new Matrix(rows, val.cols);
      var a = matrix;
      var b = val.matrix;
      for(int i = 0; i < result.rows; i++) {
        for(int j = 0; j < result.cols; j++) {
          var sum = 0.0;
          for(int k = 0; k < cols; k++) {
            sum += a[i][k] * b[k][j];
          }
          result.matrix[i][j] = sum;
        }
      }
      return result;
    }

    Matrix result = clone();
    for(int i = 0; i < rows; i++) {
      for(int j = 0; j < cols; j++) {
        if(val is Matrix && hadamard) {
          // Hadamard product
          result.matrix[i][j] *= val.matrix[i][j];
        } else {
          // Scalar product
          result.matrix[i][j] *= val;
        }
      }
    }
    return result;
  }

  Matrix add(var val) {
    Matrix result = clone();
    for(int i = 0; i < rows; i++) {
      for(int j = 0; j < cols; j++) {
        if(val is Matrix) {
          result.matrix[i][j] += val.matrix[i][j];
        } else {
          result.matrix[i][j] += val;
        }
      }
    }
    return result;
  }

  Matrix ones() {
    Matrix result = clone();
    for(int i = 0; i < rows; i++) {
      for(int j = 0; j < cols; j++) {
        result.matrix[i][j] = 1.0;
      }
    }
    return result;
  }

  Matrix zeros() {
    Matrix result = clone();
    for(int i = 0; i < rows; i++) {
      for(int j = 0; j < cols; j++) {
        result.matrix[i][j] = 0.0;
      }
    }
    return result;
  }

  Matrix randomize() {
    Matrix result = clone();
    for(int i = 0; i < rows; i++) {
      for(int j = 0; j < cols; j++) {
        result.matrix[i][j] = rnd.nextDouble();
      }
    }
    return result;
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