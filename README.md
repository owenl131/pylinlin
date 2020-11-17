# pylinlin [![PyPI version](https://badge.fury.io/py/pylinlin.svg)](https://badge.fury.io/py/pylinlin) [![codecov](https://codecov.io/gh/owenl131/pylinlin/branch/main/graph/badge.svg)](https://codecov.io/gh/owenl131/pylinlin)

Nice linear algebra library in python

Based on https://www.cs.utexas.edu/users/flame/laff/alaff/

In theory SVD is working well. In practice it's too slow for anything greater than a 4x4 matrix. Upcoming changes will use the python standard array type to represent the matrix internally instead of a list of lists.

## Installation

```
$ pip install pylinlin
```

## Examples

```python
from pylinlin.matrix import Matrix
from pylinlin.lu_factorization import compute_lu_factorization
from pylinlin.qr_factorization import compute_qr_factorization
from pylinlin.svd import compute_svd

# Create matrix
matrix = Matrix.from_cols([[1, 2, 3], [4, 5, 6], [7, 8, 10]])  # preferred way to initialize a matrix
matrix2 = Matrix.from_rows([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

matrix.print()  # prints the matrix in a readable format

column_as_list = [1, 2, 3, 4, 5]
column_vector = Matrix.from_cols([column_as_list])  # column vectors can be represented as matrices
magnitude_sq = column_vector.transpose().multiply(column_vector).get(0, 0)  # 55

print(matrix.size())      # Get dimensions of matrix (rows, columns)
print(matrix.get_col(0))  # Get first column from matrix
print(matrix.get_row(1))  # Get second row from matrix

print(matrix.all_cols())  # List of matrix columns

matL, matU = compute_lu_factorization(matrix)
matQ, matR = compute_qr_factorization(matrix)
print(matQ.all_cols())
print(matQ.transpose().multiply(matQ).all_cols())  # approximately an identity matrix

product = matrix.multiply(matrix)  # matrix multiplication

matU, matS, matV = compute_svd(matrix)
```

## Goals

- Test-driven development
- Profiling of performance
- Profiling of numerical stability
- Lightweight, easy to port over to other languages

## TODOs

### Algorithms

- [x] LU factorization
- [ ] LU factorization with partial pivoting
- [x] QR factorization with householder matrices
- [ ] QR factorization with pivoting
- [ ] Gram Schmidt and Modified Gram Schmidt (help wanted!)
- [ ] Spectral decomposition
- [x] SVD
- [ ] Conjugate gradient descent
- [ ] Condition number of a matrix
- [ ] Jacobi SVD
- [ ] Power iteration
- [ ] Matrix Pseudoinverse

### Profiling

- [ ] Profile time taken varying size of matrices
- [ ] Profile time taken to solve linear system comparing different algorithms
- [ ] Graph error distribution on random matrices

### Others

- [ ] Make curve fitting demonstration
- [ ] Make IK demonstration
