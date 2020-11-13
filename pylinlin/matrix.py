"""Core matrix functionality.

Contains functions for initializing a matrix, performing matrix transpose and
matrix multiplication, and accessing rows, columns or elements in a matrix.
This matrix interface is used by the other algorithms.

    Typical usage example:

    from pylinlin.matrix import Matrix

    matrix = Matrix.from_cols([[1, 2, 3], [4, 5, 6]])
    matrix.size()  # (3, 2)
    transpose = matrix.transpose()
    product = matrix.multiply(transpose)
"""

from __future__ import annotations
from typing import List
import math


class Matrix:
    """The core matrix class. This is the matrix interface used by all algorithms."""

    @staticmethod
    def from_rows(rows: List[List[float]]):
        """Initializes a matrix from a list of rows.

        Parameters
        ----------
        rows : List[List[float]]
            Rows of the matrix.
        """
        num_rows = len(rows)
        num_cols = len(rows[0])
        for row in rows:
            if len(row) != num_cols:
                raise ValueError("Rows must have equal length")
        cols = [[0] * num_rows for _ in range(num_cols)]
        for row_index, row in enumerate(rows):
            for col_index, elem in enumerate(row):
                cols[col_index][row_index] = elem
        return Matrix(cols)

    @staticmethod
    def from_cols(cols: List[List[float]]):
        """Initializes a matrix from a list of columns.

        Parameters
        ----------
        cols : List[List[float]]
            Columns of the matrix.
        """
        return Matrix(cols)

    @staticmethod
    def zeroes(num_rows: int, num_cols: int) -> Matrix:
        """Initializes a matrix of zeroes.

        Parameters
        ----------
        num_rows : int
            Number of rows in the matrix.

        num_cols : int
            Number of columns in the matrix.
        """
        return Matrix.from_cols([[0] * num_rows for _ in range(num_cols)])

    @staticmethod
    def identity(dims: int) -> Matrix:
        """Initializes a square identity matrix.

        Parameters
        ----------
        dims : int
            The number of rows and columns of the matrix.
        """
        cols = [[0] * dims for _ in range(dims)]
        for i in range(dims):
            cols[i][i] = 1
        return Matrix.from_cols(cols)

    @staticmethod
    def column_scale(col: List[float], scale: float) -> List[float]:
        """Helper function to scale a column vector by a scalar.

        Parameters
        ----------
        col : List[float]
            Column vector to be scaled.

        scale : float
            Scale factor.
        """
        return [elem * scale for elem in col]

    @staticmethod
    def column_add(col1: List[float], col2: List[float]) -> List[float]:
        """Helper function to add two column vectors.

        Parameters
        ----------
        col1 : List[float]
            First column vector to be added.

        col2 : List[float]
            Second column vector to be added.

        Returns
        -------
        List[float]
            The resulting vector.

        Raises
        ------
        ValueError
            If the vectors have different dimensions.
        """
        if len(col1) != len(col2):
            raise ValueError("Columns must have same dimension to be added")
        return [a + b for a, b in zip(col1, col2)]

    def __init__(self: Matrix, columns: List[List[float]]):
        """Initializes a matrix from a list of columns.

        Parameters
        ----------
        columns : List[List[float]]
            Columns of the matrix.

        Raises
        ------
        ValueError
            If not all the columns are of same dimension.
        """
        self._num_cols = len(columns)
        self._num_rows = len(columns[0])
        for col in columns:
            if len(col) != self.num_rows():
                raise ValueError("Columns must have equal length")
        self.columns = columns

    def print(self: Matrix):
        """Outputs a matrix in a readable format for debugging purposes."""
        print("Size: %d by %d" % (self.num_rows(), self.num_cols()))
        for i in range(self.num_rows()):
            row = self.get_row(i)
            for r in row:
                print("%8.3f " % (r), end='')
            print()

    def copy(self: Matrix) -> Matrix:
        """Makes a copy of the matrix. Mutating the copy should not affect the original.

        Returns
        -------
        Matrix
            The deep copy of the matrix.
        """
        cols = [col[:] for col in self.columns]
        return Matrix.from_cols(cols)

    def size(self: Matrix) -> (int, int):
        """Get the dimensions of the matrix.

        Returns
        -------
        (int, int)
            (Number of rows, number of columns) in a tuple
        """
        return (self._num_rows, self._num_cols)

    def num_rows(self: Matrix) -> int:
        """Get the number of rows in the matrix.

        Returns
        -------
        int
            Number of rows.
        """
        return self._num_rows

    def num_cols(self: Matrix) -> int:
        """Get the number of columns in the matrix.

        Returns
        -------
        int
            Number of columns.
        """
        return self._num_cols

    def get_row(self: Matrix, index: int) -> List[float]:
        """Extracts a row of the matrix as a list

        Parameters
        ----------
        index : int 
            Index of the row to be extracted.

        Returns
        -------
        List[float]
            The row of the matrix.

        Raises
        ------
        ValueError
            If the index given is outside the bounds of the matrix.
        """
        if index < 0 or index >= self.num_rows():
            raise ValueError("Index out of bounds")
        return [col[index] for col in self.columns]

    def get_col(self: Matrix, index: int) -> List[float]:
        """Extracts a column of the matrix as a list

        Parameters
        ----------
        index : int 
            Index of the column to be extracted.

        Returns
        -------
        List[float]
            The column of the matrix.

        Raises
        ------
        ValueError
            If the index given is outside the bounds of the matrix.
        """
        if index < 0 or index >= self.num_cols():
            raise ValueError("Index out of bounds")
        return self.columns[index]

    def get(self: Matrix, row: int, col: int) -> float:
        """Extracts an element of the matrix

        Parameters
        ----------
        row : int 
            Index of the row of the element to be extracted.
        col : int 
            Index of the column of the element to be extracted.

        Returns
        -------
        float
            The element.

        Raises
        ------
        ValueError
            If either of the indices given are outside the bounds of the matrix.
        """
        if row < 0 or row >= self.num_rows():
            raise ValueError("Index out of bounds")
        if col < 0 or col >= self.num_cols():
            raise ValueError("Index out of bounds")
        return self.columns[col][row]

    def all_cols(self: Matrix) -> List[List[float]]:
        """Get all the columns of the matrix

        Returns
        -------
        List[List[float]]
            A list of the columns of the matrix, each column being a list of numbers
        """
        return self.columns

    def transpose(self: Matrix) -> Matrix:
        """Computes the transpose of the matrix.

        Returns
        -------
        Matrix
            The transpose of the matrix.
        """
        return Matrix.from_rows(self.columns)

    def multiply_column(self: Matrix, vector: List[float]) -> List[float]:
        """Helper function for computing a matrix-vector product

        Parameters
        ----------
        vector : List[float]
            The vector to be multiplied.

        Returns
        -------
        List[float]
            The vector obtained by taking linear combinations of the matrix according to the given coefficients.

        Raises
        ------
        ValueError
            If the number of columns of the matrix does not match the dimension of the vector
        """
        if self.num_cols() != len(vector):
            raise ValueError(
                f"Incompatible sizes for multiplication: {self.size()} and {len(vector)}")
        result = [0] * self.num_rows()
        for col, multiplier in zip(self.all_cols(), vector):
            result = Matrix.column_add(
                result, Matrix.column_scale(col, multiplier))
        return result

    def multiply(self: Matrix, other: Matrix) -> Matrix:
        """Performs matrix multiplication.

        Parameters
        ----------
        other : Matrix
            The right hand side of the product.

        Returns
        -------
        Matrix
            The matrix obtained by multiplying the matrices.

        Raises
        ------
        ValueError
            If the number of columns of this matrix does not match the number of rows of the right hand side.
        """
        if self.num_cols() != other.num_rows():
            raise ValueError(
                f"Incompatible matrix sizes for multiplication: {self.size()} and {other.size()}")
        result = [self.multiply_column(col) for col in other.all_cols()]
        return Matrix.from_cols(result)

    def frobenius_norm(self: Matrix) -> float:
        """Computes the frobenius norm of the matrix.

        The frobenius norm is computed by taking square root of the sums the squares of each entry of the matrix.
        This can be used to calculate the 2-norm of a column vector.

        Returns
        -------
        float
            The frobenius norm.
        """
        sum_sq = 0
        for col in self.columns:
            for elem in col:
                sum_sq += elem * elem
        return math.sqrt(sum_sq)
