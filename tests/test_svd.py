import pytest
from pylinlin.matrix import Matrix
from pylinlin.matrix_view import MatrixView
import pylinlin.matrix_utils as utils
from pylinlin.svd import compute_svd, reduce_to_bidiagonal


class TestSVD:

    def test_reduce_to_bidiagonal(self):
        mat = Matrix.from_cols([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
        b, _, _ = reduce_to_bidiagonal(mat)
        utils.assert_upper_triangular(b)
        truncated = MatrixView.with_size(
            b, (0, 1), (mat.num_rows() - 1, mat.num_cols() - 1)
        ).to_matrix()
        utils.assert_lower_triangular(truncated)

    def test_bidiagonal_recreate(self):
        mat = Matrix.from_cols([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
        b, left, right = reduce_to_bidiagonal(mat)
        for index, hh in list(enumerate(left))[::-1]:
            b = hh.multiply_left(b, index)
        for index, hh in list(enumerate(right))[::-1]:
            b = hh.multiply_right(b, index + 1)
        utils.assert_matrix_equal(mat, b)
