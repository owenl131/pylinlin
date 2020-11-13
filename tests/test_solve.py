from pylinlin.solve import solve_lower_triangular, solve_upper_triangular
from pylinlin.matrix import Matrix
import pylinlin.matrix_utils as utils


class TestSolve:

    def test_solve_lower(self):
        L = Matrix.from_cols([[1, 2, 3], [0, 4, 5], [0, 0, 6]])
        b = Matrix.from_cols([[3, 2, 1]])
        utils.assert_lower_triangular(L)
        x = solve_lower_triangular(L, b)
        assert x.size() == (3, 1)
        utils.assert_matrix_equal(b, L.multiply(x))

    def test_solve_upper(self):
        U = Matrix.from_cols([[0.3, 0, 0], [1, 4, 0], [-1, -2, -3]])
        b = Matrix.from_cols([[3, 2, 1]])
        utils.assert_upper_triangular(U)
        x = solve_upper_triangular(U, b)
        assert x.size() == (3, 1)
        utils.assert_matrix_equal(b, U.multiply(x))
