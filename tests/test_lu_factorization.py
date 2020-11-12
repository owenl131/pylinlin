from pylinlin.matrix import Matrix
from pylinlin.lu_factorization import compute_lu_factorization
import pylinlin.matrix_utils as utils


class TestLUFactorization:

    def test_lu(self):
        matrix = Matrix.from_cols([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        mat_l, mat_u = compute_lu_factorization(matrix)
        assert utils.is_lower_triangular(mat_l)
        assert utils.is_upper_triangular(mat_u)
        product = mat_l.multiply(mat_u)
        assert product.all_cols() == matrix.all_cols()
