from pylinlin.matrix import Matrix
from pylinlin.matrix_view import MatrixView
from pylinlin.householder import Householder
from typing import List


def reduce_to_bidiagonal(mat: Matrix) -> (Matrix, List[Householder], List[Householder]):
    mat = mat.copy()
    if mat.num_rows() != mat.num_cols():
        raise ValueError("Matrix should be square")
    iterations = mat.num_rows() - 1
    acc_left = []
    acc_right = []
    for iteration in range(iterations):
        # clear zeroes below diagonal
        col = mat.get_col(iteration)[iteration:]
        householder_left = Householder(col)
        mat = householder_left.multiply_left(mat, pad_top=iteration)
        acc_left.append(householder_left)
        if iteration != iterations - 1:
            # clear zeroes above superdiagonal
            row = mat.get_row(iteration)[iteration + 1:]
            householder_right = Householder(row)
            mat = householder_right.multiply_right(mat, pad_top=iteration + 1)
            acc_right.append(householder_right)
            print(mat.get_row(0))
    return mat, acc_left, acc_right


def compute_svd(mat: Matrix) -> (Matrix, Matrix, Matrix):
    pass
