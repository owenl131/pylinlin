from __future__ import annotations
from typing import List
from .matrix import Matrix
from .matrix_view import MatrixView


class Householder:
    def __init__(self: Householder, vec: List[float]):
        # Reflects vec to e0 * ||vec||
        e0 = Matrix.zeroes(len(vec), 1)
        MatrixView(e0, (0, 0), (0, 0)).set_element(0, 0, 1)
        mat = Matrix.from_cols([vec[:]])
        norm = mat.frobenius_norm()
        MatrixView.to_end(mat, (0, 0)).scale_add(e0, -norm)
        norm2 = mat.frobenius_norm()
        MatrixView.to_end(mat, (0, 0)).scale(1 / norm2)
        self.base = mat

    def multiply_left(self: Householder, mat: Matrix) -> Matrix:
        pass

    def multiply_right(self: Householder, mat: Matrix) -> Matrix:
        pass

    def to_matrix(self: Householder) -> Matrix:
        householder_mat = Matrix.identity(self.base.num_rows())
        update = self.base.multiply(self.base.transpose())
        MatrixView.to_end(householder_mat, (0, 0)).scale_add(update, -2)
        return householder_mat
