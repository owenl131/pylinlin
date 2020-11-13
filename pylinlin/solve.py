from pylinlin.matrix import Matrix
from pylinlin.matrix_view import MatrixView


def solve_upper_triangular(U: Matrix, b: Matrix) -> Matrix:
    b = b.copy()
    if U.num_cols() != U.num_rows():
        raise ValueError("Matrix U should be square")
    if b.num_cols() != 1:
        raise ValueError("Vector b must be a column vector")
    dims = U.num_cols()
    if b.num_rows() != dims:
        raise ValueError("U and b have incompatible sizes")
    x = Matrix.zeroes(dims, 1)
    for iteration in range(dims):
        index = dims - 1 - iteration
        bEntry = b.get(index, 0)
        pivot = U.get(index, index)
        MatrixView.with_size(x, (index, 0), (1, 1)).set_element(
            0, 0, bEntry / pivot)
        if index != 0:
            bToUpdate = MatrixView(b, (0, 0), (index - 1, 0))
            bToUpdate.scale_add(
                MatrixView(U, (0, index), (index - 1, index)),
                -x.get(index, 0)
            )
    return x


def solve_lower_triangular(L: Matrix, b: Matrix) -> Matrix:
    b = b.copy()
    if L.num_cols() != L.num_rows():
        raise ValueError("Matrix L should be square")
    if b.num_cols() != 1:
        raise ValueError("Vector b must be a column vector")
    dims = L.num_cols()
    if b.num_rows() != dims:
        raise ValueError("L and b have incompatible sizes")
    x = Matrix.zeroes(dims, 1)
    for iteration in range(dims):
        bEntry = b.get(iteration, 0)
        pivot = L.get(iteration, iteration)
        MatrixView.with_size(x, (iteration, 0), (1, 1)).set_element(
            0, 0, bEntry / pivot)
        if iteration != dims - 1:
            bToUpdate = MatrixView.to_end(b, (iteration + 1, 0))
            bToUpdate.scale_add(
                MatrixView(
                    L, (iteration + 1, iteration), (L.num_rows()-1, iteration)),
                -x.get(iteration, 0))
    return x
