from pylinlin.matrix import Matrix
import random


def generate_normal() -> float:
    return random.gauss(0, 1)


def generate_matrix_normal(size: int) -> Matrix:
    columns = [[generate_normal() for _ in range(size)] for _ in range(size)]
    return Matrix.from_cols(columns)
