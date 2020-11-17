
from time import perf_counter
import os
import sys
sys.path.append(os.getcwd())


def time_svd(size: int):
    import pylinlin.generate
    import pylinlin.svd
    mat = pylinlin.generate.generate_matrix_normal(size)
    time_start = perf_counter()
    _, _, _ = pylinlin.svd.compute_svd(mat)
    time_stop = perf_counter()
    print(f"{time_stop - time_start:8.5f} seconds")


if __name__ == '__main__':
    time_svd(5)
