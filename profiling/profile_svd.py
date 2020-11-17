
from time import perf_counter
import os
import sys
sys.path.append(os.getcwd())


def time_svd(size: int, iterations: int = 5) -> float:
    import pylinlin.generate
    import pylinlin.svd

    timings = []
    for _ in range(iterations):
        mat = pylinlin.generate.generate_matrix_normal(size)
        time_start = perf_counter()
        _, _, _ = pylinlin.svd.compute_svd(mat)
        time_stop = perf_counter()
        timings.append(time_stop - time_start)

    return sum(timings) / iterations


def profile_svd():
    for i in range(3, 100):
        duration = time_svd(i)
        print(f"Time taken for n = {i:3d}: {duration:10.5f} seconds")


if __name__ == '__main__':
    profile_svd()
