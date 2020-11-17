from pylinlin.generate import generate_matrix_normal
from pylinlin.svd import compute_svd


class TestGenerate:

    def test_random_matrix(self):
        mat = generate_matrix_normal(5)
        assert mat.size() == (5, 5)
        _, s, _ = compute_svd(mat)
        for i in range(5):
            assert s.get(i, i) > 1E-8
