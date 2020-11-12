from typing import List


class Matrix:

    @staticmethod
    def from_rows(rows: List[List[float]]):
        num_rows = len(rows)
        num_cols = len(rows[0])
        for row in rows:
            if len(row) != num_cols:
                raise ValueError("Rows must have equal length")
        cols = [[0] * num_rows for _ in range(num_cols)]
        for row_index, row in enumerate(rows):
            for col_index, elem in enumerate(row):
                cols[col_index][row_index] = elem
        return Matrix(cols)

    @staticmethod
    def from_cols(cols: List[List[float]]):
        return Matrix(cols)

    def __init__(self, columns: List[List[float]]):
        self._num_cols = len(columns)
        self._num_rows = len(columns[0])
        for col in columns:
            if len(col) != self.num_rows():
                raise ValueError("Columns must have equal length")
        self.columns = columns

    def size(self) -> (int, int):
        return (self._num_rows, self._num_cols)

    def num_rows(self):
        return self._num_rows

    def num_cols(self):
        return self._num_cols

    def get_row(self, index: int) -> List[float]:
        if index < 0 or index >= self.num_rows():
            raise ValueError("Index out of bounds")
        return [col[index] for col in self.columns]

    def get_col(self, index: int) -> List[float]:
        if index < 0 or index >= self.num_cols():
            raise ValueError("Index out of bounds")
        return self.columns[index]

    def all_cols(self):
        return self.columns

    def transpose(self):
        return Matrix.from_rows(self.columns)
