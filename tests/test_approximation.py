import unittest
from implementations.algorithms.approximation import TableMean
import numpy as np
import typing


class TestTableMean(unittest.TestCase):
    def test_int(self):
        t: TableMean[int] = TableMean({})
        t.update(1, np.array(3))

        self.assertEqual(t.predict(1).item(), 3)
