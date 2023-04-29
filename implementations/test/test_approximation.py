import unittest
from algorithms.approximation import TableMean, TableExp
import numpy as np


class TestTableMean(unittest.TestCase):
    def test_int(self):
        t: TableMean[int] = TableMean({})
        t.update(1, 3.0)
        t.update(2, 1.0)
        t.update(2, 2.0)

        self.assertEqual(t.predict(0), 0)
        self.assertEqual(t.predict(1), 3)
        self.assertEqual(t.predict(2), 1.5)

    def test_default(self):
        t: TableMean[str] = TableMean({"default": np.pi})

        self.assertEqual(t.predict(""), np.pi)

    def test_inf(self):
        t: TableMean[int] = TableMean({"default": np.inf})
        t.update(1, 3.0)
        t.update(2, 1.0)
        t.update(2, 2.0)

        self.assertEqual(t.predict(0), np.inf)
        self.assertEqual(t.predict(1), 3)
        self.assertEqual(t.predict(2), 1.5)


class TestTableExp(unittest.TestCase):
    def test_int(self):
        t: TableExp[int] = TableExp({"lr": 0.5})
        t.update(1, 3.0)
        t.update(2, 1.0)
        t.update(2, 2.0)

        self.assertEqual(t.predict(0), 0)
        self.assertEqual(t.predict(1), 1.5)
        self.assertEqual(t.predict(2), 1.25)

    def test_default(self):
        t: TableExp[str] = TableExp({"lr": 0.5, "default": np.pi})

        self.assertEqual(t.predict(""), np.pi)

    def test_inf(self):
        t: TableExp[int] = TableExp({"lr": 0.5, "default": 3.0})
        t.update(1, 3.0)
        t.update(2, 1.0)
        t.update(2, 2.0)

        self.assertEqual(t.predict(0), 3)
        self.assertEqual(t.predict(1), 3)
        self.assertEqual(t.predict(2), 2)

if __name__ == "__main__":
    t: TableMean[int] = TableMean({"default": np.inf})
    print(t.predict(0))