import unittest

import numpy as np
import torch
from algorithms.approximation import DiffL2, TableExp, TableMean
from torch import nn


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


class TestDiffL2(unittest.TestCase):
    def test_parabola(self):
        appr: DiffL2[str] = DiffL2(
            nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 1)),
            transform=lambda x: torch.tensor([len(x)], dtype=torch.float),
            input_shape=(1,),
            opt_builder=lambda p: torch.optim.Adam(p, lr=0.1),
            n=1024,
            k=10,
            batch_size=32,
        )
        for i in range(100):
            x = "a" * torch.randint(0, 10, ())
            y = len(x) ** 2 + 0.5
            appr.update(x, y)
        self.assertAlmostEqual(appr.predict("b" * 5), 25.5, delta=3.0)


if __name__ == "__main__":
    pass
