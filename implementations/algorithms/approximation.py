import numpy as np
from typing import TypeVar, Generic, Hashable

X = TypeVar("X")
H = TypeVar("H", bound=Hashable)


class Approximator(Generic[X]):
    """
    Interface for various function approximation methods.
    X is the type of inputs and outputs are always np.ndarray.
    """

    def __init__(self, config: dict):
        """
        `config`: dict
            All extra parameters.
        """
        self.config = config

    def predict(x: X) -> np.ndarray:
        """
        `x`: X
            Argument for which value should be estimated.
        Returns:
        `y`: np.ndarray
            Estimated value.
        """
        raise NotImplementedError

    def update(x: X, y: np.ndarray):
        """
        Make `predict(x)` closer to `y`.
        `x`: X
            Argument with a known value.
        `y`: np.ndarray
            Possible value for that argument.
        """
        raise NotImplementedError


class TableMean(Approximator[H], Generic[H]):
    """
    Stores means and counts for all arguments in a (hash) table.
    """

    def __init__(self, config: dict):
        """
        `config`: dict
            `default` = np.zeros(): np.ndarray

        """
        super().__init__(config)
        self.default = self.config.get("default", np.zeros(()))
        self.mean: dict[H, np.ndarray] = {}
        self.cnt: dict[H, int] = {}

    def predict(self, x: H) -> np.ndarray:
        return self.mean.get(x, self.default)

    def update(self, x: H, y: np.ndarray):
        count = self.cnt.setdefault(x, self.default) + 1
        current = self.mean.setdefault(x, self.default)
        self.cnt[x] = count
        self.mean[x] += (y - current) / count
