from typing import TypeVar, Generic, Hashable

X = TypeVar("X")
H = TypeVar("H", bound=Hashable)


class Approximator(Generic[X]):
    """
    Interface for various function approximation methods.
    X is the type of inputs and outputs are always float.
    """

    def __init__(self, config: dict):
        """
        `config`: dict
            All extra parameters.
        """
        self.config = config

    def predict(self, x: X) -> float:
        """
        `x`: X
            Argument for which value should be estimated.
        Returns:
        `y`: float
            Estimated value.
        """
        raise NotImplementedError

    def update(self, x: X, y: float):
        """
        Make `predict(x)` closer to `y`.
        `x`: X
            Argument with a known value.
        `y`: float
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
            `default` = 0.0: float
                Value to use for unknown arguments.
        """
        super().__init__(config)
        self.default = self.config.get("default", 0.0)
        self.mean: dict[H, float] = {}
        self.cnt: dict[H, int] = {}

    def predict(self, x: H) -> float:
        return self.mean.get(x, self.default)

    def update(self, x: H, y: float):
        assert isinstance(y, float)
        count = self.cnt.setdefault(x, 0) + 1
        current = self.mean.setdefault(x, self.default)
        self.cnt[x] = count
        if count == 1:
            self.mean[x] = y
        else:
            self.mean[x] += (y - current) / count


class TableExp(Approximator[H], Generic[H]):
    """
    Stores exponential averages for all arguments in a (hash) table.
    """

    def __init__(self, config: dict):
        """
        `config`: dict
            `lr`: float
                Used in `mean' = mean + lr * (y - mean)`.
            `default` = 0.0: float
                Value to use for unknown arguments.

        """
        super().__init__(config)
        self.lr = self.config["lr"]
        self.default = self.config.get("default", 0.0)
        self.mean: dict[H, float] = {}

    def predict(self, x: H) -> float:
        return self.mean.get(x, self.default)

    def update(self, x: H, y: float):
        assert isinstance(y, float)
        current = self.mean.setdefault(x, self.default)
        self.mean[x] = current + self.lr * (y - current)
