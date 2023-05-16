from collections import defaultdict
from itertools import cycle, islice
from typing import Callable, Generic, Hashable, Iterator, TypeVar

import torch
from torch import nn

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


class Mean:
    def __init__(self, default=0.0):
        self.mean = default
        self.cnt = 0

    def __call__(self, new_value=None):
        if new_value is not None:
            self.cnt += 1
            self.mean += (new_value - self.mean) / self.cnt
        return self.mean


class TableMean(Approximator[H], Generic[H]):
    """
    Stores means and counts for all arguments in a table.
    """

    def __init__(self, config: dict):
        """
        `config`: dict
            `default` = 0.0: float
                Value to use for unknown arguments.
        """
        super().__init__(config)
        self.default = self.config.get("default", 0.0)
        self.mean: defaultdict[H, Mean] = defaultdict(lambda: Mean(self.default))

    def predict(self, x: H) -> float:
        return self.mean[x]()

    def update(self, x: H, y: float):
        assert isinstance(y, float)
        self.mean[x](y)


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


class DiffL2(Approximator[X], Generic[X]):
    """
    Updates are stored in a buffer with max size n.
    After adding a new update, k random batches of size batch_size are sampled from the buffer and used to train the model with L2 loss.

    n = 1, k = 1, batch_size = 1: Update once with each sample, the fastest version.
    n = inf, k = inf, batch_size = n: Converge to mean.
    n = 1 / alpha, k = inf, batch_size = n: Converge to exponential average with momentum alpha.
    """

    def __init__(
        self,
        model: nn.Module,
        transform: Callable[[X], torch.Tensor],
        input_shape: tuple[int, ...],
        opt_builder: Callable[
            [Iterator[nn.Parameter]], torch.optim.Optimizer
        ] = torch.optim.Adam,
        n: int = 1,
        k: int = 1,
        batch_size: int = 1,
    ):
        """
        Args:
        - model:
            Neural network that will be used as approximator.
        - transform:
            Preprocessing step that outputs torch tensors with the same shape.
        - opt_builder:
            Takes model.parameters() and returns optimizer.
        - n
            Buffer size.
        - k
            Number of batches per update.
        - batch_size
            Batch size to use during training.
        """
        self.model = model
        self.optim = opt_builder(model.parameters())
        self.transform = transform
        self.input_shape = input_shape
        self.n = n
        self.k = k
        self.batch_size = batch_size
        self.x_buffer = torch.zeros((n, *input_shape))
        self.y_buffer = torch.zeros((n,))
        self.size = 0

    def train_approximator(self):
        dataset = torch.utils.data.TensorDataset(self.x_buffer, self.y_buffer)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )
        for x, y in islice(cycle(dataloader), self.k):
            p = self.model(x).reshape(*y.shape)
            loss = torch.nn.functional.mse_loss(p, y)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

    def update(self, x: X, y: float):
        x_tensor = self.transform(x)
        if self.size == self.n:
            i = torch.randint(low=0, high=self.n, size=()).item()
            self.x_buffer[[i, -1]] = self.x_buffer[[-1, i]]
            self.y_buffer[[i, -1]] = self.y_buffer[[-1, i]]
            self.size -= 1
        self.x_buffer[self.size] = x_tensor
        self.y_buffer[self.size] = y
        self.size += 1
        self.train_approximator()

    def forward(self, x: X) -> torch.Tensor:
        return self.model(self.transform(x))

    def predict(self, x: X) -> float:
        return self.forward(x).item()
