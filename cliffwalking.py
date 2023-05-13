import matplotlib as mpl

mpl.use("TkAgg")  # otherwise gymnasium conflicts with matplotlib

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from algorithms.approximation import DiffL2, TableMean
from algorithms.logs import ConsoleLogger
from algorithms.online import QLearning, SoftmaxLearning
from gymnasium.wrappers import TransformReward
from torch import nn


class CliffWrapper:
    def __init__(self, render_mode):
        self.orig = gym.make("CliffWalking-v0", render_mode=render_mode)

    def new_step(self, action):
        observation, reward, terminated, truncated, info = self.orig.step(action)
        if terminated:
            reward = 1e3
        return observation, reward, terminated, truncated, info

    def __getattr__(self, name):
        return getattr(self.orig, name) if name != "step" else self.new_step


env = CliffWrapper(None)


def onehot(k, n):
    assert 0 <= k <= n
    return torch.nn.functional.one_hot(torch.tensor(k).long(), num_classes=n).float()


# agent = SoftmaxLearning({"q": TableMean({"default": 0.05}), "gamma": 0.9, "T": 0.1})


def fourier(i: int, j: int, max_i: int, max_j: int, k_i: int, k_j: int) -> torch.Tensor:
    """
    Generates 2 k_i k_j features for a point (i, j) inside max_i x max_j grid.
    """
    assert 0 <= i < max_i and 0 <= j < max_j
    features = []
    for i_freq in range(k_i):
        for j_freq in range(k_j):
            a = 2 * np.pi * (i * i_freq / max_i + j * j_freq / max_j)
            features.append(np.sin(a))
            features.append(np.cos(a))
    return torch.tensor(features, dtype=torch.float32)


agent = SoftmaxLearning(
    {
        "q": DiffL2(
            model=nn.Sequential(
                nn.LazyLinear(128),
                nn.ReLU(),
                nn.LazyLinear(32),
                nn.ReLU(),
                nn.LazyLinear(1),
            ),
            opt_builder=lambda p: torch.optim.Adam(p, lr=0.01),
            transform=lambda x: torch.cat(
                (fourier(x[0] // 12, x[0] % 12, 4, 12, 4, 3), onehot(x[1], 4))
            ),
            input_shape=(28,),
            n=1024,
            k=4,
            batch_size=512,
        ),
        "gamma": 0.9,
        "T": 0.5,
    }
)


agent.train(
    env,
    num_steps=2049,
    logger=ConsoleLogger(observations=False, actions=False, rewards=False),
)

estimates = np.array(
    [
        [
            [agent.q.predict((12 * row + col, dir)) for col in range(12)]
            for row in range(4)
        ]
        for dir in range(4)
    ]
)

print(estimates)

fig, axn = plt.subplots(4, 1, sharex=True, sharey=True)
dirs = ["up", "right", "down", "left"]
for i in range(4):
    sns.heatmap(estimates[i], ax=axn[i], cmap="bwr", vmin=-10, vmax=0)
    axn[i].set_title(f"{dirs[i]}")
plt.tight_layout()
plt.show()

env = CliffWrapper("human")
agent.T = 0.01
agent.train(
    env,
    num_steps=10**5,
    logger=ConsoleLogger(observations=False, actions=False, rewards=False),
)
