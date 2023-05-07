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
from torch import nn

env = gym.make("CliffWalking-v0", render_mode=None)


def onehot(k, n):
    assert 0 <= k <= n
    return torch.nn.functional.one_hot(torch.tensor(k).long(), num_classes=n).float()


agent = SoftmaxLearning({"q": TableMean({"default": 0.05}), "gamma": 0.9, "T": 0.1})

# agent = SoftmaxLearning(
#     {
#         "q": DiffL2(
#             model=nn.Sequential(
#                 nn.LazyLinear(128),
#                 nn.ReLU(),
#                 nn.LazyLinear(32),
#                 nn.ReLU(),
#                 nn.LazyLinear(1),
#             ),
#             opt_builder=lambda p: torch.optim.Adam(p, lr=0.01),
#             transform=lambda x: torch.cat((onehot(x[0] // 12, 4), onehot(x[0] % 12, 12), onehot(x[1], 4))),
#             input_shape=(20,),
#             n=1024,
#             k=4,
#             batch_size=512,
#         ),
#         "gamma": 0.9,
#         "T": 1.0,
#     }
# )


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

env = gym.make("CliffWalking-v0", render_mode="human")
agent.eps = 0
agent.train(
    env,
    num_steps=10**5,
    logger=ConsoleLogger(observations=False, actions=False, rewards=False),
)
