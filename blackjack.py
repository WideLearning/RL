import gymnasium as gym
import numpy as np
import torch
from algorithms.approximation import DiffL2, TableMean
from algorithms.logs import ConsoleLogger
from algorithms.online import QLearning
from torch import nn

env = gym.make("Blackjack-v1", render_mode=None)
obs, info = env.reset()
print(obs)


def onehot(k, n):
    assert 0 <= k <= n
    return torch.nn.functional.one_hot(torch.tensor(k).long(), num_classes=n).float()


def bj_encode(x):
    ((player, dealer, ace), action) = x
    return torch.cat(
        (onehot(player, 32), onehot(dealer, 11), onehot(ace, 2), onehot(action, 2))
    )


# agent = QLearning(
#     {
#         "q": DiffL2(
#             model=nn.Sequential(
#                 nn.LazyLinear(64),
#                 nn.ReLU(),
#                 nn.LazyLinear(32),
#                 nn.ReLU(),
#                 nn.LazyLinear(1),
#             ),
#             transform=bj_encode,
#             input_shape=(47,),
#             opt_builder=lambda p: torch.optim.Adam(p, lr=0.01),
#             n=1024,
#             k=8,
#             batch_size=16,
#         ),
#         "gamma": 0.5,
#         "eps": 0.1,
#     }
# )

agent = QLearning({"q": TableMean({"default": 0.0}), "gamma": 0.5, "eps": 0.1})
agent.train(env, num_steps=10**5, logger=ConsoleLogger())
# agent.eps = 0.05
# agent.train(env, num_steps=10**5, logger=ConsoleLogger())
# agent.eps = 0.0
# agent.train(env, num_steps=10**5, logger=ConsoleLogger())
