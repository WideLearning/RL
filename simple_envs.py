import matplotlib as mpl

mpl.use("TkAgg")  # otherwise gymnasium conflicts with matplotlib

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from algorithms.approximation import TableMean
from algorithms.logs import ConsoleLogger
from algorithms.online import QLearning

env = gym.make("CliffWalking-v0", render_mode=None)
agent = QLearning({"q": TableMean({"default": 0.05}), "gamma": 0.9, "eps": 0.05})
agent.train(
    env,
    num_steps=10**5,
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
    num_steps=1000,
    logger=ConsoleLogger(observations=True, actions=True, rewards=True),
)
