import matplotlib as mpl

mpl.use("TkAgg")  # otherwise gymnasium conflicts with matplotlib

from typing import Callable

import gymnasium as gym
import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
from algorithms.MultiArmedBandit import MultiArmedBanditPolicy, Optimal, EpsGreedy, UCB
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

plt.rcParams["text.usetex"] = True

import implementations


def rewards_during_training(
    build_agent: Callable[[gym.Env], MultiArmedBanditPolicy], seed=0
) -> np.ndarray[np.float32]:
    np.random.seed(seed)
    env = gym.make(
        "MultiArmedBandit",
        means=np.random.randn(5),
        stds=np.full(5, fill_value=3),
        render_mode="ansi",
    )
    agent = build_agent(env)
    num_steps = 300
    rewards = np.zeros(num_steps)
    _observation, _info = env.reset(seed=seed)
    for i in range(num_steps):
        action = agent.predict(exploration=True)
        _observation, rewards[i], _terminated, _truncated, _info = env.step(action)
        agent.learn(action, rewards[i])
    return rewards


def plot_training(build_agent: Callable[[], MultiArmedBanditPolicy], title):
    runs = 10000
    rewards = np.stack(
        [rewards_during_training(build_agent, seed) for seed in tqdm(range(runs))]
    )
    mean = gaussian_filter1d(rewards.mean(axis=0), sigma=10)
    std = rewards.std(axis=0) / np.sqrt(runs)
    plt.xlim(1, 300)
    plt.ylim(0, 2)
    plt.xticks(np.linspace(0, 300, 11))
    plt.yticks(np.linspace(0, 2, 11))
    plt.grid(visible=True)
    plt.plot(mean, "-", c="red")
    plt.plot(mean + std, "--", c="blue")
    plt.plot(mean - std, "--", c="blue")
    plt.title(title, fontdict={"size": 20})


def under_curve(build_agent: Callable[[], MultiArmedBanditPolicy]) -> tuple[np.float32, np.float32]:
    runs = 10000
    rewards = np.stack(
        [rewards_during_training(build_agent, seed) for seed in tqdm(range(runs))]
    )
    mean = rewards.mean(axis=0)
    stds = rewards.std(axis=0) / np.sqrt(runs)
    return mean.sum(), stds.sum() # just summing standard deviations, because they are correlated


greedy = lambda env: EpsGreedy(k=env.means.size, config={"eps": 0.0})
eps_greedy = lambda env: EpsGreedy(k=env.means.size, config={"eps": 0.05})
ucb = lambda env: UCB(k=env.means.size, config={"c": 3.})
optimal = lambda env: Optimal(k=env.means.size, config={"means": env.means})

current = greedy

print(under_curve(current)) # mean area under curve, its standard deviation
plot_training(current, r"Greedy")
plt.savefig("images/greedy.svg")

# optimal 350±10
# eps_greedy 242±10
# ucb 262±10
# greedy 225±10
