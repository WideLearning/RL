import gymnasium as gym
import numpy as np
from algorithms.EpsGreedy import EpsGreedy

import implementations

env = gym.make(
    "MultiArmedBandit",
    means=np.array([1, 2, 3]),
    stds=np.array([0, 1, 1]),
    render_mode="ansi",
)
agent = EpsGreedy(k=3, eps=0.1)
agent.train(env=env, num_steps=1000)

observation, info = env.reset()
print(env.render())

total = 0
for _ in range(100):
    action = agent.predict()
    observation, reward, terminated, truncated, info = env.step(action)
    # print(f"{reward:.3f}, {terminated:1}, {truncated:1}, {info}")
    if terminated or truncated:
        observation, info = env.reset()
    total += reward
env.close()
print(f"total reward: {total}")
