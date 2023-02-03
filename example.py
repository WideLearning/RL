import gymnasium as gym
import numpy as np

import implementations

env = gym.make("bandit", means=np.array([1, 2, 3]), stds=np.array([0, 1, 1]))

observation, info = env.reset()
print(observation, info)

for _ in range(10):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print(f"{reward:.3f}, {terminated:1}, {truncated:1}, {info}")
    if terminated or truncated:
        observation, info = env.reset()

env.close()
