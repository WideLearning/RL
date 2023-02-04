import gymnasium as gym
import numpy as np

import implementations

env = gym.vector.make("bandit", means=np.array([1, 2, 3]), stds=np.array([0, 1, 1]), num_envs=4)

observation, info = env.reset()
print(observation, info)

for _ in range(10):
    action = env.action_space.sample()
    observations, rewards, termination, truncation, infos = env.step(action)
    print(observation)
    # print(f"{reward:.3f}, {terminated:1}, {truncated:1}, {info}")
    if termination or truncation:
        observation, info = env.reset()

env.close()
