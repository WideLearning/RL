import gymnasium as gym
import numpy as np


class EpsGreedy:
    """
    ε-greedy algorithm for multi-argmed bandit problem.
    """

    def __init__(self, k: int, eps: float):
        """
        `k`: int
            Number of arms.
        `eps`: float in [0, 1]
            Ratio of non-greedy moves.
        """
        self.k = k
        self.eps = eps
        self.sum = np.zeros(shape=(self.k), dtype=np.float32)
        self.cnt = np.zeros(shape=(self.k,), dtype=np.float32)

    def train(self, env: gym.Env, num_steps: int):
        """
        Train for `num_steps` steps.
        """
        _observation, _info = env.reset()
        for _ in range(num_steps):
            if np.random.uniform(0, 1) < self.eps:
                action = env.action_space.sample()
            else:
                action = self.predict()
            _observation, reward, _terminated, _truncated, _info = env.step(action)
            self.sum[action] += reward
            self.cnt[action] += 1

    def value_estimates(self) -> np.array:
        """
        Computes sample average estimates of value function.
        """
        return np.where(self.cnt, self.sum / np.maximum(self.cnt, 1), -np.inf)

    def predict(self) -> np.int64:
        """
        Chooses the best action according to the current value function estimate.
        """
        return np.argmax(self.value_estimates())
