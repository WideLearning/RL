import gymnasium as gym
import numpy as np


class MultiArmedBanditPolicy:
    """
    Base class for all multi-armed bandit policies.
    """

    def __init__(self, k: int, config: dict):
        """
        `k`: int
            Number of arms.
        `config`: dict
            All extra parameters, including:
            `eps`: float in [0, 1]
                Ratio of non-greedy moves.
        """
        self.k = k

    def train(self, env: gym.Env, num_steps: int):
        """
        Train for `num_steps` steps.
        """
        _observation, _info = env.reset()
        for _ in range(num_steps):
            action = self.predict(exploration=True)
            _observation, reward, _terminated, _truncated, _info = env.step(action)
            self.learn(action, reward)

    def predict(self, exploration=True) -> np.int64:
        """
        With `exploration` set to `False` chooses the best action according to the current value function estimate.
        Otherwise, selects the next action according to the learning algorithm.
        """
        pass

    def learn(self, action: np.int64, reward: np.float32):
        """
        Update parameters given that for the last `action` the agent received `reward`.
        """
        pass


class Optimal(MultiArmedBanditPolicy):
    """
    Algorithm that always selects action with the largest mean reward.
    Uses internal information of environment to do so.
    """

    def __init__(self, k: int, config: dict):
        """
        `k`: int
            Number of arms.
        `config`: dict
            All extra parameters, including:
            `means`: np.array of floats
                Actual action values from the environment.
        """
        super().__init__(k, config)
        self.means = config["means"]

    def predict(self, exploration=True) -> np.int64:
        return np.argmax(self.means)

    def learn(self, action: np.int64, reward: np.float32):
        pass


class EpsGreedy(MultiArmedBanditPolicy):
    """
    ε-greedy algorithm for multi-armed bandit problem.
    """

    def __init__(self, k: int, config: dict):
        """
        `k`: int
            Number of arms.
        `config`: dict
            All extra parameters, including:
            `eps`: float in [0, 1]
                Ratio of non-greedy moves.
        """
        super().__init__(k, config)
        self.eps = config["eps"]
        self.sum = np.zeros(shape=(self.k), dtype=np.float32)
        self.cnt = np.zeros(shape=(self.k,), dtype=np.float32)

    def value_estimates(self) -> np.array:
        """
        Computes sample average estimates of value function.
        """
        return np.where(self.cnt, self.sum / np.maximum(self.cnt, 1), +np.inf)

    def predict(self, exploration=True) -> np.int64:
        if exploration and np.random.uniform(0, 1) < self.eps:
            return np.random.randint(low=0, high=self.k)
        else:
            return np.argmax(self.value_estimates())

    def learn(self, action: np.int64, reward: np.float32):
        self.sum[action] += reward
        self.cnt[action] += 1


class UCB(MultiArmedBanditPolicy):
    """
    Upper Confidence Bound algorithm for multi-armed bandit problem.
    Selects actions according to `Q_t(a) + c * sqrt(ln(t) / N_t(a))`.
    """

    def __init__(self, k: int, config: dict):
        """
        `k`: int
            Number of arms.
        `config`: dict
            All extra parameters, including:
            `c`: float in (0, ∞)

        """
        super().__init__(k, config)
        self.c = config["c"]
        self.sum = np.zeros(shape=(self.k), dtype=np.float32)
        self.cnt = np.zeros(shape=(self.k,), dtype=np.float32)

    def UCB_estimates(self) -> np.array:
        """
        Computes upper confidence bounds for action values.
        """
        upper_bound = 3.0  # approximate upper bound for action value distribution
        t = self.cnt.sum()
        if not t:
            return np.full_like(self.cnt, fill_value=upper_bound)
        N = np.maximum(self.cnt, 1)
        Q = self.sum / N
        UCB = Q + self.c * np.sqrt(np.log(t) / N)
        return np.where(self.cnt, UCB, upper_bound)

    def predict(self, exploration=True) -> np.int64:
        return np.argmax(self.UCB_estimates())

    def learn(self, action: np.int64, reward: np.float32):
        self.sum[action] += reward
        self.cnt[action] += 1
