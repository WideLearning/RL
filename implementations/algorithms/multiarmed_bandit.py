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
        self.config = config

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
        raise NotImplementedError

    def learn(self, action: np.int64, reward: np.float32):
        """
        Update parameters given that for the last `action` the agent received `reward`.
        """
        raise NotImplementedError


class OptimalBandit(MultiArmedBanditPolicy):
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
            `q`: Approximator
                Will be used for action value estimation.
        """
        super().__init__(k, config)
        self.eps = config["eps"]
        self.q = config["q"]

    def value_estimates(self) -> np.array:
        """
        Computes sample average estimates of value function.
        """
        return np.array([self.q.predict(a) for a in range(self.k)])

    def predict(self, exploration=True) -> np.int64:
        if exploration and np.random.uniform(0, 1) < self.eps:
            return np.random.randint(low=0, high=self.k)
        return np.argmax(self.value_estimates())

    def learn(self, action: np.int64, reward: np.float32):
        self.q.update(action, reward)


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


class OptimalGradientBandit(MultiArmedBanditPolicy):
    """
    Exact gradient step would be:

    `H_{t+1}(a) - H_t(a) = lr * \\pi_t(a) * (q(a) - E(R))`

    It performs it by using internal information from the environment.
    """

    def __init__(self, k: int, config: dict):
        """
        `k`: int
            Number of arms.
        `config`: dict
            All extra parameters, including:
            `means`: np.array of floats
                Actual action values from the environment.
            `lr`: float in (0, ∞)
                Learning rate.
        """
        super().__init__(k, config)
        self.means = config["means"]
        self.lr = config["lr"]
        self.reward_sum = 0
        self.reward_cnt = 0
        self.H = np.zeros(shape=(self.k), dtype=np.float32)

    def policy(self):
        return np.exp(self.H) / np.exp(self.H).sum()

    def predict(self, exploration=True) -> np.int64:
        if exploration:
            return np.random.choice(self.k, p=self.policy())
        return np.argmax(self.H)

    def learn(self, action: np.int64, reward: np.float32):
        # True gradient
        pi = self.policy()
        expected_rewards = (pi * self.means).sum()
        grad = pi * (self.means - expected_rewards)

        self.H += self.lr * grad
        self.H -= self.H.max()


class GradientBandit(MultiArmedBanditPolicy):
    """
    Exact gradient step would be:

    `H_{t+1}(a) - H_t(a) = lr * \\pi_t(a) * (q(a) - E(R))`

    Here the following unbiased approximation is used:

    `H_{t+1}(a) - H_t(a) = lr * ([A_t = a] - \\pi_t(a)) * (R_t - \\bar R_t)`
    """

    def __init__(self, k: int, config: dict):
        """
        `k`: int
            Number of arms.
        `config`: dict
            All extra parameters, including:
            `lr`: float in (0, ∞)
                Learning rate.
        """
        super().__init__(k, config)
        self.lr = config["lr"]
        self.reward_sum = 0
        self.reward_cnt = 0
        self.H = np.zeros(shape=(self.k), dtype=np.float32)

    def policy(self):
        return np.exp(self.H) / np.exp(self.H).sum()

    def predict(self, exploration=True) -> np.int64:
        if exploration:
            return np.random.choice(self.k, p=self.policy())
        return np.argmax(self.H)

    def learn(self, action: np.int64, reward: np.float32):
        baseline = self.reward_sum / self.reward_cnt if self.reward_cnt != 0 else reward
        self.reward_sum += reward
        self.reward_cnt += 1

        self.H -= self.lr * (reward - baseline) * self.policy()
        self.H[action] += self.lr * (reward - baseline)
        self.H -= self.H.max()


class GradientBanditBiased(MultiArmedBanditPolicy):
    """
    Exact gradient step would be:

    `H_{t+1}(a) - H_t(a) = lr * \\pi_t(a) * (q(a) - E(R))`

    Here the following (biased, but in some cases useful) approximation is used:

    `H_{t+1}(a) - H_t(a) = lr * [A_t = a] * (R_t - \\bar R_t)`
    """

    def __init__(self, k: int, config: dict):
        """
        `k`: int
            Number of arms.
        `config`: dict
            All extra parameters, including:
            `lr`: float in (0, ∞)
                Learning rate.
        """
        super().__init__(k, config)
        self.lr = config["lr"]
        self.reward_sum = 0
        self.reward_cnt = 0
        self.H = np.zeros(shape=(self.k), dtype=np.float32)

    def predict(self, exploration=True) -> np.int64:
        def softmax(x):
            return np.exp(x) / np.exp(x).sum()

        if exploration:
            return np.random.choice(self.k, p=softmax(self.H))
        return np.argmax(self.H)

    def learn(self, action: np.int64, reward: np.float32):
        baseline = self.reward_sum / self.reward_cnt if self.reward_cnt != 0 else reward
        self.reward_sum += reward
        self.reward_cnt += 1

        self.H[action] += self.lr * (reward - baseline)
        self.H -= self.H.max()
