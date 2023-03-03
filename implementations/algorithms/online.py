import gymnasium as gym
import numpy as np


class OnlinePolicy:
    """
    Abstract class for RL policies interacting with environment.
    """

    def __init__(self, config: dict):
        """
        `config`: dict
            All extra parameters.
        """
        self.config = config

    def train(self, env: gym.Env, num_steps: int):
        """
        Train for `num_steps` steps.
        """
        observation, _info = env.reset()
        for _ in range(num_steps):
            action = self.predict(exploration=True)
            new_observation, reward, terminated, truncated, _info = env.step(action)
            self.learn(observation, action, reward)
            if terminated or truncated:
                observation, _info = env.reset()
            else:
                observation = new_observation

    def predict(self, obs, exploration=True):
        """
        With `exploration` set to `False` chooses the best action (based on the last observation `obs`).
        Otherwise, selects the next action according to the learning algorithm.
        """
        pass

    def learn(self, obs, action, reward: np.float32):
        """
        Update parameters given that for the last `action` the agent received `reward`.
        """
        pass


class EpsGreedy(OnlinePolicy):
    """
    ε-greedy strategy with on-policy value iteration.
    """

    def __init__(self, k: int, config: dict):
        """
        `config`: dict
            All extra parameters, including:
            `eps`: float in [0, 1]
                Ratio of non-greedy moves.
        """
        super().__init__(k, config)
        self.eps = config["eps"]
        ...

    def predict(self, exploration=True) -> np.int64:
        if exploration and np.random.uniform(0, 1) < self.eps:
            ...
        else:
            ...

    def learn(self, action: np.int64, reward: np.float32):
        ...