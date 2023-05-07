from typing import TypeVar

import numpy as np

ObservationT = TypeVar("ObservationT")
ActionT = TypeVar("ActionT")


class TrainingLogger:
    def log_observation(self, observation: ObservationT):
        print("observation:", observation)

    def log_action(self, action: ActionT):
        print("action:", action)

    def log_reward(self, reward: float):
        raise NotImplementedError

    def new_episode(
        self,
    ):
        raise NotImplementedError


class ConsoleLogger(TrainingLogger):
    def __init__(self, observations=False, actions=False, rewards=False):
        self.current_reward = 0.0
        self.rewards = []
        self.log_observations = observations
        self.log_actions = actions
        self.log_rewards = rewards

    def log_observation(self, observation):
        if self.log_observations:
            print("observation:", observation, flush=True)

    def log_action(self, action):
        if self.log_actions:
            print("action:", action)

    def log_reward(self, reward: float):
        self.current_reward += reward
        self.rewards.append(reward)
        if self.log_rewards:
            print("reward:", reward)
        n = len(self.rewards)
        if n & (n - 1) != 0:
            return
        a = np.array(self.rewards)
        print(f"--- {a.size} episodes, {a.mean()} mean reward ---")

    def new_episode(self):
        self.current_reward = 0
