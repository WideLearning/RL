import gymnasium as gym
import numpy as np
from algorithms.approximation import TableMean
from algorithms.online import EpsGreedy, TrainingLogger


class ConsoleLogger(TrainingLogger):
    def __init__(self):
        self.current_reward = 0.0
        self.rewards = []

    def log_observation(self, observation):
        # print("observation:", observation, flush=True)
        pass

    def log_action(self, action):
        # print("action:", action)
        pass

    def log_reward(self, reward: float):
        self.current_reward += reward
        # print(reward)
        pass

    def new_episode(self):
        self.rewards.append(self.current_reward)
        self.current_reward = 0
        l = len(self.rewards)
        if l & (l - 1) != 0:
            return
        a = np.array(self.rewards)
        print(f"--- {a.size} episodes, {a.mean()} mean reward ---")


env = gym.make("Blackjack-v1", render_mode=None)
agent = EpsGreedy({"q": TableMean({"default": 0.0}), "gamma": 0.5, "eps": 0.1})
agent.train(env, num_steps=10**5, logger=ConsoleLogger())
agent.eps = 0.05
agent.train(env, num_steps=10**5, logger=ConsoleLogger())
agent.eps = 0.0
agent.train(env, num_steps=10**5, logger=ConsoleLogger())
