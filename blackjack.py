import gymnasium as gym
import numpy as np
from algorithms.approximation import TableMean
from algorithms.online import QLearning
from algorithms.logs import ConsoleLogger


env = gym.make("Blackjack-v1", render_mode=None)
agent = QLearning({"q": TableMean({"default": 0.0}), "gamma": 0.5, "eps": 0.1})
agent.train(env, num_steps=10**5, logger=ConsoleLogger())
agent.eps = 0.05
agent.train(env, num_steps=10**5, logger=ConsoleLogger())
agent.eps = 0.0
agent.train(env, num_steps=10**5, logger=ConsoleLogger())
