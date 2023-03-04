# pylint: disable=too-many-arguments
import itertools
from typing import TypeVar

import gymnasium as gym
import numpy as np

from implementations.algorithms.approximation import Approximator

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


def get_actions(space: gym.Space) -> list[ActionT]:
    """
    `space`: gym.Space
        A finite action space constructed from `types`.
    Returns:
    `actions`: list[A]
        A list of all possible actions from this space.
    """

    types = [
        gym.spaces.multi_binary.MultiBinary,
        gym.spaces.discrete.Discrete,
        gym.spaces.multi_discrete.MultiDiscrete,
        gym.spaces.dict.Dict,
        gym.spaces.tuple.Tuple,
    ]

    if type(space) not in types:
        raise ValueError(
            f"input space {space} is not constructed from spaces of types:"
            + "\n"
            + str(types)
        )
    if isinstance(space, gym.spaces.multi_binary.MultiBinary):
        return [
            np.reshape(np.array(element), space.n)
            for element in itertools.product(*[range(2)] * np.prod(space.n))
        ]
    if isinstance(space, gym.spaces.discrete.Discrete):
        return list(range(space.n))
    if isinstance(space, gym.spaces.multi_discrete.MultiDiscrete):
        return [
            np.array(element)
            for element in itertools.product(*[range(n) for n in space.nvec])
        ]
    if isinstance(space, gym.spaces.dict.Dict):
        keys = space.spaces.keys()
        values_list = itertools.product(
            *[get_actions(sub_space) for sub_space in space.spaces.values()]
        )
        return [dict(zip(keys, values)) for values in values_list]

    assert isinstance(space, gym.spaces.tuple.Tuple)
    return [
        list(element)
        for element in itertools.product(
            *[get_actions(sub_space) for sub_space in space.spaces]
        )
    ]


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

    def train(self, env: gym.Env, num_steps: int, logger: TrainingLogger):
        """
        Train for `num_steps` steps.
        """
        observation, _info = env.reset()
        actions = get_actions(env.action_space)
        for _ in range(num_steps):
            logger.log_observation(observation)
            action = self.predict(observation, actions, exploration=True)
            logger.log_action(action)
            new_observation, reward, terminated, truncated, _info = env.step(action)
            logger.log_reward(reward)
            self.learn(observation, action, reward, new_observation, actions)
            if terminated or truncated:
                observation, _info = env.reset()
                logger.new_episode()
            else:
                observation = new_observation

    def predict(
        self, observation: ObservationT, actions: list[ActionT], exploration=True
    ) -> ActionT:
        """
        With `exploration` set to `False` chooses the best action (based on the last observation `obs`).
        Otherwise, selects the next action according to the learning algorithm.
        """
        raise NotImplementedError

    def learn(
        self,
        observation: ObservationT,
        action: ActionT,
        reward: float,
        new_observation,
        actions: list[ActionT],
    ):
        """
        Update parameters given:
        `observation` -> `action` -> `reward` -> `new_observation`
        and with possible next actions in `actions`.
        """
        raise NotImplementedError


class EpsGreedy(OnlinePolicy):
    """
    ε-greedy strategy with on-policy action-value iteration.
    """

    def __init__(self, config: dict):
        """
        `config`: dict
            All extra parameters, including:
            `q`: Approximator[tuple[O, A]]
                How to estimate action-value function.
            `gamma`: float in [0, 1]
                Discount applied to returns: `G_t = R_{t+1} + gamma * R_{t+2} + gamma^2 * R_{t+2} + ...`.
            `eps`: float in [0, 1]
                Ratio of non-greedy moves.
        """
        super().__init__(config)
        self.q: Approximator[tuple[ObservationT, ActionT]] = config["q"]
        self.gamma = config["gamma"]
        self.eps = config["eps"]

    def predict(
        self, observation: ObservationT, actions: list[ActionT], exploration=True
    ) -> ActionT:
        if exploration and np.random.random() < self.eps:
            return np.random.choice(actions)
        evaluations = [(self.q.predict((observation, a)), a) for a in actions]
        return max(evaluations)[1]

    def learn(
        self,
        observation: ObservationT,
        action: ActionT,
        reward: float,
        new_observation,
        actions: list[ActionT],
    ):
        G = self.predict(new_observation, actions)
        self.q.update((observation, action), reward + self.gamma * G)
