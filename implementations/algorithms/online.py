# pylint: disable=too-many-arguments
import itertools
from typing import Any, Generic, TypeVar

import gymnasium as gym
import numpy as np

from implementations.algorithms.approximation import Approximator, Mean

from .logs import TrainingLogger

ObservationT = TypeVar("ObservationT")
ActionT = TypeVar("ActionT")


def get_actions(space: gym.Space) -> list[Any]:
    """
    `space`: gym.Space
        A finite action space constructed from `types`.

    Returns:
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


class OnlinePolicy(Generic[ObservationT, ActionT]):
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
            reward = float(reward)
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


class QLearning(OnlinePolicy[ObservationT, ActionT]):
    """
    Q-learning with ε-greedy exploration.
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
            return actions[np.random.randint(0, len(actions))]
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
        argmax_action = self.predict(new_observation, actions)
        max_return = self.q.predict((new_observation, argmax_action))
        self.q.update((observation, action), reward + self.gamma * max_return)


class SoftmaxLearning(OnlinePolicy[ObservationT, ActionT]):
    """
    Q-learning where exploration policy is the softmax of the value function. Objective is discounted reward.
    """

    def __init__(self, config: dict):
        """
        `config`: dict
            All extra parameters, including:
            `q`: Approximator[tuple[O, A]]
                How to estimate action-value function.
            `gamma`: float in [0, 1]
                Discount applied to returns: `G_t = R_{t+1} + gamma * R_{t+2} + gamma^2 * R_{t+2} + ...`.
            `T`:
                Temperature of softmax. Higher T means more random moves.
        """
        super().__init__(config)
        self.q: Approximator[tuple[ObservationT, ActionT]] = config["q"]
        self.gamma = config["gamma"]
        self.T = config["T"]

    def predict(
        self, observation: ObservationT, actions: list[ActionT], exploration=True
    ) -> ActionT:
        if exploration:
            values = np.array([self.q.predict((observation, a)) for a in actions])
            values /= self.T
            values -= values.max()
            probs = np.exp(values) / np.exp(values).sum()
            return actions[np.random.choice(len(probs), p=probs)]
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
        argmax_action = self.predict(new_observation, actions)
        max_return = self.q.predict((new_observation, argmax_action))
        self.q.update((observation, action), reward + self.gamma * max_return)


class AverageSoftmaxLearning(OnlinePolicy[ObservationT, ActionT]):
    """
    Q-learning where exploration policy is the softmax of the value function. Objective is average reward.
    """

    def __init__(self, config: dict):
        """
        `config`: dict
            All extra parameters, including:
            `q`: Approximator[tuple[O, A]]
                How to estimate action-value function.
            `e`: Approximator[tuple[O, A]]
                How to estimate action-value function errors.
            `gamma`: float in [0, 1]
                Discount applied to returns: `G_t = R_{t+1} + gamma * R_{t+2} + gamma^2 * R_{t+2} + ...`.
            `T`:
                Temperature of softmax. Higher T means more random moves.
        """
        super().__init__(config)
        # estimates expected action value
        self.q: Approximator[tuple[ObservationT, ActionT]] = config["q"]
        # estimates its dispersion
        self.e: Approximator[tuple[ObservationT, ActionT]] = config["e"]
        self.T = config["T"]
        self.gamma = config["gamma"]
        self.mean_reward = Mean(default=0.0)

    def predict(
        self, observation: ObservationT, actions: list[ActionT], exploration=True
    ) -> ActionT:
        if exploration:

            def UCL(a):
                oa = (observation, a)
                mean = self.q.predict(oa)
                std = np.sqrt(self.e.predict(oa))
                return mean + self.T * std

            values = np.array(list(map(UCL, actions)))
            return actions[np.argmax(values)]
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
        r_mean = self.mean_reward(reward)  # update mean reward estimate
        argmax_action = self.predict(new_observation, actions)
        max_return = self.q.predict((new_observation, argmax_action))
        # exploration_bonus = self.e.predict((new_observation, argmax_action))
        old_q = self.q.predict((observation, action))
        # upd_q = reward - r_mean + (max_return + self.T * exploration_bonus) * self.gamma
        upd_q = reward - r_mean + max_return * self.gamma
        self.e.update((observation, action), (old_q - upd_q) ** 2)
        self.q.update((observation, action), upd_q)
