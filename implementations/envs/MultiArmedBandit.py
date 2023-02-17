from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class MultiArmedBanditEnv(gym.Env):
    """
    An environment for multi-armed bandit problem with Gaussian rewards.

    Example:
        env = gym.make("MultiArmedBandit", means=np.array([1, 2, 3]), stds=np.array([0, 1, 1]))
        observation, info = env.reset()
        for _ in range(10):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                observation, info = env.reset()
        env.close()
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(
        self,
        render_mode: str | None = None,
        means: np.array = np.zeros(1),
        stds: np.array = np.ones(1),
    ):
        """
        render_mode: string
            Selects how to render the environment. Only "ansi" supported.
        means: nd.array with shape (self.k,) and dtype np.float32
            Sets the mean reward for each arm.
        std: nd.array with shape (self.k,) and dtype np.float32
            Sets the standard deviatino of reward for each arm.
        """
        assert means.shape == stds.shape, f"means: {means.shape} ≠ stds: {stds.shape}"
        assert means.ndim == 1, f"means.ndim = {means.ndim}"
        render_modes = [None] + self.metadata["render_modes"]
        assert render_mode in render_modes, f"render_mode = {render_mode}"
        self.render_mode = render_mode
        self.k = means.size
        self.means = means
        self.stds = stds
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,))
        self.action_space = spaces.Discrete(self.k)

    def _get_obs(self):
        return np.zeros(1, dtype=np.float32)

    def _get_info(self):
        return {}

    def reset(self, seed: int = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        # reset the state here if needed
        observation = self._get_obs()
        info = self._get_info()
        # render here if needed
        return observation, info

    def step(self, action: np.int64) -> tuple[np.float32, np.float32, bool, bool, dict]:
        """
        Because this environment is stateless, just returns the reward for a given action.

        action: np.int64 in [0, self.k - 1]
            The arm to use on this step.

        returns: (np.float32, np.float32, bool, bool, dict)
            It is (observation, reward, terminated, truncated, info).
            Only reward is meaningful in this case.

        """
        assert self.action_space.contains(action), f"bad action {action}"

        terminated = False
        truncated = False
        reward = self.np_random.normal(self.means[action], self.stds[action])
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self):
        """
        Returns a string description of reward distributions (means and standard deviations).
        """
        if self.render_mode == "ansi":
            render_one = lambda mu_sigma: f"{mu_sigma[0]:.2f}±{mu_sigma[1]:.2f}"
            return ", ".join(map(render_one, zip(self.means, self.stds)))
