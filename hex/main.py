from typing import Any, Callable, Dict, List, Optional, Union

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from board import HexBoard, HexEnv


class Net(nn.Module):
    def conv(self, n_in, n_out):
        return nn.Conv2d(in_channels=n_in, out_channels=n_out, kernel_size=(3, 3))

    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.state_shape = state_shape
        self.action_shape = action_shape

        self.model = nn.Sequential(
            self.conv(6, 8),
            nn.ReLU(inplace=True),
            self.conv(8, 16),
            nn.ReLU(inplace=True),
            self.conv(16, 64),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, np.prod(action_shape)),
        )

    def forward(self, obs, state=None, _info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        batch = obs.shape[0]
        obs = obs.view((batch,) + self.state_shape)
        logits = self.model(obs)
        return logits, state
