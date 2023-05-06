from __future__ import annotations

from collections import deque
from itertools import product
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from einops import rearrange
from gymnasium import spaces

directions = frozenset([(0, -1), (0, 1), (-1, 0), (1, 0), (-1, 1), (1, -1)])


class HexBoard:
    """
    The board consists of actual play area (inner_size x inner_size) and padding (frame_size from each side).
    Players are denoted by 1 (first) and -1 (second).
    """

    def __init__(self, inner_size: int, frame_size: int) -> None:
        """
        Args:
        - inner_size: Size of the board.
        - frame_size: Size of the padding applied from each side.
        """
        self.inner_size = inner_size
        self.frame_size = frame_size
        self.total_size = inner_size + 2 * frame_size

        self.board = np.zeros((self.total_size, self.total_size), dtype=int)
        self.board[0 : self.frame_size, :] += 1
        self.board[-self.frame_size :, :] += 1
        self.board[:, 0 : self.frame_size] -= 1
        self.board[:, -self.frame_size :] -= 1

        self.top = [(0, j) for j in range(self.inner_size)]
        self.bottom = [(self.inner_size - 1, j) for j in range(self.inner_size)]
        self.left = [(i, 0) for i in range(self.inner_size)]
        self.right = [(i, self.inner_size - 1) for i in range(self.inner_size)]

    def inside_total(self, i: int, j: int) -> bool:
        """
        Args:
        - i: Row index.
        - j: Column index.

        Returns:
        - bool: -frame_size ≤ i, j < inner_size + frame_size
        """
        interval = range(-self.frame_size, self.inner_size + self.frame_size)
        return i in interval and j in interval

    def inside_inner(self, i: int, j: int) -> bool:
        """
        Args:
        - i: Row index.
        - j: Column index.

        Returns:
        - bool: 0 ≤ i, j < inner_size
        """
        return i in range(self.inner_size) and j in range(self.inner_size)

    def neighbours_list(self, i: int, j: int) -> list[tuple[int, int]]:
        """
        Args:
        - i: Row index, -frame_size ≤ i < inner_size + frame_size
        - j: Column index, -frame_size ≤ j < inner_size + frame_size

        Returns:
        - List of neighbours of the given position.
        """
        return [
            (i + di, j + dj)
            for di, dj in directions
            if self.inside_total(i + di, j + dj)
        ]

    def bfs(self, start: list[tuple[int, int]], player: int) -> np.ndarray:
        """
        Args:
        - start: List of starting positions in [0, inner_size) coordinates.
        - player: Player to whom these positions belong, ±1.

        Returns:
        - Boolean mask of positions reachable from the starting positions.
        """
        q = deque(filter(lambda pos: self.get(*pos) == player, start))
        used = np.full((self.total_size, self.total_size), fill_value=False)
        frame = self.frame_size
        for i, j in q:
            used[frame + i, frame + j] = True

        while q:
            i, j = q.popleft()
            assert self.get(i, j) == player and used[frame + i, frame + j]
            for ni, nj in self.neighbours_list(i, j):
                if self.get(ni, nj) == player and not used[frame + ni, frame + nj]:
                    used[frame + ni, frame + nj] = True
                    q.append((ni, nj))
        return used

    def current_player(self) -> int:
        """
        Returns the current player.

        Returns:
        - int: The current player.
        """
        balance = self.board.sum()
        return 1 if balance == 0 else -1

    def get(self, i: int, j: int) -> int:
        """
        Args:
        - i: Row index, 0 ≤ i < inner_size
        - j: Column index, 0 ≤ j < inner_size

        Returns:
        - Value at the given position.
        """
        assert self.inside_total(i, j), "position out of bounds"
        return self.board[i + self.frame_size][j + self.frame_size]

    def put(self, i: int, j: int, player: int | None = None) -> HexBoard:
        """
        Args:
        - i: Row index, 0 ≤ i < inner_size
        - j: Column index, 0 ≤ j < inner_size
        - player: Value to put. If None, the current player's value is used.
        """
        assert not self.get(i, j), "cell is not empty"
        player = player or self.current_player()
        self.board[i + self.frame_size][j + self.frame_size] = player
        return self

    def __str__(self) -> str:
        """
        Returns:
        - str: A string representation of the board.
        """
        from itertools import product

        n = self.total_size
        i_vec, j_vec = np.array([[2, 1], [0, 2]])
        shape = (2 * n - 1, 3 * n - 2)
        assert list(shape) == ((i_vec + j_vec) * (n - 1) + 1).tolist()
        result = np.full(shape, fill_value=" ")
        tokens = {0: ".", 1: "O", -1: "+"}
        for i, j in product(range(n), repeat=2):
            ti, tj = i * i_vec + j * j_vec
            result[ti][tj] = tokens[self.board[i][j]]
        return "\n".join(["".join(line) for line in result])

    def to_pieces(self) -> list[tuple[int, int, int]]:
        """
        Returns:
        - List of non-empty cells in (row, column, player) format.
        """
        n = self.inner_size
        full = [(i, j, self.get(i, j)) for i, j in product(range(n), repeat=2)]
        return [x for x in full if x[2]]

    def put_list(self, pieces: list[tuple[int, int, int]]) -> HexBoard:
        """
        Adds the given list of pieces to the board.

        Args:
        - pieces: List of non-empty cells in (row, column, player) format.
        """
        for i, j, c in pieces:
            self.put(i, j, c)
        return self

    def win(self) -> int:
        """
        Determines the winner of the game.

        Returns:
        - 1 if the first player wins, -1 if the second player wins, and 0 if there is no winner.
        """
        first = (self.bfs(self.top, 1) & self.bfs(self.bottom, 1)).any()
        second = (self.bfs(self.left, -1) & self.bfs(self.right, -1)).any()
        assert not (first and second)
        return int(first) - int(second)

    def to_features(self) -> np.ndarray:
        """
        Features are:
        - First player
        - Second player
        - Reachable from the top by first
        - Reachable from the bottom by first
        - Reachable from the left by second
        - Reachable from the right by second

        Returns:
        - (6, total_size, total_size) array of float32
        """
        return np.stack(
            [
                self.board == 1,
                self.board == -1,
                self.bfs(self.top, 1),
                self.bfs(self.bottom, 1),
                self.bfs(self.left, -1),
                self.bfs(self.right, -1),
            ]
        ).astype(np.float32)


class HexEnv(gym.Env):
    metadata = {"render.modes": ["console"]}

    def __init__(self, inner_size: int, frame_size: int) -> None:
        """
        Args:
        - inner_size: Board inner size.
        - frame_size: Board frame size
        """
        super(HexEnv, self).__init__()
        self.inner_size = inner_size
        self.frame_size = frame_size
        self.total_size = self.inner_size + 2 * self.frame_size
        self.action_space = spaces.MultiDiscrete(
            np.array([self.inner_size, self.inner_size])
        )
        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(6, self.total_size, self.total_size),
            dtype=np.float32,
        )
        self.board = HexBoard(self.inner_size, self.frame_size)

    def _get_obs(self) -> np.ndarray:
        """
        Returns:
        - The current state of the board.
        """
        return rearrange(
            self.board.to_features(),
            "c h w -> 1 c h w",
            c=6,
            h=self.total_size,
            w=self.total_size,
        )

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Args:
        - action: Coordinates of the move.

        Returns:
        - The next observation, reward, status, and info.
        """
        i, j = action
        assert self.board.inside_inner(i, j)
        terminated = False
        truncated = False
        reward = 0.0
        info: dict[str, Any] = {}
        if not self.board.get(i, j):
            self.board.put(i, j)
            w = self.board.win()
            terminated = w != 0
            reward = w
        return self._get_obs(), reward, terminated, truncated, info

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Args:
        - _seed: The seed to use.
        - _options: The options to use.

        Returns:
        - np.ndarray: The initial state of the board.
        """
        super().reset(seed=seed)
        self.board = HexBoard(self.inner_size, self.frame_size)
        return self._get_obs(), {}

    def render(self, mode: str = "console") -> None:
        """
        Args:
        - mode: The mode to use.
        """
        if mode != "console":
            raise NotImplementedError()
        print(self.board)

    def close(self) -> None:
        """
        Closes the environment.
        """
        pass
