from collections import deque
from itertools import product
from typing import Optional
import numpy as np

directions = frozenset([(0, -1), (0, 1), (-1, 0), (1, 0), (-1, 1), (1, -1)])

"""
TODO:
- Rewrite all coordinates to 0...inner-1 and -frame...inner+frame-1
"""


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
        self.from_pieces(inner_size, frame_size, [])

    def inside(self, i: int, j: int) -> bool:
        """
        Args:
        - i: The row index.
        - j: The column index.

        Returns:
        - bool: 0 ≤ i, j < total_size
        """
        return i in range(0, self.total_size) and j in range(0, self.total_size)

    def neighbours_list(self, i: int, j: int) -> list[(int, int)]:
        """
        Args:
        - i: Row index.
        - j: Column index.

        Returns:
        - List of neighbours of the given position.
        """
        return [
            (i + di, j + dj) for di, dj in directions if self.inside(i + di, j + dj)
        ]

    def bfs(self, start: list[(int, int)], player: int) -> np.ndarray:
        """
        Args:
        - start: List of starting positions in [0, inner_size) coordinates.
        - player: Player to whom these positions belong, ±1.

        Returns:
        - Boolean array indicating which positions are reachable from the starting positions.
        """
        q = deque(filter(lambda pos: self.get(*pos) == player, start))
        used = np.fromfunction(
            lambda i, j: (i, j) in q, (self.total_size, self.total_size), dtype=bool
        )

        while q:
            i, j = q.popleft()
            assert self.get(i, j) == player and used[i][j]
            for ni, nj in self.neighbours_list(i, j):
                if self.get(ni, nj) != player or used[ni][nj]:
                    continue
                used[ni][nj] = True
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
        assert self.inside(
            i + self.frame_size, j + self.frame_size
        ), "position out of bounds"
        return self.board[i + self.frame_size][j + self.frame_size]

    def put(self, i: int, j: int, player: Optional[int] = None) -> None:
        """
        Args:
        - i: Row index, 0 ≤ i < inner_size
        - j: Column index, 0 ≤ j < inner_size
        - player: Value to put. If None, the current player's value is used.
        """
        assert not self.get(i, j), "cell is not empty"
        self.board[i + self.frame_size][j + self.frame_size] = (
            player or self.current_player()
        )

    def __str__(self) -> str:
        """
        Returns:
        - str: A string representation of the board.
        """
        from itertools import product

        n = self.total_size
        i_vec, j_vec = np.array([[2, 1], [0, 2]])
        assert [2 * n - 1, 3 * n - 2] == ((i_vec + j_vec) * (n - 1) + 1).tolist()
        result = np.full((2 * n - 1, 3 * n - 2), fill_value=" ")
        tokens = {0: ".", 1: "O", -1: "+"}
        for i, j in product(range(n), repeat=2):
            ti, tj = i * i_vec + j * j_vec
            result[ti][tj] = tokens[self.board[i][j]]
        return "\n".join(["".join(line) for line in result])

    def to_pieces(self) -> list[(int, int, int)]:
        """
        Returns:
        - List of non-empty cells in (row, column, player) format.
        """
        n = self.inner_size
        full = [(i, j, self.get(i, j)) for i, j in product(range(n), repeat=2)]
        return [x for x in full if x[2]]

    def from_pieces(
        self, inner_size: int, frame_size: int, pieces: list[(int, int, int)]
    ) -> None:
        """
        Initializes the board from a list of non-empty positions and their values.

        Args:
        - inner_size: Size of the board.
        - frame_size: Size of the frame.
        - history: List of non-empty cells in (row, column, player) format.
        """
        self.inner_size = inner_size
        self.frame_size = frame_size
        self.total_size = inner_size + 2 * frame_size
        self.board = np.zeros((self.total_size, self.total_size), dtype=int)
        self.board[0 : self.frame_size, :] += 1
        self.board[-self.frame_size :, :] += 1
        self.board[:, 0 : self.frame_size] -= 1
        self.board[:, -self.frame_size :] -= 1
        for i, j, c in pieces:
            self.put(i, j, c)

        self.up = [
            (self.frame_size, self.frame_size + j) for j in range(self.inner_size)
        ]
        self.down = [
            (self.frame_size + self.inner_size - 1, self.frame_size + j)
            for j in range(self.inner_size)
        ]
        self.left = [
            (self.frame_size + i, self.frame_size) for i in range(self.inner_size)
        ]
        self.right = [
            (self.frame_size + i, self.frame_size + self.inner_size - 1)
            for i in range(self.inner_size)
        ]

    def win(self) -> int:
        """
        Determines the winner of the game.

        Returns:
        - 1 if the first player wins, -1 if the second player wins, and 0 if there is no winner.
        """
        first = (self.bfs(self.up, 1) & self.bfs(self.down, 1)).any()
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
                self.bfs(self.up, 1),
                self.bfs(self.down, 1),
                self.bfs(self.left, -1),
                self.bfs(self.right, -1),
            ]
        ).astype(np.float32)


import gymnasium as gym
from gymnasium import spaces


class HexEnv(gym.Env):
    metadata = {"render.modes": ["console"]}

    def __init__(self, n, frame):
        super(HexEnv, self).__init__()
        self.n = n
        self.frame = frame
        self.s = self.n + 2 * self.frame
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.n, self.n),
        )
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(6, 7, 7), dtype=np.float32
        )
        self.board = HexBoard(self.n, self.frame)

    def _get_obs(self):
        return self.board.to_features().reshape((1,) + self.observation_space.shape)

    def step_helper(self, action):
        i, j = np.unravel_index(action, (self.n, self.n))
        done = False
        reward = -1
        info = {}
        if not self.board.get(i, j):
            self.board.put(i, j)
            w = self.board.win()
            done = w != 0
            reward = w
        return self._get_obs(), reward, done, info

    def step(self, action):
        obs, reward, done, info = self.step_helper(action)
        return (obs, reward, done, info) if done else self.step_helper(action)

    def reset(self, _seed: None = None, _options: None = None):
        self.board = HexBoard(self.n, self.frame)
        return self._get_obs()

    def render(self, mode="console"):
        if mode != "console":
            raise NotImplementedError()
        print(self.board)

    def close(self):
        pass

    def seed(self, seed):
        np.random.seed(seed)
