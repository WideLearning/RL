from copy import deepcopy
from time import sleep
from typing import Any, Literal

import gymnasium as gym
import matplotlib as mpl
import numpy as np
from gymnasium import spaces

mpl.use("TkAgg")  # otherwise gymnasium conflicts with matplotlib

from matplotlib import pyplot as plt

M = float  # length
Kg = float  # mass
N = float  # force
Sec = float  # time
N_per_M = float  # elasticity
Vector = np.ndarray[tuple[Literal[2]], np.dtype[np.float_]]  # 2D vectors


def vec(x: float = 0.0, y: float = 0.0) -> Vector:
    return np.array([x, y], dtype=float)


def perp(a: Vector) -> Vector:
    """
    Args:
        a (Vector): Given vector.

    Returns:
        Vector: Perpendicular for it on the left side.
    """
    return vec(-a[1], a[0])


def l2(a: Vector) -> float:
    return float(np.linalg.norm(a))


def rad(alpha: float) -> float:
    return alpha * np.pi / 180


def turn(a: Vector, b: Vector) -> float:
    """Difference of polar angles of b and a in radians.
    E.g.: turn((1, 0), (0, 1)) = pi / 2

    Args:
        a (Vector): First vector.
        b (Vector): Second vector.

    Returns:
        float: atan2(a x b, a * b)
    """
    vec_product = a[0] * b[1] - a[1] * b[0]
    dot_product = a[0] * b[0] + a[1] * b[1]
    return np.arctan2(vec_product, dot_product)


class LengthConstraint:
    def __init__(
        self,
        i: int,
        j: int,
        length: M,
        elasticity: N_per_M = 5e4,
    ):
        """Equivalent to an elastic spring between i-th and j-th point.

        Args:
            i (int): Index of the first endpoint.
            j (int): Index of the second endpoint.
            length (M, optional): Base length.
            elasticity (N_per_M, optional): It is k in F = k(l - l_0). Defaults to 5e4.
        """
        self.i = i
        self.j = j
        self.length = length
        self.elasticity = elasticity


class AngleConstraint:
    def __init__(
        self,
        i: int,
        m: int,
        j: int,
        min_angle: float,
        max_angle: float,
        elasticity_moment: N = 1e4,
    ):
        """Soft constraint on turn(vec(p[m] - p[i]), vec(p[j] - p[m])) to be in [min_angle, max_angle].

        Args:
            i (int): First endpoint.
            m (int): Middle point.
            j (int): Second endpoint.
            min_angle (float): Lower bound for angle, in radians.
            max_angle (float): Upper bound for angle, in radians.
            elasticity_moment (N): Product of elasticity (for constaint violation) and length from the middle point.
        """
        self.i = i
        self.m = m
        self.j = j
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.elasticity_moment = elasticity_moment


class PointSystem:
    def __init__(
        self,
        n: int,
        g: Vector,
    ):
        """Simulates a system of material points with length and angle constaints.

        Args:
            n (int): Number of points.
            g (Vector): Extra acceleration, e.g. gravity.
        """
        self.n = n
        self.g = g
        self.mass: list[Kg] = [np.random.lognormal() for i in range(n)]
        self.position: list[Vector] = [np.random.normal(size=(2,)) for i in range(n)]
        self.speed: list[Vector] = [
            np.random.normal(size=(2,), scale=0.9) for i in range(n)
        ]
        self.lengths: list[LengthConstraint] = []
        self.angles: list[AngleConstraint] = []

    def add_length(
        self,
        i: int,
        j: int,
        length: M = None,
        elasticity: N_per_M = 5e4,
    ) -> None:
        actual_length = length or 0.0
        if length is None:
            p_i = self.position[i]
            p_j = self.position[j]
            actual_length = l2(p_j - p_i)
        self.lengths.append(LengthConstraint(i, j, actual_length, elasticity))

    def add_angle(
        self,
        i: int,
        m: int,
        j: int,
        min_angle: float,
        max_angle: float,
        elasticity_moment: N = 1e4,
    ):
        self.angles.append(
            AngleConstraint(i, m, j, min_angle, max_angle, elasticity_moment)
        )

    def E_pot(self) -> float:
        return -sum(m * np.dot(pos, self.g) for m, pos in zip(self.mass, self.position))

    def E_kin(self) -> float:
        return sum(m * np.dot(v, v) / 2 for m, v in zip(self.mass, self.speed))

    def accelerations(self) -> list[Vector]:
        a = [self.g.copy() for i in range(self.n)]
        for lc in self.lengths:
            i = lc.i
            j = lc.j
            p_i = self.position[i]
            p_j = self.position[j]
            base = lc.length
            cur = M(l2(p_j - p_i) + 1e-9)
            contraction = lc.elasticity * (cur - base)
            F_ij = (p_j - p_i) / cur * contraction
            a[i] += F_ij / self.mass[i]
            a[j] -= F_ij / self.mass[j]

        for ac in self.angles:
            i = ac.i
            m = ac.m
            j = ac.j
            im = self.position[m] - self.position[i]
            mj = self.position[j] - self.position[m]
            lef = ac.min_angle
            rig = ac.max_angle
            alpha = turn(im, mj)
            torque = (max(lef - alpha, 0) + min(rig - alpha, 0)) * ac.elasticity_moment

            F_j = torque * perp(mj) / (l2(mj) + 1e-9) ** 2
            F_i = torque * perp(im) / (l2(im) + 1e-9) ** 2
            a[j] += F_j / self.mass[j]
            a[i] += F_i / self.mass[i]
            a[m] -= (F_i + F_j) / self.mass[m]

        return a

    def render(self, ax):
        for i in range(self.n):
            p = self.position[i]
            ax.plot([p[0]], [p[1]], "ok", ms=5)
        for lc in self.lengths:
            a = self.position[lc.i]
            b = self.position[lc.j]
            ax.plot([a[0], b[0]], [a[1], b[1]], "-k", lw=3, alpha=0.5)


def simulate(
    system: PointSystem,
    dt: Sec,
):
    old = deepcopy(system)
    cur = deepcopy(system)
    old_acc = old.accelerations()
    friction = 0.99
    for it in range(10):
        cur_acc = cur.accelerations()
        acc = [(a + b) / 2 for a, b in zip(old_acc, cur_acc)]
        cur.speed = [friction * (v + a * dt) for v, a in zip(old.speed, acc)]
        cur.position = [
            x + (v_0 + v_1) / 2 * dt
            for x, v_0, v_1 in zip(old.position, old.speed, cur.speed)
        ]
    return cur


class Pullup(gym.Env):
    metadata: dict[str, Any] = {"render_modes": ["human"]}

    def __init__(
        self,
        render_mode: str | None = None,
    ):
        render_modes = [None] + self.metadata["render_modes"]
        assert render_mode in render_modes, f"render_mode = {render_mode}"
        self.render_mode = render_mode
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,))
        self.action_space = spaces.MultiDiscrete([2, 2])

    def _get_obs(self) -> np.ndarray:
        return np.zeros(1, dtype=np.float32)

    def _get_info(self) -> dict[str, Any]:
        return {}

    def reset(self, seed: int = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        # reset the state here if needed
        observation = self._get_obs()
        info = self._get_info()
        self.render()
        return observation, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self.action_space.contains(action), f"bad action {action}"

        terminated = False
        truncated = False
        reward = 0.0
        observation = self._get_obs()
        info = self._get_info()

        self.render()

        return observation, reward, terminated, truncated, info

    def render(self) -> None:
        plt.clf()
        plt.cla()
        self.system.render(plt.gca())


s = PointSystem(n=8, g=vec(0, -9.81))
""" 
Idx Name        Y       Mass
0   bar         0.0     1000
1   hands       0.0     5
2   elbows      -0.4    5
3   shoulders   -0.8    30
4   head        -0.4    10
5   hips        -1.28   30
6   knees       -1.78   20
7   feet        -2.28   10
"""

s.mass = list(
    map(
        Kg,
        [
            1000,
            5,
            5,
            30,
            10,
            30,
            20,
            10,
        ],
    )
)
s.position = list(
    map(
        lambda y: vec(y=y),
        [
            0.0,
            0.0,
            -0.4,
            -0.8,
            -0.4,
            -1.28,
            -1.78,
            -2.28,
        ],
    )
)

s.add_length(0, 1)
s.add_length(1, 2)
s.add_length(2, 3)
s.add_length(3, 4)
s.add_length(3, 5)
s.add_length(5, 6)
s.add_length(6, 7)


for it in range(10**9):
    s.render(plt.gca())
    plt.xlim(-2, 2)
    plt.ylim(-3, 1)
    plt.savefig("render.png")
    plt.clf()
    # for x in s.accelerations():
    #     print(np.round(x, 3), end=" ")
    # print()
    # for x in s.position:
    #     print(np.round(x, 3), end=" ")
    # print()
    # print()
    print(s.E_pot(), s.E_kin())
    s = simulate(s, 0.01)
    s = simulate(s, 0.01)
    s = simulate(s, 0.01)
    s.position[0] = vec()
    sleep(0.17)
