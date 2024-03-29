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
        elasticity: N_per_M,
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

    def contraction(
        self,
        position: np.ndarray,
    ) -> float:
        """
        Args:
            position (np.ndarray): Coordinates of the points in the system.

        Returns:
            float: Amount of attraction between i and j that this constraint produces.
        """
        cur = M(l2(position[self.j] - position[self.i]) + 1e-9)
        return self.elasticity * (cur - self.length)


class AngleConstraint:
    def __init__(
        self,
        i: int,
        m: int,
        j: int,
        min_angle: float,
        max_angle: float,
        elasticity_moment: N,
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

    def torque(
        self,
        position: np.ndarray,
    ) -> float:
        """
        Args:
            position (np.ndarray): Coordinates of points in the system.

        Returns:
            float: Torque that this constraint produces for given positions.
        """
        im = position[self.m] - position[self.i]
        mj = position[self.j] - position[self.m]
        alpha = turn(im, mj)
        return (
            max(self.min_angle - alpha, 0) + min(self.max_angle - alpha, 0)
        ) * self.elasticity_moment


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
        self.mass: np.ndarray = np.random.lognormal((self.n,))
        self.position: np.ndarray = np.random.normal(size=(self.n, 2))
        self.speed: np.ndarray = np.zeros((self.n, 2))
        self.lengths: list[LengthConstraint] = []
        self.angles: list[AngleConstraint] = []

    def add_length(
        self,
        i: int,
        j: int,
        length: M = None,
        elasticity: N_per_M = 1e5,
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
        elasticity_moment: N = 3e4,
    ):
        self.angles.append(
            AngleConstraint(i, m, j, min_angle, max_angle, elasticity_moment)
        )

    def E_gravity(self) -> float:
        return -sum(m * np.dot(pos, self.g) for m, pos in zip(self.mass, self.position))

    def E_elastic(self) -> float:
        elastic = lambda F, k: F**2 / (2 * k)
        lengths = 0
        for lc in self.lengths:
            contraction = lc.contraction(self.position)
            lengths += elastic(contraction, lc.elasticity)

        angles = 0
        for ac in self.angles:
            torque = ac.torque(self.position)
            angles += elastic(torque, ac.elasticity_moment)
        return lengths + angles

    def E_kinetic(self) -> float:
        return sum(m * np.dot(v, v) / 2 for m, v in zip(self.mass, self.speed))

    def accelerations(self) -> np.ndarray:
        a = np.repeat(self.g.reshape(1, 2), self.n, axis=0)
        for lc in self.lengths:
            contraction = lc.contraction(self.position)
            ij = self.position[lc.j] - self.position[lc.i]
            F_ij = ij / (l2(ij) + 1e-9) * contraction
            a[lc.i] += F_ij / self.mass[lc.i]
            a[lc.j] -= F_ij / self.mass[lc.j]

        for ac in self.angles:
            torque = ac.torque(self.position)
            im = self.position[ac.m] - self.position[ac.i]
            mj = self.position[ac.j] - self.position[ac.m]
            F_j = torque * perp(mj) / (l2(mj) + 1e-9) ** 2
            F_i = torque * perp(im) / (l2(im) + 1e-9) ** 2
            a[ac.j] += F_j / self.mass[ac.j]
            a[ac.i] += F_i / self.mass[ac.i]
            a[ac.m] -= (F_i + F_j) / self.mass[ac.m]

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
    def advance(old, acc, h):
        cur = deepcopy(old)
        cur.speed = np.clip(old.speed + acc * h, -5, 5)
        cur.position = old.position + (old.speed + cur.speed) / 2 * h
        return cur

    """ 
    Using 3/8 Runge-Kutta here:
    0   | 
    1/3 | 1/3
    2/3 | -1/3  1
     1  |  1   -1   1
     -------------------
        | 1/8  3/8 3/8 1/8
    Numbers on the left are not important, because here the system is independent of time.
    """
    k1 = system.accelerations()
    k2 = advance(system, k1, dt / 3).accelerations()
    k3 = advance(system, -0.5 * k1 + 1.5 * k2, dt * 2 / 3).accelerations()
    k4 = advance(system, k1 - k2 + k3, dt).accelerations()
    return advance(system, 0.125 * k1 + 0.375 * k2 + 0.375 * k3 + 0.125 * k4, dt)

    # for it in range(10):
    #     cur_acc = cur.accelerations()
    #     acc = [(a + b) / 2 for a, b in zip(old_acc, cur_acc)]
    #     cur = advance(old, acc)
    return cur

"""
No actual logic in this class yet. You can specify observations and rewards however you like.
"""
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


s = PointSystem(n=9, g=vec(0, -9.81))

masses = [
    1000,
    5,
    5,
    15,
    5,
    40,
    20,
    10,
    20,
]
coords = [
    vec(y=0.0),
    vec(y=0.0),
    vec(y=-0.4),
    vec(y=-0.8),
    vec(y=-0.4),
    vec(y=-1.28),
    vec(y=-1.78),
    vec(y=-2.28),
    vec(x=0.15, y=-0.8),
]
s.mass = np.array(masses)
s.position = np.array(coords)

""" 
Idx Name        Y       Mass
0   bar         0.0     1000
1   hands       0.0     5
2   elbows      -0.4    5
3   shoulders   -0.8    10
4   head        -0.4    5
5   hips        -1.28   40
6   knees       -1.78   20
7   feet        -2.28   10
8   arm helper  -0.8    20
"""
s.add_length(0, 1)
s.add_length(1, 2)
s.add_length(2, 3)
s.add_length(3, 4)
s.add_length(3, 8)
s.add_length(3, 5)
s.add_length(5, 6)
s.add_length(6, 7)
s.add_angle(1, 2, 3, rad(0), rad(120))  # arms elbows shoulders
s.add_angle(8, 3, 2, rad(-90), rad(120))  # helper shoulders elbows
s.add_angle(5, 3, 8, rad(-90), rad(-90))  # hips shoulders helper
s.add_angle(5, 3, 4, rad(-1), rad(1))  # hips shoulders head
s.add_angle(3, 5, 6, rad(-135), rad(10))  # shoulders hips knees
s.add_angle(5, 6, 7, rad(0), rad(120))  # hips kneees feet

simulated_time = 0.0
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
    mgh, kx2, mv2 = s.E_gravity(), s.E_elastic(), s.E_kinetic()
    print(
        f"{simulated_time:.3f}: {mgh:.3f} + {kx2:.3f} + {mv2:.3f} = {mgh + kx2 + mv2:.3f}"
    )
    h = 0.003
    for inner in range(10):
        s = simulate(s, h)
        simulated_time += h
    s.position[0] = vec()
    s.speed[0] = vec()
    sleep(0.05)
    if it < 20:
        s.speed += np.random.randn(*s.speed.shape) * 0.5
