from copy import deepcopy
from random import random
from time import sleep, time
from typing import Any, Literal

import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np

mpl.use("TkAgg")  # otherwise gymnasium conflicts with matplotlib

from matplotlib import pyplot as plt

M = float  # length
Kg = float  # mass
N = float  # force
Sec = float  # time
N_per_M = float  # elasticity
Vector = np.ndarray[tuple[Literal[2]], np.dtype[np.float_]]  # 2D vectors

NDIM = 3


def vec(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> Vector:
    return np.array([x, y, z], dtype=float)


def l2(a: Vector) -> float:
    return float(np.linalg.norm(a))


class PointSystem:
    def __init__(
        self,
        n: int,
        m: int,
        g: Vector,
    ):
        """Simulates a system of material points with length constaints.

        Args:
            n (int): Number of points.
            m (int): Number of constraints.
            g (Vector): Extra acceleration, e.g. gravity.
        """
        self.n = n
        self.m = m
        self.g = g
        self.mass: np.ndarray = np.random.lognormal((self.n,))
        self.position: np.ndarray = np.random.normal(size=(self.n, NDIM))
        self.speed: np.ndarray = np.zeros((self.n, NDIM))
        self.l_i: np.ndarray = np.zeros((self.m,), dtype=np.int64)
        self.l_j: np.ndarray = np.zeros((self.m,), dtype=np.int64)
        self.l_l: np.ndarray = np.zeros((self.m,))
        self.l_e: np.ndarray = np.zeros((self.m,))
        self.constraints = 0

    def add_length(
        self,
        i: int,
        j: int,
        length: M = None,
        elasticity: N_per_M = 1e3,  # ???
    ) -> None:
        assert self.constraints < self.m
        if length is None:
            length = l2(self.position[i] - self.position[j])
        self.l_i[self.constraints] = i
        self.l_j[self.constraints] = j
        self.l_l[self.constraints] = length
        self.l_e[self.constraints] = elasticity
        self.constraints += 1

    def E_gravity(self) -> float:
        return -sum(m * np.dot(pos, self.g) for m, pos in zip(self.mass, self.position))

    def E_elastic(self) -> float:
        current_length = (
            np.linalg.norm(self.position[self.l_i] - self.position[self.l_j], axis=1)
            + 1e-9
        )
        contraction = self.l_e * (current_length - self.l_l)
        energy = contraction**2 / (2 * self.l_e)
        return energy.sum()

    def E_kinetic(self) -> float:
        return sum(m * np.dot(v, v) / 2 for m, v in zip(self.mass, self.speed))

    def accelerations(self) -> np.ndarray:
        a = np.repeat(self.g.reshape(1, NDIM), self.n, axis=0)
        dpos = self.position[self.l_j] - self.position[self.l_i]
        dspeed = self.speed[self.l_j] - self.speed[self.l_i]
        current_length = np.linalg.norm(dpos, axis=1) + 1e-9
        contraction = self.l_e * (current_length - self.l_l)
        F_ij = dpos * (contraction / current_length).reshape(self.m, 1)
        friction = 10.0 * np.sign(np.einsum("ij,ij->i", F_ij, dspeed))
        F_ij += friction.reshape(self.m, 1) * F_ij
        a[self.l_i] += F_ij / self.mass[self.l_i].reshape(self.m, 1)
        a[self.l_j] -= F_ij / self.mass[self.l_j].reshape(self.m, 1)
        return a

    def render(self, ax, axes):
        def xyc(p):
            return [p[axes[0]], p[axes[1]], (p[axes[2]] + 1) / 2]

        cmap = cm.get_cmap("jet")
        for i in range(self.n):
            x, y, c = xyc(self.position[i])
            ax.plot([x], [y], "o", ms=5, color=cmap(c))
        for i, j in zip(self.l_i, self.l_j):
            a = xyc(self.position[i])
            b = xyc(self.position[j])
            ax.plot([a[0], b[0]], [a[1], b[1]], "-k", lw=1, alpha=0.5)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)


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


s = PointSystem(n=6, m=13, g=vec(z=-0.0))

masses = [
    1,
    1,
    1,
    1,
    1,
    1,
]
coords = [
    vec(1.0, 0.0),
    vec(0.0, 1.0),
    vec(-1.0, 0.0),
    vec(0.0, -1.0),
    vec(z=0.1),
    vec(z=-0.1),
]
s.mass = np.array(masses)
s.position = np.array(coords)

for i in range(4):
    s.add_length(i, (i + 1) % 4)
    s.add_length(i, 4)
    s.add_length(i, 5)
s.add_length(4, 5)
a0 = vec(z=0.1)
s.speed[0] += a0
s.speed[2] -= a0

simulated_time = 0.0
t0 = time()
for it in range(10**9):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    s.render(plt.gca(), (0, 2, 1))
    plt.subplot(1, 2, 2)
    s.render(plt.gca(), (0, 1, 1))
    plt.tight_layout()
    plt.savefig("render.png")
    plt.clf()
    with open("coordinates.txt", "w") as f:
        for pos in s.position:
            print(*pos, file=f)

    if it % 10 == 0:
        mgh, kx2, mv2 = s.E_gravity(), s.E_elastic(), s.E_kinetic()
        elapsed_time = time() - t0
        print(
            f"{simulated_time:.3f}, x{simulated_time/elapsed_time:.3f}: {mgh:.3f} + {kx2:.3f} + {mv2:.3f} = {mgh + kx2 + mv2:.3f}"
        )
    h = 0.001
    for inner in range(100):
        s = simulate(s, h)
        simulated_time += h
