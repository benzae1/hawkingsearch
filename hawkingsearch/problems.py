"""Benchmark problems: classic test functions plus objectives built on real data.

Every problem is a minimisation problem with box bounds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable

import numpy as np


@dataclass
class Problem:
    name: str
    f: Callable[[np.ndarray], float]
    lower: np.ndarray
    upper: np.ndarray
    f_opt: float | None = None          # known optimum, if any
    description: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def dim(self) -> int:
        return len(self.lower)

    @property
    def bounds(self) -> np.ndarray:
        return np.stack([self.lower, self.upper], axis=1)

    def __call__(self, x) -> float:
        return float(self.f(np.asarray(x, dtype=float)))


def _box(d, lo, hi):
    return np.full(d, float(lo)), np.full(d, float(hi))


# ----------------------------------------------------------------- synthetic
def sphere(d=10):
    return Problem("sphere", lambda x: float(np.sum(x**2)), *_box(d, -5.12, 5.12), 0.0,
                   "Convex bowl. Tests pure refinement.")


def rosenbrock(d=10):
    def f(x):
        return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))
    return Problem("rosenbrock", f, *_box(d, -5, 10), 0.0, "Curved narrow valley.")


def rastrigin(d=10):
    def f(x):
        return float(10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))
    return Problem("rastrigin", f, *_box(d, -5.12, 5.12), 0.0, "Regular grid of ~10^d local minima.")


def ackley(d=10):
    def f(x):
        n = len(x)
        return float(-20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / n))
                     - np.exp(np.sum(np.cos(2 * np.pi * x)) / n) + 20 + np.e)
    return Problem("ackley", f, *_box(d, -32.768, 32.768), 0.0, "Flat plateau with a deep central funnel.")


def griewank(d=10):
    def f(x):
        i = np.arange(1, len(x) + 1)
        return float(1 + np.sum(x**2) / 4000 - np.prod(np.cos(x / np.sqrt(i))))
    return Problem("griewank", f, *_box(d, -600, 600), 0.0, "Wide bowl with fine ripples.")


def schwefel(d=10):
    def f(x):
        return float(418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x)))))
    return Problem("schwefel", f, *_box(d, -500, 500), 0.0,
                   "Deceptive: the best minimum is far from the second best.")


# ----------------------------------------------------------------- real data
@lru_cache(maxsize=1)
def _dem():
    import matplotlib.cbook as cbook
    data = cbook.get_sample_data("jacksboro_fault_dem.npz")
    z = np.asarray(data["elevation"], dtype=float)
    meta = {k: float(data[k]) for k in ("xmin", "xmax", "ymin", "ymax", "dx", "dy")}
    return z, meta


def terrain():
    """Find the highest summit on a real USGS elevation model.

    Data: 30 m DEM of the Jacksboro Fault region, Tennessee (bundled with
    matplotlib). Search space is (column, row) in pixel units; the objective is
    the negated elevation in metres, bilinearly interpolated.
    """
    from scipy.ndimage import map_coordinates
    z, meta = _dem()
    rows, cols = z.shape

    def f(x):
        # x = (col, row)
        v = map_coordinates(z, [[x[1]], [x[0]]], order=1, mode="nearest")[0]
        return -float(v)

    return Problem("terrain", f, np.array([0.0, 0.0]), np.array([cols - 1.0, rows - 1.0]),
                   -float(z.max()), "Real elevation map: find the highest summit (72 local summits).",
                   meta={"elevation": z, **meta})


def _kmeans_problem(name, X, k, desc):
    X = (X - X.mean(0)) / X.std(0)
    n, p = X.shape

    def f(v):
        C = v.reshape(k, p)
        d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        return float(d2.min(1).sum() / n)

    lo = np.tile(X.min(0), k)
    hi = np.tile(X.max(0), k)
    # reference optimum: best of many k-means++ restarts
    from sklearn.cluster import KMeans
    km = KMeans(k, n_init=50, random_state=0).fit(X)
    return Problem(name, f, lo, hi, float(km.inertia_ / n), desc, meta={"X": X, "k": k})


@lru_cache(maxsize=None)
def iris_kmeans():
    from sklearn.datasets import load_iris
    return _kmeans_problem("iris-kmeans", load_iris().data, 3,
                           "Place 3 cluster centres in the 4-D Iris data (12 variables).")


@lru_cache(maxsize=None)
def wine_kmeans():
    from sklearn.datasets import load_wine
    return _kmeans_problem("wine-kmeans", load_wine().data, 3,
                           "Place 3 cluster centres in the 13-D Wine data (39 variables).")


def get(name: str, dim: int = 10) -> Problem:
    synthetic = {"sphere": sphere, "rosenbrock": rosenbrock, "rastrigin": rastrigin,
                 "ackley": ackley, "griewank": griewank, "schwefel": schwefel}
    if name in synthetic:
        return synthetic[name](dim)
    return {"terrain": terrain, "iris-kmeans": iris_kmeans, "wine-kmeans": wine_kmeans}[name]()


ALL = ["sphere", "rosenbrock", "rastrigin", "ackley", "griewank", "schwefel",
       "terrain", "iris-kmeans", "wine-kmeans"]
