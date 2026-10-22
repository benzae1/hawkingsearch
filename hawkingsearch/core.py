"""Shared plumbing: a budgeted objective wrapper and a result type.

Every optimiser in this package (and every baseline) calls the objective through
`Budget`, so all of them get exactly the same number of function evaluations
and their convergence traces are directly comparable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class BudgetExhausted(Exception):
    """Raised when an optimiser asks for more evaluations than it was given."""


class Budget:
    """Wraps f(x) on the original box; tracks evaluations and best-so-far."""

    def __init__(self, f, lower, upper, max_evals: int):
        self.f = f
        self.lower = np.asarray(lower, float)
        self.upper = np.asarray(upper, float)
        self.width = self.upper - self.lower
        self.max_evals = int(max_evals)
        self.nfev = 0
        self.best_x = None
        self.best_f = np.inf
        self.trace_evals: list[int] = []
        self.trace_best: list[float] = []

    @property
    def remaining(self) -> int:
        return self.max_evals - self.nfev

    def __call__(self, x) -> float:
        if self.nfev >= self.max_evals:
            raise BudgetExhausted
        x = np.clip(np.asarray(x, float), self.lower, self.upper)
        fx = float(self.f(x))
        self.nfev += 1
        if fx < self.best_f:
            self.best_f, self.best_x = fx, x.copy()
            self.trace_evals.append(self.nfev)
            self.trace_best.append(fx)
        return fx

    # Hawking search works in the unit cube; these helpers map in and out.
    def to_real(self, u):
        return self.lower + np.asarray(u) * self.width

    def to_unit(self, x):
        return (np.asarray(x) - self.lower) / self.width

    def unit(self, u) -> float:
        return self(self.to_real(u))

    def result(self, name: str, history=None) -> "Result":
        return Result(name, self.best_x, self.best_f, self.nfev,
                      np.array(self.trace_evals), np.array(self.trace_best), history or [])


@dataclass
class Result:
    algorithm: str
    x: np.ndarray
    f: float
    nfev: int
    trace_evals: np.ndarray
    trace_best: np.ndarray
    history: list = field(default_factory=list)

    def best_at(self, evals) -> np.ndarray:
        """Best-so-far value after each eval count in `evals` (for plotting)."""
        evals = np.atleast_1d(evals)
        idx = np.searchsorted(self.trace_evals, evals, side="right") - 1
        out = np.where(idx >= 0, self.trace_best[np.maximum(idx, 0)], np.nan)
        return out


def reflect(u: np.ndarray) -> np.ndarray:
    """Fold a point back into the unit cube (mirror at the walls)."""
    u = np.mod(u, 2.0)
    return np.where(u > 1.0, 2.0 - u, u)
