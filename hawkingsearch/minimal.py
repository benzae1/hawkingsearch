"""Hawking search, minimal version: the core loop and nothing else.

Each candidate solution is a black hole with mass M.

    T = k / M                       Hawking temperature = mutation step length
    better candidate  -> accretion:  M += gain * improvement / scale
    no improvement    -> evaporation: M -= dt * alpha / M**2
                         dt = iterations since the hole last accreted ("starvation
                         clock": a feeding hole barely radiates, a starving one
                         radiates more and more), or dt = 1 if switched off
    M < M_min         -> final burst: b light holes scattered around x
    |xa - xb| < r_s(a) + r_s(b), r_s = c * M  -> merger

Search runs in the unit cube [0, 1]^d; distances and temperatures are in those
units, so the defaults work on any box.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import Budget, BudgetExhausted, Result, reflect


@dataclass(eq=False)
class Hole:
    x: np.ndarray       # position in the unit cube
    f: float            # objective value at x
    M: float            # mass
    starve: int = 0     # iterations since the hole last accreted


@dataclass
class MinimalConfig:
    n_holes: int = 20
    m0: float = 1.0           # initial mass
    k: float = 0.2            # T = k / M  (step length in unit-cube units)
    gain: float = 2.0         # accretion gain per unit of normalised improvement
    alpha: float = 0.01       # evaporation constant (with dt = 1, a stuck hole lives (M^3 - m_min^3) / 3 alpha steps)
    m_min: float = 0.3        # burst threshold
    m_max: float = 1e5        # numerical ceiling on mass
    burst_children: int = 3
    child_mass: float = 0.6
    rs_coeff: float = 0.004   # Schwarzschild radius r_s = rs_coeff * M
    rs_max: float = 0.08      # horizon radius saturates (keeps one hole from swallowing the box)
    max_holes_factor: float = 2.0
    starvation_clock: bool = True   # dt = iterations since last accretion (else dt = 1)


def temperature(M, k):
    return k / M


def evaporate(M, alpha, dt=1.0):
    """dM/dt = -alpha / M^2 (Hawking power ~ 1/M^2)."""
    return M - dt * alpha / M**2


def schwarzschild_radius(M, cfg):
    return min(cfg.rs_coeff * M, cfg.rs_max)


def minimize(f, lower, upper, budget=10_000, seed=None, config: MinimalConfig | None = None,
             callback=None) -> Result:
    cfg = config or MinimalConfig()
    rng = np.random.default_rng(seed)
    B = Budget(f, lower, upper, budget)
    d = len(B.lower)
    sq = np.sqrt(d)
    holes: list[Hole] = []
    max_holes = int(cfg.max_holes_factor * cfg.n_holes)
    it = 0
    try:
        for _ in range(cfg.n_holes):
            u = rng.random(d)
            holes.append(Hole(u, B.unit(u), cfg.m0))
        # fixed fitness scale so mass gain is unit-free
        scale = float(np.std([h.f for h in holes])) or 1.0

        while True:
            it += 1
            new: list[Hole] = []
            for h in holes:
                T = temperature(h.M, cfg.k)
                cand = reflect(h.x + rng.normal(0.0, T / sq, d))
                fc = B.unit(cand)
                if fc < h.f:                                        # accretion
                    dM = cfg.gain * (h.f - fc) / scale
                    h.M = min(h.M + min(dM, h.M), cfg.m_max)        # at most doubles per step
                    h.x, h.f = cand, fc
                    h.starve = 0
                else:                                               # evaporation
                    h.starve += 1
                    h.M = evaporate(h.M, cfg.alpha, h.starve if cfg.starvation_clock else 1.0)

                if h.M < cfg.m_min:                                 # final burst
                    room = max_holes - (len(holes) + len(new)) + 1
                    nb = max(1, min(cfg.burst_children, room))
                    Tb = temperature(cfg.m_min, cfg.k)
                    for j in range(nb):
                        u = reflect(h.x + rng.normal(0.0, Tb / sq, d))
                        child = Hole(u, B.unit(u), cfg.child_mass)
                        if j == 0:
                            h.x, h.f, h.M, h.starve = child.x, child.f, child.M, 0
                        else:
                            new.append(child)
            holes.extend(new)

            # mergers: closest pairs first
            merged = True
            while merged and len(holes) > 1:
                merged = False
                X = np.array([h.x for h in holes])
                D = np.linalg.norm(X[:, None] - X[None], axis=-1)
                R = np.array([schwarzschild_radius(h.M, cfg) for h in holes])
                hit = D < (R[:, None] + R[None])
                np.fill_diagonal(hit, False)
                if hit.any():
                    i, j = np.unravel_index(np.argmin(np.where(hit, D, np.inf)), D.shape)
                    a, b = holes[i], holes[j]
                    keep = a if a.f <= b.f else b
                    keep.M = min(a.M + b.M, cfg.m_max)
                    holes.pop(max(i, j))
                    holes[min(i, j)] = keep
                    merged = True

            # keep a working population (primordial holes from nowhere)
            while len(holes) < cfg.n_holes // 2:
                u = rng.random(d)
                holes.append(Hole(u, B.unit(u), cfg.m0))

            if callback:
                callback({"iteration": it, "nfev": B.nfev, "best_f": B.best_f,
                          "holes": [(B.to_real(h.x), h.f, h.M) for h in holes]})
    except BudgetExhausted:
        pass
    return B.result("hawking-minimal")
