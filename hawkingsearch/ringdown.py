"""Damped relaxation in a learned local potential, in unit-cube coordinates.

This is a black-hole ringdown *analogy*, not a GR simulation: paired probes
measure the potential gradient and diagonal tidal curvature; BFGS secants learn
coupled curvature. A horizon-sized trust radius and dissipative line search
keep relaxation local. Every probe uses the caller's objective budget.
"""
from __future__ import annotations

import numpy as np


def ringdown(evaluate, x, fx, max_evals, radius=0.1):
    """Return the best measured point, value, and evaluation count.

    Uses at most ``max_evals`` calls; accepts a known initial value for free.
    Central probes become one-sided at box walls. No objective derivatives
    are required. Memory is O(d^2), so this is intended for modest dimensions.
    """
    x = np.asarray(x, float).copy()
    d = len(x)
    best_x, best_f = x.copy(), float(fx)
    used = 0
    radius = float(np.clip(radius, 1e-6, 0.25))

    def sample(u):
        nonlocal used, best_x, best_f
        value = float(evaluate(u))
        used += 1
        if value < best_f:
            best_x, best_f = u.copy(), value
        return value

    def field(u, value):
        gradient, curvature = np.zeros(d), np.zeros(d)
        for j in range(d):
            # Move probes inside at a wall instead of reflecting the potential.
            h = 1e-6
            a, b = min(h, u[j]), min(h, 1.0 - u[j])
            if a > h * 0.01 and b > h * 0.01:
                minus, plus = u.copy(), u.copy()
                minus[j] -= a
                plus[j] += b
                fm, fp = sample(minus), sample(plus)
                gradient[j] = (a * (fp - value) / b + b * (value - fm) / a) / (a + b)
                curvature[j] = 2 * ((fp - value) / b - (value - fm) / a) / (a + b)
            else:
                step = h if b >= a else -h
                p, q = u.copy(), u.copy()
                p[j] += step
                q[j] += 2 * step
                fp, fq = sample(p), sample(q)
                gradient[j] = (4 * (fp - value) - (fq - value)) / (2 * step)
                curvature[j] = (fq - 2 * fp + value) / step**2
        return gradient, curvature

    if max_evals < 2 * d + 1:
        return best_x, best_f, used
    g, diagonal = field(x, fx)
    positive = diagonal[np.isfinite(diagonal) & (diagonal > 0)]
    scale = float(np.median(positive)) if len(positive) else 1.0
    diagonal = np.where(np.isfinite(diagonal), diagonal, scale)
    H = np.diag(1.0 / np.maximum(diagonal, max(scale * 1e-6, 1e-12)))
    while used < max_evals:
        projected = g.copy()
        projected[(x <= 0) & (g > 0) | (x >= 1) & (g < 0)] = 0
        if not np.all(np.isfinite(projected)) or np.linalg.norm(projected, np.inf) < 1e-10:
            break
        step = -H @ projected
        step[(x <= 0) & (step < 0) | (x >= 1) & (step > 0)] = 0
        if g @ step >= 0:
            step = -projected / max(scale, 1e-12)
        length = np.linalg.norm(step)
        if length < 1e-14:
            break
        step *= min(1.0, radius / length)
        accepted = False
        for _ in range(min(12, max_evals - used)):
            trial = np.clip(x + step, 0, 1)
            displacement = trial - x
            if np.linalg.norm(displacement) < 1e-14:
                break
            ft = sample(trial)
            if np.isfinite(ft) and ft < fx and ft <= fx + 1e-4 * (g @ displacement):
                accepted = True
                break
            step *= 0.5
        if not accepted or max_evals - used < 2 * d:
            break
        new_g, _ = field(trial, ft)
        s, y = displacement, new_g - g
        sy = float(s @ y)
        if sy > 1e-10 * np.linalg.norm(s) * np.linalg.norm(y) and np.isfinite(sy):
            # Positive secant curvature preserves a positive definite inverse tensor.
            rho = 1.0 / sy
            V = np.eye(d) - rho * np.outer(s, y)
            H = V @ H @ V.T + rho * np.outer(s, s)
            H = (H + H.T) * 0.5
        if np.linalg.norm(s) > 0.8 * radius:
            radius = min(1.5 * radius, 0.25)
        x, fx, g = trial, ft, new_g
    return best_x, best_f, used
