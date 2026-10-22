"""Reference optimisers, all run through the same `Budget` wrapper.

Implemented here (compact, textbook versions):
    random search, simulated annealing, (1+1)-ES with the 1/5th rule,
    particle swarm optimisation, CMA-ES
From SciPy:
    differential evolution, dual annealing
"""
from __future__ import annotations

import numpy as np

from .core import Budget, BudgetExhausted, Result, reflect


def _run(name, body, f, lower, upper, budget):
    B = Budget(f, lower, upper, budget)
    try:
        body(B)
    except BudgetExhausted:
        pass
    return B.result(name)


def random_search(f, lower, upper, budget=10_000, seed=None) -> Result:
    rng = np.random.default_rng(seed)

    def body(B):
        while True:
            B.unit(rng.random(len(B.lower)))
    return _run("random-search", body, f, lower, upper, budget)


def simulated_annealing(f, lower, upper, budget=10_000, seed=None, t0_quantile=0.8,
                        step0=0.2, step1=0.002) -> Result:
    """One global temperature lowered on a geometric schedule (the classic setup)."""
    rng = np.random.default_rng(seed)

    def body(B):
        d = len(B.lower)
        x = rng.random(d)
        fx = B.unit(x)
        # calibrate T0 so a typical uphill move is accepted ~80% of the time at start
        probes = [abs(B.unit(rng.random(d)) - fx) for _ in range(20)]
        T0 = max(np.quantile(probes, t0_quantile) / -np.log(t0_quantile), 1e-12)
        T1 = T0 * 1e-4
        n = B.remaining
        for i in range(n):
            frac = i / n
            T = T0 * (T1 / T0) ** frac
            step = step0 * (step1 / step0) ** frac
            y = reflect(x + rng.normal(0, step / np.sqrt(d), d))
            fy = B.unit(y)
            if fy < fx or rng.random() < np.exp(-(fy - fx) / T):
                x, fx = y, fy
    return _run("simulated-annealing", body, f, lower, upper, budget)


def one_plus_one_es(f, lower, upper, budget=10_000, seed=None) -> Result:
    """(1+1)-ES with Rechenberg's 1/5th success rule and restarts on collapse."""
    rng = np.random.default_rng(seed)

    def body(B):
        d = len(B.lower)
        while True:
            x = rng.random(d)
            fx = B.unit(x)
            s = 0.3
            while s > 1e-9:
                y = reflect(x + s * rng.normal(size=d) / np.sqrt(d))
                fy = B.unit(y)
                if fy < fx:
                    x, fx = y, fy
                    s *= np.exp(0.8)
                else:
                    s *= np.exp(-0.2)
    return _run("(1+1)-ES", body, f, lower, upper, budget)


def pso(f, lower, upper, budget=10_000, seed=None, n=30, w=0.729, c1=1.494, c2=1.494) -> Result:
    rng = np.random.default_rng(seed)

    def body(B):
        d = len(B.lower)
        X = rng.random((n, d))
        V = (rng.random((n, d)) - 0.5) * 0.2
        F = np.array([B.unit(x) for x in X])
        P, PF = X.copy(), F.copy()
        while True:
            g = P[PF.argmin()]
            r1, r2 = rng.random((n, d)), rng.random((n, d))
            V = w * V + c1 * r1 * (P - X) + c2 * r2 * (g - X)
            V = np.clip(V, -0.25, 0.25)
            X = np.clip(X + V, 0, 1)
            for i in range(n):
                fi = B.unit(X[i])
                if fi < PF[i]:
                    P[i], PF[i] = X[i], fi
    return _run("PSO", body, f, lower, upper, budget)


def cma_es(f, lower, upper, budget=10_000, seed=None, sigma0=0.3) -> Result:
    """Compact CMA-ES (Hansen's purecma recipe) with IPOP restarts."""
    rng = np.random.default_rng(seed)

    def one_run(B, lam):
        d = len(B.lower)
        mu = lam // 2
        wts = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        wts /= wts.sum()
        mueff = 1 / np.sum(wts**2)
        cc = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
        cs = (mueff + 2) / (d + mueff + 5)
        c1 = 2 / ((d + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (d + 1)) - 1) + cs
        chiN = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d * d))
        m, sigma = rng.random(d), sigma0
        pc, ps, C = np.zeros(d), np.zeros(d), np.eye(d)
        gen = 0
        while sigma > 1e-12:
            gen += 1
            Dg, Bm = np.linalg.eigh(C)
            Dg = np.sqrt(np.maximum(Dg, 1e-20))
            Z = rng.normal(size=(lam, d))
            Y = Z * Dg @ Bm.T
            X = m + sigma * Y
            Xc = np.clip(X, 0, 1)
            F = np.array([B.unit(x) + 1e-8 * np.sum((x - xc) ** 2) for x, xc in zip(X, Xc)])
            idx = F.argsort()[:mu]
            yw = wts @ Y[idx]
            m = m + sigma * yw
            invsqrtC = Bm @ np.diag(1 / Dg) @ Bm.T
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ yw
            hs = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * gen)) / chiN < 1.4 + 2 / (d + 1)
            pc = (1 - cc) * pc + hs * np.sqrt(cc * (2 - cc) * mueff) * yw
            C = ((1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hs) * cc * (2 - cc) * C)
                 + cmu * (Y[idx].T * wts) @ Y[idx])
            C = (C + C.T) / 2
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            if np.ptp(F) < 1e-12 * (1 + abs(F.min())) and gen > 10:
                break

    def body(B):
        d = len(B.lower)
        lam = 4 + int(3 * np.log(d))
        while True:
            one_run(B, lam)
            lam *= 2
    return _run("CMA-ES", body, f, lower, upper, budget)


def differential_evolution(f, lower, upper, budget=10_000, seed=None) -> Result:
    from scipy.optimize import differential_evolution as de

    def body(B):
        rng = np.random.default_rng(seed)
        while True:     # restart if DE converges before the budget is spent
            de(B, list(zip(B.lower, B.upper)), seed=int(rng.integers(2**31)), maxiter=10**6,
               tol=1e-12, atol=0, polish=False, init="latinhypercube", popsize=15)
    return _run("DE (scipy)", body, f, lower, upper, budget)


def dual_annealing(f, lower, upper, budget=10_000, seed=None) -> Result:
    from scipy.optimize import dual_annealing as da

    def body(B):
        rng = np.random.default_rng(seed)
        while True:
            da(B, list(zip(B.lower, B.upper)), seed=int(rng.integers(2**31)), maxfun=B.remaining,
               no_local_search=False)
    return _run("dual-annealing (scipy)", body, f, lower, upper, budget)


ALL = {
    "random-search": random_search,
    "simulated-annealing": simulated_annealing,
    "(1+1)-ES": one_plus_one_es,
    "PSO": pso,
    "CMA-ES": cma_es,
    "DE (scipy)": differential_evolution,
    "dual-annealing (scipy)": dual_annealing,
}
