"""Controlled study of the per-hole Hawking cooling rule (paper experiment).

Every variant runs the same loop as `hawkingsearch.minimal` (population of 20,
Gaussian steps in the unit cube, reflection at the walls, a 3-child burst
around a dying searcher, population cap 40). Mergers are off in all variants so
that the only difference is how each searcher sets its step length and when it
gives up:

    hawking        T = k/M, accretion cools, evaporation (starvation clock) heats,
                   burst when M < M_min                        (heat: yes, patience: earned)
    hawking-dt1    same with dt = 1 (lifetime ~ M^3 instead of ~ M^1.5)
    heat-fixedL    evaporation heats as in hawking, but the burst comes after L
                   consecutive failures                         (heat: yes, patience: fixed)
    cool-earned    burst timing identical to hawking, but T = k / (mass at last
                   success): no reheating                       (heat: no,  patience: earned)
    cool-fixedL    T = k / (mass at last success), burst after L failures
                                                                (heat: no,  patience: fixed)
    onefifth-L     per-searcher 1/5 success rule, burst after L failures
    onefifth-earned  1/5 success rule for the step; burst timing from the
                   evaporating mass, as in hawking
    fixed-L        constant step k/m0, burst after L failures

Each variant gets one tuned parameter, chosen on screening seeds (100-104) by
mean rank from a four-value grid: L for the fixed-patience variants, the
evaporation constant alpha for the others. The reported comparison uses seeds 0-14.

    python paper/cooling_controls.py            # screen + final, writes paper/results/
"""
from __future__ import annotations

import csv
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hawkingsearch import problems                                    # noqa: E402
from hawkingsearch.core import Budget, BudgetExhausted, reflect       # noqa: E402
from hawkingsearch.minimal import MinimalConfig, evaporate, schwarzschild_radius  # noqa: E402

CFG = MinimalConfig()
L_GRID = [8, 30, 100, 300]
ALPHA_GRID = [0.001, 0.003, 0.01, 0.03]
FIXED_L = ["heat-fixedL", "cool-fixedL", "onefifth-L", "fixed-L"]
EARNED = ["hawking", "hawking-dt1", "cool-earned", "onefifth-earned"]
VARIANTS = EARNED + FIXED_L
GRID = {v: (L_GRID if v in FIXED_L else ALPHA_GRID) for v in VARIANTS}


class Searcher:
    __slots__ = ("x", "f", "M", "M_cool", "sigma", "starve")

    def __init__(self, x, f, M):
        self.x, self.f, self.M, self.M_cool = x, f, M, M
        self.sigma = CFG.k / M          # used by onefifth / fixed
        self.starve = 0


def step_length(h, rule):
    if rule in ("hawking", "hawking-dt1", "heat-fixedL"):
        return CFG.k / h.M
    if rule in ("cool-earned", "cool-fixedL"):
        return CFG.k / h.M_cool
    return h.sigma


def on_success(h, rule, gain):
    h.M = min(h.M + min(gain, h.M), CFG.m_max)
    h.M_cool = h.M
    h.starve = 0
    if rule.startswith("onefifth"):
        h.sigma = min(h.sigma * 1.5, 1.0)


def on_failure(h, rule, param) -> bool:
    """Update after a failed step; return True if the searcher bursts.

    `param` is the patience L for fixed-patience rules, else the evaporation alpha."""
    h.starve += 1
    alpha = CFG.alpha if rule in FIXED_L else param
    if rule.startswith("onefifth"):
        h.sigma *= 1.5 ** -0.25
    if rule in ("hawking", "cool-earned", "heat-fixedL", "onefifth-earned"):
        h.M = evaporate(h.M, alpha, h.starve)
    elif rule == "hawking-dt1":
        h.M = evaporate(h.M, alpha, 1.0)
    if rule == "heat-fixedL":
        h.M = max(h.M, CFG.m_min)       # heating saturates at the burst temperature
    if rule in FIXED_L:
        return h.starve >= param
    return h.M < CFG.m_min


def minimize(f, lower, upper, budget, seed, rule="hawking", param=CFG.alpha, mergers=False):
    rng = np.random.default_rng(seed)
    B = Budget(f, lower, upper, budget)
    d = len(B.lower)
    sq = np.sqrt(d)
    max_holes = int(CFG.max_holes_factor * CFG.n_holes)
    holes: list[Searcher] = []
    try:
        for _ in range(CFG.n_holes):
            u = rng.random(d)
            holes.append(Searcher(u, B.unit(u), CFG.m0))
        scale = float(np.std([h.f for h in holes])) or 1.0
        while True:
            new = []
            for h in holes:
                T = step_length(h, rule)
                cand = reflect(h.x + rng.normal(0.0, T / sq, d))
                fc = B.unit(cand)
                if fc < h.f:
                    on_success(h, rule, CFG.gain * (h.f - fc) / scale)
                    h.x, h.f = cand, fc
                    burst = False
                else:
                    burst = on_failure(h, rule, param)
                if burst:
                    room = max_holes - (len(holes) + len(new)) + 1
                    nb = max(1, min(CFG.burst_children, room))
                    Tb = CFG.k / CFG.m_min
                    for j in range(nb):
                        u = reflect(h.x + rng.normal(0.0, Tb / sq, d))
                        child = Searcher(u, B.unit(u), CFG.child_mass)
                        if j == 0:
                            h.x, h.f, h.M, h.M_cool, h.sigma, h.starve = (
                                child.x, child.f, child.M, child.M_cool, child.sigma, 0)
                        else:
                            new.append(child)
            holes.extend(new)
            if mergers:                 # only used to check equivalence with minimal.py
                merged = True
                while merged and len(holes) > 1:
                    merged = False
                    X = np.array([h.x for h in holes])
                    D = np.linalg.norm(X[:, None] - X[None], axis=-1)
                    R = np.array([schwarzschild_radius(h.M, CFG) for h in holes])
                    hit = D < (R[:, None] + R[None])
                    np.fill_diagonal(hit, False)
                    if hit.any():
                        i, j = np.unravel_index(np.argmin(np.where(hit, D, np.inf)), D.shape)
                        a, b = holes[i], holes[j]
                        keep = a if a.f <= b.f else b
                        keep.M = min(a.M + b.M, CFG.m_max)
                        holes.pop(max(i, j))
                        holes[min(i, j)] = keep
                        merged = True
                while len(holes) < CFG.n_holes // 2:
                    u = rng.random(d)
                    holes.append(Searcher(u, B.unit(u), CFG.m0))
    except BudgetExhausted:
        pass
    return B.best_f


def _job(args):
    rule, L, pname, seed, budget = args
    p = problems.get(pname)
    fbest = minimize(p.f, p.lower, p.upper, budget, seed, rule, L)
    err = max(fbest - p.f_opt, 0.0) if p.f_opt is not None else fbest
    return rule, L, pname, seed, err


def mean_rank(rows, keys, probs):
    """Mean over problems of the rank of each key's median error."""
    ranks = {k: [] for k in keys}
    for pn in probs:
        med = {k: np.median([r[4] for r in rows if (r[0], r[1]) == k and r[2] == pn]) for k in keys}
        order = sorted(keys, key=lambda k: med[k])
        for i, k in enumerate(order):
            ranks[k].append(i + 1)
    return {k: float(np.mean(v)) for k, v in ranks.items()}


def main():
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    probs, budget = problems.ALL, 10_000

    # sanity check: hawking rule with mergers reproduces minimal.py exactly
    from hawkingsearch.minimal import minimize as minimal
    p = problems.get("rastrigin")
    assert minimize(p.f, p.lower, p.upper, 2000, 3, "hawking", mergers=True) == \
        minimal(p.f, p.lower, p.upper, budget=2000, seed=3).f

    t0 = time.time()
    with Pool() as pool:
        screen = pool.map(_job, [(v, q, pn, s, budget) for v in VARIANTS for q in GRID[v]
                                 for pn in probs for s in range(100, 105)], chunksize=4)
        best = {}
        for v in VARIANTS:
            keys = [(v, q) for q in GRID[v]]
            mr = mean_rank(screen, keys, probs)
            best[v] = min(GRID[v], key=lambda q: mr[(v, q)])
            print(f"screen {v}: " + ", ".join(f"{q}: {mr[(v, q)]:.2f}" for q in GRID[v]),
                  f"-> {best[v]}")
        final = pool.map(_job, [(v, best[v], pn, s, budget) for v in VARIANTS
                                for pn in probs for s in range(15)], chunksize=4)
    print(f"{time.time() - t0:.0f} s")

    with open(out / "controls_raw.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "param", "problem", "seed", "final_error"])
        for r in screen + final:
            w.writerow([r[0], r[1], r[2], r[3], f"{r[4]:.10g}"])

    keys = [(v, best[v]) for v in VARIANTS]
    mr = mean_rank(final, keys, probs)
    lines = ["| variant | tuned | " + " | ".join(probs) + " | mean rank | W/T/L vs hawking |",
             "|---" * (len(probs) + 4) + "|"]
    ref = {pn: [r[4] for r in final if r[0] == "hawking" and r[2] == pn] for pn in probs}
    for k in keys:
        meds, wtl = [], [0, 0, 0]
        for pn in probs:
            e = [r[4] for r in final if (r[0], r[1]) == k and r[2] == pn]
            meds.append(f"{np.median(e):.3g}")
            if k[0] != "hawking":
                pv = mannwhitneyu(ref[pn], e).pvalue
                if pv >= 0.05:
                    wtl[1] += 1
                elif np.median(ref[pn]) < np.median(e):
                    wtl[0] += 1
                else:
                    wtl[2] += 1
        q = f"L={k[1]}" if k[0] in FIXED_L else f"alpha={k[1]}"
        lines.append(f"| {k[0]} | {q} | " + " | ".join(meds) +
                     f" | {mr[k]:.2f} | {'/'.join(map(str, wtl)) if k[0] != 'hawking' else '-'} |")
    table = "\n".join(lines)
    (out / "controls.md").write_text(
        "Median final error over seeds 0-14, budget 10,000, d=10. W/T/L = hawking "
        "significantly better / no difference / worse (Mann-Whitney U, p<0.05).\n\n" + table + "\n")
    print(table)


if __name__ == "__main__":
    main()
