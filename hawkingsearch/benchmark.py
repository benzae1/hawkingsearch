"""Benchmark Hawking search against standard optimisers.

    python -m hawkingsearch.benchmark                 # full run -> results/
    python -m hawkingsearch.benchmark --quick         # fewer seeds, smaller budget
    python -m hawkingsearch.benchmark --ablation      # switch modules off one at a time

Every algorithm gets the same number of function evaluations, through the
same budget wrapper. Writes:
    results/raw.csv            final error of every run
    results/summary.md         median error table, ranks, win/tie/loss vs Hawking
    results/convergence.png    median best-so-far error vs evaluations
    results/ablation.md/.png   (with --ablation)
"""
from __future__ import annotations

import argparse
import csv
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from . import ALGORITHMS, MODULES, FullConfig, minimize_full, problems


def _error(p, f):
    return max(f - p.f_opt, 0.0) if p.f_opt is not None else f


def _job(args):
    algo, pname, seed, budget, switches = args
    p = problems.get(pname)
    t = time.perf_counter()
    if switches is not None:            # {module: on/off} overrides of the full defaults
        res = minimize_full(p.f, p.lower, p.upper, budget=budget, seed=seed,
                            config=FullConfig(**switches))
    else:
        res = ALGORITHMS[algo](p.f, p.lower, p.upper, budget=budget, seed=seed)
    grid = np.unique(np.geomspace(1, budget, 60).astype(int))
    curve = np.array([_error(p, v) for v in res.best_at(grid)])
    return algo, pname, seed, _error(p, res.f), res.nfev, time.perf_counter() - t, curve


def run(algos, probs, seeds, budget, ablate=None, workers=None):
    jobs = []
    for pn in probs:
        for s in range(seeds):
            if ablate is None:
                jobs += [(a, pn, s, budget, None) for a in algos]
            else:
                jobs += [(name, pn, s, budget, off) for name, off in ablate]
    # multiprocessing.Pool rather than ProcessPoolExecutor: the latter intermittently
    # dies on Windows with "WinError 6: the handle is invalid" while starting workers
    with Pool(workers) as pool:
        return pool.map(_job, jobs, chunksize=2)


def _mwu(a, b):
    from scipy.stats import mannwhitneyu
    if np.allclose(a, b):
        return 1.0
    return mannwhitneyu(a, b, alternative="two-sided").pvalue


def summarise(results, algos, probs, ref, out: Path, title):
    E = {(a, p): np.array([r[3] for r in results if r[0] == a and r[1] == p]) for a in algos for p in probs}
    lines = [f"# {title}", "",
             f"Median final error (f - f_opt) over {len(E[algos[0], probs[0]])} seeds; "
             "lower is better. **Bold** = best on that problem. "
             f"W/T/L = {ref} significantly better / no significant difference / worse "
             "(Mann-Whitney U, p < 0.05).", ""]
    hdr = "| algorithm | " + " | ".join(probs) + " | mean rank |"
    lines += [hdr, "|" + "---|" * (len(probs) + 2)]
    med = np.array([[np.median(E[a, p]) for p in probs] for a in algos])
    ranks = np.array([np.argsort(np.argsort(med[:, j], kind="stable"), kind="stable") + 1
                      for j in range(len(probs))]).T
    for i, a in enumerate(algos):
        cells = []
        for j, p in enumerate(probs):
            v = f"{med[i, j]:.3g}"
            cells.append(f"**{v}**" if med[i, j] <= med[:, j].min() else v)
        lines.append(f"| {a} | " + " | ".join(cells) + f" | {ranks[i].mean():.2f} |")
    lines += ["", f"## {ref} vs each method (W/T/L across problems)", "",
              "| opponent | W | T | L | problems where " + ref + " wins |", "|---|---|---|---|---|"]
    for a in algos:
        if a == ref:
            continue
        w = t = l_ = 0
        wins = []
        for p in probs:
            pv = _mwu(E[ref, p], E[a, p])
            if pv < 0.05 and np.median(E[ref, p]) < np.median(E[a, p]):
                w += 1
                wins.append(p)
            elif pv < 0.05:
                l_ += 1
            else:
                t += 1
        lines.append(f"| {a} | {w} | {t} | {l_} | {', '.join(wins) or '-'} |")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "\n".join(lines)


# colours: reference categorical palette (light mode), fixed order
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def plot_convergence(results, algos, probs, budget, out: Path, colour_overrides=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grid = np.unique(np.geomspace(1, budget, 60).astype(int))
    ncol = 3
    nrow = int(np.ceil(len(probs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(14, 3.6 * nrow), facecolor="#fcfcfb")
    # Hawking variants take blue and violet so they never sit next to a baseline's hue
    fixed = {"hawking-full": PALETTE[0], "default": PALETTE[0],
             "hawking-minimal": PALETTE[6], "random-search": "#a3a29c"}
    fixed.update(colour_overrides or {})
    rest = [c for c in PALETTE if c not in (PALETTE[0], PALETTE[6])]
    colours, ci = {}, 0
    for a in algos:
        if a in fixed:
            colours[a] = fixed[a]
        else:
            colours[a] = rest[ci % len(rest)]
            ci += 1
    for ax, p in zip(axes.flat, probs):
        ax.set_facecolor("#fcfcfb")
        for a in algos:
            C = np.array([r[6] for r in results if r[0] == a and r[1] == p])
            m = np.median(C, axis=0) + 1e-12
            lw = 2.4 if a.startswith("hawking") else 1.3
            ls = "--" if a == "random-search" else "-"
            ax.plot(grid, m, color=colours[a], lw=lw, ls=ls, label=a)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(p, loc="left", fontsize=11, color="#0b0b0b")
        ax.grid(True, color="#e6e5e0", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=8, colors="#52514e")
    for ax in list(axes.flat)[len(probs):]:
        ax.axis("off")
    axes.flat[0].set_ylabel("median error", color="#52514e")
    h, lab = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=min(len(algos), 5), frameon=False, fontsize=9)
    fig.suptitle(f"Best-so-far error vs evaluations (median over seeds, budget {budget})",
                 x=0.01, ha="left", fontsize=13, color="#0b0b0b")
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(out, dpi=110)
    plt.close(fig)


def write_raw(results, out: Path):
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["algorithm", "problem", "seed", "final_error", "nfev", "seconds"])
        for r in results:
            w.writerow([r[0], r[1], r[2], f"{r[3]:.10g}", r[4], f"{r[5]:.3f}"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=15)
    ap.add_argument("--budget", type=int, default=10_000)
    ap.add_argument("--problems", default=",".join(problems.ALL))
    ap.add_argument("--algorithms", default=",".join(ALGORITHMS))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)
    if a.quick:
        a.seeds, a.budget = 5, 3000
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    probs = a.problems.split(",")
    t0 = time.time()
    if a.ablation:
        default = FullConfig()
        variants = [("default", {})]
        for m in MODULES:
            on = getattr(default, m)
            variants.append((f"{'without' if on else 'with'} {m}", {m: not on}))
        variants += [("every module on", {m: True for m in MODULES}),
                     ("no modules", {m: False for m in MODULES})]
        res = run(None, probs, a.seeds, a.budget, ablate=variants)
        names = [v[0] for v in variants]
        write_raw(res, out / "ablation_raw.csv")
        print(summarise(res, names, probs, "default", out / "ablation.md",
                        "Ablation: Hawking full with one module switched on or off"))
        plot_convergence(res, names, probs, a.budget, out / "ablation.png")
    else:
        algos = a.algorithms.split(",")
        res = run(algos, probs, a.seeds, a.budget)
        write_raw(res, out / "raw.csv")
        print(summarise(res, algos, probs, "hawking-full", out / "summary.md",
                        "Hawking search vs standard optimisers"))
        plot_convergence(res, algos, probs, a.budget, out / "convergence.png")
    print(f"\ndone in {time.time() - t0:.0f}s -> {out}/")


if __name__ == "__main__":
    main()
