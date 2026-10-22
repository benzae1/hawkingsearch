"""Reproducible screening/validation without overwriting the original benchmark.

    python -m hawkingsearch.refinement_study --seed-start 300 --seeds 5
    python -m hawkingsearch.refinement_study --seed-start 400 --seeds 15 --out results/refinement-validation
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
from dataclasses import asdict
from importlib.metadata import version
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import scipy

from . import FullConfig, problems
from .benchmark import _job, plot_convergence, summarise


VARIANTS = [
    ("original", {}),
    ("ringdown", {"ringdown": True}),
    ("local-centre", {"local_centre": True}),
    ("ringdown+local-centre", {"ringdown": True, "local_centre": True}),
    ("quanta", {"hawking_quanta": True}),
    ("ringdown+quanta", {"ringdown": True, "hawking_quanta": True}),
    ("late-ringdown", {"ringdown": True, "ringdown_endgame_only": True}),
    ("late-ringdown+quanta", {"ringdown": True, "ringdown_endgame_only": True,
                              "hawking_quanta": True}),
    ("dual-annealing (scipy)", None),
]

COLOURS = dict(zip([name for name, _ in VARIANTS], [
    "#2a78d6", "#1baf7a", "#eda100", "#e87ba4", "#6250d6", "#008300",
    "#8c564b", "#333333", "#e34948",
]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=300)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--budget", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--variants", nargs="+", choices=[name for name, _ in VARIANTS])
    parser.add_argument("--reference", choices=[name for name, _ in VARIANTS])
    parser.add_argument("--out", type=Path, default=Path("results/refinement-screen"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    variants = [(name, cfg) for name, cfg in VARIANTS
                if args.variants is None or name in args.variants]
    names = [name for name, _ in variants]
    reference = args.reference or ("ringdown" if "ringdown" in names else names[0])
    if reference not in names:
        parser.error("--reference must be one of the selected --variants")
    jobs = [(name, p, seed, args.budget, cfg)
            for p in problems.ALL for seed in seeds for name, cfg in variants]
    results = []
    with Pool(args.workers) as pool:
        for result in pool.imap_unordered(_job, jobs, chunksize=1):
            results.append(result)
            if len(results) % 25 == 0:
                print(f"{len(results)}/{len(jobs)} runs", flush=True)
    results.sort(key=lambda r: (r[1], r[0], r[2]))
    with (args.out / "raw.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["algorithm", "problem", "seed", "error", "nfev", "seconds"])
        writer.writerows(r[:6] for r in results)
    np.savez_compressed(args.out / "curves.npz",
                        labels=np.array([r[:3] for r in results]),
                        errors=np.array([r[6] for r in results]),
                        evaluations=np.unique(np.geomspace(1, args.budget, 60).astype(int)))
    (args.out / "settings.json").write_text(json.dumps({
        "seeds": seeds, "budget": args.budget, "variants": variants,
        "reference": reference,
        "numpy": np.__version__, "scipy": scipy.__version__,
        "python": platform.python_version(), "scikit-learn": version("scikit-learn"),
        "matplotlib": version("matplotlib"), "defaults": asdict(FullConfig()),
        "note": "Same problems as original benchmark; new seeds, not held-out problems. Unadjusted p-values.",
    }, indent=2) + "\n")
    print(summarise(results, names, problems.ALL, reference, args.out / "summary.md",
                    "Optional refinement study"))
    plot_convergence(results, names, problems.ALL, args.budget, args.out / "convergence.png",
                     colour_overrides=COLOURS)


if __name__ == "__main__":
    main()
