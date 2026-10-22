"""Diagnostic for the temperature floor (Proposition 4 in the paper).

Reruns three variants of the controlled study at their selected settings on
seeds 0-14 and records the smallest proposal scale used by any searcher during
each run. Mass grows only by normalised improvement, so the Hawking and
no-reheating rules cannot shrink their step below roughly k / (largest mass);
the 1/5 controller has no such floor.

    python paper/temperature_floor.py           # writes paper/results/temperature_floor.md
"""
from __future__ import annotations

from multiprocessing import Pool
from pathlib import Path

import numpy as np

import cooling_controls as cc
from cooling_controls import problems

SELECTED = {"hawking": 0.001, "cool-earned": 0.003, "onefifth-L": 100}

_step_length = cc.step_length
_min_step = [np.inf]


def _recording_step_length(h, rule):
    T = _step_length(h, rule)
    _min_step[0] = min(_min_step[0], T)
    return T


def _job(args):
    rule, pname, seed = args
    cc.step_length = _recording_step_length
    _min_step[0] = np.inf
    p = problems.get(pname)
    cc.minimize(p.f, p.lower, p.upper, 10_000, seed, rule, SELECTED[rule])
    return rule, pname, _min_step[0]


def main():
    probs = problems.ALL
    with Pool() as pool:
        rows = pool.map(_job, [(v, pn, s) for v in SELECTED for pn in probs for s in range(15)])
    lines = ["| variant | " + " | ".join(probs) + " |", "|---" * (len(probs) + 1) + "|"]
    for v in SELECTED:
        meds = [np.median([r[2] for r in rows if r[0] == v and r[1] == pn]) for pn in probs]
        lines.append(f"| {v} | " + " | ".join(f"{m:.2g}" for m in meds) + " |")
    table = "\n".join(lines)
    (Path(__file__).parent / "results" / "temperature_floor.md").write_text(
        "Median over seeds 0-14 of the smallest proposal scale (unit-cube RMS step) "
        "used by any searcher in a run; budget 10,000, selected settings.\n\n" + table + "\n")
    print(table)


if __name__ == "__main__":
    main()
