# Curvature refinement and exploration study

The recommended first experiment is:

```python
cfg = FullConfig(ringdown=True, ringdown_endgame_only=True)
res = minimize_full(p.f, p.lower, p.upper, budget=10_000, seed=0, config=cfg)
```

Original defaults remain unchanged. All three new modules are opt-in. The results
below compare variants in the same environment; they do not reuse the old README
numbers as a control.

## What produces the drop

The existing Penrose process fits a parabola along a sampled direction, then
discards that directional curvature. Ringdown measures all coordinate slopes
with paired probes and retains coupled curvature through BFGS secant updates.
It proposes `delta = -H @ gradient`, limits the displacement using an initial
horizon/temperature radius, and dissipates overshoot with backtracking. A
quadratic bowl can be traversed in a few such steps. All numerical derivative
probes and line-search trials count against the shared evaluation budget.

The physical inspiration is a perturbed black hole relaxing by dissipating
energy; actual black-hole ringdown involves quasinormal modes
([review](https://arxiv.org/abs/1102.4014)). The objective Hessian is an analogue
of local tidal curvature. BFGS is an ordinary numerical curvature-learning rule
([SciPy BFGS reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.BFGS.html));
this implementation is not derived from GR and does not solve its field equations.

The implementation is in `hawkingsearch/ringdown.py`, using NumPy rather than an
external local-minimizer call. It handles walls with one-sided probes, skips
nonpositive secant curvature, returns the best point measured, and never exceeds
its allocation. Central slope measurement costs 2d evaluations. The dense inverse
curvature tensor uses O(d²) memory. No noise handling is implemented.

## Experimental sequence

Every run received exactly 10,000 objective evaluations on each of the same nine
problems as the original benchmark. Runs used four worker processes, with BLAS
and OpenMP threads limited to one per process. Versions and complete default
settings are stored beside the raw results.

1. Screened early ringdown and local centre attraction on seeds 300–304:
   [summary](refinement-screen/summary.md), [raw runs](refinement-screen/raw.csv).
2. Screened single-coordinate emissions on the same seeds:
   [summary](quanta-screen/summary.md), [raw runs](quanta-screen/raw.csv).
3. Checked all seven initial configurations on fresh seeds 400–414:
   [summary](refinement-validation/summary.md), [plot](refinement-validation/convergence.png),
   [raw runs](refinement-validation/raw.csv). Early ringdown improved several
   smooth problems but median Ackley error increased from 2.78e-11 to 0.437.
4. In response to that regression, screened ringdown restricted to the existing
   endgame, using the screening seeds again:
   [summary](late-ringdown-screen/summary.md).
5. Validated the revised choice on another fresh set, seeds 500–514:
   [summary](late-ringdown-validation/summary.md),
   [plot](late-ringdown-validation/convergence.png),
   [raw runs](late-ringdown-validation/raw.csv),
   [settings](late-ringdown-validation/settings.json).

This is **validation on new seeds, not on new problems**. The revised endgame
choice was informed by the earlier Ackley regression. Generalization to other
objectives, dimensions, rotations and noise levels remains unestablished.
Tests include a rotated quadratic as a mathematical check of coupled refinement;
that is not a held-out global-optimization benchmark.

## Final validation: median error over 15 runs

| Problem | Original | Early ringdown | Endgame ringdown | Endgame + quanta |
|---|---:|---:|---:|---:|
| Sphere | 1.00e-22 | 3.16e-30 | 3.16e-30 | 2.37e-30 |
| Rosenbrock | 0.491 | 5.20e-15 | 4.86e-15 | 4.85e-15 |
| Rastrigin | 7.96 | 8.95 | 7.96 | 6.96 |
| Ackley | 2.37e-11 | 3.37e-11 | 3.59e-13 | 3.02e-13 |
| Griewank | 0.0911 | 0 | 0.0911 | 0.0394 |
| Schwefel | 1,560 | 1,480 | 1,560 | 1,370 |
| Terrain | 1.66e-6 | 2.81e-5 | 1.11e-6 | 1.71 |
| Iris | 3.55e-15 | 2.22e-16 | 2.22e-16 | 2.22e-16 |
| Wine | 4.82e-9 | 4.17e-10 | 0 | 0 |

Zero means the computed error reached the reference value to floating-point
precision. The clustering references come from k-means restarts, not a global
optimality proof. Differences at floating-point precision should not be treated
as useful practical advantages.

The inherited summary uses two-sided Mann–Whitney tests at p < 0.05, without
multiple-comparison correction, and treats arrays satisfying NumPy `allclose`
as ties. This heuristic suppresses some comparisons among very small errors.
With that convention, endgame ringdown improves Rosenbrock and Wine significantly
versus the original, with seven ties and no detected losses. These tests do not
establish equivalence on the other seven problems.

Medians also hide failure rates. At error < 1e-8, endgame ringdown solved
Rosenbrock in 13/15 runs versus 0/15 originally, and Wine in 12/15 versus 8/15.
On Ackley, however, that count fell from 12/15 to 9/15 despite a better median.
The revised rule improves refinement but does not establish uniformly better
reliability; test the entire error distribution on the intended objective.

In a separate measurement on Sphere with the final validation seeds, the median
evaluation count to reach error below 1e-8 was **2,479 originally versus 133 with
early ringdown**, about 19 times fewer evaluations. Endgame ringdown preserved the
original count of 2,479 because its early trajectory is identical. Exact counts,
without the plot's coarse sampling, are in [ringdown-onset.csv](ringdown-onset.csv).

The convergence figures sample approximately 60 evaluation counts on logarithmic
axes, show medians, and add 1e-12 to errors. Values below that plotting floor are
visible in the tables and raw CSV, not distinguishable on the figures.

## Other improvement candidates

**Single-coordinate Hawking emissions:** excite one mode while preserving the
expected squared thermal step length. The original algorithm changes every
coordinate at once, which can destroy progress when only one coordinate needs
to escape a barrier. This is promising for separable objectives and clustering.
The endgame-plus-quanta variant improved several median errors, but Terrain
regressed to 1.71 from 1.66e-6 on the final seeds. It is not a universal default,
and its choice of coordinate axes needs a rotation study.

**Independent gravitational systems:** only nearby, better holes contribute to
the galactic-centre pull. On seeds 400–414 this reduced median Schwefel error
from about 1,720 to 1,130, but Rastrigin rose from 7.96 to 21.9. Keeping distant
basins alive can sacrifice beneficial population convergence. The opt-in
`local_centre` switch supports further work; it is not recommended globally.

**Preserve an active best remnant:** the budget wrapper always remembers the
best result, but shocks and evaporation can remove that position from the
living population. During one Sphere run, a subsequent ringdown refined a
worse new leader despite an excellent incumbent already having been found.
A protected best remnant or reinjection from the archive could prevent wasted
refinement. This is an inspected code-path opportunity, not an implemented or
benchmarked improvement in this change.

**Validate the landscape assumptions:** smooth finite-difference refinement is
suited to deterministic potentials. Noisy objectives need larger/adaptive probe
distances and repeated measurements. Higher dimensions call for a limited-memory
curvature representation. Rotated and shifted multimodal problems should be the
next benchmark, since the current suite has substantial coordinate structure.

## Verification

`tests/test_ringdown.py`: 14 passing cases covering a coupled rotated quadratic,
wall minima, partial allocations, exact shared budget accounting, allocation
caps, determinism, single-mode emissions, and identical exploration before late
ringdown.

Full suite: **91 passed, 2 failed, 1 skipped**. Both failing tests were reproduced
with the untouched HEAD source in a separate temporary directory under the same
dependencies:

- `test_beats_random_search[hawking-full-terrain]`: the eight-seed significance
  threshold is not met.
- `test_page_curve_switches_burst_children_to_informed_sampling`: this fixed run
  produces no informed burst.

The existing assertions were not weakened to accommodate those failures. The
original default trajectory is preserved when the new modules are disabled.
