"""Curvature refinement must earn its precision within the shared budget."""
import numpy as np
import pytest

from hawkingsearch import FullConfig, minimize_full, problems
from hawkingsearch.ringdown import ringdown


def test_coupled_rotated_quadratic_relaxes_within_its_budget():
    rng = np.random.default_rng(81)
    Q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    A = Q @ np.diag([1, 3, 20, 100]) @ Q.T
    target = np.array([0.2, 0.7, 0.4, 0.8])
    calls = []

    def f(x):
        calls.append(x.copy())
        assert np.all((x >= 0) & (x <= 1))
        return float((x - target) @ A @ (x - target))

    x0 = np.full(4, 0.5)
    f0 = f(x0)
    calls.clear()
    x, fx, used = ringdown(f, x0, f0, 400)
    assert used == len(calls) <= 400
    assert fx < 1e-14
    assert np.linalg.norm(x - target) < 1e-6


@pytest.mark.parametrize("cap", [0, 2, 3, 4, 5, 8, 17, 100])
def test_wall_minimum_and_partial_budgets(cap):
    calls = []

    def f(x):
        calls.append(x.copy())
        assert np.all((x >= 0) & (x <= 1))
        return float((x[0] + 1)**2 + (x[1] - 0.3)**2)

    initial = np.array([0.0, 1.0])
    initial_f = f(initial)
    calls.clear()
    x, fx, used = ringdown(f, initial, initial_f, cap)
    assert used == len(calls) <= cap
    assert fx <= initial_f
    if cap == 100:
        assert x == pytest.approx([0, 0.3], abs=1e-6)


@pytest.mark.parametrize("local", [False, True])
def test_ringdown_counts_probes_and_respects_total_allocation(local):
    p = problems.sphere(4)
    cfg = FullConfig(ringdown=True, local_centre=local)
    measured = []

    def f(x):
        measured.append(x.copy())
        assert np.all((x >= p.lower) & (x <= p.upper))
        return p.f(x)

    r = minimize_full(f, p.lower, p.upper, budget=2000, seed=7, config=cfg, record=True)
    assert r.nfev == len(measured) == 2000
    episodes = [e for s in r.history for e in s["events"] if e["type"] == "ringdown"]
    assert episodes
    assert sum(e["nfev"] for e in episodes) <= int(2000 * cfg.ringdown_fraction)
    assert all(e["f_after"] <= e["f_before"] for e in episodes)
    assert r.f < 1e-16
    again = minimize_full(p.f, p.lower, p.upper, budget=2000, seed=7, config=cfg)
    assert np.array_equal(r.x, again.x) and r.f == again.f


def test_zero_allocation_matches_original_exactly():
    p = problems.rastrigin(4)
    a = minimize_full(p.f, p.lower, p.upper, budget=600, seed=19)
    b = minimize_full(p.f, p.lower, p.upper, budget=600, seed=19,
                      config=FullConfig(ringdown=True, ringdown_fraction=0))
    assert a.f == b.f and np.array_equal(a.trace_best, b.trace_best)


def test_single_quantum_emission_changes_one_coordinate_without_external_forces():
    p = problems.sphere(5)
    cfg = FullConfig(n_holes=1, hawking_quanta=True, gravity=False, spin=False,
                     galactic_centre=False, page_curve=False, penrose=False,
                     endgame=False, alpha=0)
    r = minimize_full(p.f, p.lower, p.upper, budget=100, seed=8, config=cfg, record=True)
    pairs = list(zip(r.history, r.history[1:]))
    assert pairs
    for prev, cur in pairs:
        origin = prev["holes"][0]["x"]
        for point, mode in cur["candidates"]:
            assert mode == "quantum"
            assert np.count_nonzero(point != origin) == 1


def test_late_ringdown_leaves_exploration_trajectory_unchanged():
    p = problems.rosenbrock(4)
    budget = 2000
    original = minimize_full(p.f, p.lower, p.upper, budget=budget, seed=7)
    cfg = FullConfig(ringdown=True, ringdown_endgame_only=True)
    late = minimize_full(p.f, p.lower, p.upper, budget=budget, seed=7,
                         config=cfg, record=True)
    cutoff = (1 - cfg.endgame_fraction) * budget
    grid = np.arange(1, int(cutoff))
    assert np.array_equal(original.best_at(grid), late.best_at(grid), equal_nan=True)
    episodes = [e for s in late.history for e in s["events"] if e["type"] == "ringdown"]
    assert episodes and all(e["start_eval"] >= cutoff for e in episodes)
