"""Every optimiser obeys the same contract: exact budget, bounds, determinism."""
import numpy as np
import pytest

from hawkingsearch import ALGORITHMS, FullConfig, MinimalConfig, minimize_full, minimize_minimal, problems


@pytest.fixture(scope="module")
def rastrigin():
    return problems.rastrigin(5)


@pytest.mark.parametrize("name", list(ALGORITHMS))
def test_uses_exact_budget_and_stays_in_bounds(name, rastrigin):
    p = rastrigin
    r = ALGORITHMS[name](p.f, p.lower, p.upper, budget=1500, seed=0)
    assert r.nfev == 1500
    assert np.all(r.x >= p.lower) and np.all(r.x <= p.upper)
    assert r.f == pytest.approx(p(r.x))
    assert np.all(np.diff(r.trace_best) < 0)          # best-so-far only improves


@pytest.mark.parametrize("name", list(ALGORITHMS))
def test_deterministic_given_seed(name, rastrigin):
    p = rastrigin
    a = ALGORITHMS[name](p.f, p.lower, p.upper, budget=800, seed=7)
    b = ALGORITHMS[name](p.f, p.lower, p.upper, budget=800, seed=7)
    assert a.f == b.f and np.array_equal(a.x, b.x)


def test_minimal_population_stays_bounded(rastrigin):
    p = rastrigin
    cfg = MinimalConfig()
    sizes = []
    minimize_minimal(p.f, p.lower, p.upper, budget=4000, seed=1, config=cfg,
                     callback=lambda s: sizes.append(len(s["holes"])))
    assert max(sizes) <= cfg.max_holes_factor * cfg.n_holes
    assert min(sizes) >= cfg.n_holes // 2


def test_minimal_masses_follow_the_rules(rastrigin):
    """Mass rises only on improvement and falls by exactly alpha/M^2 otherwise (dt = 1)."""
    p = rastrigin
    cfg = MinimalConfig(n_holes=1, rs_coeff=0.0, max_holes_factor=1.0, starvation_clock=False)
    states = []
    minimize_minimal(p.f, p.lower, p.upper, budget=300, seed=3, config=cfg, callback=states.append)
    for prev, cur in zip(states, states[1:]):
        (_, f0, M0), (_, f1, M1) = prev["holes"][0], cur["holes"][0]
        if M1 == cfg.child_mass and M0 - cfg.alpha / M0**2 < cfg.m_min:
            continue                                   # final burst replaced the hole
        if f1 < f0:
            assert M1 > M0
        else:
            assert M1 == pytest.approx(M0 - cfg.alpha / M0**2)


def test_starvation_clock_makes_losses_grow_while_starving(rastrigin):
    """With dt = iterations since last accretion, consecutive losses increase."""
    p = rastrigin
    cfg = MinimalConfig(n_holes=1, rs_coeff=0.0, max_holes_factor=1.0)
    states = []
    minimize_minimal(p.f, p.lower, p.upper, budget=400, seed=3, config=cfg, callback=states.append)
    M = [s["holes"][0][2] for s in states]
    f = [s["holes"][0][1] for s in states]
    starve, checked = 0, 0
    for i in range(1, len(M)):
        if f[i] < f[i - 1] or M[i] == cfg.child_mass:
            starve = 0
            continue
        starve += 1
        assert M[i - 1] - M[i] == pytest.approx(starve * cfg.alpha / M[i - 1] ** 2)
        checked += 1
    assert checked > 10


def test_full_records_history_for_animation():
    p = problems.terrain()
    r = minimize_full(p.f, p.lower, p.upper, budget=600, seed=0, record=True)
    assert len(r.history) > 5
    s = r.history[-1]
    assert {"holes", "events", "candidates", "best_f", "nfev"} <= set(s)
    h = s["holes"][0]
    assert h["T"] == pytest.approx(FullConfig().k / h["M"])


def test_ablation_rejects_unknown_module():
    with pytest.raises(ValueError):
        FullConfig.ablate("wormholes")
