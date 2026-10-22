"""Head-to-head comparisons at a reduced scale (the full study is benchmark.py).

These pin down the claims the README makes, using a fixed budget and seeds,
so a change that quietly breaks the algorithm shows up as a failing test.
"""
import numpy as np
import pytest
from scipy.stats import mannwhitneyu

from hawkingsearch import ALGORITHMS, problems

SEEDS = range(8)
BUDGET = 5000


def finals(algo, p):
    return np.array([ALGORITHMS[algo](p.f, p.lower, p.upper, budget=BUDGET, seed=s).f for s in SEEDS])


def significantly_better(a, b):
    return np.median(a) < np.median(b) and mannwhitneyu(a, b, alternative="less").pvalue < 0.05


@pytest.fixture(scope="module")
def cache():
    store = {}
    def get(algo, pname):
        if (algo, pname) not in store:
            store[algo, pname] = finals(algo, problems.get(pname))
        return store[algo, pname]
    return get


@pytest.mark.parametrize("pname", ["sphere", "rastrigin", "rosenbrock", "ackley", "terrain", "iris-kmeans"])
@pytest.mark.parametrize("algo", ["hawking-minimal", "hawking-full"])
def test_beats_random_search(cache, algo, pname):
    if algo == "hawking-minimal" and pname == "ackley":
        pytest.skip("known limitation, see test_minimal_is_lost_on_ackley_plateau")
    if algo == "hawking-full" and pname == "terrain":
        # Terrain outcomes are bimodal (summit or a 1,040 m side peak), so 8 seeds at
        # this budget cannot separate the methods (p ~ 0.1). The full benchmark
        # (15 seeds, 10,000 evaluations) does: see results/summary.md.
        pytest.skip("bimodal at test scale; covered by the full benchmark")
    assert significantly_better(cache(algo, pname), cache("random-search", pname))


def test_minimal_is_lost_on_ackley_plateau(cache):
    """Known limitation: without gravity the holes are independent local searches,
    and Ackley's huge flat plateau gives them nothing to follow. The minimal
    version is no better than random search there; the full version is."""
    mini, rand = cache("hawking-minimal", "ackley"), cache("random-search", "ackley")
    assert not significantly_better(mini, rand)
    assert significantly_better(cache("hawking-full", "ackley"), rand)


@pytest.mark.parametrize("pname", ["rastrigin", "ackley", "rosenbrock"])
def test_full_beats_minimal_on_hard_landscapes(cache, pname):
    """The modules (mainly gravity) are what make it competitive on multimodal problems."""
    assert significantly_better(cache("hawking-full", pname), cache("hawking-minimal", pname))


def test_full_beats_simulated_annealing_on_rastrigin(cache):
    """Per-hole temperatures vs one global schedule."""
    assert significantly_better(cache("hawking-full", "rastrigin"), cache("simulated-annealing", "rastrigin"))


def test_full_does_not_beat_simulated_annealing_on_ackley(cache):
    """Honesty check: SA's slow global cooling suits Ackley's single funnel. At 10k
    evaluations (benchmark.py) SA is clearly ahead; here it is bimodal (usually
    ~0.1, sometimes stuck at ~17), so the claim pinned down is the weaker one."""
    assert not significantly_better(cache("hawking-full", "ackley"), cache("simulated-annealing", "ackley"))


def test_full_beats_one_plus_one_es_on_rastrigin(cache):
    assert significantly_better(cache("hawking-full", "rastrigin"), cache("(1+1)-ES", "rastrigin"))


def test_full_beats_cma_es_on_sphere(cache):
    """The leader's Penrose steps (with an adaptive step and an endgame) remove the
    precision cap of mass-based cooling; on a smooth bowl it now out-refines CMA-ES."""
    assert significantly_better(cache("hawking-full", "sphere"), cache("CMA-ES", "sphere"))


def test_dual_annealing_still_wins_on_rastrigin(cache):
    """Honesty check: dual annealing's long jumps plus local search solve Rastrigin
    outright; Hawking search still ends in a local minimum."""
    assert significantly_better(cache("dual-annealing (scipy)", "rastrigin"),
                                cache("hawking-full", "rastrigin"))
