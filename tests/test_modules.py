"""Integration checks: each module does what it claims inside a real run."""
import numpy as np
import pytest

from hawkingsearch import MODULES, FullConfig, minimize_full, problems


# alpha = 0.01 makes holes evaporate fast enough that every mechanism fires
# within a short run (the tuned default 0.003 is slower).
FAST = dict(alpha=0.01)


def events_of(cfg, prob=None, budget=6000, seed=0):
    p = prob or problems.rastrigin(5)
    r = minimize_full(p.f, p.lower, p.upper, budget=budget, seed=seed, config=cfg, record=True)
    return [e for s in r.history for e in s["events"]], r


@pytest.fixture(scope="module")
def full_run():
    return events_of(FullConfig(**FAST))


def test_every_merger_respects_the_area_theorem(full_run):
    events, _ = full_run
    mergers = [e for e in events if e["type"] == "merger"]
    assert mergers, "expected some mergers"
    for e in mergers:
        Ma, Mb = e["masses"]
        assert e["mass"] ** 2 >= Ma**2 + Mb**2 - 1e-9
        assert e["radiated"] == pytest.approx(0.05 * (Ma + Mb))


def test_blocked_mergers_are_the_lopsided_ones(full_run):
    events, _ = full_run
    blocked = [e for e in events if e["type"] == "merger-blocked"]
    assert blocked, "expected the area theorem to block something"
    for e in blocked:
        Ma, Mb = e["masses"]
        assert max(Ma, Mb) / min(Ma, Mb) > 17


def test_without_area_theorem_nothing_is_blocked():
    events, _ = events_of(FullConfig.ablate("area_theorem"), budget=3000)
    assert not [e for e in events if e["type"] == "merger-blocked"]


def test_gravitational_waves_push_holes_outward(full_run):
    events, _ = full_run
    pushes = [e for e in events if e["type"] == "gw-push"]
    assert pushes
    assert not [e for e in events_of(FullConfig.ablate("gravitational_waves"), budget=3000)[0]
                if e["type"] == "gw-push"]


def test_page_curve_switches_burst_children_to_informed_sampling():
    # A single run can contain only one or two bursts, so pool a few seeds.
    bursts = [e for seed in range(4) for e in events_of(FullConfig(**FAST), seed=seed)[0]
              if e["type"] == "burst"]
    assert any(e["informed"] for e in bursts), "some holes should burst after their Page time"
    off, _ = events_of(FullConfig.ablate("page_curve", **FAST), budget=3000)
    assert not any(e["informed"] for e in off if e["type"] == "burst")


def test_informed_children_land_closer_together_than_thermal_ones():
    """After the Page time, children come from the accreted distribution, not a random scatter.

    Uses a larger population cap so bursts have room for several children, and pools
    a few seeds so both kinds of burst occur."""
    bursts = []
    for seed in range(4):
        events, _ = events_of(FullConfig(max_holes_factor=2.0, **FAST), seed=seed)
        bursts += [e for e in events if e["type"] == "burst" and len(e["children"]) > 1]

    def spread(evs):
        return np.median([np.mean(np.std(np.array(e["children"]), axis=0)) for e in evs])
    informed = [e for e in bursts if e["informed"]]
    thermal = [e for e in bursts if not e["informed"]]
    assert len(informed) >= 3 and len(thermal) >= 3
    assert spread(informed) < spread(thermal)


def test_spin_appears_only_with_spin_module():
    p = problems.rastrigin(5)
    for cfg, expect in [(FullConfig(), True), (FullConfig.ablate("spin"), False)]:
        r = minimize_full(p.f, p.lower, p.upper, budget=3000, seed=0, config=cfg, record=True)
        spun = any(h["chi"] > 0 for s in r.history for h in s["holes"])
        assert spun is expect


def test_all_modules_off_never_uses_module_events():
    events, _ = events_of(FullConfig.ablate(*MODULES), budget=3000)
    kinds = {e["type"] for e in events}
    assert kinds <= {"merger", "burst"}


# ---------------------------------------------------------------- second-generation modules
def test_white_holes_are_never_entered():
    """No candidate may land inside a white hole that already existed."""
    p = problems.rastrigin(5)
    r = minimize_full(p.f, p.lower, p.upper, budget=6000, seed=0, record=True,
                      config=FullConfig(alpha=0.01, wh_lifetime=10**9, white_holes=True))
    w = (p.upper - p.lower)
    born = 0
    for prev, cur in zip(r.history, r.history[1:]):
        born += sum(e["type"] == "white-hole" for e in cur["events"])
        for c, rad in prev["white_holes"]:
            for x, _ in cur["candidates"]:
                assert np.linalg.norm((x - c) / w) >= rad * (1 - 1e-9)
    assert born > 0


def test_white_holes_never_seal_the_best_basin():
    p = problems.rastrigin(5)
    r = minimize_full(p.f, p.lower, p.upper, budget=6000, seed=0, record=True,
                      config=FullConfig(alpha=0.01, white_holes=True))
    for s in r.history:
        for e in s["events"]:
            if e["type"] == "white-hole":
                assert p(e["x"]) > s["best_f"] - 1e-9


def test_penrose_only_the_leader_and_at_most_three_probes_per_iteration():
    p = problems.sphere(5)
    r = minimize_full(p.f, p.lower, p.upper, budget=4000, seed=0, record=True)
    counts = [sum(m == "penrose" for _, m in s["candidates"]) for s in r.history]
    assert max(counts) <= 3 and sum(counts) > 0
    off = minimize_full(p.f, p.lower, p.upper, budget=2000, seed=0, record=True,
                        config=FullConfig.ablate("penrose"))
    assert not any(m == "penrose" for s in off.history for _, m in s["candidates"])


def test_slingshot_with_zero_momentum_equals_no_slingshot():
    """Velocity is the slingshot's only effect."""
    p = problems.rastrigin(5)
    a = minimize_full(p.f, p.lower, p.upper, budget=3000, seed=4,
                      config=FullConfig(slingshot=True, momentum=0.0))
    b = minimize_full(p.f, p.lower, p.upper, budget=3000, seed=4, config=FullConfig(slingshot=False))
    assert a.f == b.f
    c = minimize_full(p.f, p.lower, p.upper, budget=3000, seed=4, config=FullConfig(slingshot=True))
    assert c.f != a.f


def test_maximal_switches_everything_on():
    from hawkingsearch import MODULES
    cfg = FullConfig.maximal()
    assert all(getattr(cfg, m) for m in MODULES)


# ---------------------------------------------------------------- third-round mechanisms
def test_endgame_gives_every_evaluation_to_the_leader():
    p = problems.sphere(5)
    cfg = FullConfig()
    r = minimize_full(p.f, p.lower, p.upper, budget=4000, seed=0, record=True, config=cfg)
    # skip the iteration that straddles the threshold (it started before the endgame)
    late = [s for s in r.history[1:] if s["nfev"] > (1 - cfg.endgame_fraction) * 4000 + 40]
    assert late
    for s in late:
        assert len(s["candidates"]) <= 3
        assert all(m == "penrose" for _, m in s["candidates"])


def test_cmb_spawn_spreads_the_starting_population():
    p = problems.sphere(4)

    def min_gap(cmb):
        gaps = []
        for seed in range(10):
            r = minimize_full(p.f, p.lower, p.upper, budget=200, seed=seed, record=True,
                              config=FullConfig(cmb_spawn=cmb))
            X = np.array([h["x"] for h in r.history[0]["holes"]])
            D = np.linalg.norm(X[:, None] - X[None], axis=-1) + np.eye(len(X)) * 1e9
            gaps.append(D.min())
        return np.mean(gaps)
    assert min_gap(True) > 1.3 * min_gap(False)


def test_levy_jumps_only_when_switched_on():
    p = problems.rastrigin(5)
    def modes(on):
        r = minimize_full(p.f, p.lower, p.upper, budget=3000, seed=0, record=True,
                          config=FullConfig(levy_radiation=on))
        return {m for s in r.history for _, m in s["candidates"]}
    assert "levy" in modes(True)
    assert "levy" not in modes(False)
