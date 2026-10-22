"""The core formulas behave like the physics they borrow from."""
import numpy as np
import pytest

from hawkingsearch.full import (
    archive_gaussian, area_theorem_allows, frame_drag, gravity_pull, gw_push_distance,
    radiated_mass, remnant_spin, tidal_stretch)
from hawkingsearch.minimal import MinimalConfig, evaporate, schwarzschild_radius, temperature


def test_temperature_is_inverse_mass():
    assert temperature(2.0, 0.2) == pytest.approx(0.1)
    assert temperature(4.0, 0.2) == pytest.approx(temperature(2.0, 0.2) / 2)


def test_evaporation_accelerates_as_mass_shrinks():
    loss = [M - evaporate(M, 0.01) for M in (4.0, 2.0, 1.0, 0.5)]
    assert all(a < b for a, b in zip(loss, loss[1:]))
    # dM = -alpha / M^2 exactly
    assert loss[2] == pytest.approx(0.01)


def test_negative_heat_capacity_runaway():
    """Losing mass makes a hole hotter, which makes it lose mass faster."""
    M, k, alpha = 1.0, 0.2, 0.01
    temps, losses = [], []
    while M > 0.3:
        newM = evaporate(M, alpha)
        temps.append(temperature(newM, k))
        losses.append(M - newM)
        M = newM
    assert np.all(np.diff(temps) > 0)
    assert np.all(np.diff(losses) > 0)


def test_lifetime_matches_hawking_cubic_law():
    """Integrating dM/dt = -alpha/M^2 gives a lifetime of (M0^3 - Mmin^3) / (3 alpha)."""
    alpha, M0, Mmin = 0.001, 1.0, 0.3
    M, steps = M0, 0
    while M > Mmin:
        M = evaporate(M, alpha)
        steps += 1
    assert steps == pytest.approx((M0**3 - Mmin**3) / (3 * alpha), rel=0.02)


def test_schwarzschild_radius_proportional_to_mass_then_saturates():
    cfg = MinimalConfig(rs_coeff=0.01, rs_max=0.05)
    assert schwarzschild_radius(2.0, cfg) == pytest.approx(2 * schwarzschild_radius(1.0, cfg))
    assert schwarzschild_radius(1000.0, cfg) == 0.05


def test_merger_radiates_five_percent():
    assert radiated_mass(3.0, 1.0, 0.05) == pytest.approx(0.2)


@pytest.mark.parametrize("ratio, allowed", [(1, True), (5, True), (17, True), (20, False), (100, False)])
def test_area_theorem_forbids_lopsided_mergers(ratio, allowed):
    Ma, Mb = float(ratio), 1.0
    Mf = Ma + Mb - radiated_mass(Ma, Mb, 0.05)
    assert area_theorem_allows(Ma, Mb, Mf) is allowed


def test_area_theorem_always_allows_lossless_merger():
    for Ma, Mb in [(1, 1), (1000, 1), (0.01, 50)]:
        assert area_theorem_allows(Ma, Mb, Ma + Mb)


def test_equal_mass_remnant_spin_is_069():
    assert remnant_spin(1.0, 1.0, 0.0, 0.0) == pytest.approx(0.69)
    assert remnant_spin(100.0, 1.0, 0.0, 0.0) < 0.05          # tiny companion adds little
    assert remnant_spin(1.0, 1.0, 0.998, 0.998) <= 0.998      # Kerr bound


def test_gw_push_falls_off_with_distance_and_mass():
    near, far = gw_push_distance(1.0, 1.0, 0.05, 0.03), gw_push_distance(1.0, 1.0, 0.3, 0.03)
    light, heavy = gw_push_distance(1.0, 0.5, 0.1, 0.03), gw_push_distance(1.0, 50.0, 0.1, 0.03)
    assert near > far and light > heavy


def test_gravity_pulls_toward_heavier_mass_and_never_overshoots():
    x = np.zeros(3)
    target = np.array([0.1, 0.0, 0.0])
    pull = gravity_pull(x, [target], [1e6], G=1.0, softening=0.01)
    assert pull[0] > 0 and abs(pull[1]) < 1e-12
    assert np.linalg.norm(pull) <= 0.05 + 1e-12            # at most half the gap
    far = np.array([1.0, 0.0, 0.0])                      # far enough that the cap is not hit
    weak = gravity_pull(x, [far], [1.0], G=0.002, softening=0.02)
    strong = gravity_pull(x, [far], [10.0], G=0.002, softening=0.02)
    assert np.linalg.norm(strong) == pytest.approx(10 * np.linalg.norm(weak), rel=1e-6)
    assert np.linalg.norm(strong) > np.linalg.norm(weak)


def test_spaghettification_stretches_along_and_squeezes_across():
    rng = np.random.default_rng(0)
    d, s = 5, 4.0
    e = np.eye(d)[0]
    Z = np.array([tidal_stretch(rng.normal(size=d), e, s) for _ in range(20000)])
    var = Z.var(0)
    assert var[0] == pytest.approx(s**2, rel=0.05)
    assert var[1:].mean() == pytest.approx(s ** (-2 / (d - 1)), rel=0.05)
    # volume preserving: product of the per-axis scale factors is 1
    assert s * (s ** (-1 / (d - 1))) ** (d - 1) == pytest.approx(1.0)


def test_frame_drag_is_a_rotation_in_the_spin_plane():
    plane = np.stack([np.eye(4)[0], np.eye(4)[1]])
    rel = np.array([0.1, 0.0, 0.03, 0.0])
    disp = frame_drag(rel, plane, np.pi / 2)
    new = rel + disp
    assert np.linalg.norm(new) == pytest.approx(np.linalg.norm(rel))     # orbit, not fall in
    assert new == pytest.approx(np.array([0.0, 0.1, 0.03, 0.0]))         # rotated 90 degrees
    assert disp[2] == 0 and disp[3] == 0                                  # only inside the plane


def test_archive_gaussian_weights_better_points_more():
    P = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    F = np.array([0.0, 5.0, 10.0])
    mu, C = archive_gaussian(P, F)
    assert mu[0] < 1.0                      # pulled toward the best point
    assert np.allclose(C, C.T) and np.all(np.linalg.eigvalsh(C) >= -1e-12)


# ---------------------------------------------------------------- second-generation modules
from hawkingsearch.full import expel, parabola_minimum  # noqa: E402


def test_white_hole_expels_points_inside_and_ignores_points_outside():
    rng = np.random.default_rng(0)
    c = np.array([0.5, 0.5, 0.5])
    inside = np.array([0.52, 0.5, 0.5])
    out = expel(inside, c, 0.1, rng)
    assert np.linalg.norm(out - c) >= 0.1
    assert np.allclose((out - c) / np.linalg.norm(out - c), [1, 0, 0])   # pushed straight out
    outside = np.array([0.8, 0.5, 0.5])
    assert np.array_equal(expel(outside, c, 0.1, rng), outside)
    assert np.linalg.norm(expel(c.copy(), c, 0.1, rng) - c) >= 0.1       # even from the centre


def test_penrose_parabola_finds_quadratic_minimum_exactly():
    f = lambda t: 3 * (t - 0.7) ** 2 + 1          # noqa: E731
    assert parabola_minimum(f(-1), f(0), f(1)) == pytest.approx(0.7)
    assert parabola_minimum(f(-1), f(0), f(1), t_max=0.5) == 0.5         # clipped
    assert parabola_minimum(1.0, 2.0, 1.0) is None                       # concave: no minimum
