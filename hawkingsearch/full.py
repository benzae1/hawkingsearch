"""Hawking search, full version: the minimal core loop plus every physics module.

Modules (each can be switched off in `FullConfig`, see `FullConfig.ablate`):

gravity
    Not in the original module list, but spin ("orbit instead of falling
    straight in") and spaghettification both assume holes fall toward heavier
    ones. Each candidate drifts toward every hole that is heavier and better
    (lower f) with a = G * M / r^2, softened, and never more than half the
    distance. "Better" makes f the potential: holes fall downhill, and the
    leading hole is never dragged off its own minimum. Besides mergers this is
    the only channel through which holes share information.

gravitational_waves
    A merger radiates ~5% of the total mass. The radiated energy becomes a
    shockwave that pushes nearby holes outward, push ~ E / (M_h * distance),
    so light holes fly and heavy ones barely move. A pushed hole's next move is
    accepted unconditionally (it was knocked out of its spot), so the shock
    costs no extra evaluations.

area_theorem
    Total horizon area (sum M^2) may not drop in a merger. With 5% radiated,
    (0.95 (Ma+Mb))^2 >= Ma^2 + Mb^2 forbids mergers more lopsided than ~18:1,
    so one heavy hole cannot swallow every light explorer that falls toward
    it. Without radiation a merger always increases area, so this module only
    bites together with gravitational_waves.

page_curve
    Each hole records the points it accreted. Before its Page time (the hole
    has not yet lost half of its peak mass) its radiation (mutations) is
    isotropic, and burst children scatter uniformly. After the Page time,
    mutations follow the covariance of the accreted points, and burst children
    are sampled from the distribution of those good points.

spin
    Holes spin up when they accrete along curved paths, and from mergers (an
    equal-mass merger leaves chi ~ 0.69, as in general relativity). A heavy
    spinning hole drags the candidates of nearby holes around itself in its
    spin plane (frame dragging), with angle ~ chi * (r_s / r)^3.

spaghettification
    The candidate of a hole near a heavier (and better) hole is stretched
    along the line to that hole and squeezed across it, preserving volume.
    Stretch factor 1 + kappa * (r_s / r)^3: the tidal fall-off.

Second-generation modules (added after the first benchmark). Penrose is on
by default; slingshot and white holes are opt-in because neither improved
results (see README):

slingshot
    Holes have inertia. After a successful move a hole keeps a fraction of
    that step as velocity; failures act as friction. Because the step includes
    the gravitational pull, a hole that falls toward a heavy hole picks up
    speed and can swing past it: a gravity assist.

white_holes
    The time reverse of a black hole: nothing can enter. When a hole dies
    after occupying a basin that is worse than the best known point, a white
    hole forms at that basin's bottom. Candidates that would land inside are
    expelled to its edge, and holes inside it when it forms are thrown out.
    White holes are unstable and decay after `wh_lifetime` iterations.

penrose
    In a spinning hole's ergosphere a particle can split: one half falls in,
    the other escapes with extra energy. The leading hole, if it is heavy and
    spinning, probes x + d and x - d as a pair. If f-, f, f+ curve upward it
    fits a parabola and also evaluates the parabola's minimum (a one-
    dimensional Newton step). This removes the precision cap of mass-based
    cooling. Letting every heavy hole do it starves exploration, so by default
    only the leader does.

ringdown (opt-in)
    A cold spinning leader relaxes in the local potential. Paired probes
    estimate slopes and diagonal curvature, secant updates learn coupled
    curvature, and dissipative bounded steps reduce the potential. This is
    a numerical ringdown analogy, not a simulation of GR. Refinement has a
    hard evaluation allocation and pauses in the presence of white holes.

local_centre / hawking_quanta (opt-in)
    Independent gravitational systems use nearby better holes for the centre
    pull; individual radiation quanta excite a single coordinate instead of
    all coordinates. Neither experimental rule is universally beneficial.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .core import Budget, BudgetExhausted, Result, reflect
from .minimal import MinimalConfig, evaporate, schwarzschild_radius, temperature
from .ringdown import ringdown

MODULES = ("gravity", "gravitational_waves", "area_theorem", "page_curve", "spin",
           "spaghettification", "slingshot", "white_holes", "penrose",
           "penrose_page_dirs", "penrose_adaptive", "endgame", "galactic_centre",
           "levy_radiation", "cmb_spawn", "gw_adaptive", "ringdown", "local_centre",
           "hawking_quanta")


@dataclass
class FullConfig(MinimalConfig):
    # tuned defaults: random search over 40 settings on tuning seeds 200-205, best five
    # re-checked on seeds 100-119; the reported benchmark uses seeds 0-14 (never tuned on)
    n_holes: int = 10
    max_holes_factor: float = 1.0
    k: float = 0.15
    alpha: float = 0.001
    rs_max: float = 0.05

    gravity: bool = True
    gravitational_waves: bool = True
    area_theorem: bool = True
    page_curve: bool = True
    spin: bool = True
    spaghettification: bool = True
    penrose: bool = True
    # experimental: implemented and tested, but they did not help in the benchmark
    slingshot: bool = False
    white_holes: bool = False
    # third round: screened on tuning seeds, all seven on together (see README)
    penrose_page_dirs: bool = True   # Penrose probes along the leader's learned directions
    penrose_adaptive: bool = True    # Penrose step grows on success, shrinks on failure
    endgame: bool = True             # last part of the budget goes to the leader only
    galactic_centre: bool = True     # pull toward the population's centre of mass
    levy_radiation: bool = True      # hot holes take heavy-tailed (Cauchy) jumps
    cmb_spawn: bool = True           # new random holes go where nothing has been evaluated
    gw_adaptive: bool = True         # shockwaves weaken while the population is spread out
    ringdown: bool = False           # optional coupled-curvature relaxation of a cold leader
    local_centre: bool = False       # centre attraction uses only nearby better holes
    hawking_quanta: bool = False     # thermal emission excites one coordinate at a time
    ringdown_fraction: float = 0.25  # hard cap on the total refinement evaluation share
    ringdown_evals_per_dim: int = 80 # maximum evaluations in one relaxation episode / d
    ringdown_endgame_only: bool = False # let exploration finish before coupled relaxation

    G: float = 0.004                # gravitational constant (unit-cube units)
    softening: float = 0.02
    gravity_downhill: bool = True   # only fall toward holes that are heavier AND better
    gw_eps: float = 0.05            # radiated fraction of the merged mass
    gw_push: float = 0.1            # shock strength
    gw_range: float = 0.35          # shock reach (unit cube)
    page_fraction: float = 0.5      # Page time: M <= page_fraction * peak mass
    page_inherit: float = 1.0       # informed children share this fraction of the peak mass
    archive_size: int = 40
    spin_mass: float = 2.0          # only holes at least this heavy drag others
    spin_gain: float = 0.3
    drag_strength: float = 1.5      # radians at r = r_s for chi = 1
    tidal_kappa: float = 8.0
    influence_radii: float = 4.0    # neighbours within this many r_s exert tides and drag
    momentum: float = 0.25          # fraction of a successful step kept as velocity
    friction: float = 0.5           # velocity kept after a failed step
    wh_size: float = 0.5            # white-hole radius = wh_size * k / parent peak mass
    wh_lifetime: int = 150          # iterations before a white hole decays
    penrose_chi: float = 0.3        # spin needed for the Penrose process
    penrose_starve: int = 0         # only after this many failed random steps in a row
    penrose_leader_only: bool = True   # only the hole holding the best position
    endgame_fraction: float = 0.2   # share of the budget reserved for the leader
    galactic_strength: float = 0.25 # fraction of the distance to the centre of mass per step
    levy_mass: float = 1.0          # holes lighter than this radiate heavy-tailed jumps
    levy_cap: float = 20.0          # longest jump, in units of the hole's temperature
    cmb_candidates: int = 16        # random proposals compared when spawning in the dark
    gw_diversity_scale: float = 0.05 # shock strength x exp(-spread / scale)

    @classmethod
    def ablate(cls, *off: str, **kw) -> "FullConfig":
        """Config with the named modules switched off."""
        unknown = set(off) - set(MODULES)
        if unknown:
            raise ValueError(f"unknown modules: {unknown}")
        return cls(**{m: False for m in off}, **kw)

    @classmethod
    def maximal(cls, **kw) -> "FullConfig":
        """Every module on, including the experimental ones."""
        return cls(**{m: True for m in MODULES}, **kw)


# ------------------------------------------------------------------ physics
def radiated_mass(Ma: float, Mb: float, eps: float) -> float:
    """Energy carried away by gravitational waves in a merger."""
    return eps * (Ma + Mb)


def area_theorem_allows(Ma: float, Mb: float, Mf: float) -> bool:
    """Hawking's area theorem: horizon area ~ M^2 must not decrease."""
    return Mf**2 >= Ma**2 + Mb**2


def remnant_spin(Ma, Mb, chi_a, chi_b) -> float:
    """Orbital angular momentum (0.69 for equal masses) plus carried spins."""
    Mt = Ma + Mb
    eta = Ma * Mb / Mt**2
    return min(0.998, 0.69 * 4 * eta + (chi_a * Ma**2 + chi_b * Mb**2) / Mt**2)


def gw_push_distance(E, M_h, r, strength, cap=0.25) -> float:
    """Shock displacement: amplitude ~ 1/r, acceleration ~ 1/M of the pushed hole."""
    return min(strength * E / (M_h * (r + 0.02)), cap)


def gravity_pull(x, X, Ms, G, softening) -> np.ndarray:
    """Sum of G*M/r^2 pulls toward the points X (masses Ms), each <= half the gap."""
    pull = np.zeros_like(x)
    for xo, Mo in zip(X, Ms):
        rv = xo - x
        r = float(np.linalg.norm(rv))
        if r > 1e-12:
            a = G * Mo / (r * r + softening**2)
            pull += min(a, 0.5 * r) * rv / r
    return pull


def tidal_stretch(z, e, s) -> np.ndarray:
    """Stretch z by s along unit vector e, squeeze across it (volume preserving)."""
    d = len(z)
    zr = (z @ e) * e
    return s * zr + (z - zr) * s ** (-1.0 / max(d - 1, 1))


def frame_drag(rel, plane, theta) -> np.ndarray:
    """Displacement that rotates `rel` by theta inside the 2-D spin plane."""
    a, b = rel @ plane[0], rel @ plane[1]
    c, s = np.cos(theta), np.sin(theta)
    return (a * c - b * s - a) * plane[0] + (a * s + b * c - b) * plane[1]


def archive_gaussian(P, F):
    """Rank-weighted mean and covariance of accreted points P with values F."""
    P = np.asarray(P)[np.argsort(F)]
    w = np.log(len(P) + 1.0) - np.log(np.arange(1, len(P) + 1))
    w = w / w.sum()
    mu = w @ P
    C = ((P - mu).T * w) @ (P - mu)
    return mu, C


def shaped_noise(C, rng) -> np.ndarray:
    """z ~ N(0, C) rescaled so E|z|^2 = d; blended with the identity so it never degenerates."""
    d = len(C)
    tr = np.trace(C)
    if tr <= 1e-14:
        return rng.normal(size=d)
    C = 0.7 * C * (d / tr) + 0.3 * np.eye(d)
    return np.linalg.cholesky(C + 1e-9 * np.eye(d)) @ rng.normal(size=d)


def expel(u, centre, radius, rng) -> np.ndarray:
    """Move u to just outside a white hole of the given radius (nothing can enter)."""
    rv = u - centre
    r = float(np.linalg.norm(rv))
    if r >= radius:
        return u
    if r < 1e-12:
        rv = rng.normal(size=len(u))
        r = float(np.linalg.norm(rv))
    return centre + rv / r * radius * 1.05


def parabola_minimum(f_minus, f0, f_plus, t_max=4.0):
    """Minimiser t of the parabola through (-1, f-), (0, f0), (1, f+), or None if not convex."""
    c = f_plus + f_minus - 2 * f0
    if not np.isfinite(c) or c <= 0:
        return None
    t = (f_minus - f_plus) / (2 * c)
    return float(np.clip(t, -t_max, t_max))


# ------------------------------------------------------------------ state
@dataclass(eq=False)
class Hole:
    x: np.ndarray
    f: float
    M: float
    uid: int
    peak: float = 0.0
    chi: float = 0.0                       # dimensionless spin, 0 .. 0.998
    plane: np.ndarray | None = None        # 2 x d orthonormal spin plane
    last_step: np.ndarray | None = None
    archive_x: list = field(default_factory=list)
    archive_f: list = field(default_factory=list)
    accreted: int = 0
    starve: int = 0                        # iterations since last accretion
    v: np.ndarray | None = None            # velocity (slingshot module)
    pen_scale: float = 1.0                 # Penrose step multiplier (penrose_adaptive)

    def __post_init__(self):
        self.peak = max(self.peak, self.M)

    def past_page_time(self, frac) -> bool:
        return self.accreted >= 2 and self.M <= frac * self.peak


# ------------------------------------------------------------------ optimiser
def minimize(f, lower, upper, budget=10_000, seed=None, config: FullConfig | None = None,
             callback=None, record: bool = False) -> Result:
    cfg = config or FullConfig()
    rng = np.random.default_rng(seed)
    B = Budget(f, lower, upper, budget)
    d = len(B.lower)
    sq = np.sqrt(d)
    max_holes = int(cfg.max_holes_factor * cfg.n_holes)
    seen: list[np.ndarray] = []             # recent evaluated points (cmb_spawn)
    if cfg.cmb_spawn:
        _unit = B.unit

        def _logged_unit(u):
            seen.append(np.asarray(u, float))
            if len(seen) > 3000:
                del seen[:1000]
            return _unit(u)
        B.unit = _logged_unit
    history: list[dict] = []
    events: list[dict] = []
    candidates: list[tuple] = []
    white: list[list] = []                  # [centre, radius, birth iteration]
    next_id = [0]
    it = 0
    ring_used = 0
    ring_last_eval = -10 * d
    ring_last_x = None
    ring_last_f = np.inf

    def new_hole(u, M):
        next_id[0] += 1
        return Hole(u, B.unit(u), M, next_id[0])

    def rs(h):
        return schwarzschild_radius(h.M, cfg)

    def dark_point():
        """A random point, or with cmb_spawn the emptiest of several proposals."""
        if not cfg.cmb_spawn or not seen:
            return rng.random(d)
        P = rng.random((cfg.cmb_candidates, d))
        S = np.asarray(seen[-1500:])
        dist = np.min(((P[:, None, :] - S[None, :, :]) ** 2).sum(-1), axis=1)
        return P[int(np.argmax(dist))]

    def attracts(o, h):
        return o is not h and o.M > h.M and not (cfg.gravity_downhill and o.f >= h.f)

    def dominant_neighbour(h, holes):
        """Heaviest attracting hole whose tidal zone contains h."""
        best, best_r = None, 0.0
        for o in holes:
            if attracts(o, h):
                r = float(np.linalg.norm(h.x - o.x))
                if r < cfg.influence_radii * rs(o) and (best is None or o.M > best.M):
                    best, best_r = o, r
        return best, best_r

    def spin_up(h, step):
        if h.last_step is not None:
            na, nb = np.linalg.norm(h.last_step), np.linalg.norm(step)
            if na > 0 and nb > 0:
                e1 = h.last_step / na
                v = step - (step @ e1) * e1
                nv = np.linalg.norm(v)
                if nv > 1e-12:
                    curvature = nv / nb            # sine of the turning angle
                    h.chi = min(0.998, h.chi + cfg.spin_gain * curvature * (1 - h.chi))
                    h.plane = np.stack([e1, v / nv])
        h.last_step = step

    def form_white_hole(h, holes):
        """A dead hole that held a worse-than-best basin leaves a white hole behind."""
        if not h.archive_x:
            return
        j = int(np.argmin(h.archive_f))
        if h.archive_f[j] <= B.best_f + 1e-12 * (1 + abs(B.best_f)):
            return                          # never seal off the best basin
        centre = np.asarray(h.archive_x[j])
        radius = float(np.clip(cfg.wh_size * cfg.k / h.peak, 0.01, 0.25))
        white.append([centre, radius, it])
        events.append({"type": "white-hole", "x": B.to_real(centre), "radius": radius})
        for o in holes:                     # everything inside is expelled
            if o is not h and np.linalg.norm(o.x - centre) < radius:
                o.x = reflect(expel(o.x, centre, radius, rng))
                o.f = np.inf

    def repel(u):
        for c, r, _ in white:
            u = expel(u, c, r, rng)
        return u

    def burst(h, holes, new):
        if cfg.white_holes:
            form_white_hole(h, holes)
        room = max_holes - (len(holes) + len(new)) + 1
        n_kids = max(1, min(cfg.burst_children, room))
        informed = cfg.page_curve and h.past_page_time(cfg.page_fraction)
        Tb = temperature(cfg.m_min, cfg.k)
        if informed:
            mu, C = archive_gaussian(h.archive_x, h.archive_f)
            L = np.linalg.cholesky(2.0 * C + (0.02 / sq) ** 2 * np.eye(d))
        kids = []
        for _ in range(n_kids):
            if informed:                    # late radiation carries information
                u = mu + L @ rng.normal(size=d)
            elif cfg.page_curve:            # early radiation is thermal: uniform scatter
                u = h.x + rng.uniform(-Tb, Tb, d)
            else:
                u = h.x + rng.normal(0.0, Tb / sq, d)
            kids.append(reflect(u))
        events.append({"type": "burst", "x": B.to_real(h.x), "informed": informed,
                       "children": [B.to_real(u) for u in kids]})
        kid_mass = cfg.child_mass
        if informed:   # late radiation carries what fell in: children inherit the peak mass
            kid_mass = max(cfg.child_mass, cfg.page_inherit * h.peak / n_kids)
        for j, u in enumerate(kids):
            child = new_hole(u, kid_mass)
            if j == 0:                      # the hole itself is replaced by the first child
                h.x, h.f, h.M, h.peak, h.uid = child.x, child.f, child.M, child.M, child.uid
                h.archive_x, h.archive_f, h.accreted, h.starve = [], [], 0, 0
                h.chi, h.plane, h.last_step, h.v = 0.0, None, None, None
            else:
                new.append(child)

    def gw_shock(site, E, holes, remnant):
        for h in holes:
            if h is remnant:
                continue
            rv = h.x - site
            r = float(np.linalg.norm(rv))
            if r > cfg.gw_range or r < 1e-12:
                continue
            push = gw_push_distance(E, h.M, r, cfg.gw_push * gw_factor[0])
            if push < 1e-3:
                continue
            old = B.to_real(h.x)
            h.x = reflect(h.x + push * rv / r)
            h.f = np.inf                    # unknown until its next move, which is accepted
            events.append({"type": "gw-push", "from": old, "to": B.to_real(h.x)})

    def merge_pass(holes):
        """Merge overlapping horizons, closest pairs first, until none can merge."""
        while len(holes) > 1:
            X = np.array([h.x for h in holes])
            D = np.linalg.norm(X[:, None] - X[None], axis=-1)
            R = np.array([rs(h) for h in holes])
            hit = D < (R[:, None] + R[None])
            np.fill_diagonal(hit, False)
            merged = False
            for flat in np.argsort(np.where(hit, D, np.inf), axis=None):
                i, j = np.unravel_index(flat, D.shape)
                if not hit[i, j]:
                    break
                if i >= j:
                    continue
                a, b = holes[i], holes[j]
                E = radiated_mass(a.M, b.M, cfg.gw_eps) if cfg.gravitational_waves else 0.0
                Mf = a.M + b.M - E
                if cfg.area_theorem and not area_theorem_allows(a.M, b.M, Mf):
                    events.append({"type": "merger-blocked", "x": B.to_real((a.x + b.x) / 2),
                                   "masses": (a.M, b.M)})
                    continue
                keep, other = (a, b) if a.f <= b.f else (b, a)
                events.append({"type": "merger", "x": B.to_real(keep.x), "radiated": E,
                               "masses": (a.M, b.M), "mass": Mf})
                if cfg.spin:
                    keep.chi = remnant_spin(a.M, b.M, a.chi, b.chi)
                    sep = other.x - keep.x
                    if keep.plane is None and np.linalg.norm(sep) > 1e-12:
                        e1 = sep / np.linalg.norm(sep)
                        v = rng.normal(size=d)
                        v -= (v @ e1) * e1
                        keep.plane = np.stack([e1, v / np.linalg.norm(v)])
                keep.M = min(Mf, cfg.m_max)
                keep.peak = max(keep.peak, keep.M)
                keep.archive_x = (keep.archive_x + other.archive_x)[-cfg.archive_size:]
                keep.archive_f = (keep.archive_f + other.archive_f)[-cfg.archive_size:]
                keep.accreted += other.accreted
                holes.remove(other)
                if E > 0:
                    gw_shock(keep.x.copy(), E, holes, keep)
                merged = True
                break
            if not merged:
                return

    def snapshot(it):
        state = {
            "iteration": it, "nfev": B.nfev, "best_f": B.best_f,
            "best_x": None if B.best_x is None else B.best_x.copy(),
            "holes": [{"x": B.to_real(h.x), "f": h.f, "M": h.M, "T": temperature(h.M, cfg.k),
                       "rs": rs(h), "chi": h.chi, "page": h.past_page_time(cfg.page_fraction),
                       "uid": h.uid} for h in holes],
            "events": list(events),
            "candidates": list(candidates),
            "white_holes": [(B.to_real(c), r) for c, r, _ in white],
        }
        if callback:
            callback(state)
        if record:
            history.append(state)

    holes: list[Hole] = []
    gw_factor = [1.0]
    try:
        for _ in range(cfg.n_holes):
            holes.append(new_hole(dark_point(), cfg.m0))
        scale = float(np.std([h.f for h in holes])) or 1.0
        snapshot(0)

        while True:
            it += 1
            events.clear()
            candidates.clear()
            white[:] = [w for w in white if it - w[2] < cfg.wh_lifetime]
            new: list[Hole] = []
            finite = [o for o in holes if np.isfinite(o.f)]
            leader = min(finite, key=lambda o: o.f) if finite else None
            in_endgame = cfg.endgame and B.nfev >= (1 - cfg.endgame_fraction) * B.max_evals
            if cfg.galactic_centre or cfg.gw_adaptive:
                Xs = np.array([o.x for o in holes])
                Ms = np.array([o.M for o in holes])
                centre = (Ms[:, None] * Xs).sum(0) / Ms.sum()
                spread = float(np.mean(np.linalg.norm(Xs - centre, axis=1)))
                if cfg.gw_adaptive:
                    gw_factor[0] = float(np.exp(-spread / cfg.gw_diversity_scale))
            for h in ([leader] if in_endgame and leader is not None else list(holes)):
                T = temperature(h.M, cfg.k)
                # radiation: isotropic, or shaped by accreted points after the Page time
                if cfg.page_curve and h.past_page_time(cfg.page_fraction) and len(h.archive_x) >= 3:
                    z = shaped_noise(archive_gaussian(h.archive_x, h.archive_f)[1], rng)
                    mode = "informed"
                else:
                    z = rng.normal(size=d)
                    mode = "thermal"
                    if cfg.levy_radiation and h.M < cfg.levy_mass:
                        # high-energy tail of the radiation of a nearly evaporated hole
                        z = z * min(abs(rng.standard_cauchy()), cfg.levy_cap)
                        mode = "levy"
                if cfg.hawking_quanta and mode in ("thermal", "levy"):
                    # A single emitted quantum excites one mode. Preserve the
                    # expected squared step length of the isotropic radiation.
                    j = int(rng.integers(d))
                    amplitude = sq * z[j]
                    z = np.zeros(d)
                    z[j] = amplitude
                    mode = "quantum"
                pull = np.zeros(d)
                drag = np.zeros(d)
                if cfg.gravity:
                    att = [o for o in holes if attracts(o, h)]
                    pull = gravity_pull(h.x, [o.x for o in att], [o.M for o in att],
                                        cfg.G, cfg.softening)
                if cfg.galactic_centre and h is not leader:
                    if cfg.local_centre:
                        # Separate gravitational systems retain their own centres;
                        # distant or worse basins must not drag an explorer uphill.
                        neighbours = [o for o in holes if o is not h and o.f < h.f
                                      and np.linalg.norm(o.x - h.x) <= cfg.influence_radii * rs(o)]
                        if neighbours:
                            weights = np.array([o.M for o in neighbours])
                            local = weights @ np.array([o.x for o in neighbours]) / weights.sum()
                            pull = pull + cfg.galactic_strength * (local - h.x)
                    else:
                        pull = pull + cfg.galactic_strength * (centre - h.x)
                if cfg.spaghettification or cfg.spin:
                    big, r = dominant_neighbour(h, holes)
                    if big is not None and r > 1e-12:
                        tide = min((rs(big) / r) ** 3, 1.0)     # capped at the horizon
                        if cfg.spaghettification:
                            z = tidal_stretch(z, (big.x - h.x) / r, 1.0 + cfg.tidal_kappa * tide)
                        if cfg.spin and big.M >= cfg.spin_mass and big.plane is not None:
                            theta = min(cfg.drag_strength * big.chi * tide, np.pi / 3)
                            drag = frame_drag(h.x - big.x, big.plane, theta)
                use_penrose = (cfg.penrose and h.M >= cfg.spin_mass and h.chi >= cfg.penrose_chi
                               and np.isfinite(h.f) and h.starve >= cfg.penrose_starve
                               and (not cfg.penrose_leader_only
                                    or h.f <= min(o.f for o in holes)))
                if in_endgame and cfg.penrose and np.isfinite(h.f):
                    use_penrose = True              # endgame: the leader always refines
                ring_allowance = min(int(cfg.ringdown_fraction * B.max_evals) - ring_used,
                                     cfg.ringdown_evals_per_dim * d, B.remaining)
                use_ringdown = (cfg.ringdown and h is leader and use_penrose
                                and (not cfg.ringdown_endgame_only or in_endgame)
                                and not white and ring_allowance >= 2 * d + 1
                                and B.nfev - ring_last_eval >= 10 * d
                                and (ring_last_x is None
                                     or np.linalg.norm(h.x - ring_last_x) > 0.01
                                     or h.f < ring_last_f - 0.1 * max(abs(ring_last_f), 1e-12)))
                if use_ringdown:
                    start_f, start_eval = h.f, B.nfev

                    def measure(u):
                        value = B.unit(u)
                        if record:
                            candidates.append((B.to_real(u), "ringdown"))
                        return value

                    cand, fc, spent = ringdown(measure, h.x, h.f, ring_allowance,
                                               radius=max(rs(h), T))
                    ring_used += spent
                    ring_last_eval, ring_last_x, ring_last_f = B.nfev, cand.copy(), fc
                    events.append({"type": "ringdown", "uid": h.uid, "nfev": spent,
                                   "start_eval": start_eval, "f_before": start_f, "f_after": fc})
                elif use_penrose:
                    # ergosphere split: probe +/- delta, then the parabola's minimum
                    if cfg.penrose_page_dirs and len(h.archive_x) >= 3:
                        z = shaped_noise(archive_gaussian(h.archive_x, h.archive_f)[1], rng)
                    delta = (T / sq) * z * (h.pen_scale if cfg.penrose_adaptive else 1.0)
                    probes = [h.x + delta, h.x - delta]
                    if cfg.white_holes:
                        probes = [repel(p) for p in probes]
                    probes = [reflect(p) for p in probes]
                    vals = [B.unit(p) for p in probes]
                    t = parabola_minimum(vals[1], h.f, vals[0])
                    if t is not None and abs(abs(t) - 1) > 0.1 and abs(t) > 1e-3:
                        p = h.x + t * delta
                        if cfg.white_holes:
                            p = repel(p)
                        probes.append(reflect(p))
                        vals.append(B.unit(probes[-1]))
                    if record:
                        candidates.extend((B.to_real(p), "penrose") for p in probes)
                    j = int(np.argmin(vals))
                    cand, fc = probes[j], vals[j]
                    if cfg.penrose_adaptive:
                        h.pen_scale = (min(h.pen_scale * 1.5, 4.0) if fc < h.f
                                       else max(h.pen_scale * 0.6, 1e-6))
                else:
                    move = pull + drag + (T / sq) * z
                    if cfg.slingshot and h.v is not None:
                        move = move + h.v
                    cand = h.x + move
                    if cfg.white_holes and white:
                        cand = repel(cand)
                    cand = reflect(cand)
                    fc = B.unit(cand)
                    if record:
                        candidates.append((B.to_real(cand), mode))

                if not np.isfinite(h.f):                         # knocked by a shockwave
                    h.x, h.f = cand, fc
                elif fc < h.f:                                   # accretion
                    dM = cfg.gain * (h.f - fc) / scale
                    step = cand - h.x
                    h.M = min(h.M + min(dM, h.M), cfg.m_max)
                    h.peak = max(h.peak, h.M)
                    h.x, h.f = cand, fc
                    h.archive_x = (h.archive_x + [cand])[-cfg.archive_size:]
                    h.archive_f = (h.archive_f + [fc])[-cfg.archive_size:]
                    h.accreted += 1
                    h.starve = 0
                    if cfg.spin:
                        spin_up(h, step)
                    if cfg.slingshot:
                        h.v = cfg.momentum * step
                else:                                            # evaporation
                    if h.v is not None:
                        h.v = h.v * cfg.friction
                    h.starve += 1
                    if not in_endgame:                       # the leader does not evaporate in the endgame
                        h.M = evaporate(h.M, cfg.alpha, h.starve if cfg.starvation_clock else 1.0)
                if h.M < cfg.m_min:                              # final burst
                    burst(h, holes, new)
            holes.extend(new)
            merge_pass(holes)
            while len(holes) < cfg.n_holes // 2:                 # primordial holes
                holes.append(new_hole(dark_point(), cfg.m0))
            snapshot(it)
    except BudgetExhausted:
        pass
    default = FullConfig()
    diff = [("+" if getattr(cfg, m) else "-") + m for m in MODULES
            if getattr(cfg, m) != getattr(default, m)]
    name = "hawking-full" + (" " + ",".join(diff) if diff else "")
    return B.result(name, history)
