"""Animated visualisations on a real elevation map.

    python -m hawkingsearch.animate                     # full version -> hawking_full.gif
    python -m hawkingsearch.animate --variant minimal
    python -m hawkingsearch.animate --variant maximal   # incl. slingshot + white holes
    python -m hawkingsearch.animate --race              # 4 algorithms side by side

The landscape is the USGS 30 m DEM of the Jacksboro Fault, Tennessee (bundled
with matplotlib). Objective: find the highest summit.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LightSource, LogNorm  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from . import baselines, problems  # noqa: E402
from .full import MODULES, FullConfig  # noqa: E402
from .full import minimize as full_minimize  # noqa: E402
from .minimal import MinimalConfig  # noqa: E402

BG = "#0e0e10"
INK = "#e8e6e1"
MUTED = "#8f8d86"
HAWK = "#3987e5"     # palette slot 1 (dark mode)
PAGE = "#199e70"     # palette slot 3: informed / post-Page radiation
PENROSE = "#9085e9"  # palette slot 7 (dark): Penrose probes
BURST = "#d95926"    # palette slot 2
BLOCK = "#e66767"
TEMP_CMAP = "inferno"


def _terrain_image(p):
    z = p.meta["elevation"]
    ls = LightSource(azdeg=315, altdeg=40)
    rgb = ls.shade(z, cmap=plt.get_cmap("gist_earth"), blend_mode="soft", vert_exag=2,
                   vmin=z.min() - 150, vmax=z.max() + 50)
    return rgb, z


def _style(ax):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color("#3a3a3e")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(INK)


def _record_full(variant, budget, seed):
    p = problems.terrain()
    if variant == "full":
        cfg = FullConfig()
    elif variant == "maximal":
        cfg = FullConfig.maximal()
    else:  # minimal physics, but run through the recorder of the full engine
        cfg = FullConfig.ablate(*MODULES, **vars(MinimalConfig()))
    res = full_minimize(p.f, p.lower, p.upper, budget=budget, seed=seed, config=cfg, record=True)
    return p, res, cfg


def animate_hawking(variant="full", budget=2500, seed=3, out=None, fps=12, max_frames=180):
    p, res, cfg = _record_full(variant, budget, seed)
    hist = res.history
    if len(hist) > max_frames:
        idx = np.unique(np.linspace(0, len(hist) - 1, max_frames).astype(int))
    else:
        idx = np.arange(len(hist))
    rgb, z = _terrain_image(p)
    rows, cols = z.shape
    px = (cols + rows) / 2          # unit-cube distance -> pixels (approx.)
    summit = np.unravel_index(z.argmax(), z.shape)

    fig = plt.figure(figsize=(13, 7.2), facecolor=BG)
    gs = fig.add_gridspec(3, 2, width_ratios=[2.3, 1], hspace=0.55, wspace=0.12,
                          left=0.03, right=0.97, top=0.9, bottom=0.08)
    ax = fig.add_subplot(gs[:, 0])
    axc = fig.add_subplot(gs[0, 1])
    axm = fig.add_subplot(gs[1, 1])
    axt = fig.add_subplot(gs[2, 1])
    for a in (ax, axc, axm, axt):
        _style(a)
    ax.imshow(rgb, origin="upper", extent=(0, cols, rows, 0), interpolation="bilinear")
    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.plot(summit[1], summit[0], marker="^", ms=11, mec="white", mfc="none", mew=1.5, zorder=3)
    ax.annotate(f"true summit {z.max():.0f} m", (summit[1], summit[0]), xytext=(8, -12),
                textcoords="offset points", color="white", fontsize=8)
    title = {"full": "Hawking search, full (default modules)",
             "maximal": "Hawking search, maximal (every module, incl. experimental)",
             "minimal": "Hawking search, minimal"}[variant]
    fig.suptitle(f"{title}  |  find the highest point on a real elevation map", color=INK,
                 fontsize=13, x=0.03, ha="left")
    fig.text(0.03, 0.925, "USGS 30 m DEM, Jacksboro Fault, Tennessee  |  disc = horizon (r_s ∝ M), "
             "ring colour = Hawking temperature T = k/M, dashed circle = step length",
             color=MUTED, fontsize=8.5, ha="left")

    norm = LogNorm(vmin=cfg.k / 30, vmax=cfg.k / cfg.m_min)
    # trim inferno's black end so cold (heavy) holes keep a visible ring
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(plt.get_cmap(TEMP_CMAP)(np.linspace(0.3, 1.0, 256)))

    # convergence panel
    evals = np.array([h["nfev"] for h in hist])
    best = -np.array([h["best_f"] for h in hist])
    axc.set_title("best elevation found", fontsize=10, loc="left", color=INK)
    axc.axhline(z.max(), color=MUTED, lw=1, ls=":")
    axc.text(evals[-1], z.max(), "summit", color=MUTED, fontsize=7, ha="right", va="bottom")
    axc.set_xlim(0, budget)
    axc.set_ylim(np.nanmin(best[np.isfinite(best)]) - 10, z.max() + 15)
    axc.set_xlabel("evaluations", fontsize=8)
    (conv_line,) = axc.plot([], [], color=HAWK, lw=2)

    # mass-temperature panel
    axm.set_title("each hole: mass vs temperature", fontsize=10, loc="left", color=INK)
    Ms = np.logspace(np.log10(cfg.m_min * 0.8), np.log10(400), 50)
    axm.plot(Ms, cfg.k / Ms, color=MUTED, lw=1, ls="--")
    axm.text(Ms[5], cfg.k / Ms[5] * 1.3, "T = k / M", color=MUTED, fontsize=7)
    axm.set_xscale("log")
    axm.set_yscale("log")
    axm.set_xlim(Ms[0], Ms[-1])
    axm.set_ylim(cfg.k / Ms[-1], cfg.k / Ms[0] * 1.5)
    axm.set_xlabel("mass M", fontsize=8)
    axm.set_ylabel("T", fontsize=8)
    mt_scatter = axm.scatter([], [], s=34, c=[], cmap=cmap, norm=norm, edgecolors=BG, linewidths=0.6)

    # event counters
    axt.set_title("events so far", fontsize=10, loc="left", color=INK)
    axt.axis("off")
    counter_text = axt.text(0.0, 0.95, "", color=INK, fontsize=9, va="top", family="monospace",
                            transform=axt.transAxes)

    dyn: list = []
    rings: list = []           # (x, y, age, strength) expanding GW rings
    totals = {"merger": 0, "merger-blocked": 0, "burst": 0, "informed burst": 0, "gw-push": 0,
              "penrose probes": 0, "white-hole": 0}

    def draw(fi):
        for art in dyn:
            art.remove()
        dyn.clear()
        s = hist[idx[fi]]
        # replay events between the previous drawn frame and this one
        lo = idx[fi - 1] + 1 if fi > 0 else 0
        for j in range(lo, idx[fi] + 1):
            for e in hist[j]["events"]:
                t = e["type"]
                if t == "burst" and e["informed"]:
                    totals["informed burst"] += 1
                totals[t] = totals.get(t, 0) + 1
                if t == "merger":
                    rings.append([e["x"][0], e["x"][1], 0, e["radiated"]])
        # candidates (this iteration's radiation)
        cand = s["candidates"]
        if cand:
            C = np.array([c[0] for c in cand])
            mode = np.array([c[1] for c in cand])
            for m, col, size, al in (("thermal", "white", 3, 0.45), ("informed", PAGE, 5, 0.9),
                                     ("levy", BURST, 6, 0.9), ("penrose", PENROSE, 14, 1.0)):
                sel = mode == m
                if sel.any():
                    dyn.append(ax.scatter(C[sel, 0], C[sel, 1], s=size, c=col, alpha=al,
                                          lw=0.5 if m == "penrose" else 0, ec="white", zorder=4))
            totals["penrose probes"] += int((mode == "penrose").sum())
        # white holes: nothing can enter
        for c, r in s.get("white_holes", []):
            dyn.append(ax.add_patch(Circle((c[0], c[1]), r * px, fill=True, fc=(1, 1, 1, 0.14),
                                           ec=(1, 1, 1, 0.9), lw=1.8, zorder=5)))
        # events of this iteration
        for e in s["events"]:
            if e["type"] == "burst":
                col = PAGE if e["informed"] else BURST
                for ch in e["children"]:
                    dyn.append(ax.plot([e["x"][0], ch[0]], [e["x"][1], ch[1]], color=col, lw=0.8,
                                       alpha=0.8, zorder=5)[0])
                dyn.append(ax.scatter([e["x"][0]], [e["x"][1]], marker="*", s=160, c=col,
                                      edgecolors="white", linewidths=0.6, zorder=8))
            elif e["type"] == "merger-blocked":
                dyn.append(ax.scatter([e["x"][0]], [e["x"][1]], marker="x", s=14, c=BLOCK, lw=1,
                                      alpha=0.7, zorder=6))
            elif e["type"] == "gw-push":
                dyn.append(ax.annotate("", xy=e["to"], xytext=e["from"], zorder=6,
                                       arrowprops=dict(arrowstyle="->", color="#c3c2b7", lw=0.8)))
        # gravitational-wave rings
        for r in rings:
            rad = 6 + 9 * r[2]
            dyn.append(ax.add_patch(Circle((r[0], r[1]), rad, fill=False, ec="#9ec5f4",
                                           lw=1.5 * (1 - r[2] / 6), alpha=1 - r[2] / 6, zorder=6)))
            r[2] += 1
        rings[:] = [r for r in rings if r[2] < 6]
        # holes
        for h in s["holes"]:
            x, y = h["x"]
            col = cmap(norm(np.clip(h["T"], norm.vmin, norm.vmax)))
            rad = max(h["rs"] * px, 2.2)
            dyn.append(ax.add_patch(Circle((x, y), h["T"] * px, fill=False, ec=col, lw=0.6,
                                           ls=(0, (2, 3)), alpha=0.55, zorder=5)))
            # horizon: translucent so the terrain stays readable; ring = temperature
            dyn.append(ax.add_patch(Circle((x, y), rad, fill=True, fc="black", alpha=0.62 if rad > 6 else 1.0,
                                           ec="none", zorder=6)))
            dyn.append(ax.add_patch(Circle((x, y), rad, fill=False, ec=col, lw=2.6, zorder=7)))
            if h["page"]:
                dyn.append(ax.add_patch(Circle((x, y), rad + 3, fill=False, ec=PAGE, lw=1.2, zorder=7)))
            if h["chi"] > 0.3 and h["M"] >= 2:
                dyn.append(ax.annotate("", xy=(x + rad * 2.1, y), xytext=(x, y - rad * 2.1),
                                       zorder=7, arrowprops=dict(arrowstyle="->", color="white",
                                                                 lw=0.9, connectionstyle="arc3,rad=-0.45")))
        if getattr(cfg, "galactic_centre", False) and s["holes"]:
            Xh = np.array([h["x"] for h in s["holes"]])
            Mh = np.array([h["M"] for h in s["holes"]])
            gc = (Mh[:, None] * Xh).sum(0) / Mh.sum()
            dyn.append(ax.scatter([gc[0]], [gc[1]], marker="o", s=70, facecolors="none",
                                  edgecolors="#eda100", linewidths=1.6, zorder=9))
            dyn.append(ax.scatter([gc[0]], [gc[1]], marker=".", s=20, c="#eda100", zorder=9))
        if s["best_x"] is not None:
            bx = s["best_x"]
            dyn.append(ax.scatter([bx[0]], [bx[1]], marker="+", s=120, c="white", lw=1.6, zorder=9))
        conv_line.set_data(evals[: idx[fi] + 1], best[: idx[fi] + 1])
        H = s["holes"]
        mt_scatter.set_offsets(np.array([[h["M"], h["T"]] for h in H]))
        mt_scatter.set_array(np.array([h["T"] for h in H]))
        txt = (f"iteration {s['iteration']:4d}   evals {s['nfev']:5d}\n"
               f"holes alive      {len(H):4d}\n"
               f"bursts           {totals['burst']:4d}  (informed {totals['informed burst']})\n"
               f"mergers          {totals['merger']:4d}\n")
        if variant != "minimal":
            txt += (f"blocked (area)   {totals['merger-blocked']:4d}\n"
                    f"shockwave pushes {totals['gw-push']:4d}\n"
                    f"Penrose probes   {totals['penrose probes']:4d}\n")
        if variant == "maximal":
            txt += f"white holes      {totals['white-hole']:4d}\n"
        if getattr(cfg, "endgame", False) and s["nfev"] >= (1 - cfg.endgame_fraction) * budget:
            txt += "ENDGAME: only the leader moves\n"
        txt += f"\nbest so far  {-s['best_f']:7.1f} m"
        counter_text.set_text(txt)
        return dyn

    # legend (static)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], marker="o", ls="", mfc="black", mec=cmap(0.85), mew=3, ms=9, label="black hole"),
        Line2D([], [], marker="o", ls="", ms=3, mfc="white", mec="none", label="thermal radiation"),
        Line2D([], [], marker="*", ls="", ms=12, mfc=BURST, mec="white", label="final burst"),
        Line2D([], [], marker="+", ls="", ms=10, mec="white", mew=1.6, label="best so far"),
    ]
    if variant != "minimal":
        handles[2:2] = [Line2D([], [], marker="o", ls="", ms=4, mfc=PAGE, mec="none",
                               label="informed radiation (after Page time)"),
                        Line2D([], [], marker="o", ls="", ms=6, mfc=PENROSE, mec="white", mew=0.5,
                               label="Penrose probes (leader)"),
                        Line2D([], [], marker="o", ls="", ms=4, mfc=BURST, mec="none",
                               label="heavy-tailed jump (hot hole)"),
                        Line2D([], [], marker="o", ls="", ms=8, mfc="none", mec="#eda100", mew=1.6,
                               label="centre of mass")]
        handles += [Line2D([], [], marker="o", ls="", mfc="none", mec="#9ec5f4", ms=10,
                           label="merger: gravitational wave"),
                    Line2D([], [], marker="x", ls="", mec=BLOCK, ms=6, label="merger blocked (area theorem)")]
    if variant == "maximal":
        handles.append(Line2D([], [], marker="o", ls="", ms=10, mfc=(1, 1, 1, 0.25), mec="white",
                              label="white hole (no entry)"))
    leg = ax.legend(handles=handles, loc="lower left", fontsize=7.5, facecolor=BG, edgecolor="#3a3a3e",
                    framealpha=0.85, labelcolor=INK)
    leg.set_zorder(20)

    out = Path(out or f"hawking_{variant}.gif")
    save_gif(fig, draw, len(idx), out, fps)
    return out, res


def save_gif(fig, draw, n_frames, out: Path, fps, dpi=80):
    """Render frames and write a GIF with one shared palette.

    Matplotlib's Pillow writer quantises each frame separately, so the static
    terrain gets a different dither every frame and nothing compresses. A
    shared palette without dithering lets unchanged pixels stay identical.
    """
    from PIL import Image
    fig.set_dpi(dpi)
    frames = []
    for i in range(n_frames):
        draw(i)
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
    plt.close(fig)
    # palette from a mosaic of a few frames so late-appearing colours are covered
    sample = [frames[j] for j in np.linspace(0, n_frames - 1, 4).astype(int)]
    w, h = sample[0].size
    mosaic = Image.new("RGB", (w, h * len(sample)))
    for j, im in enumerate(sample):
        mosaic.paste(im, (0, h * j))
    pal = mosaic.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    idx = [np.asarray(im.quantize(palette=pal, dither=Image.Dither.NONE)) for im in frames]
    palette = pal.getpalette()[: 255 * 3] + [0, 0, 0]          # index 255 = transparent
    out_frames = []
    for j, a in enumerate(idx):
        b = a.copy()
        if j:
            b[a == idx[j - 1]] = 255        # unchanged pixels -> transparent, compress to ~nothing
        im = Image.fromarray(b.astype(np.uint8), mode="P")
        im.putpalette(palette)
        out_frames.append(im)
    out_frames[0].save(out, save_all=True, append_images=out_frames[1:], duration=int(1000 / fps),
                       loop=0, optimize=False, disposal=1, transparency=255)


def _record_points(algo, p, budget, seed):
    """Run any optimiser and capture all evaluated points in order."""
    pts = []

    def f(x):
        pts.append(np.array(x, float))
        return p.f(x)
    res = algo(f, p.lower, p.upper, budget=budget, seed=seed)
    return np.array(pts), res


def animate_race(budget=2500, seed=3, out="race.gif", fps=12, frames=120):
    """Hawking full vs minimal vs simulated annealing vs PSO on the same terrain."""
    from .full import minimize as fmin
    from .minimal import minimize as mmin
    p = problems.terrain()
    rgb, z = _terrain_image(p)
    rows, cols = z.shape
    summit = np.unravel_index(z.argmax(), z.shape)
    algos = [("Hawking full", fmin), ("Hawking minimal", mmin),
             ("Simulated annealing", baselines.simulated_annealing), ("PSO", baselines.pso)]
    runs = [(name, *_record_points(a, p, budget, seed)) for name, a in algos]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.6), facecolor=BG)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.02, wspace=0.04, hspace=0.12)
    fig.suptitle(f"Same terrain, same budget ({budget} evaluations): where each method looks",
                 color=INK, fontsize=13, x=0.02, ha="left")
    fig.text(0.02, 0.93, "white dots = last 150 evaluations, + = best so far, triangle = true summit",
             color=MUTED, fontsize=9, ha="left")
    arts = []
    for ax, (name, pts, res) in zip(axes.flat, runs):
        _style(ax)
        ax.imshow(rgb, origin="upper", extent=(0, cols, rows, 0))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.plot(summit[1], summit[0], marker="^", ms=10, mec="white", mfc="none", mew=1.4)
        sc = ax.scatter([], [], s=10, c="white", ec=BG, lw=0.4, alpha=0.9)
        (bm,) = ax.plot([], [], marker="+", ms=14, mew=2, color=HAWK if "Hawking" in name else BURST)
        tt = ax.set_title(name, color=INK, fontsize=11, loc="left")
        arts.append((sc, bm, tt, name, pts))

    cuts = np.linspace(1, budget, frames).astype(int)
    fvals = {name: np.array([p.f(q) for q in pts]) for name, pts, _ in runs}

    def draw(fi):
        n = cuts[fi]
        for sc, bm, tt, name, pts in arts:
            m = min(n, len(pts))
            win = pts[max(0, m - 150):m]
            sc.set_offsets(win if len(win) else np.empty((0, 2)))
            if m:
                fs = fvals[name][:m]
                b = int(np.argmin(fs))
                bm.set_data([pts[b, 0]], [pts[b, 1]])
                tt.set_text(f"{name}   best {-fs[b]:.0f} m  (summit {z.max():.0f} m)")
        return []

    save_gif(fig, draw, frames, Path(out), fps, dpi=70)
    return Path(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=["full", "maximal", "minimal"], default="full")
    ap.add_argument("--race", action="store_true", help="4-way race instead of a single run")
    ap.add_argument("--budget", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.race:
        print("wrote", animate_race(a.budget, a.seed, a.out or "race.gif"))
    else:
        out, res = animate_hawking(a.variant, a.budget, a.seed, a.out)
        print(f"wrote {out}   best elevation {-res.f:.1f} m after {res.nfev} evaluations")


if __name__ == "__main__":
    main()
