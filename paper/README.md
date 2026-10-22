# Hawking Cooling: paper and reproduction guide

`hawking_cooling.tex` is the editable manuscript; `hawking_cooling.pdf` is its
compiled version. The paper distinguishes the controlled cooling experiment
from exploratory results for the larger **Hawking search** optimiser. The
appendices describe those experiments without requiring familiarity with the
source tree. This guide contains the software-specific commands and data locations.

The associated software project is <https://github.com/benzae1/hawkingsearch>.
The paths below are relative to the repository root, not to this `paper`
directory.

## Install and build

Use Python 3.10 or newer. From the project root:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The package declares NumPy, SciPy, Matplotlib and scikit-learn dependencies.
Terrain is Matplotlib's bundled Jacksboro Fault elevation sample; Iris and Wine
come from scikit-learn. These tasks need no external data download.

Build the PDF with [Tectonic](https://tectonic-typesetting.github.io/):

```sh
cd paper
tectonic hawking_cooling.tex
```

Tectonic resolves the bibliography and cross-references automatically. Its first
build may need network access to fetch TeX packages. A conventional LaTeX build
can instead use `pdflatex`, `bibtex`, and two further `pdflatex` passes in `paper`.

## Controlled experiment: main-text results

From the project root:

```sh
python paper/cooling_controls.py
```

This runs all eight variants on all nine problems. For each variant it screens
four parameter settings on seeds 100–104, chooses the setting with the lowest
mean rank, then evaluates that setting on seeds 0–14. Each run receives 10,000
objective evaluations, including initialisation and burst children. The script
uses a multiprocessing pool, so this is a full experiment, not a quick build step.
It also checks one seeded run against the minimal implementation with mergers
before starting the experiment.

Outputs (replaced when rerun):

| File | Contents |
| --- | --- |
| `paper/results/controls_raw.csv` | Screening and evaluation records: variant, parameter, problem, seed, final clipped error |
| `paper/results/controls.md` | Selected settings, medians, mean ranks, and comparisons against Hawking |

Only evaluation seeds 0–14 at each selected setting belong in the paper's main
result table. The six synthetic problems have dimension 10; terrain, Iris and
Wine have dimensions 2, 12 and 39 respectively. The existing generated Markdown
header says `d=10`; that shorthand applies only to the synthetic problems.

### Temperature-floor diagnostic (Table 2)

```sh
python paper/temperature_floor.py
```

Reruns the selected Hawking, `cool-earned` and `onefifth-L` settings on seeds
0–14 and writes `paper/results/temperature_floor.md`. It contains the median,
over seeds, of the smallest proposal scale used by any searcher in a run.
It takes about 15 seconds on a multicore machine.

## Extended optimiser: appendix results

The appendix compares a different configuration, with separately tuned defaults
and additional modules. Its “minimal, mergers” row also enables mergers, unlike
the controlled main-text Hawking variant.

```sh
python -m hawkingsearch.benchmark --seeds 15 --budget 10000
python -m hawkingsearch.benchmark --ablation --seeds 15 --budget 10000
```

| File | Contents |
| --- | --- |
| `results/raw.csv` | Baseline comparison, seeds 0–14 |
| `results/summary.md` | Baseline medians, ranks and significance counts |
| `results/ablation_raw.csv` | Module ablation, seeds 0–14 |
| `results/ablation.md` | Complete ablation summary, including contrasts selected for the paper |

These commands replace the corresponding saved outputs. Development used the
same nine problems as reporting. The extended defaults received parameter and
module selection; baseline configurations did not receive an equal tuning budget.
The benchmark runner uses current defaults, so retaining the source revision is
necessary to reproduce historical results.

## Separate endgame-refinement experiment

The final appendix paragraph refers specifically to **late-ringdown**, which
permits finite-difference BFGS refinement only in the final 20% of the evaluation
budget. It does not refer to the earlier unrestricted `ringdown` variant.

```sh
python -m hawkingsearch.refinement_study \
  --seed-start 500 --seeds 15 --budget 10000 \
  --variants original ringdown late-ringdown late-ringdown+quanta 'dual-annealing (scipy)' \
  --reference late-ringdown --out results/late-ringdown-validation
```

The output directory contains `raw.csv`, `summary.md`, `curves.npz`,
`convergence.png` and `settings.json`. The settings record seeds, configuration
and dependency versions. The stored experiment used Python 3.12.14, NumPy 2.5.4,
SciPy 1.18.1, scikit-learn 1.9.1 and Matplotlib 3.11.2. These are the recorded
versions for this experiment, not a version lock for the older studies.

## Interpretation and release requirements

All errors are clipped at zero against a reference value. Clustering references
are the best of 50 k-means++ starts, not certified global minima. Tiny errors and
zeros should be interpreted with floating-point and reference accuracy in mind.

The main control script applies unadjusted two-sided Mann–Whitney tests directly.
The extended benchmark, ablation and refinement summaries first treat arrays
satisfying NumPy's default `allclose` criterion as ties, then test the remaining
comparisons. The manuscript explicitly distinguishes these conventions. A
non-significant result is not an equivalence result. Exact median ties receive
sequential ranks in a fixed implementation order rather than average ranks.

The manuscript cites the project URL, which points to a moving branch. To refer
to the exact code and data behind the reported numbers, cite a tagged release or
commit of this repository.

## Licence

The manuscript is licensed under CC BY 4.0 and the scripts under MIT; see
[LICENSE.md](LICENSE.md).
