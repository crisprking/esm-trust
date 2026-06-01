# esm-trust

**When can you trust ESM-C's zero-shot variant-effect rankings — and when can't you?**
A small, reproducible benchmark and a calibration tool for working scientists.

[![tests](https://github.com/crisprking/esm-trust/actions/workflows/ci.yml/badge.svg)](https://github.com/crisprking/esm-trust/actions/workflows/ci.yml)
&nbsp;MIT&nbsp;·&nbsp;Python ≥ 3.10

---

## The finding in one paragraph

ESM-C ranks the effects of mutations well when a phenotype is coupled to evolutionary
conservation (β-lactamase resistance, PABP binding: Spearman ≈ 0.72 against experiment) and
poorly when it is not (GFP brightness ≈ 0.29; GB1 ≈ 0.12). The danger is that **the model's
internal self-consistency looks the same in both regimes** — a no-data signal like 300M↔600M
cross-size agreement does not separate the reliable assays from the unreliable ones. So you
cannot tell, for free, which regime your assay is in. You *can* buy the answer cheaply: about
**100 measured variants** pin the achievable correlation to roughly ±0.10. This repo is the
benchmark behind that claim and a small tool that does the calibration bookkeeping.

## Provenance — read this first

Every number in the figures and the README is computed by the code here and written to
[`results/results.csv`](results/results.csv). Nothing is hard-coded into the plotting script.
See [`DATA.md`](DATA.md) for exactly which rows are present, how they were produced, and which
figures each column drives.

The shipped `results.csv` contains a **7-assay panel** that spans phenotype classes (four
β-lactamase scans, PABP, GFP, GB1) plus a two-assay masked-marginal robustness check, with the
per-assay **cross-size agreement** column already populated for every assay — so the keystone
"confidence ≠ accuracy" figure renders straight from the shipped file. Re-running the benchmark
regenerates the whole CSV, so every figure traces to numbers this code produced — not to anything
typed by hand.

## Reproduce in ~10 minutes

```bash
git clone https://github.com/crisprking/esm-trust.git
cd esm-trust
pip install -e ".[model,figures]"      # math layer + torch/esm + matplotlib

# 1) regenerate the benchmark (downloads the open 300M & 600M ESM-C models; GPU recommended).
#    No EvolutionaryScale Forge token is needed.
python -m esm_trust.benchmark                       # -> results/results.csv (incl. cross_size_agree)

# 2) (optional) add the masked-marginal robustness check on two assays
python -m esm_trust.benchmark \
    --masked-marginal BLAT_ECOLX_Stiffler_2015 GFP_AEQVI_Sarkisyan_2016

# 3) render every figure from the CSV
python make_figures.py                              # -> figures/*.png
```

On Kaggle, set the accelerator to GPU and run the notebook in
[`notebooks/reproduce.ipynb`](notebooks/reproduce.ipynb); it does the three steps above.

## Use it on your own protein

```python
from esm_trust import report

# no ground truth yet -> you only get the weak guardrail
report(wt_sequence, variants)

# with ~50-100 measured variants -> an honest reliability estimate + verdict
report(wt_sequence, variants, measured={"M1A": 0.12, "K2R": -0.4, ...})
```

`report()` returns a Spearman estimate with a bootstrap CI and a plain
`RELIABLE / MARGINAL / UNRELIABLE` verdict, and tells you how many more variants to measure to
hit ±0.10. `recommend_n(rho_hat)` answers the budgeting question directly.

## What's here

| Path | What it is |
|---|---|
| `esm_trust/core.py` | The lightweight user API: `report`, `recommend_n`. |
| `esm_trust/bench.py` | The audited engine — scoring (WT-marginal, masked-marginal), cross-size agreement, ProteinGym loading, calibration math. The math/IO layer imports with no `torch`/`esm`. |
| `esm_trust/benchmark.py` | The runner. Produces `results/results.csv`; hard-codes no measurements. |
| `make_figures.py` | Reads `results.csv`, renders each figure only when its columns exist. |
| `results/results.csv` | The single source of truth for every reported number. |
| `tests/` | 16 unit tests for the analysis methods (run on CPU, no model). |
| `.github/workflows/ci.yml` | Runs the tests on every push (Python 3.10–3.12). |
| `figures/` | Generated figures. |

## Key results (all from `results/results.csv`)

![competence boundary](figures/fig1_competence_boundary.png)

- **A sharp competence boundary.** Reliable where conservation-coupled, unreliable where the
  phenotype is decoupled (engineered or escape-driven). Even *within* β-lactamase, the achievable
  correlation runs ≈ 0.50–0.72 across four scans of the same protein — the experiment you
  calibrate against matters as much as the protein.
- **Confidence ≠ accuracy.** A no-data self-consistency signal (300M↔600M agreement,
  `ESMScorer.cross_size_agreement`) is a **necessary-not-sufficient** guardrail: low agreement
  warrants distrust, but high agreement does **not** prove reliability. Measure it for your own
  assay; never assume it. *(Figure `fig2` renders directly from the shipped CSV — `cross_size_agree` is already present.)*
- **Scaling stays inside the regime.** 300M → 600M buys a few hundredths on the coupled assays,
  rescues nothing on the decoupled ones, and can regress (GB1). The 6B model — where ESM-C's
  *structural* scaling law is steepest — needs gated Forge access and is out of scope here.
- **Calibration.** ~100 measured variants pin the achievable Spearman to about ±0.10
  (`calibration_se`, `recommend_n`).

## Scope and honesty

This is a stress test of **one component (ESM-C, the language model) on one task (zero-shot
variant ranking)**. It says nothing about ESMFold2 structure prediction or binder design, which
are different models doing a different job. Scoring is the wild-type-marginal log-likelihood
ratio unless a row is explicitly marked masked-marginal; multi-site variants are scored
additively, which caps performance on combinatorial libraries (this is why GB1/Wu, a four-site
library, is a *weak* example and the headline contrast leans on BLAT vs GFP). Reframed
constructively, the result is a **calibration** gap, not a **capability** gap.

## Citation

If this is useful, cite the repository and the underlying ProteinGym benchmark
(Notin et al., 2023). Full references are in the companion write-up.
