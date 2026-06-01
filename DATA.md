# DATA.md — provenance of `results/results.csv`

Every figure and every headline number in this repository is computed from
`results/results.csv`. This file documents exactly what that CSV contains, how each
column is produced, and which figure consumes it — so a reader can audit the claims
without trusting any number that was typed by hand.

## Columns

| Column | Meaning | Produced by |
|---|---|---|
| `dms_id` | ProteinGym deep-mutational-scanning assay id (resolved). | `bench.resolve_assay` against the live ProteinGym shard index. |
| `label` | Short human label used in figures. | The benchmark panel definition. |
| `category` | ProteinGym selection type, when available. | `bench.load_assay` (best-effort; may be blank). |
| `n` | Variants actually scored (NaN scores dropped). | `ESMScorer.score`. |
| `rho_300M` | Spearman(ESM-C 300M score, experiment) — **true reliability**. | `ESMScorer.score(model="esmc_300m")` + `spearman_safe`. |
| `rho_600M` | Spearman(ESM-C 600M score, experiment) — **true reliability**. | `ESMScorer.score(model="esmc_600m")` + `spearman_safe`. |
| `cross_size_agree` | Spearman(300M score, 600M score) — the **no-data guardrail**. | `ESMScorer.cross_size_agreement`. |
| `rho_wt_marginal` | WT-marginal Spearman (only on `--masked-marginal` assays). | runner. |
| `rho_masked_marginal` | Masked-marginal Spearman (only on `--masked-marginal` assays). | runner (`L` forward passes). |

All ρ values are Spearman correlations computed by `esm_trust.benchmark`. There are no
hand-entered measurements anywhere in the figure pipeline.

## Which figure needs which column

| Figure | Requires | Renders when |
|---|---|---|
| `fig1_competence_boundary.png` | `rho_600M` | always (shipped CSV has it) |
| `fig2_confidence_vs_accuracy.png` (keystone) | `cross_size_agree`, `rho_600M` | **after** you run the benchmark |
| `fig3_scaling.png` | `rho_300M`, `rho_600M` | always |
| `fig4_calibration.png` | — (pure math) | always |
| `fig5_masked_marginal.png` | `rho_wt_marginal`, `rho_masked_marginal` | when a masked-marginal run is present |

`make_figures.py` skips any figure whose columns are missing rather than inventing values.

## State of the shipped CSV

The committed `results.csv` contains a **7-assay panel** with verified `rho_300M` / `rho_600M`,
a populated `cross_size_agree` column for **every** assay, and a two-assay **masked-marginal**
robustness check (BLAT/Stiffler and GFP). Every figure — including the keystone "confidence ≠
accuracy" plot — renders directly from this shipped CSV; no extra run is needed to reproduce them.
Re-running

```bash
python -m esm_trust.benchmark
```

recomputes the entire CSV — including `cross_size_agree` for every assay — so the whole table is
reproducible end to end from measured numbers.

To extend the panel (e.g. to the broader ProteinGym selection-type cohort), add assay tokens to
`DEFAULT_PANEL` in `esm_trust/benchmark.py` and re-run; the runner resolves each token against
ProteinGym and records the resolved `dms_id`.

## A note on the masked-marginal rows

The shipped masked-marginal check ran on **BLAT/Stiffler** (the headline β-lactamase scan,
ρ ≈ 0.71) and GFP — the same β-lactamase assay named in the headline. The conclusion — the
field-standard scoring widens the gap (GFP drops 0.29 → 0.14 while BLAT/Stiffler edges up
0.71 → 0.74) — does not depend on which BLAT scan is used. To re-run the same pair:

```bash
python -m esm_trust.benchmark --masked-marginal BLAT_ECOLX_Stiffler_2015 GFP_AEQVI_Sarkisyan_2016
```

and `fig5` will redraw from the same pair.
