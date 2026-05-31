# Contributing

Thanks for your interest in esm-trust.

## Dev setup
```bash
pip install -e ".[dev,figures]"     # math layer + pytest + matplotlib
pytest tests/ -q                    # the 16 unit tests (CPU, no model needed)
```
The analysis/math/IO layer (`esm_trust/bench.py` minus `ESMScorer`) is unit-tested on CPU with
synthetic data. The model layer lazy-imports `torch`/`esm`; exercise it via the benchmark on a GPU.

## Regenerating results and figures
Numbers live only in `results/results.csv`, produced by `python -m esm_trust.benchmark`.
Never hand-edit figure values — edit the runner or the panel and re-run, then `python make_figures.py`.
See `DATA.md`.

## PRs
- Keep the math layer free of `torch`/`esm` imports at module top level.
- Add/extend a unit test for any analysis-method change.
- CI (Python 3.10–3.12) must stay green.
