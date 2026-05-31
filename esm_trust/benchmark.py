"""esm_trust.benchmark — the reproducible benchmark runner.

This is the script that produces ``results/results.csv``, the single file every
figure and every headline number in this repository is computed from. Run it once
on a GPU (Kaggle T4 is enough) and every claim in the README and the article
becomes traceable to a CSV that *this code* generated:

    python -m esm_trust.benchmark                 # default panel -> results/results.csv
    python -m esm_trust.benchmark --assays BLAT_ECOLX_Stiffler_2015 GFP_AEQVI_Sarkisyan_2016
    python -m esm_trust.benchmark --masked-marginal BLAT_ECOLX_Deng_2012 GFP_AEQVI_Sarkisyan_2016

No EvolutionaryScale Forge token is needed: the 300M and 600M ESM-C models are
openly downloadable and run locally, and cross-size agreement is computed from
those two. (The 6B model would need Forge and is out of scope here.)

What each row of results.csv contains
-------------------------------------
    dms_id              resolved ProteinGym assay id
    label               short human label (for figures)
    category            ProteinGym selection type, when available
    n                   number of variants actually scored (NaNs dropped)
    rho_300M            Spearman(ESM-C 300M score, experiment)   -- true reliability
    rho_600M            Spearman(ESM-C 600M score, experiment)   -- true reliability
    cross_size_agree    Spearman(300M score, 600M score)         -- no-data guardrail
    rho_wt_marginal     filled only for --masked-marginal assays
    rho_masked_marginal filled only for --masked-marginal assays

Every number is a Spearman correlation this script computed; nothing is hard-coded.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import bench

# The default panel: a cohort chosen to SPAN phenotype classes, so the
# competence boundary is visible rather than cherry-picked. Tokens are resolved
# against the live ProteinGym shard index at run time (resolve_assay), so a
# token that matches exactly one assay is enough; the resolved id is recorded.
DEFAULT_PANEL = [
    ("BLAT_ECOLX_Stiffler_2015", "BLAT (Stiffler)"),
    ("BLAT_ECOLX_Firnberg_2014", "BLAT (Firnberg)"),
    ("BLAT_ECOLX_Jacquier_2013", "BLAT (Jacquier)"),
    ("BLAT_ECOLX_Deng_2012",     "BLAT (Deng)"),
    ("PABP_YEAST_Melamed_2013",  "PABP"),
    ("GFP_AEQVI_Sarkisyan_2016", "GFP"),
    ("SPG1_STRSG_Wu_2016",       "GB1 (Wu)"),
    # Extend the panel by adding tokens here, e.g.:
    # ("KKA2_KLEPN_Melnikov_2014", "KKA2"), ("PTEN_HUMAN_Mighell_2018", "PTEN"),
    # ("TPMT_HUMAN_Matreyek_2018", "TPMT"), ("DLG4_RAT_McLaughlin_2012", "DLG4"),
]

# T4s have no flash-attention and limited memory; keep weights in fp32 on GPU,
# which is exactly the configuration the released numbers were validated under.
def _make_scorer():
    import torch  # noqa: F401  (import error here means GPU layer deps are missing)

    orig_load = bench.ESMScorer._load

    def fp32_load(self, name):
        m = orig_load(self, name)
        if self.device != "cpu":
            m = m.float()
            self._cache[name] = m
        return m

    bench.ESMScorer._load = fp32_load
    return bench.ESMScorer("esmc_600m", small_model="esmc_300m")


def _category(dms_id: str, idx) -> str:
    """Best-effort ProteinGym selection type; empty string if unavailable."""
    try:
        df = bench.load_assay(dms_id, idx)
        for col in ("selection_type", "coarse_selection_type", "selection"):
            if col in df.columns and df[col].notna().any():
                return str(df[col].dropna().iloc[0])
    except Exception:
        pass
    return ""


def _rho_vs_experiment(scorer, wt, variants, y, model):
    s = scorer.score(wt, variants, model=model)
    keep = [i for i, v in enumerate(variants) if not bench._isnan(s[v])]
    if len(keep) < 3:
        return len(keep), float("nan")
    return len(keep), bench.spearman_safe(
        [s[variants[i]] for i in keep], [y[i] for i in keep]
    )


def _rho_masked(scorer, wt, variants, y, model="esmc_600m"):
    """Masked-marginal Spearman (L forward passes) for one assay."""
    lp = scorer.masked_logp(wt, model)  # [L, vocab], 0-based positions
    aid = scorer._aa_ids(scorer._load(model))
    scores = {}
    for v in variants:
        try:
            subs = bench.parse_variant(v)
        except ValueError:
            scores[v] = float("nan")
            continue
        tot, ok = 0.0, True
        for (w, pos, mut) in subs:
            i = pos - 1
            if i < 0 or i >= len(wt) or wt[i] != w or mut not in aid:
                ok = False
                break
            tot += float(lp[i, aid[mut]] - lp[i, aid[w]])
        scores[v] = tot if ok else float("nan")
    keep = [i for i, v in enumerate(variants) if not bench._isnan(scores[v])]
    if len(keep) < 3:
        return float("nan")
    return bench.spearman_safe([scores[variants[i]] for i in keep], [y[i] for i in keep])


def run(panel=None, masked_marginal=None, out="results/results.csv"):
    """Run the benchmark and write results.csv. Returns the DataFrame."""
    bench.ensure_esm()
    idx = bench.build_shard_index()
    scorer = _make_scorer()

    panel = panel if panel is not None else DEFAULT_PANEL
    # Normalize: accept ["ID", ...] or [("ID","label"), ...]
    panel = [(p, p.split("_")[0]) if isinstance(p, str) else p for p in panel]
    mm_set = set(masked_marginal or [])

    rows = []
    print(f"{'label':<16}{'n':>8}{'rho_300M':>10}{'rho_600M':>10}{'cross_size':>12}")
    print("-" * 56)
    for token, label in panel:
        dms_id = bench.resolve_assay(token, idx)
        if not dms_id:
            print(f"{label:<16}  [skipped] token '{token}' did not resolve to an assay")
            continue
        d = bench.load_assay(dms_id, idx)
        wt = bench.validate_sequence(d["target_seq"].iloc[0])
        variants = d["mutant"].tolist()
        y = d["DMS_score"].to_numpy(float)

        n, r300 = _rho_vs_experiment(scorer, wt, variants, y, "esmc_300m")
        _, r600 = _rho_vs_experiment(scorer, wt, variants, y, "esmc_600m")
        agree = scorer.cross_size_agreement(wt, variants)

        row = dict(dms_id=dms_id, label=label, category=_category(dms_id, idx),
                   n=n, rho_300M=r300, rho_600M=r600, cross_size_agree=agree,
                   rho_wt_marginal=np.nan, rho_masked_marginal=np.nan)

        if token in mm_set or dms_id in mm_set:
            row["rho_wt_marginal"] = r600
            row["rho_masked_marginal"] = _rho_masked(scorer, wt, variants, y)
        rows.append(row)
        print(f"{label:<16}{n:>8}{r300:>10.3f}{r600:>10.3f}{agree:>12.3f}")

    df = pd.DataFrame(rows)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nwrote {len(df)} rows -> {out}")
    print("Now regenerate figures from this CSV:  python make_figures.py")
    return df


def _cli(argv=None):
    ap = argparse.ArgumentParser(description="ESM-Trust reproducible benchmark runner.")
    ap.add_argument("--assays", nargs="*", default=None,
                    help="ProteinGym assay tokens (default: the built-in panel).")
    ap.add_argument("--masked-marginal", nargs="*", default=None,
                    help="Assays to ALSO score under masked-marginal (slow; L passes).")
    ap.add_argument("--out", default="results/results.csv")
    args = ap.parse_args(argv)
    panel = args.assays if args.assays else None
    try:
        run(panel=panel, masked_marginal=args.masked_marginal, out=args.out)
    except ImportError as e:
        print("This runner needs the model layer (torch + esm) and a GPU.\n"
              f"Import failed: {e}\nInstall with:  pip install -e '.[model]'  (or run on Kaggle GPU).",
              file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    _cli()
