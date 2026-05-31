"""
Unit tests for the pure-math / I/O-free layer of esm_trust_bench.

These run on CPU with no torch, no esm, no network — they validate the analysis
*methods* on synthetic data with KNOWN ground truth, which is what lets you trust
the numbers the GPU pipeline produces. Run:  pytest -q test_esm_trust_bench.py
"""
import math

import numpy as np
import pandas as pd
import pytest

from esm_trust import bench


# --------------------------------------------------------------------------- #
# variant parsing + validation
# --------------------------------------------------------------------------- #
def test_parse_variant_basic_and_multisite():
    assert bench.parse_variant("A24V") == [("A", 24, "V")]
    assert bench.parse_variant("a12v:K41r") == [("A", 12, "V"), ("K", 41, "R")]
    assert bench.mutation_count("A1V:B2C:D3E") == 3


def test_parse_variant_rejects_garbage():
    for bad in ("not-a-variant", "", "A2", "24V", "AAV"):
        with pytest.raises(ValueError):
            bench.parse_variant(bad)


def test_validate_sequence():
    assert bench.validate_sequence("  mkt ".replace(" ", "")) == "MKT"
    with pytest.raises(ValueError):
        bench.validate_sequence("")


# --------------------------------------------------------------------------- #
# parse_assay + assay_qc on a tiny hand-built table
# --------------------------------------------------------------------------- #
def _toy_df():
    # WT = "MKTAYIAK"; positions 1..8
    rows = [
        ("M1K", 0.5),       # single, WT matches
        ("A4V", -1.0),      # single, WT matches
        ("M1K:A4V", -0.4),  # double, both match
        ("Q2A", 9.9),       # WT mismatch (pos2 is K, not Q) -> dropped
        ("A99V", 9.9),      # out of range -> dropped
        ("garbage", 9.9),   # unparseable -> dropped
    ]
    return pd.DataFrame({"mutant": [r[0] for r in rows], "DMS_score": [r[1] for r in rows]})


def test_parse_assay_filters_correctly():
    wt = "MKTAYIAK"
    singles, doubles = bench.parse_assay(_toy_df(), wt)
    assert singles == {(1, "K"): 0.5, (4, "V"): -1.0}
    assert doubles == [((1, "K"), (4, "V"), -0.4)]


def test_assay_qc_surfaces_problems():
    qc = bench.assay_qc(_toy_df(), "MKTAYIAK")
    assert qc["n_variants"] == 6
    assert qc["frac_unparseable"] == pytest.approx(1 / 6)
    assert qc["frac_wt_mismatch"] == pytest.approx(1 / 6)
    assert qc["frac_out_of_range"] == pytest.approx(1 / 6)
    assert qc["frac_multi_site"] == pytest.approx(1 / 6)
    assert qc["wt_length"] == 8
    assert qc["nonstandard_wt"] == []


# --------------------------------------------------------------------------- #
# spearman_safe guards
# --------------------------------------------------------------------------- #
def test_spearman_safe_guards():
    assert math.isnan(bench.spearman_safe([1, 2, 3], [1, 2, 3]))          # <10 points
    assert math.isnan(bench.spearman_safe([1] * 20, list(range(20))))     # constant input
    r = bench.spearman_safe(list(range(20)), list(range(20)))
    assert r == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# isotonic decomposition properties
# --------------------------------------------------------------------------- #
def test_isotonic_is_monotone_and_partitions():
    rng = np.random.default_rng(0)
    phi = rng.normal(size=400)
    D = np.tanh(1.5 * phi) + rng.normal(scale=0.2, size=400)   # monotone saturation + noise
    global_nl, spec, g = bench.isotonic_decompose(phi, D)
    # g is a non-decreasing function of phi
    order = np.argsort(phi)
    assert np.all(np.diff(g[order]) >= -1e-9)
    # exact partition: g + spec == D, and global_nl == g - phi
    assert np.allclose(g + spec, D)
    assert np.allclose(global_nl, g - phi)
    # the global fit removes most of the variance -> residual std < raw std
    assert spec.std() < D.std()


# --------------------------------------------------------------------------- #
# THE DECISIVE QA: within/between distinguishes the two epistasis regimes
# --------------------------------------------------------------------------- #
def _build_regime(kind: str, seed: int = 0):
    """Synthesize doubles with KNOWN epistatic structure.

    Each of 40 position-pairs has a pair-level 'activity' that drives both the
    additive model magnitude (phi) and the specific epistasis (spec). The only
    difference between regimes is whether spec ALSO tracks the within-pair
    variation of phi (substitution-level coupling) or not (position aliasing).
    """
    rng = np.random.default_rng(seed)
    n_pairs, per_pair = 40, 10
    phi, spec, pc = [], [], []
    for p in range(n_pairs):
        pair_activity = rng.normal()                      # hub-ness of this position pair
        within_phi = rng.normal(scale=0.7, size=per_pair)  # substitution-level phi variation
        phi_pair = pair_activity + within_phi
        if kind == "position_only":
            # spec set by pair identity + noise INDEPENDENT of within_phi
            spec_pair = 1.0 * pair_activity + rng.normal(scale=0.5, size=per_pair)
        elif kind == "substitution":
            # spec ALSO tracks within-pair phi -> real coupling at fixed positions
            spec_pair = 1.0 * pair_activity + 0.9 * within_phi + rng.normal(scale=0.15, size=per_pair)
        else:
            raise ValueError(kind)
        phi.extend(phi_pair); spec.extend(spec_pair); pc.extend([p] * per_pair)
    return np.array(phi), np.array(spec), np.array(pc)


def test_within_between_position_only_regime():
    phi, spec, pc = _build_regime("position_only", seed=1)
    out = bench.within_between(phi, spec, pc)
    # between-pair signal is strong (both driven by pair activity)...
    assert out["between"] > 0.4
    # ...but WITHIN ~ 0: no substitution-level coupling. "no separate channel" holds.
    assert abs(out["within"]) < 0.20


def test_within_between_substitution_regime():
    phi, spec, pc = _build_regime("substitution", seed=2)
    out = bench.within_between(phi, spec, pc)
    # now the within-pair correlation survives removing pair means -> real coupling
    assert out["within"] > 0.45
    assert out["n_within"] > 300


def test_within_between_distinguishes_the_two():
    """The crown-jewel guarantee: the decomposition separates the regimes that
    cells 33-34 must tell apart on real assays."""
    _, _, _ = None, None, None
    pos = bench.within_between(*_build_regime("position_only", seed=3))
    sub = bench.within_between(*_build_regime("substitution", seed=3))
    assert sub["within"] - pos["within"] > 0.4


# --------------------------------------------------------------------------- #
# reliability math
# --------------------------------------------------------------------------- #
def test_calibration_se_monotone_decreasing():
    ns = [10, 20, 30, 50, 100, 200, 400]
    ses = [bench.calibration_se(0.5, n) for n in ns]
    assert all(ses[i] > ses[i + 1] for i in range(len(ses) - 1))
    assert math.isinf(bench.calibration_se(0.5, 2))   # too few points


def test_recommend_n_inverts_and_clamps():
    # a strong assay needs fewer measurements than a weak one
    assert bench.recommend_n(0.7, 0.10) < bench.recommend_n(0.4, 0.10)
    # midrange rho lands near ~100 for +/-0.10 (the article's headline number)
    assert 60 <= bench.recommend_n(0.4, 0.10) <= 120
    # clamps
    assert bench.recommend_n(0.99, 0.10) >= 12
    assert bench.recommend_n(0.0, 0.0) == 1000


def test_recommend_n_roundtrips_against_calibration_se():
    # asking for the SE that n variants buys, then asking how many variants that
    # SE needs, should return ~n
    for rho in (0.3, 0.5, 0.7):
        n = 100
        se = bench.calibration_se(rho, n)
        assert abs(bench.recommend_n(rho, se) - n) <= 2


def test_bootstrap_ci_covers_truth():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    y = x + rng.normal(scale=0.8, size=200)      # strong positive association
    rho, (lo, hi) = bench.bootstrap_ci(x, y, n_boot=400, seed=0)
    assert 0.4 < rho < 0.8
    assert lo < rho < hi
    assert np.isfinite(lo) and np.isfinite(hi)


def test_verdict_thresholds():
    assert bench.verdict(0.62) == "RELIABLE"
    assert bench.verdict(0.40) == "MARGINAL"
    assert bench.verdict(0.10) == "UNRELIABLE"


# --------------------------------------------------------------------------- #
# import hygiene: the pure layer must not drag in torch/esm
# --------------------------------------------------------------------------- #
def test_pure_layer_has_no_heavy_imports_at_module_load():
    import sys
    # esm_trust_bench is already imported above; importing it must not have
    # pulled torch or esm into sys.modules.
    assert "torch" not in sys.modules, "torch leaked into the pure layer's import"
    assert "esm" not in sys.modules, "esm leaked into the pure layer's import"