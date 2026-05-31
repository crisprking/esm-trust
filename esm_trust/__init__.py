"""esm-trust — reliability calibration for ESM-C zero-shot variant-effect predictions."""
from .core import report, recommend_n
from .bench import (
    ESMScorer, calibration_se, verdict, bootstrap_ci,
    build_shard_index, load_assay, resolve_assay, spearman_safe,
    RELIABLE_THRESHOLD, MARGINAL_THRESHOLD,
)

__all__ = [
    "report", "recommend_n", "ESMScorer", "calibration_se", "verdict",
    "bootstrap_ci", "build_shard_index", "load_assay", "resolve_assay",
    "spearman_safe", "RELIABLE_THRESHOLD", "MARGINAL_THRESHOLD",
]
__version__ = "0.2.0"
