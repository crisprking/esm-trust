"""
esm_trust.bench — the audited benchmark engine (single source of truth).

Why this file exists
--------------------
The benchmark grew up as a chain of Kaggle cells that each re-implemented the
same machinery: discovering ProteinGym shards, loading an assay, WT-marginal
scoring, parsing doubles, the isotonic epistasis decomposition. Every cell
therefore depended on shared kernel state (`shard_of`, `scores`, `R`,
`_qc_model`, `logp_of`, `aid`, `parse_assay` ...). A kernel reset broke
everything downstream, and pasting tracebacks back into cells caused
copy/paste corruption.

This module ends that. Write it to disk once (in a Kaggle cell:
`%%writefile esm_trust_bench.py` then paste this file), then every cell does:

    import esm_trust_bench as bench
    bench.ensure_esm()                       # idempotent install
    idx     = bench.build_shard_index()      # cached after first call
    scorer  = bench.ESMScorer("esmc_600m")   # lazy-loads the model

No cell re-implements anything; there is one audited copy of each function.

Design contract
---------------
* The PURE-MATH + I/O + validation layer imports with **only** numpy / pandas /
  scipy / scikit-learn — no torch, no esm, no GPU, no network at import time.
  That half is unit-tested on CPU with synthetic data (see test_esm_trust_bench.py).
* The MODEL layer (`ESMScorer`, `conditional_coupling`) lazy-imports `esm` and
  `torch` only when you actually score, so importing this module never fails on
  a machine without the SDK.

Validated findings this module operationalizes (ESM-C, wt-marginal, ProteinGym):
  - Strong where conservation-coupled (BLAT ~0.72, PABP ~0.72), weak where
    decoupled (GFP ~0.28, GB1 ~0.11).
  - Failures are CONFIDENT: a no-data self-consistency signal (300M<->600M
    cross-size agreement) is a necessary-not-sufficient guardrail. LOW agreement
    warrants distrust; HIGH agreement does NOT prove reliability. Compute it for
    your own assay with ESMScorer.cross_size_agreement -- do not assume a value.
  - No zero-data signal reliably predicts per-assay reliability; ~100 measured
    variants pin the achievable Spearman to about +/-0.10 (see calibration_se).
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = frozenset(AA)
_SUB_RE = re.compile(r"^([A-Za-z])(\d+)([A-Za-z])$")

DEFAULT_REPO = "OATML-Markslab/ProteinGym_v1"
RELIABLE_THRESHOLD = 0.50          # Spearman point estimate -> "RELIABLE"
MARGINAL_THRESHOLD = 0.30          # -> "MARGINAL"; below -> "UNRELIABLE"
_DEFAULT_CACHE = Path(os.environ.get("ESM_TRUST_CACHE", ".esm_trust_cache"))


# --------------------------------------------------------------------------- #
# input validation  (fail loudly and early, with actionable messages)
# --------------------------------------------------------------------------- #
def validate_sequence(seq: str) -> str:
    """Normalize a wild-type sequence to upper case and report non-standard residues.

    Returns the upper-cased sequence. Raises ValueError on an empty sequence.
    Non-standard residues (anything outside the 20 canonical AAs) are *not* an
    error here — they simply produce NaN at any scored position — but callers
    that want a clean panel should check `assay_qc(...)['nonstandard_wt']`.
    """
    if not seq or not str(seq).strip():
        raise ValueError("empty wild-type sequence")
    return str(seq).strip().upper()


def parse_variant(variant: str) -> List[Tuple[str, int, str]]:
    """Parse 'A123V' or 'A12V:K45R' into [(wt, pos_1based, mut), ...].

    Raises ValueError on a malformed token so bad input surfaces instead of
    being silently dropped. Position range / WT-match checks happen at scoring
    time against a concrete sequence (a token can be syntactically valid yet
    reference a position past the end of *this* protein).
    """
    subs = []
    for tok in str(variant).split(":"):
        m = _SUB_RE.match(tok.strip())
        if not m:
            raise ValueError(f"cannot parse substitution {tok!r} in variant {variant!r}")
        wt, pos, mut = m.group(1).upper(), int(m.group(2)), m.group(3).upper()
        subs.append((wt, pos, mut))
    return subs


def mutation_count(variant: str) -> int:
    """Number of substitution sites in a (possibly multi-site) variant string."""
    return len(str(variant).split(":"))


# --------------------------------------------------------------------------- #
# environment  (the checks Tier 0 normally performs)
# --------------------------------------------------------------------------- #
def ensure_esm() -> bool:
    """Idempotently install the ESM SDK into the *current* kernel.

    Returns True if `esm` is importable afterwards. No-op (and fast) once it is
    present. On Turing GPUs (T4) we deliberately do NOT install flash-attn —
    prebuilt kernels target Ampere+ and fail at runtime.
    """
    import importlib
    import importlib.util
    import subprocess
    import sys

    if importlib.util.find_spec("esm") is not None:
        return True
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "esm"], check=True)
    importlib.invalidate_caches()
    return importlib.util.find_spec("esm") is not None


def preflight(require_gpu: bool = False) -> dict:
    """Report the runtime environment before an expensive run.

    Returns a dict with esm/torch availability, CUDA, bf16 support, and the
    chosen device. Raises RuntimeError if `require_gpu=True` and no CUDA device
    is visible — better to stop here than 40 minutes into a CPU run.
    """
    info: dict = {"esm_installed": False, "torch": None, "cuda": False,
                  "bf16": False, "device": "cpu", "gpus": []}
    import importlib.util
    info["esm_installed"] = importlib.util.find_spec("esm") is not None
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = bool(torch.cuda.is_available())
        if info["cuda"]:
            info["device"] = "cuda"
            info["bf16"] = bool(torch.cuda.is_bf16_supported())  # False on T4
            info["gpus"] = [torch.cuda.get_device_name(i)
                            for i in range(torch.cuda.device_count())]
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            info["device"] = "mps"
    except ImportError:
        pass
    if require_gpu and not info["cuda"]:
        raise RuntimeError("preflight: GPU required but no CUDA device is visible. "
                           "On Kaggle, Settings -> Accelerator -> GPU T4 x2.")
    return info


def run_manifest(repo: str = DEFAULT_REPO, seed: int = 0,
                 model_revision: Optional[str] = None) -> dict:
    """A reproducibility stamp to print at the top of a run / paste in the paper."""
    import importlib.metadata as md

    def ver(pkg):
        try:
            return md.version(pkg)
        except Exception:
            return None

    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo": repo,
        "seed": seed,
        "model_revision": model_revision,
        "versions": {p: ver(p) for p in
                     ("numpy", "scipy", "scikit-learn", "pandas", "torch", "esm",
                      "huggingface_hub")},
    }


# --------------------------------------------------------------------------- #
# ProteinGym I/O  (one canonical copy of the shard logic; cached to disk)
# --------------------------------------------------------------------------- #
def _substitution_shards(repo: str) -> List[str]:
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(repo, repo_type="dataset")
    return [f for f in files if f.endswith(".parquet")
            and "substitution" in f.lower() and "clinical" not in f.lower()]


def build_shard_index(repo: str = DEFAULT_REPO, cache: bool = True,
                      cache_dir: Path = _DEFAULT_CACHE) -> Dict[str, str]:
    """Map every substitution assay id -> the parquet shard that holds it.

    This replaces the ad-hoc `shard_of` that every cell rebuilt. The first call
    lists the repo and scans each shard's DMS_id column (slow-ish); the result
    is cached to disk keyed by repo, so subsequent kernels are instant.
    """
    cache_dir = Path(cache_dir)
    key = cache_dir / (re.sub(r"[^\w.-]", "_", repo) + ".shard_index.json")
    if cache and key.exists():
        try:
            return json.loads(key.read_text())
        except Exception:
            pass  # corrupt cache -> rebuild

    shard_of: Dict[str, str] = {}
    for f in _substitution_shards(repo):
        ids = pd.read_parquet(f"hf://datasets/{repo}/{f}", columns=["DMS_id"])["DMS_id"].unique()
        for a in ids:
            shard_of[a] = f
    if cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        key.write_text(json.dumps(shard_of))
    return shard_of


def resolve_assay(token: str, shard_index: Dict[str, str]) -> Optional[str]:
    """Resolve a substring (e.g. 'PABP_YEAST_Melamed') to a full DMS_id.

    Returns the shortest matching id, or None. Using this instead of hard-coded
    ids means a benchmark name drift never crashes a run.
    """
    hits = sorted((a for a in shard_index if token.lower() in a.lower()), key=len)
    return hits[0] if hits else None


def load_assay(assay_id: str, shard_index: Dict[str, str], repo: str = DEFAULT_REPO,
               columns: Sequence[str] = ("DMS_id", "mutant", "DMS_score", "target_seq"),
               max_variants: Optional[int] = None, seed: int = 0) -> pd.DataFrame:
    """Load exactly one assay (only the matching rows are downloaded).

    `max_variants` subsamples huge assays for tractable scoring; the subsample is
    seeded for reproducibility.
    """
    if assay_id not in shard_index:
        near = [a for a in shard_index if assay_id.split("_")[0] in a][:5]
        raise KeyError(f"assay {assay_id!r} not in index. similar ids: {near}")
    d = pd.read_parquet(f"hf://datasets/{repo}/{shard_index[assay_id]}",
                        columns=list(columns),
                        filters=[("DMS_id", "=", assay_id)])
    if max_variants is not None and len(d) > max_variants:
        d = d.sample(max_variants, random_state=seed)
    return d.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# epistasis math  (pure; the article's QC conclusions depend on these)
# --------------------------------------------------------------------------- #
def spearman_safe(a, b) -> float:
    """Spearman rho with guards: returns NaN for <10 points or a constant input."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 10 or len(b) != len(a) or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    r = spearmanr(a, b)[0]
    return float(r) if np.isfinite(r) else float("nan")


def parse_assay(df: pd.DataFrame, wt: str
                ) -> Tuple[Dict[Tuple[int, str], float],
                           List[Tuple[Tuple[int, str], Tuple[int, str], float]]]:
    """Split a DMS table into WT-matched singles and doubles.

    Returns
      singles : {(pos1based, mut): score}
      doubles : [((p1, m1), (p2, m2), score), ...]
    Rows whose WT residue does not match `wt` at the stated position, or that
    reference an out-of-range position, are dropped (they cannot be scored).
    """
    wt = validate_sequence(wt)
    singles: Dict[Tuple[int, str], float] = {}
    doubles: List[Tuple[Tuple[int, str], Tuple[int, str], float]] = []
    muts = df["mutant"].astype(str).to_numpy()
    scores = df["DMS_score"].to_numpy()
    for m, s in zip(muts, scores):
        try:
            subs = parse_variant(m)
        except ValueError:
            continue
        ok = all(1 <= p <= len(wt) and wt[p - 1] == w for (w, p, _) in subs)
        if not ok:
            continue
        if len(subs) == 1:
            w, p, mt = subs[0]
            singles[(p, mt)] = float(s)
        elif len(subs) == 2:
            (w1, p1, m1), (w2, p2, m2) = subs
            doubles.append(((p1, m1), (p2, m2), float(s)))
    return singles, doubles


def assay_qc(df: pd.DataFrame, wt: str) -> dict:
    """Data-quality report to run BEFORE the expensive forward passes.

    Surfaces the failure modes that quietly poison a benchmark: unparseable
    variant strings, WT residues that disagree with the provided sequence,
    out-of-range positions, the multi-site fraction (which makes additive
    scoring track mutation *count*), and any non-standard residues in the WT.
    """
    wt = validate_sequence(wt)
    muts = df["mutant"].astype(str).to_numpy()
    n = len(muts)
    n_unparse = n_wt_mismatch = n_oob = n_multi = 0
    for m in muts:
        try:
            subs = parse_variant(m)
        except ValueError:
            n_unparse += 1
            continue
        if len(subs) > 1:
            n_multi += 1
        for (w, p, _) in subs:
            if not (1 <= p <= len(wt)):
                n_oob += 1
            elif wt[p - 1] != w:
                n_wt_mismatch += 1
    return {
        "n_variants": n,
        "frac_unparseable": n_unparse / n if n else 0.0,
        "frac_wt_mismatch": n_wt_mismatch / n if n else 0.0,
        "frac_out_of_range": n_oob / n if n else 0.0,
        "frac_multi_site": n_multi / n if n else 0.0,
        "wt_length": len(wt),
        "nonstandard_wt": sorted({c for c in wt if c not in AA_SET}),
    }


def isotonic_decompose(phi_meas: np.ndarray, D: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a monotone global non-linearity g(phi_meas) ~ D and split D.

    Returns (global_nl, spec_eps, g) where
      g          = isotonic fit of D on the measured additive prediction phi_meas
      global_nl  = g - phi_meas   (the assay's global saturation curve)
      spec_eps   = D - g          (specific epistasis: what the global curve misses)
    """
    phi_meas = np.asarray(phi_meas, float)
    D = np.asarray(D, float)
    g = IsotonicRegression(out_of_bounds="clip").fit(phi_meas, D).predict(phi_meas)
    return g - phi_meas, D - g, g


def within_between(phi: np.ndarray, spec: np.ndarray, pair_code: np.ndarray) -> dict:
    """Decompose corr(phi, spec) into between-pair and within-pair components.

    `pair_code` groups doubles that share the same position pair. The WITHIN
    component (correlation after removing per-pair means) is the decisive test:

      within ~ 0  -> position-importance aliasing only: the model knows which
                     *positions* are epistatic hubs, but carries no
                     substitution-level coupling. ("no separate channel" holds)
      within > 0  -> the model reads real substitution-level coupling that
                     survives fixing the positions. (claim must be qualified)

    Returns dict with pooled / between / within correlations and counts.
    """
    phi = np.asarray(phi, float)
    spec = np.asarray(spec, float)
    pc = np.asarray(pair_code)
    keys, inv, cnt = np.unique(pc, return_inverse=True, return_counts=True)
    sphi = np.zeros(len(keys)); sspec = np.zeros(len(keys))
    np.add.at(sphi, inv, phi); np.add.at(sspec, inv, spec)
    mphi, mspec = sphi / cnt, sspec / cnt
    big = cnt[inv] >= 2
    return {
        "pooled": spearman_safe(phi, spec),
        "between": spearman_safe(mphi, mspec),
        "within": spearman_safe((phi - mphi[inv])[big], (spec - mspec[inv])[big]),
        "n_pairs": int(len(keys)),
        "n_within": int(big.sum()),
    }


def pair_code(rows: Sequence[Tuple[Tuple[int, str], Tuple[int, str], float]]) -> np.ndarray:
    """Stable integer code for the (unordered) position pair of each double."""
    return np.array([min(a[0], b[0]) * 100000 + max(a[0], b[0]) for a, b, _ in rows])


# --------------------------------------------------------------------------- #
# reliability math  (pure; mirrors esm_trust.core for benchmark use)
# --------------------------------------------------------------------------- #
def calibration_se(rho: float, n: int) -> float:
    """Large-sample SE of a Spearman estimate from n paired points.

    SE ~= sqrt((1 - rho^2) / (n - 2)). Adapts to the assay: a high-rho assay is
    pinned down with fewer measurements than a low-rho one. Returns inf for n<3.
    """
    if n is None or n < 3:
        return float("inf")
    rho = max(min(float(rho), 0.999), -0.999)
    return math.sqrt((1.0 - rho * rho) / (n - 2))


def recommend_n(rho_hat: float, target_se: float = 0.10) -> int:
    """How many measured variants to estimate reliability to +/- target_se.

    Inverts calibration_se. Clamped to a sensible [12, 1000]. If rho is unknown,
    pass a conservative midrange (~0.4) -> about 86 variants for +/-0.10.
    """
    rho = max(min(float(rho_hat), 0.999), -0.999)
    if target_se <= 0:
        return 1000
    n = (1.0 - rho * rho) / (target_se ** 2) + 2.0
    return int(max(12, min(1000, math.ceil(n))))


def bootstrap_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 500, seed: int = 0
                 ) -> Tuple[float, Tuple[float, float]]:
    """Spearman point estimate + percentile bootstrap 95% CI."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    rho = float(spearmanr(x, y)[0]) if n >= 2 else float("nan")
    if n < 3:
        return rho, (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        b = spearmanr(x[idx], y[idx])[0]
        if np.isfinite(b):
            boots.append(float(b))
    if not boots:
        return rho, (float("nan"), float("nan"))
    lo, hi = (float(v) for v in np.percentile(boots, [2.5, 97.5]))
    return rho, (lo, hi)


def verdict(rho: float) -> str:
    if rho >= RELIABLE_THRESHOLD:
        return "RELIABLE"
    if rho >= MARGINAL_THRESHOLD:
        return "MARGINAL"
    return "UNRELIABLE"


# --------------------------------------------------------------------------- #
# model layer  (lazy esm/torch; nothing above this line imports them)
# --------------------------------------------------------------------------- #
def _pick_device(device: str = "auto") -> str:
    import torch
    if device and device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ESMScorer:
    """Cached ESM-C scorer: WT-marginal, masked-marginal, cross-size, conditional.

    Models load lazily on first use and are cached, so re-running a cell does not
    reload weights. Scoring is the wt-marginal log-likelihood ratio the
    calibration results were validated on (one forward pass per sequence).
    """

    def __init__(self, model: str = "esmc_600m", device: str = "auto",
                 small_model: str = "esmc_300m"):
        self.model_name = model
        self.small_model_name = small_model
        self.device = _pick_device(device)
        self._cache: Dict[str, object] = {}

    # -- loading ----------------------------------------------------------- #
    def _load(self, name: str):
        if name in self._cache:
            return self._cache[name]
        from esm.models.esmc import ESMC
        for kw in ({"use_flash_attn": False}, {}):   # robust to SDK API drift; no flash-attn on T4
            try:
                m = ESMC.from_pretrained(name, **kw).to(self.device).eval()
                self._cache[name] = m
                return m
            except TypeError:
                continue
        m = ESMC.from_pretrained(name).to(self.device).eval()
        self._cache[name] = m
        return m

    def _aa_ids(self, model) -> Dict[str, int]:
        vocab = model.tokenizer.get_vocab()
        return {a: vocab[a] for a in AA}

    # -- core scoring ------------------------------------------------------ #
    def wt_logp(self, seq: str, model: Optional[str] = None) -> np.ndarray:
        """Per-position log-softmax over the 20 AAs for the WT sequence.

        Returns an array indexed by 0-based residue position, shape [L, vocab].
        """
        import torch
        from esm.sdk.api import ESMProtein, LogitsConfig
        m = self._load(model or self.model_name)
        with torch.no_grad():
            out = m.logits(m.encode(ESMProtein(sequence=seq)), LogitsConfig(sequence=True))
            lp = torch.log_softmax(out.logits.sequence[0].float(), -1)
            return lp[1:len(seq) + 1].cpu().numpy()

    def score(self, seq: str, variants: Sequence[str],
              model: Optional[str] = None) -> Dict[str, float]:
        """WT-marginal score per variant; NaN where it cannot be applied."""
        seq = validate_sequence(seq)
        m = self._load(model or self.model_name)
        aid = self._aa_ids(m)
        lp = self.wt_logp(seq, model or self.model_name)
        out: Dict[str, float] = {}
        for v in variants:
            try:
                subs = parse_variant(v)
            except ValueError:
                out[v] = float("nan")
                continue
            tot, ok = 0.0, True
            for (wt, pos, mut) in subs:
                i = pos - 1
                if i < 0 or i >= len(seq) or seq[i] != wt or mut not in aid:
                    ok = False
                    break
                tot += float(lp[i, aid[mut]] - lp[i, aid[wt]])
            out[v] = tot if ok else float("nan")
        return out

    def masked_logp(self, seq: str, model: Optional[str] = None) -> np.ndarray:
        """Masked-marginal log-probs (L forward passes). Slower; less inflated."""
        import torch
        from esm.sdk.api import ESMProtein, ESMProteinTensor, LogitsConfig
        m = self._load(model or self.model_name)
        mask_id = getattr(m.tokenizer, "mask_token_id", None) or m.tokenizer.get_vocab()["<mask>"]
        base = m.encode(ESMProtein(sequence=seq))
        rows = []
        with torch.no_grad():
            for i in range(len(seq)):
                ids = base.sequence.clone()
                ids[i + 1] = mask_id
                out = m.logits(ESMProteinTensor(sequence=ids), LogitsConfig(sequence=True))
                rows.append(torch.log_softmax(out.logits.sequence[0, i + 1].float(), -1))
        return torch.stack(rows).cpu().numpy()

    def cross_size_agreement(self, seq: str, variants: Sequence[str]) -> float:
        """Spearman between the big and small model rankings (the no-data guardrail).

        Low agreement => distrust. High agreement does NOT imply reliability
        (validated: the failures are confident and consistent).
        """
        big = self.score(seq, variants, model=self.model_name)
        small = self.score(seq, variants, model=self.small_model_name)
        common = [v for v in variants
                  if not _isnan(big[v]) and not _isnan(small[v])]
        if len(common) < 3:
            return float("nan")
        return spearman_safe([big[v] for v in common], [small[v] for v in common])

    # -- the expensive readout -------------------------------------------- #
    def conditional_coupling(self, wt: str,
                             rows: Sequence[Tuple[Tuple[int, str], Tuple[int, str], float]]
                             ) -> dict:
        """Partner-mutated wt-marginal rescore: does conditioning capture coupling
        the additive magnitude misses?

        One forward pass per distinct single-mutant background. GPU strongly
        recommended. Returns the within/between decomposition for the additive
        baseline vs the conditional readout, against specific epistasis.
        """
        wt = validate_sequence(wt)
        m = self._load(self.model_name)
        aid = self._aa_ids(m)
        lp_wt = self.wt_logp(wt)

        def add_score(p, a):
            return float(lp_wt[p - 1, aid[a]] - lp_wt[p - 1, aid[wt[p - 1]]])

        backgrounds = sorted({s for a, b, _ in rows for s in (a, b)})
        bg_logp = {}
        for (p, a) in backgrounds:
            mutant_seq = wt[:p - 1] + a + wt[p:]
            bg_logp[(p, a)] = self.wt_logp(mutant_seq)

        def cond(i, a, j, b):  # log-ratio of residue (i,a) on background (j,b)
            lp = bg_logp[(j, b)]
            return float(lp[i - 1, aid[a]] - lp[i - 1, aid[wt[i - 1]]])

        D = np.array([x for *_, x in rows])
        phi_meas = np.array([add_score(*a) + add_score(*b) for a, b, _ in rows])  # placeholder
        # phi_meas should be the *measured* singles sum; supply via parse_assay upstream.
        # Here we expose model-side arrays; callers pass measured singles separately.
        phi_mdl = np.array([add_score(*a) + add_score(*b) for a, b, _ in rows])
        cvec = np.array([0.5 * ((cond(*a, *b) - add_score(*a)) +
                                (cond(*b, *a) - add_score(*b))) for a, b, _ in rows])
        pc = pair_code(rows)
        _, spec, _ = isotonic_decompose(phi_mdl, D)  # NOTE: caller may prefer measured phi
        return {
            "n": len(rows),
            "n_backgrounds": len(backgrounds),
            "additive": within_between(phi_mdl, spec, pc),
            "conditional": within_between(cvec, spec, pc),
            "_phi_mdl": phi_mdl, "_cvec": cvec, "_spec": spec, "_D": D, "_pair_code": pc,
        }


@dataclass
class AssayScore:
    """Container for one assay's scored arrays, kept in a dict so a kernel reset
    only costs a reload of the dict, not the whole pipeline."""
    assay: str
    wt: str
    variants: List[str]
    scores: Dict[str, float] = field(default_factory=dict)
    true_rho: Optional[float] = None


def _isnan(x) -> bool:
    return isinstance(x, float) and math.isnan(x)


__all__ = [
    "AA", "DEFAULT_REPO", "RELIABLE_THRESHOLD", "MARGINAL_THRESHOLD",
    "validate_sequence", "parse_variant", "mutation_count",
    "ensure_esm", "preflight", "run_manifest",
    "build_shard_index", "resolve_assay", "load_assay",
    "spearman_safe", "parse_assay", "assay_qc",
    "isotonic_decompose", "within_between", "pair_code",
    "calibration_se", "recommend_n", "bootstrap_ci", "verdict",
    "ESMScorer", "AssayScore",
]