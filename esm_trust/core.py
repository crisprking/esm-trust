# esm_trust.core - reliability calibration for ESM-C zero-shot variant-effect predictions
import numpy as np
from scipy.stats import spearmanr

AA = "ACDEFGHIKLMNPQRSTVWY"
_CACHE = {}

def _load(name):
    if name not in _CACHE:
        from esm.models.esmc import ESMC
        m = None
        for kw in ({"use_flash_attn": False}, {}):
            try:
                m = ESMC.from_pretrained(name, **kw); break
            except TypeError:
                continue
        try:
            import torch
            m = m.to("cuda" if torch.cuda.is_available() else "cpu")
        except Exception:
            pass
        _CACHE[name] = m.eval()
    return _CACHE[name]

def _wt_logp(model, seq):
    import torch
    from esm.sdk.api import ESMProtein, LogitsConfig
    with torch.no_grad():
        out = model.logits(model.encode(ESMProtein(sequence=seq)), LogitsConfig(sequence=True))
    return torch.log_softmax(out.logits.sequence[0].float(), -1)[1:len(seq)+1].cpu().numpy()

def _score(seq, variants, name="esmc_600m"):
    model = _load(name)
    aid = {a: model.tokenizer.get_vocab()[a] for a in AA}
    lp = _wt_logp(model, seq)
    out = {}
    for v in variants:
        tot, ok = 0.0, True
        for sub in str(v).split(":"):
            wt, pos, mt = sub[0], int(sub[1:-1]), sub[-1]; i = pos - 1
            if i >= len(seq) or seq[i] != wt or mt not in aid:
                ok = False; break
            tot += float(lp[i, aid[mt]] - lp[i, aid[wt]])
        out[v] = tot if ok else float("nan")
    return out

def recommend_n(rho_hat, target_se=0.10):
    # Minimum measured variants to estimate reliability to +/- target_se (Spearman SE).
    rho = min(max(abs(rho_hat), 0.05), 0.95)
    return int(np.ceil((1 - rho ** 2) / (target_se ** 2) + 2))

def report(seq, variants, measured=None, model="esmc_600m", n_boot=500, verbose=True):
    # Score variants with ESM-C and report how much to trust the ranking.
    # measured: optional {variant: value}. ~50-100 entries give reliability to ~+/-0.10.
    sc = _score(seq, variants, model)
    ranked = sorted([v for v in variants if not np.isnan(sc[v])], key=lambda v: sc[v], reverse=True)
    rep = {"scores": sc, "ranked": ranked, "model": model}
    frac_multi = float(np.mean([(":" in str(v)) for v in variants])) if variants else 0.0
    if frac_multi > 0.25:
        rep["multi_mutant_warning"] = (
            str(int(frac_multi * 100)) + "% of variants are multi-site; additive scoring makes "
            "the ranking partly track mutation count, not just per-mutation quality.")
    if measured:
        common = [v for v in variants if v in measured and not np.isnan(sc[v])]
        x = np.array([sc[v] for v in common]); y = np.array([float(measured[v]) for v in common])
        n = len(common); rho = spearmanr(x, y)[0]
        rng = np.random.default_rng(0); boots = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, n); b = spearmanr(x[idx], y[idx])[0]
            if np.isfinite(b): boots.append(b)
        lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (float("nan"), float("nan")))
        verdict = "RELIABLE" if rho >= 0.5 else "MARGINAL" if rho >= 0.3 else "UNRELIABLE"
        rep.update(mode="few-shot", n_cal=n, rho_hat=float(rho), ci=(float(lo), float(hi)),
                   verdict=verdict, recommend_n=recommend_n(rho))
        if verbose:
            print("[few-shot calibration]  n =", n, "measured variants")
            print("  reliability rho_hat = %+.3f  (95%% CI %+.2f .. %+.2f)" % (rho, lo, hi))
            print("  WARNING: <30 calibration points - very noisy." if n < 30
                  else "  note: 50-100 points recommended for +/-0.10." if n < 50
                  else "  calibration set is adequate.")
            print("  VERDICT: ESM-C is", verdict, "for this assay.")
            if verdict != "RELIABLE":
                print("  -> treat the ranking as a weak prior; the model's confidence will NOT warn you.")
    else:
        sc2 = _score(seq, variants, "esmc_300m")
        common = [v for v in variants if not np.isnan(sc[v]) and not np.isnan(sc2[v])]
        agree = spearmanr([sc[v] for v in common], [sc2[v] for v in common])[0]
        rep.update(mode="no-ground-truth", agreement=float(agree))
        if verbose:
            print("[no calibration data] - only a weak, one-sided guardrail is available")
            print("  cross-size agreement = %+.3f" % agree)
            print("  low agreement (<~0.80) => distrust. HIGH agreement does NOT prove reliability.")
            print("  STRONGLY RECOMMENDED: measure ~50-100 variants and pass them as measured=.")
    if verbose and "multi_mutant_warning" in rep:
        print("  note:", rep["multi_mutant_warning"])
    return rep
