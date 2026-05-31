"""make_figures.py — render every figure from results/results.csv.

Reads numbers from the CSV that ``python -m esm_trust.benchmark`` produced and
plots them; it hard-codes no measurements. A figure renders only when the CSV
has the columns it needs. Each figure is a function so the layout can be unit-
tested (see fig2's collision-free label placement, verified in tests/qc).

    python make_figures.py [RESULTS_CSV] [OUT_DIR]   # defaults: results/results.csv -> figures/
"""
import os, sys, math
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "savefig.dpi": 200, "figure.facecolor": "white", "savefig.facecolor": "white",
    "font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
})
TEAL, CORAL, NAVY, SKY, GREY = "#1f7a6e", "#c1543b", "#244d6e", "#9fbcd4", "#6b6b6b"
RELIABLE = 0.50


def load(csv):
    df = pd.read_csv(csv)
    df["label"] = df["label"].fillna(df["dms_id"])
    return df


def _overlap_pairs(fig, annos):
    """Count pairwise overlaps among the given text/annotation artists (display coords)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bb = [a.get_window_extent(renderer=r) for a in annos]
    return sum(1 for i in range(len(bb)) for j in range(i + 1, len(bb)) if bb[i].overlaps(bb[j]))


def fig1_competence(df, out):
    d = df.dropna(subset=["rho_600M"]).sort_values("rho_600M")
    colors = [TEAL if v >= RELIABLE else CORAL for v in d["rho_600M"]]
    fig, ax = plt.subplots(figsize=(9.2, 0.55 * len(d) + 1.8))
    ax.barh(d["label"], d["rho_600M"], color=colors, edgecolor="white")
    ax.axvline(RELIABLE, color=GREY, ls="--", lw=1.2)
    ax.set_xlim(0, max(0.84, d["rho_600M"].max() + 0.1))
    ax.set_xlabel("Spearman correlation with experiment  (ESM-C 600M)")
    ax.set_title("A sharp competence boundary on variant ranking")
    for i, v in enumerate(d["rho_600M"]):
        ax.text(v + 0.012, i, f"{v:.2f}", va="center", fontsize=11)
    ax.grid(axis="y", visible=False); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(f"{out}/fig1_competence_boundary.png"); plt.close(fig)


def fig2_confidence(df, out, save=True):
    """Keystone scatter. Returns the number of label-overlap pairs (0 == clean)."""
    d = df.dropna(subset=["cross_size_agree", "rho_600M"]).reset_index(drop=True)
    colors = [TEAL if v >= RELIABLE else CORAL for v in d["rho_600M"]]
    fig, ax = plt.subplots(figsize=(9.9, 7.0))
    ax.axhspan(RELIABLE, 1.0, color=TEAL, alpha=0.05)
    ax.axhspan(0, 0.30, color=CORAL, alpha=0.05)
    ax.scatter(d["cross_size_agree"], d["rho_600M"], s=95, c=colors,
               edgecolor="white", lw=1.3, zorder=4)

    xs = d["cross_size_agree"].to_numpy(float)
    ys = d["rho_600M"].to_numpy(float)
    names = [f"{l} ({v:.2f})" for l, v in zip(d["label"], d["rho_600M"])]
    xmin, xmax = xs.min(), xs.max()
    ax.set_xlim(xmin - 0.094, xmax + 0.006)
    ax.set_ylim(0, 0.92)

    # collision-free label y-positions: deterministic min-gap projection
    # (guarantees >= gap spacing for all labels; no oscillation).
    lo, hi, gap = 0.05, 0.88, 0.095
    idx = np.argsort(ys)
    s = ys[idx].astype(float).copy()
    for i in range(1, len(s)):                 # forward: enforce gap upward
        s[i] = max(s[i], s[i-1] + gap)
    if s[-1] > hi:                             # if it overflows the top, slide the block down
        s -= (s[-1] - hi)
    for i in range(len(s) - 2, -1, -1):        # backward: restore gap after the slide
        s[i] = min(s[i], s[i+1] - gap)
    s = np.clip(s, lo, hi)
    ly = np.empty_like(s); ly[idx] = s

    lx = xmin - 0.013                        # tidy left column; leaders fan to points
    annos = []
    for xi, yi, lyi, nm in zip(xs, ys, ly, names):
        ax.annotate("", xy=(xi, yi), xytext=(lx, lyi), textcoords="data",
                    arrowprops=dict(arrowstyle="-", color=GREY, lw=0.6, alpha=0.7,
                                    shrinkA=2, shrinkB=4))           # leader line (decoration)
        t = ax.text(lx, lyi, nm, ha="right", va="center", fontsize=7.8, color="#222")
        annos.append(t)                                             # text-only bbox for QC

    ax.set_xlabel("Model self-consistency  (300M↔600M agreement, no data needed)")
    ax.set_ylabel("True reliability  (Spearman vs experiment)")
    ax.set_title("Confidence ≠ accuracy")
    # explanatory note as a bottom caption (outside the plot area -> can't collide)
    fig.text(0.5, 0.012, "High self-consistency across the entire range of true reliability — "
             "the model fails confidently.", ha="center", fontsize=9.5, style="italic", color="#555")
    fig.subplots_adjust(bottom=0.16)
    ov = _overlap_pairs(fig, annos)
    if save:
        fig.savefig(f"{out}/fig2_confidence_vs_accuracy.png")
    plt.close(fig)
    return ov


def fig3_scaling(df, out):
    d = df.dropna(subset=["rho_300M", "rho_600M"]).sort_values("rho_600M", ascending=False)
    x = np.arange(len(d)); w = 0.38
    fig, ax = plt.subplots(figsize=(max(7.5, 1.15 * len(d) + 2), 5.2))
    ax.bar(x - w/2, d["rho_300M"], w, label="ESM-C 300M", color=SKY, edgecolor="white")
    ax.bar(x + w/2, d["rho_600M"], w, label="ESM-C 600M", color=NAVY, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(d["label"], rotation=20, ha="right")
    ax.set_ylim(0, max(0.82, d["rho_600M"].max() + 0.1)); ax.set_ylabel("Spearman")
    ax.set_title("Scaling 300M → 600M: gains stay inside each regime")
    for xi, a, b in zip(x, d["rho_300M"], d["rho_600M"]):
        ax.text(xi - w/2, a + 0.012, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(xi + w/2, b + 0.012, f"{b:.2f}", ha="center", fontsize=8)
    ax.legend(frameon=False, fontsize=10); ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(f"{out}/fig3_scaling.png"); plt.close(fig)


def se(rho, n):    return math.sqrt((1 - rho**2) / (n - 2))     # == bench.calibration_se
def rec_n(rho, t): return math.ceil((1 - rho**2) / t**2 + 2)    # == bench.recommend_n


def fig4_calibration(out):
    ns = np.unique(np.round(np.logspace(np.log10(10), np.log10(2000), 60)).astype(int))
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for rho, c, off in [(0.7, NAVY, (-6, -15)), (0.5, TEAL, (1, 13)), (0.3, CORAL, (15, -4))]:
        ax.plot(ns, [se(rho, int(n)) for n in ns], color=c, lw=2, label=f"true ρ = {rho}")
        nstar = rec_n(rho, 0.10)
        ax.scatter([nstar], [0.10], color=c, zorder=5)
        ax.annotate(f"n={nstar}", (nstar, 0.10), textcoords="offset points", xytext=off,
                    fontsize=10, color=c)
    ax.axhline(0.10, color=GREY, ls="--", lw=1)
    ax.set_xscale("log"); ax.set_xlabel("measured variants (n)")
    ax.set_ylabel("std. error of the estimated Spearman")
    ax.set_title("Calibration math: ~100 variants pin the correlation to ±0.10")
    ax.legend(frameon=False, fontsize=11)
    fig.tight_layout(); fig.savefig(f"{out}/fig4_calibration.png"); plt.close(fig)


def fig5_masked(df, out):
    d = df.dropna(subset=["rho_wt_marginal", "rho_masked_marginal"])
    if len(d) == 0:
        return
    x = np.arange(len(d)); w = 0.36
    fig, ax = plt.subplots(figsize=(max(6.5, 2.2 * len(d) + 2), 5.0))
    ax.bar(x - w/2, d["rho_wt_marginal"], w, label="wild-type-marginal (cheap)", color=SKY, edgecolor="white")
    ax.bar(x + w/2, d["rho_masked_marginal"], w, label="masked-marginal (field standard)", color=NAVY, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(d["label"])
    ax.set_ylim(0, max(0.6, d[["rho_wt_marginal", "rho_masked_marginal"]].max().max() + 0.1))
    ax.set_ylabel("Spearman"); ax.set_title("The “better” scoring method widens the gap")
    for xi, a, b in zip(x, d["rho_wt_marginal"], d["rho_masked_marginal"]):
        ax.text(xi - w/2, a + 0.01, f"{a:.2f}", ha="center", fontsize=10)
        ax.text(xi + w/2, b + 0.01, f"{b:.2f}", ha="center", fontsize=10)
    ax.legend(frameon=False, fontsize=9.5); ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(f"{out}/fig5_masked_marginal.png"); plt.close(fig)


def main():
    csv = sys.argv[1] if len(sys.argv) > 1 else "results/results.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "figures"
    os.makedirs(out, exist_ok=True)
    df = load(csv)
    written = []
    if df["rho_600M"].notna().any():
        fig1_competence(df, out); written.append("fig1_competence_boundary.png")
    if df["cross_size_agree"].notna().any() and df["rho_600M"].notna().any():
        ov = fig2_confidence(df, out); written.append(f"fig2_confidence_vs_accuracy.png (label-overlap pairs: {ov})")
    else:
        p = f"{out}/fig2_confidence_vs_accuracy.png"
        if os.path.exists(p): os.remove(p)
        print("SKIP fig2: results.csv has no 'cross_size_agree' (run python -m esm_trust.benchmark).")
    if df[["rho_300M", "rho_600M"]].notna().all(axis=1).any():
        fig3_scaling(df, out); written.append("fig3_scaling.png")
    fig4_calibration(out); written.append("fig4_calibration.png")
    if df[["rho_wt_marginal", "rho_masked_marginal"]].notna().all(axis=1).any():
        fig5_masked(df, out); written.append("fig5_masked_marginal.png")
    print("wrote:", written)


if __name__ == "__main__":
    main()
