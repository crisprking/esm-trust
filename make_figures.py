"""make_figures.py — render every figure from results/results.csv.

This script is deliberately *dumb*: it reads numbers from the CSV that
``python -m esm_trust.benchmark`` produced and plots them. It hard-codes no
measurements. A figure is rendered only when the CSV actually contains the
columns it needs; otherwise it is skipped with a message telling you which
benchmark run fills the gap. This is what keeps the figures honest.

    python make_figures.py [RESULTS_CSV] [OUT_DIR]
    (defaults: results/results.csv  ->  figures/)
"""
import os
import sys
import math

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = sys.argv[1] if len(sys.argv) > 1 else "results/results.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "figures"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "savefig.dpi": 200, "figure.facecolor": "white", "savefig.facecolor": "white",
    "font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
})
TEAL, CORAL, NAVY, SKY, GREY = "#1f7a6e", "#c1543b", "#244d6e", "#9fbcd4", "#6b6b6b"
RELIABLE = 0.50  # threshold used throughout (matches esm_trust.bench.RELIABLE_THRESHOLD)

df = pd.read_csv(CSV)
df["label"] = df["label"].fillna(df["dms_id"])
have = lambda *cols: all(c in df.columns and df[c].notna().any() for c in cols)
written = []


# ---- FIG 1: competence boundary (every assay with a 600M reliability) --------
if have("rho_600M"):
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
    fig.tight_layout(); fig.savefig(f"{OUT}/fig1_competence_boundary.png"); plt.close()
    written.append("fig1_competence_boundary.png")


# ---- FIG 2: confidence vs accuracy (the keystone) — needs cross_size_agree ---
if have("cross_size_agree", "rho_600M"):
    d = df.dropna(subset=["cross_size_agree", "rho_600M"])
    colors = [TEAL if v >= RELIABLE else CORAL for v in d["rho_600M"]]
    fig, ax = plt.subplots(figsize=(8.4, 6.0))
    ax.axhspan(RELIABLE, 1.0, color=TEAL, alpha=0.05)
    ax.axhspan(0, 0.30, color=CORAL, alpha=0.05)
    ax.scatter(d["cross_size_agree"], d["rho_600M"], s=90, c=colors,
               edgecolor="white", lw=1.2, zorder=3)
    # place labels to avoid collisions: nudge by rank within near-equal x clusters
    xpad = (d["cross_size_agree"].max() - d["cross_size_agree"].min() or 0.05)
    ax.set_xlim(d["cross_size_agree"].min() - 0.18 * xpad - 0.02,
                d["cross_size_agree"].max() + 0.06 * xpad + 0.01)
    ds = d.sort_values("rho_600M", ascending=False).reset_index(drop=True)
    for k, r in ds.iterrows():
        dy = 11 if (k % 2 == 0) else -13      # alternate up/down to de-overlap the cluster
        ax.annotate(f"{r['label']} ({r['rho_600M']:.2f})",
                    (r["cross_size_agree"], r["rho_600M"]),
                    xytext=(-10, dy), textcoords="offset points",
                    ha="right", va="center", fontsize=8.5,
                    arrowprops=dict(arrowstyle="-", color=GREY, lw=0.6, alpha=0.6))
    ax.set_xlabel("Model self-consistency  (300M↔600M agreement, no data needed)")
    ax.set_ylabel("True reliability  (Spearman vs experiment)")
    ax.set_title("Confidence ≠ accuracy")
    ax.text(0.02, 0.98, "high self-consistency across the full range of true reliability:\n"
            "the model fails confidently", transform=ax.transAxes, va="top",
            fontsize=9.5, style="italic", color="#444")
    ax.set_ylim(0, 0.88)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig2_confidence_vs_accuracy.png"); plt.close()
    written.append("fig2_confidence_vs_accuracy.png")
else:
    stale = f"{OUT}/fig2_confidence_vs_accuracy.png"
    if os.path.exists(stale):
        os.remove(stale)
    print("SKIP fig2 (confidence vs accuracy): results.csv has no 'cross_size_agree'.")
    print("     Fill it with:  python -m esm_trust.benchmark   (computes it locally, no Forge).")


# ---- FIG 3: scaling 300M vs 600M (assays with both) --------------------------
if have("rho_300M", "rho_600M"):
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
    fig.tight_layout(); fig.savefig(f"{OUT}/fig3_scaling.png"); plt.close()
    written.append("fig3_scaling.png")


# ---- FIG 4: calibration curve (pure math; mirrors esm_trust.bench) -----------
def se(rho, n):       # == bench.calibration_se
    return math.sqrt((1 - rho**2) / (n - 2))
def rec_n(rho, t):    # == bench.recommend_n (unclamped)
    return math.ceil((1 - rho**2) / t**2 + 2)
ns = np.unique(np.round(np.logspace(np.log10(10), np.log10(2000), 60)).astype(int))
fig, ax = plt.subplots(figsize=(8.2, 5.0))
for rho, c, off in [(0.7, NAVY, (4, -16)), (0.5, TEAL, (6, 9)), (0.3, CORAL, (10, -16))]:
    ax.plot(ns, [se(rho, int(n)) for n in ns], color=c, lw=2, label=f"true ρ = {rho}")
    nstar = rec_n(rho, 0.10)
    ax.scatter([nstar], [0.10], color=c, zorder=5)
    ax.annotate(f"n={nstar}", (nstar, 0.10), textcoords="offset points", xytext=off, fontsize=10, color=c)
ax.axhline(0.10, color=GREY, ls="--", lw=1)
ax.set_xscale("log"); ax.set_xlabel("measured variants (n)")
ax.set_ylabel("std. error of the estimated Spearman")
ax.set_title("Calibration math: ~100 variants pin the correlation to ±0.10")
ax.legend(frameon=False, fontsize=11)
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_calibration.png"); plt.close()
written.append("fig4_calibration.png")


# ---- FIG 5: masked-marginal robustness (assays scored both ways) -------------
if have("rho_wt_marginal", "rho_masked_marginal"):
    d = df.dropna(subset=["rho_wt_marginal", "rho_masked_marginal"])
    x = np.arange(len(d)); w = 0.36
    fig, ax = plt.subplots(figsize=(max(6.5, 2.2 * len(d) + 2), 5.0))
    ax.bar(x - w/2, d["rho_wt_marginal"], w, label="wild-type-marginal (cheap)",
           color=SKY, edgecolor="white")
    ax.bar(x + w/2, d["rho_masked_marginal"], w, label="masked-marginal (field standard)",
           color=NAVY, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(d["label"])
    ax.set_ylim(0, max(0.6, d[["rho_wt_marginal", "rho_masked_marginal"]].max().max() + 0.1))
    ax.set_ylabel("Spearman")
    ax.set_title("The “better” scoring method widens the gap")
    for xi, a, b in zip(x, d["rho_wt_marginal"], d["rho_masked_marginal"]):
        ax.text(xi - w/2, a + 0.01, f"{a:.2f}", ha="center", fontsize=10)
        ax.text(xi + w/2, b + 0.01, f"{b:.2f}", ha="center", fontsize=10)
    ax.legend(frameon=False, fontsize=9.5); ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig5_masked_marginal.png"); plt.close()
    written.append("fig5_masked_marginal.png")


print("wrote:", written)
print("figures dir now:", sorted(f for f in os.listdir(OUT) if f.endswith(".png")))
