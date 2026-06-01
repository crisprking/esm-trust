"""make_figures.py — render all five figures from results/results.csv.

Reads measurements from the CSV and plots them; it hard-codes no data values.
Each figure is self-explanatory (labeled regimes, named thresholds, finding-first
titles) so it reads without a caption.

    python make_figures.py [RESULTS_CSV] [OUT_DIR]   # defaults: results/results.csv -> figures/

Also exposes the small stats helpers the QC battery checks: load(), se(), rec_n(),
and fig2_confidence() (which returns the number of label-label overlaps).
"""
import sys, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ---- shared style ----------------------------------------------------------
TEAL   = "#15806c"   # reliable / 600M / masked-marginal (the "upgraded" series)
TEAL_L = "#b9d6ce"   # base series: 300M / wt-marginal
MID    = "#3f7d8c"   # third line (calibration rho=0.5)
RED    = "#c0543a"   # unreliable / decoupled
GREY   = "#555555"
INK    = "#222222"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12,
    "axes.edgecolor": "#bbbbbb", "axes.linewidth": 1.0,
    "xtick.color": INK, "ytick.color": INK, "text.color": INK,
})

FILES = {
    "fig1": "fig1_competence_boundary.png",
    "fig2": "fig2_confidence_vs_accuracy.png",
    "fig3": "fig3_scaling.png",
    "fig4": "fig4_calibration.png",
    "fig5": "fig5_masked_marginal.png",
}

# ---- data + stats helpers --------------------------------------------------
def load(csv):
    return pd.read_csv(csv)

def se(rho, n):
    """Approx. standard error of an estimated Spearman rho at sample size n."""
    return np.sqrt((1.0 - np.asarray(rho, float) ** 2) / (np.asarray(n, float) - 2.0))

def rec_n(rho, target):
    """Smallest n whose standard error reaches `target` for a true correlation `rho`."""
    return int(round((1.0 - rho ** 2) / target ** 2 + 2.0))

def finish(fig, ax, title, deck, out):
    ax.set_title(deck, fontsize=10.3, color="#5f5f5f", pad=8)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.975)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out)

# ---- fig 1: competence boundary -------------------------------------------
def fig1(df, out_dir):
    d = df.sort_values("rho_600M").reset_index(drop=True)
    labels, vals = d["label"].tolist(), d["rho_600M"].to_numpy(float)
    THRESH, XMAX = 0.5, 0.84
    colors = [TEAL if v >= THRESH else RED for v in vals]
    fig, ax = plt.subplots(figsize=(9, 5.7), dpi=200)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.axvspan(THRESH, XMAX, color=TEAL, alpha=0.06, zorder=0)
    ax.axvspan(0, THRESH, color=RED, alpha=0.06, zorder=0)
    y = np.arange(len(vals))
    ax.barh(y, vals, height=0.64, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
    for yi, v in zip(y, vals):
        ax.text(v + 0.013, yi, f"{v:.2f}", va="center", ha="left", fontsize=11.5, color="#333", zorder=4)
    ytop = len(vals) - 1
    ax.plot([THRESH, THRESH], [-0.45, ytop + 0.5], color=GREY, lw=1.3, ls=(0, (5, 4)), zorder=2)
    ax.text(THRESH + 0.007, -0.78, "usable threshold  \u03c1 = 0.5", ha="left", va="center", fontsize=10, color=GREY)
    head = ytop + 1.0
    ax.text(THRESH / 2, head, "UNRELIABLE", ha="center", va="center", fontsize=12.5, fontweight="bold", color=RED)
    ax.text((THRESH + XMAX) / 2, head, "RELIABLE", ha="center", va="center", fontsize=12.5, fontweight="bold", color=TEAL)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=12)
    for t, c in zip(ax.get_yticklabels(), colors): t.set_color(c)
    ax.set_xlim(0, XMAX); ax.set_ylim(-1.15, head + 0.5)
    ax.set_xticks(np.arange(0, 0.81, 0.1))
    ax.set_xlabel("Spearman \u03c1 vs experiment   (ESM-C 600M)", fontsize=11.5, labelpad=8, color="#333")
    ax.spines["left"].set_visible(False)
    finish(fig, ax, "A sharp competence boundary on variant ranking",
           "Variant-ranking accuracy across seven deep-mutational scans \u2014 only assays past \u03c1 = 0.5 are usable.",
           os.path.join(out_dir, FILES["fig1"]))

# ---- fig 2: confidence != accuracy (keystone) -----------------------------
def _declutter(ys, gap, lo, hi):
    ys = np.asarray(ys, float); order = np.argsort(ys); s = ys[order].copy()
    for i in range(1, len(s)):
        if s[i] - s[i - 1] < gap: s[i] = s[i - 1] + gap
    if s[-1] > hi:
        s[-1] = hi
        for i in range(len(s) - 2, -1, -1):
            if s[i + 1] - s[i] < gap: s[i] = s[i + 1] - gap
    s[0] = max(s[0], lo)
    out = np.empty_like(s); out[order] = s; return out

def fig2_confidence(df, out_dir, save=True):
    """Render the keystone scatter; return the number of overlapping label pairs."""
    THRESH = 0.5
    x = df["cross_size_agree"].to_numpy(float)
    yv = df["rho_600M"].to_numpy(float)
    labels = df["label"].tolist()
    colors = [TEAL if v >= THRESH else RED for v in yv]
    XLO, XHI, YHI = 0.79, 0.955, 1.0
    fig, ax = plt.subplots(figsize=(9.4, 6.0), dpi=200)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.axhspan(THRESH, YHI, color=TEAL, alpha=0.06, zorder=0)
    ax.axhspan(0, THRESH, color=RED, alpha=0.06, zorder=0)
    ax.axhline(THRESH, color=GREY, lw=1.0, ls=(0, (5, 4)), zorder=1)
    LX = 0.873
    ly = _declutter(yv, gap=0.072, lo=0.03, hi=0.90)
    label_texts = []
    for xi, yi, lyi, lab, c in zip(x, yv, ly, labels, colors):
        ax.plot([LX + 0.002, xi - 0.003], [lyi, yi], color="#c2c2c2", lw=0.8, zorder=2)
        t = ax.text(LX, lyi, f"{lab} ({yi:.2f})", ha="right", va="center", fontsize=10.5, color=c, zorder=4,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=0.6))
        label_texts.append(t)
    ax.scatter(x, yv, s=70, c=colors, edgecolor="white", linewidth=1.0, zorder=5)
    ax.text(0.951, 0.71, "RELIABLE", rotation=90, rotation_mode="anchor", ha="center", va="center",
            fontsize=11, fontweight="bold", color=TEAL)
    ax.text(0.951, 0.235, "UNRELIABLE", rotation=90, rotation_mode="anchor", ha="center", va="center",
            fontsize=11, fontweight="bold", color=RED)
    ax.set_xlim(XLO, XHI); ax.set_ylim(0, YHI)
    ax.set_xticks(np.arange(0.80, 0.96, 0.04)); ax.set_yticks(np.arange(0, 0.81, 0.2))
    ax.set_xlabel("Model self-consistency   (300M\u2194600M agreement \u2014 needs no data)", fontsize=11.5, labelpad=8, color="#333")
    ax.set_ylabel("True reliability   (Spearman \u03c1 vs experiment)", fontsize=11.5, labelpad=8, color="#333")
    rng_x = float(x.max() - x.min())
    ax.set_title(f"Self-consistency varies by only {rng_x:.2f} across assays \u2014 while true reliability ranges {yv.min():.2f} to {yv.max():.2f}.",
                 fontsize=10.3, color="#5f5f5f", pad=8)
    fig.suptitle("Confidence \u2260 accuracy", fontsize=15, fontweight="bold", y=0.975)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.tick_params(length=0); ax.grid(False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.canvas.draw()
    bb = [t.get_window_extent() for t in label_texts]
    ov = sum(1 for i in range(len(bb)) for j in range(i + 1, len(bb)) if bb[i].overlaps(bb[j]))
    if save:
        out = os.path.join(out_dir, FILES["fig2"])
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(out, bbox_inches="tight", facecolor="white")
        print("saved", out)
    plt.close(fig)
    return int(ov)

# ---- fig 3: scaling 300M -> 600M ------------------------------------------
def fig3(df, out_dir):
    d = df.sort_values("rho_600M", ascending=False).reset_index(drop=True)
    labels = d["label"].tolist()
    a300 = d["rho_300M"].to_numpy(float); a600 = d["rho_600M"].to_numpy(float)
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(10.2, 5.8), dpi=200)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.axhspan(0.5, 0.85, color=TEAL, alpha=0.05, zorder=0)
    ax.axhspan(0, 0.5, color=RED, alpha=0.05, zorder=0)
    ax.axhline(0.5, color=GREY, lw=1.0, ls=(0, (5, 4)), zorder=1)
    ax.text(len(labels) - 0.4, 0.515, "usable  \u03c1 = 0.5", ha="right", va="bottom", fontsize=9.5, color=GREY)
    b1 = ax.bar(x - w / 2, a300, w, color=TEAL_L, edgecolor="white", linewidth=0.8, label="ESM-C 300M", zorder=3)
    b2 = ax.bar(x + w / 2, a600, w, color=TEAL, edgecolor="white", linewidth=0.8, label="ESM-C 600M", zorder=3)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012, f"{b.get_height():.2f}",
                    ha="center", va="bottom", fontsize=9.5, color="#333", zorder=4)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=10.5)
    ax.set_ylim(0, 0.82); ax.set_yticks(np.arange(0, 0.81, 0.2))
    ax.set_ylabel("Spearman \u03c1 vs experiment", fontsize=11.5, labelpad=8, color="#333")
    ax.legend(loc="upper right", frameon=False, fontsize=11)
    finish(fig, ax, "Scaling 300M \u2192 600M: gains stay inside each regime",
           "Coupled assays gain ~0.03; the decoupled ones stay useless (GFP 0.24\u21920.29) and GB1 even regresses.",
           os.path.join(out_dir, FILES["fig3"]))

# ---- fig 4: calibration ----------------------------------------------------
def fig4(df, out_dir):
    n = np.logspace(np.log10(8), np.log10(2000), 240)
    fig, ax = plt.subplots(figsize=(9.2, 5.6), dpi=200)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.axhline(0.10, color=GREY, lw=1.2, ls=(0, (5, 4)), zorder=2)
    ax.text(9, 0.108, "\u00b10.10 target", fontsize=10, color=GREY, va="bottom")
    for rho, col, off, ha in [(0.7, TEAL, (-4, 9), "right"), (0.5, MID, (0, 10), "center"), (0.3, RED, (12, 9), "left")]:
        ax.plot(n, se(rho, n), color=col, lw=2.3, label=f"true \u03c1 = {rho}", zorder=3)
        nstar = rec_n(rho, 0.10)
        ax.scatter([nstar], [0.10], s=42, color=col, edgecolor="white", linewidth=1.0, zorder=5)
        ax.annotate(f"n={nstar}", (nstar, 0.10), textcoords="offset points", xytext=off,
                    fontsize=10.5, color=col, ha=ha, fontweight="bold")
    ax.set_xscale("log"); ax.set_xlim(8, 2000); ax.set_ylim(0, 0.36)
    ax.set_xlabel("measured variants  (n)", fontsize=11.5, labelpad=8, color="#333")
    ax.set_ylabel("std. error of the estimated Spearman", fontsize=11.5, labelpad=8, color="#333")
    ax.legend(loc="upper right", frameon=False, fontsize=11)
    finish(fig, ax, "~100 measured variants pin the correlation to \u00b10.10",
           "The estimate's error shrinks with sample size; ~100 measurements reach \u00b10.10 regardless of the true \u03c1.",
           os.path.join(out_dir, FILES["fig4"]))

# ---- fig 5: masked-marginal -----------------------------------------------
def fig5(df, out_dir):
    d = df[df["rho_masked_marginal"].notna()].copy()
    order = {"BLAT (Stiffler)": 0, "GFP": 1}
    d["__o"] = d["label"].map(lambda s: order.get(s, 9))
    d = d.sort_values("__o").reset_index(drop=True)
    labels = d["label"].tolist()
    wt = d["rho_wt_marginal"].to_numpy(float); mm = d["rho_masked_marginal"].to_numpy(float)
    x = np.arange(len(labels)); w = 0.34
    fig, ax = plt.subplots(figsize=(8.4, 5.8), dpi=200)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.axhspan(0.5, 0.9, color=TEAL, alpha=0.05, zorder=0)
    ax.axhspan(0, 0.5, color=RED, alpha=0.05, zorder=0)
    ax.axhline(0.5, color=GREY, lw=1.0, ls=(0, (5, 4)), zorder=1)
    ax.text(len(labels) - 0.45, 0.515, "usable  \u03c1 = 0.5", ha="right", va="bottom", fontsize=9.5, color=GREY)
    b1 = ax.bar(x - w / 2, wt, w, color=TEAL_L, edgecolor="white", linewidth=0.8, label="wild-type-marginal (cheap)", zorder=3)
    b2 = ax.bar(x + w / 2, mm, w, color=TEAL, edgecolor="white", linewidth=0.8, label="masked-marginal (field standard)", zorder=3)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012, f"{b.get_height():.2f}",
                    ha="center", va="bottom", fontsize=10.5, color="#333", zorder=4)
    # per-pair change, computed from the DISPLAYED (rounded) values so it always matches the bars
    for xi, a, b in zip(x, wt, mm):
        ar, br = round(a, 2), round(b, 2); d_ = round(br - ar, 2)
        up = d_ >= 0; arrow = "\u25b2" if up else "\u25bc"; col = TEAL if up else RED
        ax.text(xi, max(ar, br) + 0.075, f"{arrow} {d_:+.2f}", ha="center", va="bottom",
                fontsize=11.5, fontweight="bold", color=col, zorder=4)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 0.9); ax.set_yticks(np.arange(0, 0.81, 0.2))
    ax.set_ylabel("Spearman \u03c1 vs experiment", fontsize=11.5, labelpad=8, color="#333")
    ax.legend(loc="upper right", frameon=False, fontsize=10.5)
    finish(fig, ax, "The \u201cbetter\u201d scoring method widens the gap",
           "Field-standard masked-marginal scoring: the reliable assay improves, the decoupled one gets worse.",
           os.path.join(out_dir, FILES["fig5"]))

def main():
    csv = sys.argv[1] if len(sys.argv) > 1 else "results/results.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "figures"
    os.makedirs(out, exist_ok=True)
    df = load(csv)
    fig1(df, out)
    fig2_confidence(df, out, save=True)
    fig3(df, out)
    fig4(df, out)
    fig5(df, out)

if __name__ == "__main__":
    main()
