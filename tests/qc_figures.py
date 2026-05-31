"""qc_figures.py — QC battery for the figures. Run from repo root after make_figures.py.
Verifies data integrity, the thesis numbers, the calibration math, file outputs, and
(headline check) that fig2 has ZERO label overlaps under 100 randomized row orderings.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import make_figures as MF

df = MF.load("results/results.csv")
checks = []
def chk(name, cond):
    checks.append(bool(cond)); print(("PASS" if cond else "FAIL"), "-", name)

need = ["dms_id","label","rho_300M","rho_600M","cross_size_agree","rho_wt_marginal","rho_masked_marginal"]
chk("schema: expected columns present", all(c in df.columns for c in need))
chk("7 assays present", len(df) == 7)
for c in ["rho_300M","rho_600M","cross_size_agree"]:
    chk(f"{c}: values within [-1,1]", df[c].dropna().between(-1,1).all())
chk("no NaN in rho_600M / cross_size_agree", df[["rho_600M","cross_size_agree"]].notna().all().all())

g = df.set_index("dms_id")
gfp = g.loc["GFP_AEQVI_Sarkisyan_2016"]
chk("GFP fails (rho_600M < 0.40)", gfp.rho_600M < 0.40)
chk("GFP looks confident (cross_size_agree > 0.85)", gfp.cross_size_agree > 0.85)
agree = df["cross_size_agree"]
chk("self-consistency essentially flat (range < 0.06)", (agree.max()-agree.min()) < 0.06)
rel = df[df.rho_600M >= 0.50]["cross_size_agree"]
chk("GFP agreement indistinguishable from reliable band",
    rel.min()-0.02 <= gfp.cross_size_agree <= rel.max()+0.02)

gb1 = g.loc["SPG1_STRSG_Wu_2016"]
chk("GB1 regresses with scale (600M < 300M)", gb1.rho_600M < gb1.rho_300M)
for k in ["BLAT_ECOLX_Stiffler_2015","PABP_YEAST_Melamed_2013","GFP_AEQVI_Sarkisyan_2016"]:
    chk(f"{k.split('_')[0]}: 600M >= 300M", g.loc[k].rho_600M >= g.loc[k].rho_300M)

chk("GFP MM worse than WT (gap widens)", gfp.rho_masked_marginal < gfp.rho_wt_marginal)
stf = g.loc["BLAT_ECOLX_Stiffler_2015"]
chk("BLAT(Stiffler) MM >= WT (reliable holds)", stf.rho_masked_marginal >= stf.rho_wt_marginal)

chk("recommend_n(0.7,0.10) == 53", MF.rec_n(0.7,0.10) == 53)
chk("recommend_n(0.5,0.10) == 77", MF.rec_n(0.5,0.10) == 77)
chk("recommend_n(0.3,0.10) == 93", MF.rec_n(0.3,0.10) == 93)
ses = [MF.se(0.5,n) for n in (20,50,100,500,1000)]
chk("SE strictly decreasing in n", all(x > y for x,y in zip(ses, ses[1:])))
chk("SE(0.5,100) ~ 0.0876", abs(MF.se(0.5,100)-0.0876) < 0.002)
chk("rho rounds consistently (0.7147->0.71)", f"{stf.rho_600M:.2f}" == "0.71")

os.makedirs("/tmp/qcfig", exist_ok=True)
maxov = 0; rng = np.random.default_rng(0)
for _ in range(100):
    ov = MF.fig2_confidence(df.sample(frac=1.0, random_state=int(rng.integers(1_000_000_000))),
                            "/tmp/qcfig", save=False)
    maxov = max(maxov, ov)
chk("fig2: 0 label overlaps across 100 shuffled orderings", maxov == 0)

for f in ["fig1_competence_boundary.png","fig2_confidence_vs_accuracy.png","fig3_scaling.png",
          "fig4_calibration.png","fig5_masked_marginal.png"]:
    p = f"figures/{f}"; chk(f"{f} exists & >5KB", os.path.exists(p) and os.path.getsize(p) > 5000)

n_pass = sum(checks); n = len(checks)
print(f"\n=== QC: {n_pass}/{n} checks passed | fig2 max label-overlaps over 100 runs = {maxov} ===")
sys.exit(0 if n_pass == n else 1)
