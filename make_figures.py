import sys, os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = sys.argv[1] if len(sys.argv) > 1 else "figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi":120,"savefig.dpi":300,"font.size":11,
    "axes.spines.top":False,"axes.spines.right":False,"axes.titleweight":"bold",
    "axes.titlesize":12.5,"figure.facecolor":"white","savefig.facecolor":"white"})
GREEN,AMBER,RED,BLUE,GREY="#1a7f37","#9a6700","#cf222e","#0969da","#8b949e"
band=lambda r: GREEN if r>=0.5 else AMBER if r>=0.3 else RED
A=[("PABP",0.717,0.919),("BLAT (Stiffler)",0.715,0.929),("BLAT (Jacquier)",0.696,0.931),
   ("KKA2",0.635,0.901),("DLG4",0.552,0.794),("TPMT",0.546,0.924),("PTEN",0.527,0.932),
   ("DYR",0.508,0.933),("HSP82",0.493,0.911),("UBE4B",0.440,0.880),("A4 (amyloid)",0.438,0.690),
   ("SUMO1",0.351,0.730),("GFP",0.283,0.926),("SPG1 (Olson)",0.254,0.904),
   ("CALM1",0.241,0.739),("GB1 (Wu)",0.107,0.887)]
names=[a for a,_,_ in A]; rho=np.array([r for _,r,_ in A]); agr=np.array([g for _,_,g in A])
fig,ax=plt.subplots(figsize=(7.2,5.6))
ax.axhspan(0.5,0.8,color=GREEN,alpha=0.05); ax.axhspan(0,0.3,color=RED,alpha=0.05)
ax.scatter(agr,rho,s=85,c=[band(r) for r in rho],edgecolor="white",lw=1.2,zorder=3)
for tag in ["BLAT (Stiffler)","GFP"]:
    j=names.index(tag); x,y=agr[j],rho[j]
    ax.scatter([x],[y],s=260,facecolor="none",edgecolor="black",lw=1.8,zorder=4)
    ax.annotate(tag.split(" ")[0]+" (r="+("%.2f"%y)+")",(x,y),xytext=(x-0.016,y),ha="right",va="center",fontsize=9.5)
ax.set_xlabel("Model self-consistency (300M vs 600M agreement)")
ax.set_ylabel("True reliability (Spearman vs experiment)")
ax.set_title("Confidence does not predict accuracy")
ax.text(0.015,0.985,"same self-consistency, opposite reliability:\nthe model fails confidently",
    transform=ax.transAxes,va="top",fontsize=9,color="#444",style="italic")
ax.set_ylim(0,0.8); fig.tight_layout(); fig.savefig(OUT+"/fig1_confidence_vs_accuracy.png",bbox_inches="tight"); plt.close()
GRP=[("Binding",4,0.311,0.11,0.55),("Expression",4,0.341,0.07,0.55),
     ("OrganismalFitness",18,0.443,0.04,0.72),("Activity",6,0.483,0.28,0.69),("Stability",8,0.518,0.36,0.74)]
fig,ax=plt.subplots(figsize=(8.2,5.6))
for i,(c,n,mean,lo,hi) in enumerate(GRP):
    ax.vlines(i,lo,hi,color=BLUE,lw=2,alpha=0.55)
    ax.plot([i-0.12,i+0.12],[lo,lo],color=BLUE,lw=2,alpha=0.55)
    ax.plot([i-0.12,i+0.12],[hi,hi],color=BLUE,lw=2,alpha=0.55)
    ax.scatter([i],[mean],s=120,color="black",zorder=4)
    ax.annotate("n="+str(n),(i,hi),xytext=(i,hi+0.02),ha="center",fontsize=8.5,color="#555")
ax.set_xticks(range(len(GRP))); ax.set_xticklabels([g[0] for g in GRP],rotation=18,ha="right")
ax.set_ylabel("True reliability (Spearman)")
ax.set_title("Selection type does NOT predict reliability")
ax.text(0.015,0.985,"leave-one-out prior vs truth: Spearman = -0.17\nmean (dot) and full min-max range (bar)",
    transform=ax.transAxes,va="top",fontsize=9,color="#444",style="italic")
ax.set_ylim(0,0.8); fig.tight_layout(); fig.savefig(OUT+"/fig2_selection_type.png",bbox_inches="tight"); plt.close()
KS=np.array([10,20,30,50,100,200,400]); SE=np.array([0.269,0.183,0.146,0.111,0.076,0.053,0.037])
fig,ax=plt.subplots(figsize=(7.2,5.6))
ax.plot(KS,SE,"-o",color=BLUE,lw=2.2,ms=7); ax.axhline(0.10,color=RED,ls="--",lw=1.5)
ax.text(KS[-1],0.105,"+/-0.10 target",color=RED,ha="right",fontsize=9)
kh=int(KS[np.argmax(SE<0.10)]); ax.axvline(kh,color=GREY,ls=":",lw=1.3)
ax.annotate("~"+str(kh)+" variants -> +/-0.10",(kh,0.076),xytext=(kh*1.15,0.13),fontsize=9.5,
    arrowprops=dict(arrowstyle="->",color=GREY))
ax.set_xscale("log"); ax.set_xticks(KS); ax.set_xticklabels(KS)
ax.set_xlabel("Measured variants used to calibrate (k)")
ax.set_ylabel("Uncertainty of the reliability estimate (std)")
ax.set_title("You cannot infer trust - but ~100 measurements buy it")
fig.tight_layout(); fig.savefig(OUT+"/fig3_calibration_curve.png",bbox_inches="tight"); plt.close()
SC=[("BLAT",0.684,0.715),("GFP",0.235,0.283),("PABP",0.689,0.718),
    ("DLG4",0.549,0.552),("GB1 (Wu)",0.174,0.115),("A4",0.526,0.438)]
s3=np.array([a for _,a,_ in SC]); s6=np.array([b for _,_,b in SC])
fig,ax=plt.subplots(figsize=(6.2,6.0))
ax.plot([0,0.8],[0,0.8],color=GREY,ls="--",lw=1.3)
ax.scatter(s3,s6,s=85,c=[band(b) for b in s6],edgecolor="white",lw=1.2,zorder=3)
for (t,a,b) in SC:
    if b-a<-0.01: ax.annotate(t,(a,b),xytext=(a+0.01,b-0.03),fontsize=8.5,color=RED)
ax.set_xlim(0,0.8); ax.set_ylim(0,0.8); ax.set_aspect("equal")
ax.set_xlabel("Reliability - ESM-C 300M"); ax.set_ylabel("Reliability - ESM-C 600M")
ax.set_title("Scaling 300M -> 600M barely moves reliability")
ax.text(0.03,0.97,"on/below diagonal: bigger model ~ or worse",
    transform=ax.transAxes,va="top",fontsize=9,color="#444",style="italic")
fig.tight_layout(); fig.savefig(OUT+"/fig4_scaling.png",bbox_inches="tight"); plt.close()
print("figures written to", OUT)
