"""Selection study analysis: does the validation score pick good configurations under drift?

Input: the 12 tuning configs x 3 seeds of 8 methods (results/tuning/<ds>/<m>/) and the SAME configs re-run
(run_selection_study.py) on (a) the main test split ("test": UCI main 1-5 -> 6-10, SmellNet loso... see that script)
and (b) a second validation fold ("fold2": UCI tune2 1-2 -> 3, SmellNet tune2 1-3 -> 4).

Per dataset x method, over the 12 configs (scores = mean over 3 seeds; diverged run = 0):
  rho_val_test   Spearman(val, test)          does validation rank configs like the test does?
  rho_val_fold2  Spearman(val, fold2)         does a second validation fold agree with the first?
  regret_<rule>  oracle test - test of the config the rule picks, rules:
     val        highest mean validation score (the protocol used in the paper)
     mean_sd    highest (mean - sd) over seeds on validation (risk-averse)
     twofold    highest average of fold-1 and fold-2 validation scores
     random     expected regret of picking a config uniformly at random (baseline)
Writes results/analysis/selection_summary.csv and figures/selection_val_vs_test.{png,pdf}.
"""
import glob
import json
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
A = ROOT / "results" / "analysis"; FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
METHODS = ["sgd", "adamw", "sam", "msam", "disam", "coral", "groupdro", "recency"]
NAMES = {"sgd": "SGD", "adamw": "AdamW", "sam": "SAM", "msam": "MSAM", "disam": "DISAM", "coral": "CORAL",
         "groupdro": "GroupDRO", "recency": "Recency"}
DS_NAMES = {"uci": "UCI Gas Drift", "smell": "SmellNet"}
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a64", "#e4e3dd"


def score_of(r, kind):
    if r["diverged"]:
        return 0.0
    if kind == "test":
        mo = r["final_eval"]["mean_over_domains"]
        return mo.get("score", mo.get("macro_f1_present"))
    bv = r.get("best_val") or {}
    return bv.get("score", bv.get("macro_f1", np.nan))


def load():
    rows = []
    for ds in DS_NAMES:
        for m in METHODS:
            for tf in glob.glob(str(ROOT / "results" / "tuning" / ds / m / "*.json")):
                tag = os.path.basename(tf); cfg, seed = tag.rsplit("_s", 1)
                row = dict(ds=ds, m=m, cfg=cfg, seed=int(seed[:-5]), val=score_of(json.load(open(tf)), "val"))
                for kind in ("test", "fold2"):
                    f = ROOT / "results" / "selection" / ds / kind / m / tag
                    row[kind] = score_of(json.load(open(f)), kind) if f.exists() else np.nan
                rows.append(row)
    return pd.DataFrame(rows)


def main():
    d = load()
    d = d.dropna(subset=["test", "fold2"])
    g = d.groupby(["ds", "m", "cfg"]).agg(val=("val", "mean"), val_sd=("val", "std"), test=("test", "mean"),
                                         fold2=("fold2", "mean"), n=("seed", "size")).reset_index()
    out = []
    for (ds, m), x in g.groupby(["ds", "m"]):
        x = x.reset_index(drop=True); oracle = x.test.max()
        pick = dict(val=x.val.idxmax(), mean_sd=(x.val - x.val_sd).idxmax(), twofold=((x.val + x.fold2) / 2).idxmax())
        r = dict(dataset=ds, method=m, n_configs=len(x),
                 rho_val_test=spearmanr(x.val, x.test).correlation, rho_val_fold2=spearmanr(x.val, x.fold2).correlation,
                 oracle_test=oracle, test_of_val_pick=x.test[pick["val"]])
        # rank agreement among configs that TRAINED (val >= 0.5): the all-config rho is inflated by failed configs
        # (val and test both ~0), which any rule avoids; this asks whether validation ranks the good configs
        ok = x.val >= 0.5
        r["n_ok"] = int(ok.sum())
        r["rho_val_test_ok"] = spearmanr(x.val[ok], x.test[ok]).correlation if ok.sum() >= 4 else np.nan
        for k, i in pick.items():
            r[f"regret_{k}"] = oracle - x.test[i]
        r["regret_random"] = oracle - x.test.mean()
        out.append(r)
    S = pd.DataFrame(out)
    S.to_csv(A / "selection_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"{len(d)} runs with val + test + fold2\n")
    print(S.round(3).to_string(index=False))
    print("\nmean over methods:")
    print(S.groupby("dataset")[["rho_val_test", "rho_val_test_ok", "rho_val_fold2", "regret_val", "regret_mean_sd", "regret_twofold",
                                "regret_random"]].mean().round(3).to_string())
    print("\nmethods where validation-picked config is no better than a random pick (regret_val >= regret_random):")
    print(S[S.regret_val >= S.regret_random][["dataset", "method", "regret_val", "regret_random"]].round(3).to_string(index=False))

    # figure: small multiples, one panel per method; x = validation score, y = test score, 12 configs each
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(2, len(METHODS), figsize=(13, 3.8), constrained_layout=True)
    for i, ds in enumerate(DS_NAMES):
        for j, m in enumerate(METHODS):
            ax = axes[i, j]; x = g[(g.ds == ds) & (g.m == m)].reset_index(drop=True)
            s = S[(S.dataset == ds) & (S.method == m)].iloc[0]
            ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
            ax.scatter(x.val, x.test, s=22, color=BLUE, edgecolor="white", lw=1.0, zorder=3)
            k = x.val.idxmax()
            ax.scatter(x.val[k], x.test[k], s=60, facecolor="none", edgecolor=ORANGE, lw=2, zorder=4)
            ax.axhline(x.test.max(), color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=2)
            ax.set_title(f"{NAMES[m]}   ρ = {s.rho_val_test:+.2f}" + (f" ({s.rho_val_test_ok:+.2f})" if s.n_ok < s.n_configs else ""), fontsize=8, color=INK, loc="left")
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if j == 0:
                ax.set_ylabel(f"{DS_NAMES[ds]}\ntest macro-F1")
            if i == 1:
                ax.set_xlabel("validation macro-F1")
    h = [plt.Line2D([], [], ls="", marker="o", color=BLUE, label="one configuration (mean of 3 seeds)"),
         plt.Line2D([], [], ls="", marker="o", mfc="none", mec=ORANGE, mew=2, ms=8, label="picked by validation"),
         plt.Line2D([], [], color=MUTED, lw=0.8, ls=(0, (3, 2)), label="best test score (oracle)")]
    fig.legend(handles=h, loc="outside upper center", ncol=3, frameon=False, fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"selection_val_vs_test.{ext}", dpi=200)
    print(f"\nfigure -> {FIG / 'selection_val_vs_test.png'}")


if __name__ == "__main__":
    main()
