"""Analyse the final test runs in results/final/.

Outputs (results/analysis/, figures/):
  final_runs.csv            one row per run (method, split, seed, test macro-F1, sharpness measures, passes, time)
  table_test_f1.csv         mean +- sd over seeds of test macro-F1 (present classes, mean over test domains), method x split
  table_uci_main_batches.csv UCI main split: per test batch (6..10) mean macro-F1
  sharpness_corr.csv        Spearman rho between each sharpness measure and test macro-F1, computed WITHIN each split
                            (across methods x seeds) so that split difficulty doesn't drive the correlation
  figures/final_sharpness_vs_f1_<ds>.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIN = ROOT / "results" / "final"
OUT = ROOT / "results" / "analysis"
FIG = ROOT / "figures"
SHARP = ["hessian_top_eig", "adv_sharp", "random_sharp"]
ORDER = ["sgd", "adamw", "erm_mixed", "sam", "msam", "sagm", "disam", "recency", "gi", "coral", "groupdro",
         "samsam_extrap", "samsam_disagree", "samsam_random", "swa", "swa_full", "driftaug", "ccpca", "rf", "domnorm"]
COLLAPSE_F1 = 0.3   # older runs have no train_acc; a test macro-F1 below this is also counted as collapsed


def load():
    rows = []
    for f in FIN.glob("*/*/*.json"):
        r = json.loads(f.read_text())
        method = f.stem.rsplit("_s", 1)[0]
        fe = r.get("final_eval", {})
        row = dict(dataset=r["dataset"], split=r["split"], method=method, seed=r["seed"], diverged=r.get("diverged", False),
                   train_acc=r.get("train_acc", np.nan),
                   # test_f1 = the task score: macro-F1 (classification) or R2 (twinreg regression)
                   test_f1=fe.get("mean_over_domains", {}).get("score", fe.get("mean_over_domains", {}).get("macro_f1_present", np.nan)),
                   test_acc=fe.get("mean_over_domains", {}).get("acc", np.nan),
                   pooled_f1=fe.get("pooled", {}).get("macro_f1_present", np.nan),
                   passes=r.get("fwd_bwd_passes"), time_s=r.get("train_time_s"), epochs=r.get("epochs"))
        for k, v in fe.items():
            if k not in ("pooled", "mean_over_domains"):
                row[f"f1_dom{k}"] = v.get("macro_f1_present", np.nan)
        row.update({k: (r.get("sharpness") or {}).get(k, np.nan) for k in SHARP})
        rows.append(row)
    return pd.DataFrame(rows)


def split_key(s):
    return (0, 0) if s == "main" else (1, int("".join(c for c in s if c.isdigit())))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    # collapse: diverged, train accuracy (train R2 for twinreg) < 0.5, or (classification only) test F1 < 0.3
    df["collapsed"] = df.diverged | (df.train_acc < 0.5) | ((df.dataset != "twinreg") & (df.test_f1 < COLLAPSE_F1))
    # Regression (twinreg): R2 is unbounded below, so a run whose TRAINING failed (diverged / train R2 < 0.5) can
    # dominate a mean (seen: R2 = -305354). Such runs are scored R2 = 0, i.e. no better than predicting the training
    # mean -- the regression analogue of a collapsed classifier scoring near chance. Runs that trained fine keep their
    # real test R2, even if negative (that is genuine failure to transfer). Raw value kept in test_f1_raw.
    df["test_f1_raw"] = df.test_f1
    reg_c = (df.dataset == "twinreg") & df.collapsed
    df.loc[reg_c, "test_f1"] = df.loc[reg_c, "test_f1"].clip(lower=0.0)
    df.to_csv(OUT / "final_runs.csv", index=False)
    methods = [m for m in ORDER if m in set(df.method)]
    print(f"{len(df)} runs loaded")

    for ds, g in df.groupby("dataset"):
        splits = sorted(g.split.unique(), key=split_key)
        mean = g.pivot_table(index="method", columns="split", values="test_f1", aggfunc="mean").reindex(methods)[splits]
        sd = g.pivot_table(index="method", columns="split", values="test_f1", aggfunc="std").reindex(methods)[splits]
        n = g.pivot_table(index="method", columns="split", values="test_f1", aggfunc="count").reindex(methods)[splits]
        mean["avg"] = mean.mean(axis=1)
        mean["mean_rank"] = mean[splits].rank(axis=0, ascending=False).mean(axis=1)
        cell = mean[splits].round(3).astype(str) + " ± " + sd.round(3).astype(str) + " (n=" + n.fillna(0).astype(int).astype(str) + ")"
        cell["avg"], cell["mean_rank"] = mean["avg"].round(3), mean["mean_rank"].round(2)
        cell.to_csv(OUT / f"table_test_f1_{ds}.csv")
        # robust summary: per-method median over all (split, seed) runs, and collapse counts
        rob = g.groupby("method").agg(median_f1=("test_f1", "median"), q25=("test_f1", lambda x: x.quantile(.25)),
                                      q75=("test_f1", lambda x: x.quantile(.75)), n_runs=("test_f1", "size"),
                                      n_collapsed=("collapsed", "sum")).reindex(methods)
        rob.to_csv(OUT / f"table_robust_{ds}.csv")
        print(f"\n== {ds}: test macro-F1 (mean over seeds)")
        print(mean.round(3).to_string())
        print(f"\n== {ds}: median [IQR] over all runs, collapsed runs")
        print(rob.round(3).to_string())

        if ds == "uci" and "main" in splits:
            m = g[g.split == "main"]
            cols = sorted([c for c in m.columns if c.startswith("f1_dom") and m[c].notna().any()], key=lambda c: int(c[6:]))
            t = m.groupby("method")[cols].mean().reindex(methods)
            t.to_csv(OUT / "table_uci_main_batches.csv")
            print("\n== uci main: per test batch")
            print(t.round(3).to_string())

    # sharpness vs test F1, within split
    corr = []
    for (ds, sp), g in df[~df.diverged].groupby(["dataset", "split"]):
        for k in SHARP:
            x = g[[k, "test_f1"]].dropna()
            if len(x) >= 5:
                rho, p = spearmanr(x[k], x.test_f1)
                corr.append(dict(dataset=ds, split=sp, measure=k, n=len(x), spearman=rho, p=p))
    corr = pd.DataFrame(corr)
    corr.to_csv(OUT / "sharpness_corr.csv", index=False)
    if len(corr):
        print("\n== Spearman(sharpness, test F1) within split: mean over splits [min, max]")
        for (ds, k), c in corr.groupby(["dataset", "measure"]):
            print(f"  {ds:5s} {k:16s} {c.spearman.mean():+.2f}  [{c.spearman.min():+.2f}, {c.spearman.max():+.2f}]  "
                  f"splits with p<0.05: {(c.p < 0.05).sum()}/{len(c)}")

    for ds, g in df[~df.diverged].groupby("dataset"):
        fig, axes = plt.subplots(1, len(SHARP), figsize=(4.2 * len(SHARP), 3.6))
        cmap = plt.get_cmap("tab10")
        for ax, k in zip(axes, SHARP):
            for i, m in enumerate(methods):
                h = g[g.method == m]
                ax.scatter(h[k], h.test_f1, s=12, alpha=0.7, color=cmap(i % 10), label=m)
            ax.set_xscale("log"); ax.set_xlabel(k); ax.set_ylabel("test macro-F1")
        axes[0].legend(fontsize=7, ncol=2)
        fig.suptitle(f"{ds}: sharpness vs test macro-F1 (all splits)")
        fig.tight_layout()
        fig.savefig(FIG / f"final_sharpness_vs_f1_{ds}.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
