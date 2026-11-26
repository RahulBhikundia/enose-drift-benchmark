"""Significance tests on the final test runs (results/analysis/final_runs.csv; run analyze_final.py first).

Unit of analysis = SPLIT (per-split mean over seeds), because splits are the independent shifts;
seeds within a split are not independent replicates of the shift.
  * each method vs a reference (default AdamW-ERM): mean difference over splits with a 95% bootstrap CI
    (resampling splits, 10k draws), Wilcoxon signed-rank p (two-sided), Holm-corrected across methods;
    "wins" = number of splits where the method beats the reference
  * Friedman test over all methods (blocks = splits), and the average rank of each method
Note the small number of splits (UCI 9, SmellNet 6): the smallest attainable two-sided Wilcoxon p is
0.0039 (n=9) and 0.031 (n=6), so after Holm correction over ~15 methods, SmellNet differences
essentially cannot reach significance. Report effect sizes (CI) first.
Writes results/analysis/stats_vs_<ref>_<ds>.csv.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

ROOT = Path(__file__).resolve().parents[2]
A = ROOT / "results" / "analysis"


def holm(p):
    p = np.asarray(p, float); order = np.argsort(p); m = len(p)
    adj = np.empty(m); run = 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (m - i) * p[j])); adj[j] = run
    return adj


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ref", default="adamw")
    a = ap.parse_args()
    df = pd.read_csv(A / "final_runs.csv")
    rng = np.random.default_rng(0)
    for ds, g in df.groupby("dataset"):
        M = g.pivot_table(index="split", columns="method", values="test_f1", aggfunc="mean").dropna(axis=1)
        ref = M[a.ref]; rows = []
        for m in M.columns:
            if m == a.ref:
                continue
            d = (M[m] - ref).to_numpy()
            boot = rng.choice(d, size=(10000, len(d)), replace=True).mean(1)
            p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            rows.append(dict(method=m, mean_diff=d.mean(), ci_lo=np.percentile(boot, 2.5), ci_hi=np.percentile(boot, 97.5),
                             wins=int((d > 0).sum()), n_splits=len(d), p_wilcoxon=p))
        R = pd.DataFrame(rows); R["p_holm"] = holm(R.p_wilcoxon)
        R = R.sort_values("mean_diff", ascending=False)
        R.to_csv(A / f"stats_vs_{a.ref}_{ds}.csv", index=False)
        fr = friedmanchisquare(*[M[c] for c in M.columns])
        ranks = M.rank(axis=1, ascending=False).mean().sort_values()
        print(f"\n== {ds}: each method minus {a.ref} ({"test R2" if ds == "twinreg" else "test macro-F1"}, per-split means; {len(M)} splits)")
        print(R.round(3).to_string(index=False))
        print(f"Friedman over {M.shape[1]} methods: chi2={fr.statistic:.1f}, p={fr.pvalue:.3g}")
        print("average rank (1 = best): " + ", ".join(f"{k} {v:.1f}" for k, v in ranks.items()))


if __name__ == "__main__":
    main()
