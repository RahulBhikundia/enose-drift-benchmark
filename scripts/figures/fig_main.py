"""Main-results figure: each method's test score minus AdamW, per dataset (mean over splits, 95% bootstrap CI over
splits, from stats_tests.py -> results/analysis/stats_vs_adamw_<ds>.csv). Methods in the same row order on every
panel (sorted by mean rank across datasets) so a method can be followed across datasets.
Writes figures/main_effects.{png,pdf}."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
A = ROOT / "results" / "analysis"; FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
DS = {"uci": "UCI Gas Drift\n(macro-F1, 9 splits)", "smell": "SmellNet\n(macro-F1, 6 splits)",
      "twin": "Twin: gas ID\n(macro-F1, 5 units)", "twinreg": "Twin: concentration\n(R², 5 units)"}
NAMES = {"sgd": "SGD", "sam": "SAM", "msam": "MSAM", "sagm": "SAGM", "disam": "DISAM", "coral": "CORAL",
         "groupdro": "GroupDRO", "erm_mixed": "Mixed sampling", "recency": "Recency", "gi": "GI",
         "samsam_extrap": "SA-MSAM (extrap.)", "samsam_disagree": "SA-MSAM (disagree)",
         "samsam_random": "SA-MSAM (random dir.)", "swa": "SWA (early-stopped)", "swa_full": "SWA (full)",
         "driftaug": "Drift augmentation", "ccpca": "CC-PCA", "domnorm": "Per-domain norm.*", "rf": "Random forest"}
BLUE, INK, MUTED, GRID, RED = "#2a78d6", "#1f1f1e", "#6b6a64", "#e4e3dd", "#eb6834"


def main():
    S = {ds: pd.read_csv(A / f"stats_vs_adamw_{ds}.csv").set_index("method") for ds in DS}
    order = (pd.concat([s.mean_diff.rank(ascending=False) for s in S.values()], axis=1).mean(1)
             .sort_values().index.tolist())
    order = [m for m in order if m in NAMES]
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": INK})
    fig, axes = plt.subplots(1, 4, figsize=(12, 5.2), sharey=True, constrained_layout=True)
    y = np.arange(len(order))[::-1]
    for ax, (ds, title) in zip(axes, DS.items()):
        s = S[ds].reindex(order)
        lo = s.ci_lo.clip(lower=-0.6); hi = s.ci_hi
        ax.axvline(0, color=INK, lw=0.8)
        ax.grid(True, axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
        ax.hlines(y, lo, hi, color=MUTED, lw=1.5)
        sig = (s.ci_lo > 0) | (s.ci_hi < 0)
        ax.scatter(s.mean_diff.clip(lower=-0.6), y, s=28, color=np.where(sig, BLUE, "white"), edgecolor=BLUE, lw=1.5,
                   zorder=3)
        for yi, (m, r) in zip(y, s.iterrows()):
            if not np.isnan(r.mean_diff):
                ax.text(0.995, yi, f"{int(r.wins)}/{int(r.n_splits)}", transform=ax.get_yaxis_transform(),
                        ha="right", va="center", fontsize=7, color=MUTED)
            else:
                ax.text(0.995, yi, "n/a", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=7,
                        color=MUTED)
        ax.set_xlim(-0.62, 0.62)
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xlabel("difference from AdamW")
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y); axes[0].set_yticklabels([NAMES[m] for m in order])
    fig.text(0.01, -0.02, "Dot = mean difference over splits; line = 95% bootstrap CI over splits (clipped at −0.6); "
             "filled = CI excludes 0; right column = splits where the method beat AdamW.\n"
             "* Per-domain normalization uses unlabeled test-domain statistics (transductive).",
             fontsize=7, color=MUTED, ha="left", va="top")
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"main_effects.{ext}", dpi=200, bbox_inches="tight")
    print(FIG / "main_effects.png")


if __name__ == "__main__":
    main()
