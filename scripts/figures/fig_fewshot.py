"""Few-shot recalibration figure: test score vs number of labeled shots from the new domain, per dataset.
Mean over splits of the per-split seed means (results/fewshot/final_<ds>.csv). Writes figures/fewshot.{png,pdf}."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FS = ROOT / "results" / "fewshot"; FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
DS = {"uci": ("UCI Gas Drift", "labeled samples per class", "macro-F1"),
      "smell": ("SmellNet", "first windows per recording", "macro-F1"),
      "twin": ("Twin: gas ID", "recordings per gas", "macro-F1"),
      "twinreg": ("Twin: concentration", "recordings per gas", "R²")}
# validated categorical palette (slots 1-4); contrast WARN for slots 3-4 -> every line is direct-labeled + own marker
METH = {"none": ("no recalibration", "#2a78d6", "o"), "affine": ("affine (gain + offset)", "#eb6834", "s"),
        "ft_last": ("fine-tune last layer", "#1baf7a", "^"), "ft_all": ("fine-tune all layers", "#eda100", "D")}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"


def main():
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.3), constrained_layout=True)
    for ax, (ds, (title, xl, yl)) in zip(axes, DS.items()):
        d = pd.read_csv(FS / f"final_{ds}.csv")
        s = d.groupby(["split", "k", "method"]).f1.mean().groupby(["k", "method"]).mean().unstack()
        ks = list(s.index)
        x = range(len(ks))
        ax.grid(True, axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
        for m, (lab, c, mk) in METH.items():
            ax.plot(x, s[m], color=c, lw=2, marker=mk, ms=6, mec="white", mew=1, zorder=3)
            # zero-shot reference shown as a flat line (it does not use the shots)
        ys = s.iloc[-1].sort_values()
        last = [ys.iloc[0]]
        for v in ys.iloc[1:]:                           # nudge direct labels apart
            last.append(max(v, last[-1] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.055))
        for (m, _), yv in zip(ys.items(), last):
            ax.text(len(ks) - 1 + 0.12, yv, METH[m][0], va="center", fontsize=7, color=INK)
        ax.set_xticks(list(x)); ax.set_xticklabels(ks)
        ax.set_xlim(-0.2, len(ks) - 1 + 1.6)
        ax.set_xlabel(xl); ax.set_ylabel(f"test {yl}" if ds in ("uci", "twinreg") else "")
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    h = [plt.Line2D([], [], color=c, lw=2, marker=mk, ms=6, mec="white", label=lab) for lab, c, mk in METH.values()]
    fig.legend(handles=h, loc="outside upper center", ncol=4, frameon=False, fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fewshot.{ext}", dpi=200)
    print(FIG / "fewshot.png")


if __name__ == "__main__":
    main()
