"""Label-free test-time adaptation figure (2026-11-26): gain over no adaptation per method and dataset,
mean over test splits of per-domain seed means (5 seeds), from results/tta/criteria_final.csv (same aggregation as
run_tta.py report). Few-shot (labelled) shown for reference in a second colour. Writes figures/tta.{png,pdf}."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "figures"
C = pd.read_csv(ROOT / "results" / "tta" / "criteria_final.csv")
DS = {"twin": "Twin: gas ID\n(device shift, F1)", "twinreg": "Twin: concentration\n(device shift, R²)",
      "smell": "SmellNet\n(session shift, F1)", "uci": "UCI\n(ageing, F1)"}
METH = [("standardize", "standardize inputs"), ("feat_align", "align features"), ("ent_affine", "entropy: input gain/offset"),
        ("shot_im", "SHOT-IM: adapt extractor"), ("fewshot", "few-shot fine-tune (labels)")]
LABEL_FREE, LABELLED = "#2a78d6", "#eb6834"          # validated categorical slots 1-2 (same as other figures)
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"

plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
fig, axes = plt.subplots(1, 4, figsize=(12, 3.0), sharey=True, constrained_layout=True)
cols = ["none"] + [m for m, _ in METH]
S = C.groupby(["dataset", "split"])[cols].mean().groupby("dataset").mean()
for ax, (ds, title) in zip(axes, DS.items()):
    ys = list(range(len(METH)))[::-1]
    ax.axvline(0, color=MUTED, lw=0.8); ax.grid(True, axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for y, (m, lab) in zip(ys, METH):
        v = S.loc[ds, m] - S.loc[ds, "none"] if pd.notna(S.loc[ds, m]) else None
        if v is None:
            ax.text(0.005, y, "n/a (regression)", va="center", fontsize=7, color=MUTED); continue
        ax.barh(y, v, height=0.62, color=LABELLED if m == "fewshot" else LABEL_FREE, edgecolor="white", lw=2)
        ax.text(v + (0.006 if v >= 0 else -0.006), y, f"{v:+.3f}", va="center", ha="left" if v >= 0 else "right",
                fontsize=7, color=INK)
    ax.set_yticks(ys); ax.set_yticklabels([lab for _, lab in METH])
    ax.set_xlim(-0.16, 0.24)
    ax.set_title(f"{title}\nno adaptation = {S.loc[ds, 'none']:.3f}", fontsize=8, color=INK, loc="left")
    ax.set_xlabel("gain over no adaptation")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
h = [plt.Rectangle((0, 0), 1, 1, color=LABEL_FREE, label="label-free (unlabelled target data)"),
     plt.Rectangle((0, 0), 1, 1, color=LABELLED, label="uses labelled target samples")]
fig.legend(handles=h, loc="outside upper center", ncol=2, frameon=False, fontsize=8)
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"tta.{ext}", dpi=200)
print(S.round(3).to_string())
