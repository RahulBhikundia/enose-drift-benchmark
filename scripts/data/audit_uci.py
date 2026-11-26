"""Audit of the UCI Gas Sensor Array Drift dataset (proposal Section 3: "Required audit tasks").

Outputs (results/audit_uci/):
  class_by_batch.csv       class counts per batch
  label_runs.csv           within-batch clustering of classes in file order
  drift_vs_batch1.csv      feature shift and batch separability relative to batch 1
  time_only_baseline.csv   how well the batch ID alone predicts the gas
  sanity_temporal_lr.csv   logistic regression trained on early batches, tested per later batch
and figures/audit_uci_*.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data_uci import GAS_NAMES, feature_columns, load_uci  # noqa: E402

OUT = ROOT / "results" / "audit_uci"
FIG = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(exist_ok=True)
rng = np.random.default_rng(0)

df = load_uci()
F = feature_columns()
gas_order = [GAS_NAMES[i + 1] for i in range(6)]

# 1. Class-by-batch table -----------------------------------------------------------
tab = pd.crosstab(df["batch"], df["gas"])[gas_order]
tab["total"] = tab.sum(1)
tab.to_csv(OUT / "class_by_batch.csv")
print("=== Class counts per batch ===\n", tab, "\n")
missing = {b: [g for g in gas_order if tab.loc[b, g] == 0] for b in tab.index}
print("Missing classes per batch:", {b: m for b, m in missing.items() if m}, "\n")

fig, ax = plt.subplots(figsize=(7, 4.5))
im = ax.imshow(np.log1p(tab[gas_order].values), cmap="viridis", aspect="auto")
for i in range(tab.shape[0]):
    for j in range(6):
        ax.text(j, i, tab[gas_order].values[i, j], ha="center", va="center",
                color="white" if tab[gas_order].values[i, j] < 300 else "black", fontsize=8)
ax.set_xticks(range(6), gas_order, rotation=30)
ax.set_yticks(range(10), [f"batch {b}" for b in tab.index])
ax.set_title("UCI gas drift: samples per class and batch")
fig.colorbar(im, label="log(1+count)")
fig.tight_layout(); fig.savefig(FIG / "audit_uci_class_by_batch.png", dpi=150); plt.close(fig)

# 2. Within-batch clustering of labels in file order ------------------------------------
# If the files are sorted by class, the number of label runs equals the number of classes,
# and file order tells us nothing about acquisition time.
runs = []
for b, g in df.groupby("batch"):
    lab = g.sort_values("row_in_batch")["label"].to_numpy()
    n_runs = 1 + int(np.sum(lab[1:] != lab[:-1]))
    perm = [1 + int(np.sum(p[1:] != p[:-1])) for p in (rng.permutation(lab) for _ in range(500))]
    runs.append(dict(batch=b, n_classes=len(np.unique(lab)), observed_runs=n_runs,
                     random_runs_mean=np.mean(perm), random_runs_sd=np.std(perm)))
runs = pd.DataFrame(runs)
runs.to_csv(OUT / "label_runs.csv", index=False)
print("=== Label runs in file order (observed vs random shuffle) ===\n", runs, "\n")

# 3. Drift relative to batch 1 --------------------------------------------------------
# (a) median |standardised mean difference| of the features, computed within each gas and
#     then averaged over gases present in both batches, so class mix does not cause it.
# (b) batch separability: 5-fold cross-validated balanced accuracy of a logistic regression
#     telling batch 1 from batch k, again within each gas (0.5 = no detectable drift).
rows = []
for k in range(2, 11):
    smds, seps = [], []
    for gname in gas_order:
        a = df[(df.batch == 1) & (df.gas == gname)][F].to_numpy()
        c = df[(df.batch == k) & (df.gas == gname)][F].to_numpy()
        if len(a) < 10 or len(c) < 10:
            continue
        pooled = np.sqrt((a.var(0) + c.var(0)) / 2) + 1e-9
        smds.append(np.median(np.abs(a.mean(0) - c.mean(0)) / pooled))
        n = min(len(a), len(c))
        X = np.vstack([a[rng.choice(len(a), n, replace=False)], c[rng.choice(len(c), n, replace=False)]])
        y = np.r_[np.zeros(n), np.ones(n)]
        Xs = StandardScaler().fit_transform(X)  # scaling only, so leakage across folds is harmless here
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        seps.append(cross_val_score(LogisticRegression(max_iter=2000, C=0.1), Xs, y, cv=cv,
                                    scoring="balanced_accuracy").mean())
    rows.append(dict(batch=k, n_shared_gases=len(smds), median_abs_smd=np.mean(smds) if smds else np.nan,
                     batch_sep_bal_acc=np.mean(seps) if seps else np.nan))
drift = pd.DataFrame(rows)
drift.to_csv(OUT / "drift_vs_batch1.csv", index=False)
print("=== Drift vs batch 1 (within-gas) ===\n", drift.round(3), "\n")

fig, ax1 = plt.subplots(figsize=(6.5, 3.8))
ax1.plot(drift.batch, drift.median_abs_smd, "o-", label="median |SMD| (left)")
ax1.set_xlabel("batch"); ax1.set_ylabel("median |standardised mean diff.|")
ax2 = ax1.twinx()
ax2.plot(drift.batch, drift.batch_sep_bal_acc, "s--", color="C1", label="batch-1-vs-k separability (right)")
ax2.set_ylabel("balanced acc. (0.5 = no drift)"); ax2.set_ylim(0.45, 1.02)
fig.legend(loc="lower right", fontsize=8); ax1.set_title("UCI: within-gas drift relative to batch 1")
fig.tight_layout(); fig.savefig(FIG / "audit_uci_drift.png", dpi=150); plt.close(fig)

# 4. Time-only baseline: predict the gas from the batch ID alone --------------------------
# Predict each batch's majority class. If this beats chance by a lot, time and label are
# entangled, and pooled metrics can reward learning "when" instead of "what".
maj = df.groupby("batch")["label"].agg(lambda s: s.value_counts().idxmax())
pred = df["batch"].map(maj)
time_only = dict(
    accuracy=float((pred == df.label).mean()),
    balanced_accuracy=float(balanced_accuracy_score(df.label, pred)),
    chance_accuracy_majority_overall=float(df.label.value_counts(normalize=True).max()),
)
pd.DataFrame([time_only]).to_csv(OUT / "time_only_baseline.csv", index=False)
print("=== Batch-ID-only predictor of gas ===\n", time_only, "\n")

# 5. Sanity check: temporal split with logistic regression --------------------------------
# Train on batches 1-5 (scaler fit on training data only), report each later batch separately,
# plus a random 80/20 split of the same data for contrast.
tr = df[df.batch <= 5]
sc = StandardScaler().fit(tr[F])
clf = LogisticRegression(max_iter=5000, C=1.0).fit(sc.transform(tr[F]), tr.label)
res = []
for k in range(6, 11):
    te = df[df.batch == k]
    p = clf.predict(sc.transform(te[F]))
    res.append(dict(test_batch=k, n=len(te), acc=(p == te.label).mean(),
                    macro_f1=f1_score(te.label, p, average="macro"),
                    bal_acc=balanced_accuracy_score(te.label, p)))
idx = rng.permutation(len(df)); cut = int(0.8 * len(df))
a, b_ = df.iloc[idx[:cut]], df.iloc[idx[cut:]]
sc2 = StandardScaler().fit(a[F])
p2 = LogisticRegression(max_iter=5000).fit(sc2.transform(a[F]), a.label).predict(sc2.transform(b_[F]))
res.append(dict(test_batch="random 80/20 (all batches)", n=len(b_), acc=(p2 == b_.label).mean(),
                macro_f1=f1_score(b_.label, p2, average="macro"), bal_acc=balanced_accuracy_score(b_.label, p2)))
res = pd.DataFrame(res)
res.to_csv(OUT / "sanity_temporal_lr.csv", index=False)
print("=== Logistic regression: train batches 1-5, test later batches ===\n", res.round(3))
