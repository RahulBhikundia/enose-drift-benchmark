"""Audit of SmellNet-Base: split leakage and session (day) confounding.

Questions:
  Q1. How much does a random-window split (windows from the same session in train and test)
      overstate accuracy compared with leave-one-session-out (LOSO)?
  Q2. Session/day confound: does the *start* of a session (first seconds) already identify the
      substance? If so, the ambient state of that day, not the smell, may be what is learned.
      Also: can we tell which session a window came from, within one substance?
  Q3. Per-session statistics: lengths, channel levels, and whether session 6 (the official test)
      looks different from sessions 1-5.

Model: a random forest on simple window summary features (mean/std/min/max/slope per channel).
It is deliberately model-agnostic and fast; the deep models come later.
Outputs: results/audit_smellnet/*.csv, figures/audit_smellnet_*.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data_smellnet import CHANNELS, load_sessions, make_windows, substance_index, window_features  # noqa: E402

OUT = ROOT / "results" / "audit_smellnet"; OUT.mkdir(parents=True, exist_ok=True)
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
WIN, STRIDE = 60, 10          # 60 s windows, 10 s stride
SEED = 0

sessions = load_sessions()
cls = substance_index(sessions)
n_cls = len(cls)


def rf():
    return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED, min_samples_leaf=2)


def build(sess_list, start=0, max_len=None):
    X, y, sid, sess = [], [], [], []
    for k, s in enumerate(sess_list):
        arr = s["data"] if max_len is None else s["data"][: start + max_len]
        w = make_windows(arr, WIN, STRIDE, start)
        if len(w) == 0:
            continue
        X.append(window_features(w)); y += [cls[s["substance"]]] * len(w)
        sid += [k] * len(w); sess += [s["session"]] * len(w)
    return np.vstack(X), np.array(y), np.array(sid), np.array(sess)


def scores(y, p):
    return dict(acc=accuracy_score(y, p), bal_acc=balanced_accuracy_score(y, p),
                macro_f1=f1_score(y, p, average="macro"))


X, y, sid, sess = build(sessions)
print(f"{len(X)} windows from {len(sessions)} sessions, {n_cls} classes (chance = {1/n_cls:.3f})")

# Q3. Per-session statistics -------------------------------------------------------------
stat = pd.DataFrame([dict(substance=s["substance"], session=s["session"], length=len(s["data"]),
                          **{f"mean_{c}": s["data"][:, i].mean() for i, c in enumerate(CHANNELS)})
                     for s in sessions])
stat.to_csv(OUT / "session_stats.csv", index=False)
print("\n=== Mean channel level by session index (averaged over substances) ===")
print(stat.groupby("session")[[f"mean_{c}" for c in CHANNELS] + ["length"]].mean().round(1))

# Q1a. Random-window split (leaky) --------------------------------------------------------
rng = np.random.default_rng(SEED)
perm = rng.permutation(len(X)); cut = int(0.8 * len(X))
m = rf().fit(X[perm[:cut]], y[perm[:cut]])
rows = [dict(protocol="random 80/20 windows (leaky)", **scores(y[perm[cut:]], m.predict(X[perm[cut:]])))]

# Q1b. Official split: sessions 1-5 train, session 6 test ---------------------------------
tr, te = sess <= 5, sess == 6
m = rf().fit(X[tr], y[tr])
rows.append(dict(protocol="official: train s1-5, test s6", **scores(y[te], m.predict(X[te]))))

# Q1c. Leave-one-session-out over all six sessions -----------------------------------------
loso = []
for hold in range(1, 7):
    tr, te = sess != hold, sess == hold
    m = rf().fit(X[tr], y[tr])
    loso.append(dict(held_out_session=hold, **scores(y[te], m.predict(X[te]))))
loso = pd.DataFrame(loso)
loso.to_csv(OUT / "loso_per_session.csv", index=False)
rows.append(dict(protocol="leave-one-session-out (mean of 6)", **loso[["acc", "bal_acc", "macro_f1"]].mean().to_dict()))
print("\n=== LOSO per held-out session ===\n", loso.round(3))

# Q2a. Session-start confound: classify the substance from the first window only ----------
# Uses the first WIN seconds of each session (one window per session), with LOSO.
first = []
for hold in range(1, 7):
    trs = [s for s in sessions if s["session"] != hold]; tes = [s for s in sessions if s["session"] == hold]
    Xa, ya, _, _ = build(trs, max_len=WIN); Xb, yb, _, _ = build(tes, max_len=WIN)
    m = rf().fit(Xa, ya); first.append(scores(yb, m.predict(Xb)))
rows.append(dict(protocol="first 60 s only, LOSO (start-of-session confound)", **pd.DataFrame(first).mean().to_dict()))

# Q2b. Can the session be identified within a substance? (day signature)
# For every substance, classify which of its 6 sessions each window came from (random split).
sess_id = []
for c in range(n_cls):
    mk = y == c; Xc, sc = X[mk], sess[mk]
    p = rng.permutation(len(Xc)); k = int(0.8 * len(Xc))
    mm = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED).fit(Xc[p[:k]], sc[p[:k]])
    sess_id.append(accuracy_score(sc[p[k:]], mm.predict(Xc[p[k:]])))
rows.append(dict(protocol="identify session within substance (chance 1/6)", acc=float(np.mean(sess_id)),
                 bal_acc=np.nan, macro_f1=np.nan))

summary = pd.DataFrame(rows)
summary.to_csv(OUT / "split_comparison.csv", index=False)
print("\n=== Split / confound summary (chance acc = 0.02) ===\n", summary.round(3).to_string(index=False))

# Figure: average normalised trajectory of each channel over a session ----------------------
fig, axes = plt.subplots(2, 3, figsize=(10, 5), sharex=True)
L = 500
for i, (ax, ch) in enumerate(zip(axes.ravel(), CHANNELS)):
    for sn in range(1, 7):
        tr_ = np.stack([s["data"][:L, i] for s in sessions if s["session"] == sn and len(s["data"]) >= L])
        ax.plot(tr_.mean(0), label=f"s{sn}", lw=1)
    ax.set_title(ch, fontsize=9)
axes[0, 0].legend(fontsize=6, ncol=2); fig.supxlabel("seconds since session start")
fig.suptitle("SmellNet: mean raw channel trajectory per session index (all substances)")
fig.tight_layout(); fig.savefig(FIG / "audit_smellnet_trajectories.png", dpi=150); plt.close(fig)
