"""Day-1 style audit of the Twin Gas Sensor Arrays dataset (64 Vergara-style features, 4 gases).

1. class balance per unit and repetition
2. random split vs leave-one-unit-out (LOUO): how much does a random split overstate accuracy?
3. how separable are units (device signature) within a gas? and repetitions (days) within a unit?
4. per-unit feature shift: median |standardized mean difference| between units, per gas
Writes results/audit_twin/summary.txt.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data_twin import GAS_NAMES, feature_columns, features  # noqa: E402

OUT = ROOT / "results" / "audit_twin"; OUT.mkdir(parents=True, exist_ok=True)
lines = []


def say(s=""):
    print(s); lines.append(str(s))


df = features(); F = feature_columns(df); X = df[F].to_numpy(); y = df.gas.to_numpy(); u = df.unit.to_numpy()
say(f"{len(df)} recordings, {len(F)} features, gases {GAS_NAMES}")
say("1. counts unit x gas:"); say(pd.crosstab(df.unit, df.gas.map(GAS_NAMES)).to_string())
say("   counts unit x repetition:"); say(pd.crosstab(df.unit, df.rep).to_string())


def models():
    return {"logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0)),
            "rf": RandomForestClassifier(500, n_jobs=-1, random_state=0)}


say("\n2. random 5-fold (stratified) vs leave-one-unit-out; macro-F1 [balanced acc]")
for name, mk in models().items():
    p = cross_val_predict(mk, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0))
    say(f"   {name:6s} random 5-fold : F1 {f1_score(y, p, average='macro'):.3f} [{balanced_accuracy_score(y, p):.3f}]")
    res = []
    for h in range(1, 6):
        m = models()[name].fit(X[u != h], y[u != h]); ph = m.predict(X[u == h])
        res.append(f1_score(y[u == h], ph, average="macro"))
    say(f"   {name:6s} LOUO per unit : " + " ".join(f"u{h}={r:.3f}" for h, r in zip(range(1, 6), res)) + f"  mean {np.mean(res):.3f}")

say("\n3. device signature: classify UNIT from features within each gas (5-fold, logreg); chance = 0.2")
for g in range(4):
    m = df.gas == g
    p = cross_val_predict(models()["logreg"], X[m], u[m], cv=StratifiedKFold(5, shuffle=True, random_state=0))
    say(f"   {GAS_NAMES[g]:9s}: balanced acc {balanced_accuracy_score(u[m], p):.3f}")
say("   day signature: classify REPETITION within unit (units 1-3: chance 0.25; 4-5: chance 0.5)")
for h in range(1, 6):
    m = df.unit == h
    p = cross_val_predict(models()["logreg"], X[m], df.rep[m], cv=StratifiedKFold(5, shuffle=True, random_state=0))
    say(f"   unit {h}: balanced acc {balanced_accuracy_score(df.rep[m], p):.3f}")

say("\n4. median |SMD| of each unit vs unit 1, same gas (features standardized on all data)")
Z = (X - X.mean(0)) / (X.std(0) + 1e-12)
for h in range(2, 6):
    smd = []
    for g in range(4):
        a, b = Z[(u == 1) & (y == g)], Z[(u == h) & (y == g)]
        smd.append(np.median(np.abs(a.mean(0) - b.mean(0)) / (np.sqrt((a.var(0) + b.var(0)) / 2) + 1e-12)))
    say(f"   unit {h}: " + " ".join(f"{GAS_NAMES[g]}={s:.2f}" for g, s in enumerate(smd)))
(OUT / "summary.txt").write_text("\n".join(lines))
