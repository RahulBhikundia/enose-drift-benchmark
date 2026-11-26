"""Features vs model on SmellNet: per-split test macro-F1 (mean over seeds) of CNN (raw windows), MLP on the 30 RF
summary features (smellfeat), and random forest on the same 30 features. Differences vs CNN-AdamW."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
F = pd.read_csv(ROOT / "results" / "analysis" / "final_runs.csv")
S = F[(F.dataset == "smell") & F.method.isin(["adamw", "coral", "rf"])]
rows = [dict(model=f"CNN raw / {m}" if m != "rf" else "RF / 30 features", split=r.split, seed=r.seed, f1=r.test_f1)
        for m, g in S.groupby("method") for r in g.itertuples()]
for ds, lab in (("smellfeat", "MLP 30 features"), ("smellgru", "GRU raw")):
  for m in ("adamw", "coral"):
    for f in (ROOT / "results" / "final" / ds / m).glob("*.json"):
        r = json.loads(f.read_text()); mo = r["final_eval"]["mean_over_domains"]
        rows.append(dict(model=f"{lab} / {m}", split=r["split"], seed=r["seed"],
                         f1=0.0 if r["diverged"] else mo.get("score", mo["macro_f1_present"])))
d = pd.DataFrame(rows)
P = d.groupby(["split", "model"]).f1.mean().unstack()
print(P.round(3).to_string()); print("\nmean over splits:"); print(P.mean().round(3).sort_values(ascending=False).to_string())
ref = P["CNN raw / adamw"]
print("\nminus CNN raw / adamw: mean [wins/6]")
for c in P.columns:
    if c != "CNN raw / adamw":
        x = P[c] - ref; print(f"  {c:24s} {x.mean():+.3f}  [{(x > 0).sum()}/{len(x)}]")
