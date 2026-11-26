"""TTA hyperparameter sensitivity (2026-11-14). Uses only the existing validation-split tuning runs
(results/tta/tune.csv; no test data, no new training). Gain = method score - no-adaptation score,
averaged over tuning domains and seeds. Writes results/analysis/tta_sensitivity.csv."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
T = pd.read_csv(ROOT / "results" / "tta" / "tune.csv")
sel = json.loads((ROOT / "results" / "tta" / "selected.json").read_text())["hp"]
rows = []
for c in [c for c in T.columns if "|" in c]:
    m, hp = c.split("|"); hp = json.loads(hp)
    for ds, g in T.groupby("dataset"):
        if g[c].isna().all():
            continue
        rows.append(dict(dataset=ds, method=m, lr=hp["lr"], steps=hp["steps"],
                         gain=(g[c] - g["none"]).mean(), selected=sel.get(ds, {}).get(m) == hp))
R = pd.DataFrame(rows)
R.to_csv(ROOT / "results" / "analysis" / "tta_sensitivity.csv", index=False)
P = R.pivot_table(index=["method", "lr", "steps"], columns="dataset", values="gain")
print("validation gain over no adaptation (mean over tuning domains x seeds):")
print(P.round(3).to_string())
print("\nselected configs:"); print(R[R.selected].round(3).to_string(index=False))
