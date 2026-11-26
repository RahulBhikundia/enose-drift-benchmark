"""SWA protocol fix ("swa_full"), added 2026-11-12 after finding that the standard selection (best-epoch
validation score, epoch budget = median best epoch) picked SWA configs whose averaging starts AFTER the
selected stopping epoch (UCI: start 10 > 8 epochs; SmellNet: start 50 > 42), so no averaging ever happened.

SWA is meant to be evaluated at the END of training. swa_full re-selects among the SAME tuning runs
(no new tuning) by the LAST-epoch validation score of the averaged model, and trains for the full tuning
budget. Writes results/tuning/selected_stage5.json, picked up by run_final.py as method "swa_full".
Datasets: uci, smell (stage-4 tuning) and twin, twinreg (run_twin.py tuning, if present).
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "results" / "tuning"
EPOCHS = {"uci": 60, "smell": 100, "twin": 100, "twinreg": 100}


def main():
    sel = {}
    for ds in EPOCHS:
        groups = defaultdict(list)
        for f in (T / ds / "swa").glob("*.json"):
            r = json.loads(f.read_text())
            last = -1e9 if r["diverged"] or not r["curve"] else r["curve"][-1].get("val_macro_f1", -1e9)
            groups[json.dumps(dict(hp=r["hp"], mhp=r["mhp"]), sort_keys=True)].append(last)
        if not groups:
            continue
        k, v = max(groups.items(), key=lambda kv: np.mean(kv[1]))
        c = json.loads(k)
        sel[ds] = {"swa_full": dict(opt="adamw", method="swa", hp=c["hp"], mhp=c["mhp"], sampler="domain", norm="train",
                                    epochs=EPOCHS[ds], val_f1_mean=float(np.mean(v)), val_f1_sd=float(np.std(v)),
                                    n_configs=len(groups))}
        print(f"{ds:8s} swa_full: hp {c['hp']} {c['mhp']}  last-epoch val {np.mean(v):.3f} +- {np.std(v):.3f}  epochs {EPOCHS[ds]}")
    (T / "selected_stage5.json").write_text(json.dumps(sel, indent=2))


if __name__ == "__main__":
    main()
