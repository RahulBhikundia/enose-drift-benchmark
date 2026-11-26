"""Select one configuration per (dataset, method) from the tuning runs.

Criterion: highest mean (over 3 seeds) of the best-epoch validation macro-F1 (over classes present).
Diverged runs score 0. The epoch budget for final runs = median best epoch across the 3 seeds
of the selected configuration. Writes results/tuning/selected_stage{N}.json and a summary CSV.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "tuning"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", type=int, required=True)
    a = ap.parse_args()
    methods = {1: ["sgd", "adamw", "sam", "msam"],
               2: ["samsam_extrap", "samsam_disagree", "samsam_random", "recency", "gi"],
               3: ["domnorm", "erm_mixed", "sagm", "disam", "coral", "groupdro"],
               4: ["ccpca", "driftaug", "swa"]}[a.stage]
    sel, rows = {}, []
    for ds in ("uci", "smell"):
        sel[ds] = {}
        for m in methods:
            groups = defaultdict(list)
            for f in (OUT / ds / m).glob("*.json"):
                r = json.loads(f.read_text())
                key = json.dumps(dict(opt=r["opt"], method=r.get("method", "erm"), hp=r["hp"], mhp=r.get("mhp", {}),
                                      sampler=r.get("sampler", "domain"), norm=r.get("norm", "train")), sort_keys=True)
                bv = r.get("best_val") or {}
                groups[key].append((0.0 if r["diverged"] else bv.get("score", 0.0), bv.get("epoch", r["epochs"])))
            if not groups:
                continue
            stats = []
            for k, v in groups.items():
                sc = [x[0] for x in v]
                stats.append(dict(dataset=ds, method=m, hp=k, n_seeds=len(v), val_f1_mean=np.mean(sc),
                                  val_f1_sd=np.std(sc), best_epoch_median=int(np.median([x[1] for x in v]))))
            stats.sort(key=lambda d: -d["val_f1_mean"])
            rows += stats
            b = stats[0]
            cfg = json.loads(b["hp"])
            sel[ds][m] = dict(opt=cfg["opt"], method=cfg["method"], hp=cfg["hp"], mhp=cfg["mhp"], sampler=cfg["sampler"], norm=cfg["norm"], epochs=max(5, b["best_epoch_median"]),
                              val_f1_mean=b["val_f1_mean"], val_f1_sd=b["val_f1_sd"], n_configs=len(stats))
    (OUT / f"selected_stage{a.stage}.json").write_text(json.dumps(sel, indent=2))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"tuning_summary_stage{a.stage}.csv", index=False)
    for ds in sel:
        print(f"== {ds}")
        for m, v in sel[ds].items():
            print(f"  {m:16s} val_F1 {v['val_f1_mean']:.3f} +- {v['val_f1_sd']:.3f}  epochs {v['epochs']:3d}  hp {v['hp']}")


if __name__ == "__main__":
    main()
