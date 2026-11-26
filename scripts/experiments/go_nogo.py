"""Go/no-go table for SA-MSAM: compare every method's selected config on VALIDATION only.

Reads results/tuning/selected_stage{1,2}.json and the per-seed tuning JSONs, and prints, per dataset,
mean +- sd of best-epoch val macro-F1, the per-seed scores, and the gap to MSAM (the method SA-MSAM
modifies). No test data is touched. Writes results/tuning/go_nogo.csv.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "tuning"


def seed_scores(ds, m, cfg):
    scores = {}
    for f in (OUT / ds / m).glob("*.json"):
        r = json.loads(f.read_text())
        if (r["hp"] == cfg["hp"] and r.get("mhp", {}) == cfg["mhp"] and r.get("method", "erm") == cfg["method"]
                and r.get("sampler", "domain") == cfg.get("sampler", "domain") and r.get("norm", "train") == cfg.get("norm", "train")):
            bv = r.get("best_val") or {}
            scores[r.get("seed", f.stem)] = 0.0 if r["diverged"] else bv.get("score", 0.0)
    return dict(sorted(scores.items()))


def main():
    sel = {}
    for st in (1, 2, 3, 4):
        if not (OUT / f'selected_stage{st}.json').exists(): continue
        p = OUT / f"selected_stage{st}.json"
        for ds, d in json.loads(p.read_text()).items():
            sel.setdefault(ds, {}).update(d)
    rows = []
    for ds, d in sel.items():
        ref = np.array(list(seed_scores(ds, "msam", d["msam"]).values())) if "msam" in d else None
        print(f"== {ds}  (validation macro-F1, 3 seeds)")
        for m, cfg in d.items():
            s = seed_scores(ds, m, cfg)
            v = np.array(list(s.values()))
            gap = (v.mean() - ref.mean()) if ref is not None else np.nan
            rows.append(dict(dataset=ds, method=m, mean=v.mean(), sd=v.std(), gap_vs_msam=gap,
                             seeds=" ".join(f"{x:.3f}" for x in v), hp=json.dumps(cfg["hp"]), mhp=json.dumps(cfg["mhp"]),
                             epochs=cfg["epochs"]))
            print(f"  {m:16s} {v.mean():.3f} +- {v.std():.3f}  gap_vs_msam {gap:+.3f}  seeds [{rows[-1]['seeds']}]  "
                  f"hp {cfg['hp']} {cfg['mhp'] or ''}")
    pd.DataFrame(rows).to_csv(OUT / "go_nogo.csv", index=False)


if __name__ == "__main__":
    main()
