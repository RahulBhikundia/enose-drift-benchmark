"""Twin Gas Sensor Arrays pipeline (third dataset): tuning -> selection -> (final runs via run_final.py).

Datasets: twin    = gas classification (4 classes), score = macro-F1
          twinreg = log-concentration regression of a known gas, score = R2
Tuning split "tune": train units 1-3 -> validate on unit 4. Same budget as UCI/SmellNet: 12 configs x 3 seeds,
the same grids (run_grid.py). Methods that assume time-ordered domains are skipped (recency, gi,
samsam_extrap): Twin units have no temporal order. SA-MSAM disagree/random inherit MSAM's selected rho.
Selection: highest mean best-epoch validation score; epochs = median best epoch (>= 5).
Writes results/tuning/<ds>/<method>/*.json and results/tuning/selected_twin.json. Resumable.

Usage: python scripts/experiments/run_twin.py --phase tune     (all methods except SA-MSAM, then select)
       python scripts/experiments/run_twin.py --phase tune2    (SA-MSAM with MSAM's rho, then select again)
"""
import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.train import parse, run  # noqa: E402
import run_grid as G  # noqa: E402

OUT = ROOT / "results" / "tuning"
DATASETS = ["twin", "twinreg"]
EPOCHS = 100
SWA_STARTS = [5, 15, 30, 50]
PHASE1 = ["sgd", "adamw", "sam", "msam", "sagm", "disam", "coral", "groupdro", "erm_mixed", "domnorm",
          "ccpca", "driftaug", "swa"]
PHASE2 = ["samsam_disagree", "samsam_random"]


def configs(ds, name):
    if name in G.GRIDS:
        return [(name, "erm", hp, {}, []) for hp in G.GRIDS[name]]
    if name in G.STAGE3:
        return G.STAGE3[name]
    if name == "swa":
        return [("adamw", "swa", dict(lr=lr), dict(swa_start=st), []) for lr, st in itertools.product(G.LR3, SWA_STARTS)]
    if name in G.STAGE4:
        return G.STAGE4[name]
    if name in PHASE2:
        rho = json.loads((OUT / "selected_twin.json").read_text())[ds]["msam"]["hp"]["rho"]
        return [(name, "erm", dict(lr=lr, rho=rho, beta=b), {}, [])
                for lr, b in itertools.product([0.01, 0.03, 0.1], [-1.0, -0.5, 0.5, 1.0])]
    raise ValueError(name)


def select(methods):
    f = OUT / "selected_twin.json"
    sel = json.loads(f.read_text()) if f.exists() else {}
    for ds in DATASETS:
        sel.setdefault(ds, {})
        for m in methods:
            groups = defaultdict(list)
            for p in (OUT / ds / m).glob("*.json"):
                r = json.loads(p.read_text())
                key = json.dumps(dict(opt=r["opt"], method=r.get("method", "erm"), hp=r["hp"], mhp=r.get("mhp", {}),
                                      sampler=r.get("sampler", "domain"), norm=r.get("norm", "train")), sort_keys=True)
                bv = r.get("best_val") or {}
                sc = -1e9 if r["diverged"] else bv.get("score", -1e9)
                groups[key].append((sc, bv.get("epoch", r["epochs"])))
            if not groups:
                continue
            best_key, v = max(groups.items(), key=lambda kv: np.mean([x[0] for x in kv[1]]))
            cfg = json.loads(best_key)
            sel[ds][m] = dict(cfg, epochs=max(5, int(np.median([x[1] for x in v]))),
                              val_score_mean=float(np.mean([x[0] for x in v])), val_score_sd=float(np.std([x[0] for x in v])),
                              n_configs=len(groups))
    f.write_text(json.dumps(sel, indent=2))
    for ds in DATASETS:
        print(f"== {ds}")
        for m, c in sel[ds].items():
            print(f"  {m:16s} val {c['val_score_mean']:.3f} +- {c['val_score_sd']:.3f}  epochs {c['epochs']:3d}  hp {c['hp']} {c['mhp'] or ''}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--phase", choices=["tune", "tune2"], required=True)
    a = ap.parse_args()
    methods = PHASE1 if a.phase == "tune" else PHASE2
    jobs = [(ds, m, *c, s) for ds in DATASETS for m in methods for c in configs(ds, m) for s in G.SEEDS]
    print(f"{len(jobs)} jobs", flush=True)
    for i, (ds, name, opt, method, hp, mhp, extra, s) in enumerate(jobs):
        out = OUT / ds / name / f"{G.hp_tag({**hp, **mhp})}_s{s}.json"
        if out.exists():
            continue
        r = run(parse(["--dataset", ds, "--split", "tune", "--opt", opt, "--hp", json.dumps(hp), "--method", method,
                       "--mhp", json.dumps(mhp), "--seed", str(s), "--epochs", str(EPOCHS), "--out", str(out)] + extra))
        if i % 25 == 0:
            bv = (r.get("best_val") or {}).get("score")
            print(f"[{i+1}/{len(jobs)}] {ds} {name} {hp} {mhp} s{s} best_val={bv}", flush=True)
    select(methods)
    print(f"DONE twin {a.phase}", flush=True)


if __name__ == "__main__":
    main()
