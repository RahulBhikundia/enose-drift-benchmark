"""Hyperparameter tuning on the VALIDATION splits only (UCI: train 1-3 -> val 4-5; SmellNet: train
sessions 1-4 -> val 5). No test data is touched here.

Equal budget: every method gets exactly 12 configurations x 3 seeds.
Stage 1: sgd, adamw, sam, msam.
Stage 2: SA-MSAM variants (extrap, disagree, random-control). Each inherits rho from MSAM's
         selected config (read from results/tuning/selected_stage1.json) and tunes lr x beta (12 configs).
Resumable: existing result files are skipped.

Usage: python scripts/experiments/run_grid.py --stage 1   (then scripts/experiments/select_hp.py --stage 1, then --stage 2)
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.train import parse, run  # noqa: E402

OUT = ROOT / "results" / "tuning"
EPOCHS = {"uci": 60, "smell": 100}   # smell raised from 60: best epochs hit the cap (51-56 of 60)
SEEDS = [0, 1, 2]

GRIDS = {
    "sgd":   [dict(lr=lr, weight_decay=wd) for lr, wd in itertools.product([0.003, 0.01, 0.03, 0.1], [0.0, 1e-4, 1e-3])],
    "adamw": [dict(lr=lr, weight_decay=wd) for lr, wd in itertools.product([1e-4, 3e-4, 1e-3, 3e-3], [0.0, 1e-2, 1e-1])],
    "sam":   [dict(lr=lr, rho=r) for lr, r in itertools.product([0.01, 0.03, 0.1], [0.02, 0.05, 0.1, 0.2])],
    "msam":  [dict(lr=lr, rho=r) for lr, r in itertools.product([0.01, 0.03, 0.1], [0.1, 0.3, 1.0, 3.0])],
}
STAGE2 = ["samsam_extrap", "samsam_disagree", "samsam_random", "recency", "gi"]
# Baseline training methods (stage 2) run on AdamW; each also gets 12 configs (lr x method hp).
METHOD_GRIDS = {
    "recency": [(dict(lr=lr), dict(gamma=g)) for lr, g in itertools.product([3e-4, 1e-3, 3e-3], [0.3, 0.5, 0.7, 0.9])],
    "gi":      [(dict(lr=lr), dict(lam=l, delta=d)) for lr, l, d in itertools.product([3e-4, 1e-3, 3e-3], [0.5, 1.0], [0.5, 1.0])],
}


# Stage 3 (added 2026-11-12): extra baselines, same budget (12 configs x 3 seeds). Entries are
# (opt, method, hp, mhp, extra CLI args). Method hyperparameter defaults follow the official code:
# SAGM alpha = 0.001 (fixed), DISAM lambda = 0.1 (fixed); rho and lr are tuned.
ADAMW_GRID = [dict(lr=lr, weight_decay=wd) for lr, wd in itertools.product([1e-4, 3e-4, 1e-3, 3e-3], [0.0, 1e-2, 1e-1])]
MIXED = ["--sampler", "mixed"]
STAGE3 = {
    "domnorm":   [("adamw", "erm", hp, {}, ["--norm", "domain"]) for hp in ADAMW_GRID],
    "erm_mixed": [("adamw", "erm", hp, {}, MIXED) for hp in ADAMW_GRID],
    "sagm":      [("sagm", "erm", dict(lr=lr, rho=r, alpha=0.001), {}, []) for lr, r in itertools.product([0.01, 0.03, 0.1], [0.02, 0.05, 0.1, 0.2])],
    "disam":     [("disam", "erm", dict(lr=lr, rho=r, lam=0.1), {}, MIXED) for lr, r in itertools.product([0.01, 0.03, 0.1], [0.02, 0.05, 0.1, 0.2])],
    "coral":     [("adamw", "coral", dict(lr=lr), dict(lam=l), MIXED) for lr, l in itertools.product([3e-4, 1e-3, 3e-3], [0.1, 0.3, 1.0, 3.0])],
    "groupdro":  [("adamw", "groupdro", dict(lr=lr), dict(eta=e), MIXED) for lr, e in itertools.product([3e-4, 1e-3, 3e-3], [0.001, 0.01, 0.1, 1.0])],
}

# Stage 4 (added 2026-11-12): CC-PCA drift removal, drift augmentation, SWA (all on AdamW, 12 configs).
# For ccpca, k is also stored in mhp so that result filenames/keys are unique (mhp is unused by ERM).
LR3 = [3e-4, 1e-3, 3e-3]
STAGE4 = {
    "ccpca":    [("adamw", "erm", dict(lr=lr), dict(k=k), ["--norm", "ccpca", "--norm_k", str(k)]) for lr, k in itertools.product(LR3, [1, 2, 3, 4])],
    "driftaug": [("adamw", "driftaug", dict(lr=lr), dict(gain=g, offset=o), []) for lr, (g, o) in itertools.product(LR3, [(0.1, 0.1), (0.2, 0.3), (0.3, 0.5), (0.5, 1.0)])],
}
SWA_STARTS = {"uci": [1, 3, 5, 10], "smell": [5, 15, 30, 50]}


def stage4_grid(dataset):
    g = dict(STAGE4)
    g["swa"] = [("adamw", "swa", dict(lr=lr), dict(swa_start=st), []) for lr, st in itertools.product(LR3, SWA_STARTS[dataset])]
    return g


def stage2_grid(dataset, name):
    """Returns list of (opt, method, hp, mhp)."""
    if name in METHOD_GRIDS:
        return [("adamw", name, hp, mhp) for hp, mhp in METHOD_GRIDS[name]]
    sel = json.loads((OUT / "selected_stage1.json").read_text())
    rho = sel[dataset]["msam"]["hp"]["rho"]
    return [(name, "erm", dict(lr=lr, rho=rho, beta=b), {})
            for lr, b in itertools.product([0.01, 0.03, 0.1], [-1.0, -0.5, 0.5, 1.0])]


def hp_tag(hp):
    return "_".join(f"{k}{v:g}" for k, v in sorted(hp.items()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", type=int, required=True)
    ap.add_argument("--datasets", nargs="+", default=["uci", "smell"])
    a = ap.parse_args()
    jobs = []
    for ds in a.datasets:
        if a.stage == 1:
            grids = {opt: [(opt, "erm", hp, {}, []) for hp in g] for opt, g in GRIDS.items()}
        elif a.stage == 2:
            grids = {name: [(*c, []) for c in stage2_grid(ds, name)] for name in STAGE2}
        elif a.stage == 3:
            grids = STAGE3
        else:
            grids = stage4_grid(ds)
        for name, grid in grids.items():
            for opt, method, hp, mhp, extra in grid:
                for s in SEEDS:
                    jobs.append((ds, name, opt, method, hp, mhp, extra, s))
    print(f"{len(jobs)} jobs", flush=True)
    for i, (ds, name, opt, method, hp, mhp, extra, s) in enumerate(jobs):
        out = OUT / ds / name / f"{hp_tag({**hp, **mhp})}_s{s}.json"
        if out.exists():
            continue
        r = run(parse(["--dataset", ds, "--split", "tune", "--opt", opt, "--hp", json.dumps(hp),
                       "--method", method, "--mhp", json.dumps(mhp),
                       "--seed", str(s), "--epochs", str(EPOCHS[ds]), "--out", str(out)] + extra))
        if i % 20 == 0:
            bv = r["best_val"]["score"] if r.get("best_val") else None
            print(f"[{i+1}/{len(jobs)}] {ds} {name} {hp} {mhp} s{s} best_val={bv}", flush=True)
    print("DONE stage", a.stage, flush=True)


if __name__ == "__main__":
    main()
