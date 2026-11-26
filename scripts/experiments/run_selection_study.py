"""Selection study: how well does validation-based hyperparameter selection work under real drift?

For each method below, EVERY tuning configuration (the same 12 as in run_grid.py) is also run
  (a) on the main TEST split (UCI: train 1-5 -> test 6-10; SmellNet: loso6 = official split), 3 seeds,
      for that config's own epoch budget (median best epoch of its 3 tuning seeds, like the final runs);
  (b) on a second, earlier forward validation fold "tune2" (UCI 1-2 -> 3; SmellNet 1-3 -> 4), 3 seeds.
This lets us measure the val->test rank correlation across configs, the regret of validation selection
versus an oracle (test-best config, NOT a legitimate method), and alternative selection rules.
None of this changes the frozen main results. Resumable.

Usage: python scripts/experiments/run_selection_study.py [--parts test fold2]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.train import parse, run  # noqa: E402
import run_grid as G  # noqa: E402

TUNE = ROOT / "results" / "tuning"
OUT = ROOT / "results" / "selection"
METHODS = ["sgd", "adamw", "sam", "msam", "recency", "coral", "groupdro", "disam"]
TEST_SPLIT = {"uci": "main", "smell": "loso6"}


def configs(ds, name):
    """(opt, method, hp, mhp, extra) for every tuning config of `name`, exactly as in run_grid.py."""
    if name in G.GRIDS:
        return [(name, "erm", hp, {}, []) for hp in G.GRIDS[name]]
    if name in G.METHOD_GRIDS:
        return [("adamw", name, hp, mhp, []) for hp, mhp in G.METHOD_GRIDS[name]]
    if name in G.STAGE3:
        return G.STAGE3[name]
    raise ValueError(name)


def tuned_epochs(ds, name, tag):
    eps = []
    for s in G.SEEDS:
        f = TUNE / ds / name / f"{tag}_s{s}.json"
        r = json.loads(f.read_text())
        eps.append((r.get("best_val") or {}).get("epoch", r["epochs"]))
    return max(5, int(np.median(eps)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--parts", nargs="+", default=["test", "fold2"])
    a = ap.parse_args()
    jobs = []
    for part in a.parts:
        for ds in ("uci", "smell"):
            for name in METHODS:
                for opt, method, hp, mhp, extra in configs(ds, name):
                    for s in G.SEEDS:
                        jobs.append((part, ds, name, opt, method, hp, mhp, extra, s))
    print(f"{len(jobs)} jobs", flush=True)
    for i, (part, ds, name, opt, method, hp, mhp, extra, s) in enumerate(jobs):
        tag = G.hp_tag({**hp, **mhp})
        out = OUT / ds / part / name / f"{tag}_s{s}.json"
        if out.exists():
            continue
        if part == "test":
            split, epochs = TEST_SPLIT[ds], tuned_epochs(ds, name, tag)
        else:
            split, epochs = "tune2", G.EPOCHS[ds]
        r = run(parse(["--dataset", ds, "--split", split, "--opt", opt, "--hp", json.dumps(hp), "--method", method,
                       "--mhp", json.dumps(mhp), "--seed", str(s), "--epochs", str(epochs), "--out", str(out)] + extra))
        if i % 25 == 0:
            fe = (r.get("final_eval") or {}).get("mean_over_domains", {})
            print(f"[{i+1}/{len(jobs)}] {part} {ds} {name} {tag} s{s} F1={fe.get('macro_f1_present')}", flush=True)
    print("DONE selection", flush=True)


if __name__ == "__main__":
    main()
