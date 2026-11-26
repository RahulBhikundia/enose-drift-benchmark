"""Final (test) runs with FROZEN configurations from results/tuning/selected_stage{1,2}.json.

For each dataset × split × method × seed: train with the selected (opt, method, hp, mhp) for the
selected number of epochs, evaluate once on the test domain(s), compute sharpness. Resumable.

  UCI   : main (train 1-5 -> test 6..10) and roll2..roll9 (train 1..k -> test k+1)
  Smell : loso1..loso6 (loso6 = official split)

Usage: python scripts/experiments/run_final.py --methods sgd adamw sam msam samsam_disagree recency gi \
                                   --datasets uci smell --seeds 0 1 2 3 4 [--uci-splits main]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.train import parse, run  # noqa: E402

TUNE = ROOT / "results" / "tuning"
OUT = ROOT / "results" / "final"
SPLITS = {"uci": ["main"] + [f"roll{k}" for k in range(2, 10)], "smell": [f"loso{h}" for h in range(1, 7)],
          "twin": [f"louo{h}" for h in range(1, 6)], "twinreg": [f"louo{h}" for h in range(1, 6)]}


def load_selected():
    sel = {}
    for st in (1, 2, 3, 4, 5, "twin"):
        f = TUNE / (f"selected_stage{st}.json" if st != "twin" else "selected_twin.json")
        if f.exists():
            for ds, d in json.loads(f.read_text()).items():
                sel.setdefault(ds, {}).update(d)
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--datasets", nargs="+", default=["uci", "smell"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--uci-splits", nargs="+", default=None)
    a = ap.parse_args()
    sel = load_selected()
    jobs = []
    for ds in a.datasets:
        splits = a.uci_splits if (ds == "uci" and a.uci_splits) else SPLITS[ds]
        for split in splits:
            for m in a.methods:
                for s in a.seeds:
                    jobs.append((ds, split, m, s))
    print(f"{len(jobs)} jobs", flush=True)
    for i, (ds, split, m, s) in enumerate(jobs):
        cfg = sel[ds][m]
        out = OUT / ds / split / f"{m}_s{s}.json"
        if out.exists():
            continue
        r = run(parse(["--dataset", ds, "--split", split, "--opt", cfg.get("opt", m),
                       "--method", cfg.get("method", "erm"), "--mhp", json.dumps(cfg.get("mhp", {})),
                       "--hp", json.dumps(cfg["hp"]), "--epochs", str(cfg["epochs"]),
                       "--sampler", cfg.get("sampler", "domain"), "--norm", cfg.get("norm", "train"),
                       "--norm_k", str(cfg.get("mhp", {}).get("k", 1)),
                       "--seed", str(s), "--sharpness", "--out", str(out)]))
        if i % 25 == 0:
            mm = r.get("final_eval", {}).get("mean_over_domains", {})
            print(f"[{i+1}/{len(jobs)}] {ds} {split} {m} s{s} macroF1={mm.get('macro_f1_present')}", flush=True)
    print("DONE final", flush=True)


if __name__ == "__main__":
    main()
