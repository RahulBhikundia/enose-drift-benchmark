"""Features vs model on SmellNet (added 2026-11-13).

Random forest on 30 window summary features beat the CNN on raw windows on 6/6 SmellNet splits. Is that the
features or the model? Here an MLP is trained on the SAME 30 features ("smellfeat"), same protocol as everything
else: AdamW and CORAL, 12 configs x 3 seeds on the tuning split, select by mean best-epoch validation score,
epochs = median best epoch, then frozen runs on loso1-6 x 5 seeds.
Writes results/tuning/smellfeat/<m>/, results/tuning/selected_smellfeat.json, results/final/smellfeat/<m>/.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.train import parse, run  # noqa: E402
import run_grid as G  # noqa: E402

DS = sys.argv[1] if len(sys.argv) > 1 else "smellfeat"; EPOCHS = 100   # also "smellgru" (2026-11-14)
TUNE = ROOT / "results" / "tuning"; FINAL = ROOT / "results" / "final" / DS


def configs(m):
    return [("adamw", "erm", hp, {}, []) for hp in G.GRIDS["adamw"]] if m == "adamw" else G.STAGE3[m]


def main():
    sel = {}
    for m in ("adamw", "coral"):
        groups = defaultdict(list)
        for opt, method, hp, mhp, extra in configs(m):
            for s in G.SEEDS:
                out = TUNE / DS / m / f"{G.hp_tag({**hp, **mhp})}_s{s}.json"
                if out.exists():
                    r = json.loads(out.read_text())
                else:
                    r = run(parse(["--dataset", DS, "--split", "tune", "--opt", opt, "--hp", json.dumps(hp), "--method",
                                   method, "--mhp", json.dumps(mhp), "--seed", str(s), "--epochs", str(EPOCHS),
                                   "--out", str(out)] + extra))
                bv = r.get("best_val") or {}
                groups[json.dumps(dict(opt=opt, method=method, hp=hp, mhp=mhp, extra=extra), sort_keys=True)].append(
                    (-1e9 if r["diverged"] else bv.get("score", -1e9), bv.get("epoch", EPOCHS)))
        k, v = max(groups.items(), key=lambda kv: np.mean([x[0] for x in kv[1]]))
        sel[m] = dict(json.loads(k), epochs=max(5, int(np.median([x[1] for x in v]))),
                      val_score_mean=float(np.mean([x[0] for x in v])))
        print(m, sel[m], flush=True)
    (TUNE / f"selected_{DS}.json").write_text(json.dumps(sel, indent=2))
    for m, c in sel.items():
        for split in [f"loso{h}" for h in range(1, 7)]:
            for s in range(5):
                out = FINAL / m / f"{split}_s{s}.json"
                if out.exists():
                    continue
                run(parse(["--dataset", DS, "--split", split, "--opt", c["opt"], "--hp", json.dumps(c["hp"]),
                           "--method", c["method"], "--mhp", json.dumps(c["mhp"]), "--seed", str(s),
                           "--epochs", str(c["epochs"]), "--out", str(out)] + c["extra"]))
            print(f"final {m} {split} done", flush=True)
    print(f"DONE {DS}", flush=True)


if __name__ == "__main__":
    main()
