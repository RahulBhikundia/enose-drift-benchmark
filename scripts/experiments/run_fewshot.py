"""Phase B: few-shot recalibration. How much of the drift can a few LABELED samples from the new
domain (batch / day) undo?  This is a different setting from domain generalization (it uses labeled
target data) and is reported separately.

Base model: AdamW-ERM with its frozen selected config (results/tuning/selected_stage1.json), trained
exactly like the final runs. For each test domain and each k:
  shots  UCI   : k random labeled samples per class from that test batch (seeded)
         Smell : the FIRST k windows of each recording ("calibrate with the first k*50+50 s");
                 the next window (overlapping the shots) is skipped, evaluation starts at window k+1
  eval   all remaining samples of that domain; every adaptation method is scored on the SAME samples
Adaptations (each fitted per test domain):
  none      the base model as is
  affine    per-feature gain/offset mapping the domain's shot class means onto the training class
            means (ridge towards gain 1, strength lam; lam=1e6 ~ offset only). SmellNet: per channel,
            from window means. Also a diagnostic: how much of the shift is per-feature affine?
  ft_last   fine-tune only the last layer on the shots (full batch, AdamW)
  ft_all    fine-tune all weights on the shots
Adaptation hyperparameters are chosen on the VALIDATION split ("tune", k = 5 UCI / 2 SmellNet,
seeds 0-2) and frozen:  python scripts/experiments/run_fewshot.py --mode tune ; then --mode final.
"""
import argparse
import copy
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.train import DEV, metrics, parse, predict, run  # noqa: E402

OUT = ROOT / "results" / "fewshot"
# Twin (added 2026-11-13): shots = k random recordings PER GAS from the held-out unit (twinreg: concentrations of
# those recordings are random; the gas is known). Units 4-5 have 20 recordings per gas, units 1-3 have 40.
KS = {"uci": [1, 5, 10], "smell": [1, 2, 4], "twin": [1, 2, 5], "twinreg": [1, 2, 5]}
K_TUNE = {"uci": 5, "smell": 2, "twin": 2, "twinreg": 2}
SPLITS = {"uci": ["main"] + [f"roll{k}" for k in range(2, 10)], "smell": [f"loso{h}" for h in range(1, 7)],
          "twin": [f"louo{h}" for h in range(1, 6)], "twinreg": [f"louo{h}" for h in range(1, 6)]}
N_SEEDS = {"uci": 5, "smell": 5, "twin": 10, "twinreg": 10}
GRID = {"none": [{}],
        "affine": [dict(lam=l) for l in (0.1, 1.0, 10.0, 1e6)],
        "ft_last": [dict(lr=lr, steps=s) for lr, s in itertools.product([1e-4, 1e-3, 1e-2], [50, 200])],
        "ft_all": [dict(lr=lr, steps=s) for lr, s in itertools.product([1e-4, 1e-3, 1e-2], [50, 200])]}


def shots_and_eval(D, dom, k, seed):
    """Indices (into the eval arrays) of the shots and of the evaluation samples for one test domain."""
    y, dv = D.get("gev", D["yev"]), D["dev"]                 # strata: class (UCI, twin) / gas (twinreg)
    shots, ev = [], []
    if D["Xev"].ndim == 2:                                   # UCI / Twin: random k per class (per gas)
        rng = np.random.default_rng(seed * 1000 + k)
        for c in np.unique(y[dv == dom]):
            idx = rng.permutation(np.where((dv == dom) & (y == c))[0])
            n = min(k, len(idx) - 1)
            shots += list(idx[:n]); ev += list(idx[n:])
    else:                                                    # SmellNet: first k windows per recording
        pos, prev = 0, None
        for i in np.where(dv == dom)[0]:                     # windows are stored in time order per recording
            key = (dv[i], y[i])
            pos = pos + 1 if key == prev else 0; prev = key
            if pos < k:
                shots.append(i)
            elif pos >= k + 1:
                ev.append(i)
    return np.array(shots), np.array(ev)


def feats(X):
    return X.mean(1) if X.ndim == 3 else X


def affine_fit(D, Xs, ys, lam):
    """Per-feature x_src ~ a*x_tgt + b over shot classes; ridge lam*(a-1)^2. Returns (a, b)."""
    Ftr, Fs = feats(D["Xtr"]), feats(Xs)
    cls = np.unique(ys)                                           # ys = strata labels (gas for twinreg)
    s = np.stack([Ftr[D.get("gtr", D["ytr"]) == c].mean(0) for c in cls])      # (C, d) source class means
    t = np.stack([Fs[ys == c].mean(0) for c in cls])              # (C, d) target shot class means
    C = len(cls)
    A11 = (t ** 2).sum(0) + lam; A12 = t.sum(0); A22 = C
    r1 = (t * s).sum(0) + lam; r2 = s.sum(0)
    det = A11 * A22 - A12 ** 2
    a = (r1 * A22 - A12 * r2) / det; b = (A11 * r2 - A12 * r1) / det
    return a, b


def head_params(model):
    return list((model.net[-1] if hasattr(model, "net") else model.head).parameters())


def finetune(model, Xs, ys, lr, steps, last_only):
    m = copy.deepcopy(model); m.train()
    for p in m.parameters():
        p.requires_grad_(not last_only)
    params = head_params(m) if last_only else list(m.parameters())
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    reg = ys.dtype.kind == "f"
    X = torch.tensor(Xs, dtype=torch.float32, device=DEV)
    y = torch.tensor(ys, dtype=torch.float32 if reg else torch.long, device=DEV)
    if reg:
        _mse = torch.nn.MSELoss()
        lf = lambda o, t: _mse(o.squeeze(-1), t)                 # noqa: E731
    else:
        lf = torch.nn.CrossEntropyLoss()
    for _ in range(steps):
        opt.zero_grad(); lf(m(X), y).backward(); opt.step()
    return m


def evaluate(model, D, method, hp, k, seed):
    """Mean over test domains of the task score (macro-F1; R2 for twinreg) on the eval samples, adapting per domain."""
    scores = []
    for dom in D["eval_domains"]:
        sh, ev = shots_and_eval(D, dom, k, seed)
        if len(ev) == 0 or len(sh) == 0:
            continue
        Xs, ys, Xe, ye = D["Xev"][sh], D["yev"][sh], D["Xev"][ev], D["yev"][ev]
        if method == "none":
            m, Xuse = model, Xe
        elif method == "affine":
            gs = D["gev"][sh] if "gev" in D else ys
            a, b = affine_fit(D, Xs, gs, hp["lam"]); m, Xuse = model, Xe * a + b
        else:
            torch.manual_seed(seed)
            m, Xuse = finetune(model, Xs, ys, hp["lr"], hp["steps"], method == "ft_last"), Xe
        p = predict(m, torch.tensor(Xuse, dtype=torch.float32, device=DEV))
        scores.append(metrics(ye, p, D["n_cls"], D.get("y_sd", 1.0))["score"])
    return float(np.mean(scores)), len(scores)


def base_model(ds, split, seed):
    f = "selected_twin.json" if ds in ("twin", "twinreg") else "selected_stage1.json"
    sel = json.loads((ROOT / "results" / "tuning" / f).read_text())[ds]["adamw"]
    out = OUT / "base" / ds / f"{split}_s{seed}.json"
    r, model, D = run(parse(["--dataset", ds, "--split", split, "--opt", "adamw", "--hp", json.dumps(sel["hp"]),
                             "--epochs", str(sel["epochs"]), "--seed", str(seed), "--out", str(out)]), return_model=True)
    return model, D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["tune", "final"], required=True)
    ap.add_argument("--datasets", nargs="+", default=["uci", "smell"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.mode == "tune":
        rows = []
        for ds in a.datasets:
            for seed in (0, 1, 2):
                model, D = base_model(ds, "tune", seed)
                for method, grid in GRID.items():
                    for hp in grid:
                        f1, _ = evaluate(model, D, method, hp, K_TUNE[ds], seed)
                        rows.append(dict(dataset=ds, method=method, hp=json.dumps(hp), seed=seed, f1=f1))
                print(f"tune {ds} seed {seed} done", flush=True)
        df = pd.DataFrame(rows)
        tf = OUT / "tune_all.csv"                              # append: other datasets' tuning rows are kept
        if tf.exists():
            old = pd.read_csv(tf); df = pd.concat([old[~old.dataset.isin(a.datasets)], df])
        df.to_csv(tf, index=False)
        best = (df.groupby(["dataset", "method", "hp"]).f1.mean().reset_index()
                  .sort_values("f1", ascending=False).groupby(["dataset", "method"]).head(1))
        sf = OUT / "selected.json"
        sel = json.loads(sf.read_text()) if sf.exists() else {}
        sel.update({ds: {r.method: dict(hp=json.loads(r.hp), val_f1=r.f1) for r in best[best.dataset == ds].itertuples()}
                    for ds in a.datasets})
        (OUT / "selected.json").write_text(json.dumps(sel, indent=2))
        print(best.to_string(index=False))
        return
    sel = json.loads((OUT / "selected.json").read_text())
    for ds in a.datasets:
        fout = OUT / f"final_{ds}.csv"
        done = set()
        if fout.exists():
            prev = pd.read_csv(fout); done = set(zip(prev.split, prev.seed))
        for split in SPLITS[ds]:
            for seed in range(N_SEEDS[ds]):
                if (split, seed) in done:
                    continue
                model, D = base_model(ds, split, seed)
                rows = []
                for k in KS[ds]:
                    for method in GRID:
                        hp = sel[ds][method]["hp"]
                        f1, nd = evaluate(model, D, method, hp, k, seed)
                        rows.append(dict(dataset=ds, split=split, seed=seed, k=k, method=method, hp=json.dumps(hp),
                                         f1=f1, n_domains=nd))
                pd.DataFrame(rows).to_csv(fout, mode="a", header=not fout.exists(), index=False)
                print(f"final {ds} {split} s{seed} done", flush=True)
    print("DONE fewshot", flush=True)


if __name__ == "__main__":
    main()
