"""Shift diagnosis: what KIND of shift is each dataset's drift?  (added 2026-11-13)

For each test domain, the frozen AdamW base model (same as the few-shot study) is evaluated after correcting the
test-domain inputs with per-feature maps of increasing strength. All corrections are DIAGNOSTICS (oracles use the
test labels) and are never used for model selection.
  none            no correction
  standardize     label-free: z-score each feature with the test domain's own mean/sd, then map to the train
                  mean/sd (what per-domain normalization does; uses unlabeled target data)
  offset_oracle   per-feature offset fitted on ALL labeled test samples (class means -> train class means)
  affine_oracle   per-feature gain + offset fitted on ALL labeled test samples
  affine_k_first  (SmellNet) gain + offset from the first k windows per recording (= the few-shot protocol)
  affine_k_rand   (SmellNet) same, k windows drawn at random positions -> tests the onset-transient hypothesis:
                  if random windows work and first windows do not, the first windows are not representative
If affine_oracle recovers most of the gap, the drift is mostly per-feature gain/offset (calibration-type);
if not, the shift changes the feature geometry and needs more than recalibration.
Base models: seeds 0-2. Writes results/analysis/shift_diagnosis.csv and prints a summary.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.train import DEV, metrics, predict  # noqa: E402
import run_fewshot as FS  # noqa: E402

OUT = ROOT / "results" / "analysis" / "shift_diagnosis.csv"
K_SMELL = 4


def score(model, D, X, y):
    p = predict(model, torch.tensor(X, dtype=torch.float32, device=DEV))
    return metrics(y, p, D["n_cls"], D.get("y_sd", 1.0))["score"]


def standardize(D, Xe):
    Ftr, Fe = FS.feats(D["Xtr"]), FS.feats(Xe)
    a = Ftr.std(0) / (Fe.std(0) + 1e-8)
    if "gtr" in D and D["n_cls"] == 1:          # twinreg: leave the gas one-hot untouched
        a[-4:] = 1.0
    b = Ftr.mean(0) - a * Fe.mean(0)
    if "gtr" in D and D["n_cls"] == 1:
        b[-4:] = 0.0
    return Xe * a + b


def rand_windows(D, dom, k, seed):
    """SmellNet: k windows per recording at random positions; eval = the same eval set as the first-k protocol."""
    _, ev = FS.shots_and_eval(D, dom, k, seed)
    y, dv = D["yev"], D["dev"]
    rng = np.random.default_rng(seed)
    idx = np.where(dv == dom)[0]
    shots = []
    for c in np.unique(y[idx]):                 # one recording per (domain, class) in SmellNet's test sessions
        cand = idx[y[idx] == c]
        shots += list(rng.choice(cand, size=min(k, len(cand)), replace=False))
    return np.array(shots), ev


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--datasets", nargs="+", default=["uci", "smell", "twin", "twinreg"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    a = ap.parse_args()
    done = set()
    if OUT.exists():
        prev = pd.read_csv(OUT); done = set(zip(prev.dataset, prev.split, prev.seed))
    for ds in a.datasets:
        for split in FS.SPLITS[ds]:
            for seed in a.seeds:
                if (ds, split, seed) in done:
                    continue
                model, D = FS.base_model(ds, split, seed)
                rows = []
                for dom in D["eval_domains"]:
                    m = np.where(D["dev"] == dom)[0]
                    k0 = K_SMELL if D["Xev"].ndim == 3 else 1
                    _, ev = FS.shots_and_eval(D, dom, k0, seed)      # common eval set for every correction
                    Xe, ye = D["Xev"][ev], D["yev"][ev]
                    strata = D["gev"] if "gev" in D else D["yev"]
                    r = dict(dataset=ds, split=split, seed=seed, domain=int(dom), n_eval=len(ev))
                    r["none"] = score(model, D, Xe, ye)
                    r["standardize"] = score(model, D, standardize(D, Xe), ye)
                    for name, lam in (("offset_oracle", 1e6), ("affine_oracle", 1e-3)):
                        ga, gb = FS.affine_fit(D, D["Xev"][m], strata[m], lam)
                        r[name] = score(model, D, Xe * ga + gb, ye)
                    if D["Xev"].ndim == 3:
                        for name, fn in (("affine_k_first", FS.shots_and_eval), ("affine_k_rand", rand_windows)):
                            sh, _ = fn(D, dom, K_SMELL, seed)
                            ga, gb = FS.affine_fit(D, D["Xev"][sh], strata[sh], 10.0)   # frozen few-shot lam
                            r[name] = score(model, D, Xe * ga + gb, ye)
                    rows.append(r)
                COLS = ["dataset", "split", "seed", "domain", "n_eval", "none", "standardize", "offset_oracle",
                        "affine_oracle", "affine_k_first", "affine_k_rand"]      # fixed columns (SmellNet has 2 extra)
                pd.DataFrame(rows).reindex(columns=COLS).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
                print(f"{ds} {split} s{seed} done", flush=True)
    df = pd.read_csv(OUT)
    cols = [c for c in ["none", "standardize", "offset_oracle", "affine_oracle", "affine_k_first", "affine_k_rand"]
            if c in df]
    per_split = df.groupby(["dataset", "split"])[cols].mean()
    pd.set_option("display.width", 200)
    print("\nmean over splits (score = macro-F1; R2 for twinreg):")
    print(per_split.groupby("dataset").mean().round(3).to_string())
    print("\nsplits where the correction beats none:")
    print(per_split.sub(per_split["none"], axis=0).gt(0).groupby("dataset").sum().to_string())
    print("DONE shift", flush=True)


if __name__ == "__main__":
    main()
