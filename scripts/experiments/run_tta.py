"""Diagnose-then-adapt (added 2026-11-13; design pre-registered in notes/decisions.md).

For each target domain (test batch / session / unit), starting from the frozen AdamW base model (as in the
few-shot study), score:
  label-free test-time adaptation (uses the UNLABELED eval inputs of that domain only)
    none          no adaptation
    standardize   per-feature input z-scoring with the target's own statistics, mapped to the training statistics
    feat_align    same at the penultimate layer (target feature mean/sd -> source feature mean/sd)
    ent_affine    learnable per-feature input gain + offset (init identity), trained with the information-
                  maximization loss  mean_i H(p_i) - H(mean_i p_i); model frozen          [classification only]
    shot_im       SHOT-style: adapt the feature extractor with the same loss, head frozen   [classification only]
  labelled reference
    fewshot       fine-tune all layers on k labelled shots (frozen few-shot hp; k = 5 UCI / 4 SmellNet / 2 Twin)
  label-free diagnostic (no class labels)
    d_raw         balanced acc of a source-vs-target random-forest domain classifier (5-fold CV, class-balanced)
    d_std         the same after standardizing each side per feature (residual, non-calibration shift)
All methods are scored on the same eval samples (the few-shot protocol's eval set).
Modes: --mode tune  (validation splits: tune TTA hp, then the policy: label-free method + threshold tau)
       --mode final (all test splits, seeds 0-2; resumable)  then  --mode report
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.data_smellnet import window_features  # noqa: E402
from src.train import DEV, metrics  # noqa: E402
import run_fewshot as FS  # noqa: E402

OUT = ROOT / "results" / "tta"; OUT.mkdir(parents=True, exist_ok=True)
DATASETS = ["uci", "smell", "twin", "twinreg"]
K_REF = {"uci": 5, "smell": 4, "twin": 2, "twinreg": 2}
SEEDS = [0, 1, 2]
IM_GRID = [dict(lr=lr, steps=s) for lr, s in itertools.product([1e-3, 1e-2, 1e-1], [50, 200])]
SHOT_GRID = [dict(lr=lr, steps=s) for lr, s in itertools.product([1e-4, 1e-3, 1e-2], [50, 200])]
LABEL_FREE = ["standardize", "feat_align", "ent_affine", "shot_im"]


def is_reg(D):
    return D["n_cls"] == 1


def T(x):
    return torch.tensor(x, dtype=torch.float32, device=DEV)


@torch.no_grad()
def out_of(model, X, fwd=None, bs=4096):
    model.eval()
    f = fwd or model
    return torch.cat([f(X[i:i + bs]) for i in range(0, len(X), bs)])


def score_from_out(D, o, y):
    p = (o[:, 0] if o.shape[1] == 1 else o.argmax(1)).cpu().numpy()
    return metrics(y, p, D["n_cls"], D.get("y_sd", 1.0))["score"]


def n_onehot(D):
    return 4 if (is_reg(D) and "gtr" in D) else 0      # twinreg: last 4 inputs are the known-gas one-hot


def standardize(D, Xe):
    Ftr, Fe = FS.feats(D["Xtr"]), FS.feats(Xe)
    a = Ftr.std(0) / (Fe.std(0) + 1e-8); b = Ftr.mean(0) - a * Fe.mean(0)
    k = n_onehot(D)
    if k:
        a[-k:] = 1.0; b[-k:] = 0.0
    return Xe * a + b


def feat_align(model, D, Xe):
    idx = np.random.default_rng(0).permutation(len(D["Xtr"]))[:5000]     # random subsample of ALL train domains
    Htr = out_of(model, T(D["Xtr"][idx]), model.features)
    He = out_of(model, T(Xe), model.features)
    mu_s, sd_s, mu_t, sd_t = Htr.mean(0), Htr.std(0), He.mean(0), He.std(0)
    ok = (sd_t > 1e-6) & (sd_s > 1e-6)
    H2 = torch.where(ok, (He - mu_t) / (sd_t + 1e-8) * sd_s + mu_s, He)
    with torch.no_grad():
        return model.head(H2)


def im_loss(logits):
    p = logits.softmax(1)
    ent = -(p * torch.log(p + 1e-8)).sum(1).mean()
    pm = p.mean(0)
    return ent + (pm * torch.log(pm + 1e-8)).sum()          # ent - H(mean p)


def ent_affine(model, Xe, hp, seed):
    torch.manual_seed(seed)
    X = T(Xe); d = X.shape[-1]
    g = torch.ones(d, device=DEV, requires_grad=True); b = torch.zeros(d, device=DEV, requires_grad=True)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    opt = torch.optim.Adam([g, b], lr=hp["lr"])
    for _ in range(hp["steps"]):
        opt.zero_grad(); im_loss(model(X * g + b)).backward(); opt.step()
    for p in model.parameters():
        p.requires_grad_(True)
    with torch.no_grad():
        return out_of(model, X * g + b)


def shot_im(model, Xe, hp, seed):
    torch.manual_seed(seed)
    m = copy.deepcopy(model); m.train()
    head = list((m.net[-1] if hasattr(m, "net") else m.head).parameters())
    hid = {id(p) for p in head}
    for p in head:
        p.requires_grad_(False)
    opt = torch.optim.Adam([p for p in m.parameters() if id(p) not in hid], lr=hp["lr"])
    X = T(Xe)
    for _ in range(hp["steps"]):
        opt.zero_grad(); im_loss(m(X)).backward(); opt.step()
    return out_of(m, X)


def diag_feats(D, X):
    if X.ndim == 3:
        return window_features(X)
    k = n_onehot(D)
    return X[:, :-k] if k else X


def domain_sep(Fs, Ft, seed, per_side_std):
    rng = np.random.default_rng(seed)
    Fs = Fs[rng.permutation(len(Fs))[:3000]]; Ft = Ft[rng.permutation(len(Ft))[:3000]]
    if per_side_std:
        Fs = (Fs - Fs.mean(0)) / (Fs.std(0) + 1e-8); Ft = (Ft - Ft.mean(0)) / (Ft.std(0) + 1e-8)
    else:
        mu, sd = Fs.mean(0), Fs.std(0) + 1e-8
        Fs, Ft = (Fs - mu) / sd, (Ft - mu) / sd
    X = np.vstack([Fs, Ft]); y = np.r_[np.zeros(len(Fs)), np.ones(len(Ft))]
    # nonlinear + class-balanced: after per-side standardization only higher-order differences remain, which a
    # linear classifier cannot see; source and target sizes differ, so balance the classes
    clf = RandomForestClassifier(200, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=seed)
    p = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed))
    return balanced_accuracy_score(y, p)


def evaluate_domain(model, D, ds, dom, seed, grids, fs_hp):
    sh, ev = FS.shots_and_eval(D, dom, K_REF[ds], seed)
    Xe, ye = D["Xev"][ev], D["yev"][ev]
    r = dict(domain=int(dom), n_eval=len(ev))
    r["none"] = score_from_out(D, out_of(model, T(Xe)), ye)
    r["standardize"] = score_from_out(D, out_of(model, T(standardize(D, Xe))), ye)
    r["feat_align"] = score_from_out(D, feat_align(model, D, Xe), ye)
    if not is_reg(D):
        for name, fn in (("ent_affine", ent_affine), ("shot_im", shot_im)):
            for hp in grids[name]:
                r[f"{name}|{json.dumps(hp, sort_keys=True)}"] = score_from_out(D, fn(model, Xe, hp, seed), ye)
    torch.manual_seed(seed)
    m = FS.finetune(model, D["Xev"][sh], D["yev"][sh], fs_hp["lr"], fs_hp["steps"], last_only=False)
    r["fewshot"] = score_from_out(D, out_of(m, T(Xe)), ye)
    # label-free criteria inputs, on ALL samples of the domain (no labels): outputs of none / standardize /
    # feat_align (saved for the cross-seed agreement criterion) and the IM objective of each
    Xa = D["Xev"][D["dev"] == dom]
    outs = {"none": out_of(model, T(Xa)), "standardize": out_of(model, T(standardize(D, Xa))),
            "feat_align": feat_align(model, D, Xa)}
    pred = {}
    for k, o in outs.items():
        if is_reg(D):
            pred[k] = o[:, 0].cpu().numpy()
        else:
            pred[k] = o.argmax(1).cpu().numpy()
            with torch.no_grad():
                r[f"im_{k}"] = float(im_loss(o))
    r["_pred"] = pred
    return r


def zstd(F):
    return (F - F.mean(0)) / (F.std(0) + 1e-8)


def sep(A, B, seed):
    """Balanced acc of a class-balanced RF telling A from B (5-fold CV); <= 3000 samples per side."""
    rng = np.random.default_rng(seed)
    A = A[rng.permutation(len(A))[:3000]]; B = B[rng.permutation(len(B))[:3000]]
    X = np.vstack([A, B]); y = np.r_[np.zeros(len(A)), np.ones(len(B))]
    clf = RandomForestClassifier(200, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=seed)
    return balanced_accuracy_score(y, cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True,
                                                                                         random_state=seed)))


def diagnose(ds, split, seed=0):
    """Label-free 'excess shift' after per-feature calibration, per target domain:
      every domain (each source domain and the target) is standardized per feature with ITS OWN statistics
      (removing any per-feature gain/offset), then
      d_target = separability of target vs the pooled source domains,
      d_ref    = mean over source domains s of the separability of s vs the other source domains,
      excess   = d_target - d_ref.
    excess <= 0: after calibration the new domain is no more different than training domains are from each other
    (calibration-type shift); excess > 0: residual, non-calibration shift. Also d_raw (no standardization)."""
    from src.train import load_data
    D = load_data(ds, split)
    Ftr, Fev = diag_feats(D, D["Xtr"]), diag_feats(D, D["Xev"])
    src = {k: zstd(Ftr[D["dtr"] == k]) for k in np.unique(D["dtr"])}
    refs = [sep(src[k], np.vstack([v for j, v in src.items() if j != k]), seed) for k in src] if len(src) > 1 else [np.nan]
    pooled_raw, pooled_std = Ftr, np.vstack(list(src.values()))
    rows = []
    for dom in D["eval_domains"]:
        Ft = Fev[D["dev"] == dom]
        mu, sd = pooled_raw.mean(0), pooled_raw.std(0) + 1e-8
        d_raw = sep((pooled_raw - mu) / sd, (Ft - mu) / sd, seed)
        d_t = sep(pooled_std, zstd(Ft), seed)
        rows.append(dict(dataset=ds, split=split, domain=int(dom), d_raw=d_raw, d_target=d_t,
                         d_ref=float(np.mean(refs)), excess=d_t - float(np.mean(refs))))
    return rows


def run_diag(which):
    fout = OUT / f"diag_{which}.csv"
    done = set()
    if fout.exists():
        prev = pd.read_csv(fout); done = set(zip(prev.dataset, prev.split))
    splits = {ds: ["tune"] for ds in DATASETS} if which == "tune" else FS.SPLITS
    for ds in DATASETS:
        for split in splits[ds]:
            if (ds, split) in done:
                continue
            new = pd.DataFrame(diagnose(ds, split))
            if fout.exists():
                new = pd.concat([pd.read_csv(fout), new], ignore_index=True)
            new.to_csv(fout, index=False)
            print(f"diag {ds} {split} done", flush=True)


def with_diag(d, which):
    g = pd.read_csv(OUT / f"diag_{which}.csv")
    return d.drop(columns=[c for c in ("d_raw", "d_std") if c in d]).merge(g, on=["dataset", "split", "domain"])


def run_mode(mode, seeds=SEEDS):
    fs_sel = json.loads((ROOT / "results" / "fewshot" / "selected.json").read_text())
    if mode == "tune":
        grids, splits = {"ent_affine": IM_GRID, "shot_im": SHOT_GRID}, {ds: ["tune"] for ds in DATASETS}
        fout = OUT / "tune.csv"
    else:
        sel = json.loads((OUT / "selected.json").read_text())
        splits, fout = FS.SPLITS, OUT / "final.csv"
    done = set()
    if fout.exists():
        prev = pd.read_csv(fout); done = set(zip(prev.dataset, prev.split, prev.seed))
    for ds in DATASETS:
        if mode == "final":
            grids = {n: [hp] for n, hp in sel["hp"].get(ds, {}).items()}
        for split in splits[ds]:
            for seed in seeds:
                if (ds, split, seed) in done:
                    continue
                model, D = FS.base_model(ds, split, seed)
                rows = [dict(dataset=ds, split=split, seed=seed,
                             **evaluate_domain(model, D, ds, dom, seed, grids, fs_sel[ds]["ft_all"]["hp"]))
                        for dom in D["eval_domains"]]
                pdir = OUT / "preds" / mode; pdir.mkdir(parents=True, exist_ok=True)
                for r in rows:
                    np.savez(pdir / f"{ds}_{split}_d{r['domain']}_s{seed}.npz", **r.pop("_pred"))
                new = pd.DataFrame(rows)
                if fout.exists():
                    new = pd.concat([pd.read_csv(fout), new], ignore_index=True)
                new.to_csv(fout, index=False)
                print(f"{mode} {ds} {split} s{seed} done", flush=True)
    print(f"DONE tta {mode}", flush=True)


def select():
    """Validation-only: hyperparameters of ent_affine / shot_im per dataset (best mean score on the tune split)."""
    d = pd.read_csv(OUT / "tune.csv")
    sel = {"hp": {}}
    for ds, g in d.groupby("dataset"):
        sel["hp"][ds] = {}
        for name in ("ent_affine", "shot_im"):
            cols = [c for c in g.columns if c.startswith(name + "|") and g[c].notna().any()]
            if cols:
                sel["hp"][ds][name] = json.loads(max(cols, key=lambda c: g[c].mean()).split("|", 1)[1])
    (OUT / "selected.json").write_text(json.dumps(sel, indent=2))
    print(json.dumps(sel, indent=2))


def agreement(preds, reg):
    """Mean pairwise agreement of the seed models' predictions (regression: Pearson r)."""
    v = []
    for a, b in itertools.combinations(preds, 2):
        v.append(float(np.corrcoef(a, b)[0, 1]) if reg else float((a == b).mean()))
    return float(np.mean(v)) if v else np.nan


def criteria(mode):
    """Per target domain: label-free criteria for 'apply standardize?' and the resulting scores."""
    d = pd.read_csv(OUT / f"{mode}.csv")
    for name in ("ent_affine", "shot_im"):
        cols = [c for c in d.columns if c.startswith(name + "|")]
        if cols:
            d[name] = d[cols].bfill(axis=1).iloc[:, 0] if mode == "final" else d[cols].max(axis=1)
    rows = []
    for (ds, split, dom), g in d.groupby(["dataset", "split", "domain"]):
        reg = ds == "twinreg"
        P = [np.load(OUT / "preds" / mode / f"{ds}_{split}_d{dom}_s{s}.npz") for s in g.seed]
        r = dict(dataset=ds, split=split, domain=dom, n_seeds=len(g))
        for m in ("none", "standardize", "feat_align", "ent_affine", "shot_im", "fewshot"):
            if m in g and g[m].notna().any():
                r[m] = g[m].mean()
        r["agree_none"] = agreement([p["none"] for p in P], reg)
        r["agree_std"] = agreement([p["standardize"] for p in P], reg)
        r["crit_agree"] = r["agree_std"] > r["agree_none"]
        if not reg:
            r["d_im"] = float((g.im_standardize - g.im_none).mean())
            r["crit_im"] = r["d_im"] < 0
        rows.append(r)
    C = pd.DataFrame(rows)
    C["oracle"] = C[["none", "standardize"]].max(axis=1)
    C["by_agree"] = np.where(C.crit_agree, C.standardize, C.none)
    if "crit_im" in C:
        C["by_im"] = np.where(C.crit_im.fillna(False).astype(bool), C.standardize, C.none)
    C["std_helps"] = C.standardize > C.none
    C.to_csv(OUT / f"criteria_{mode}.csv", index=False)
    return C


def report(mode="final"):
    C = criteria(mode)
    pd.set_option("display.width", 220)
    cols = [c for c in ["none", "standardize", "feat_align", "ent_affine", "shot_im", "fewshot", "by_agree", "by_im",
                        "oracle"] if c in C]
    S = C.groupby(["dataset", "split"])[cols].mean()
    print(f"[{mode}] mean over splits of per-domain scores (seed-mean); by_* = apply standardize iff criterion says so")
    print(S.groupby("dataset").mean().round(3).to_string())
    print("\nsplits where method beats none:")
    print(S.sub(S["none"], axis=0).gt(0).groupby("dataset").sum().to_string())
    print("\ncriterion decision accuracy vs 'does standardize actually help' (per target domain):")
    for c in ("crit_agree", "crit_im"):
        if c in C:
            x = C.dropna(subset=[c])
            acc = x.groupby("dataset").apply(lambda g: (g[c].astype(bool) == g.std_helps).mean())
            print(f"  {c}: overall {(x[c].astype(bool) == x.std_helps).mean():.2f}  " +
                  "  ".join(f"{k} {v:.2f} (n={int((x.dataset == k).sum())})" for k, v in acc.items()))
    print("\nper domain:")
    print(C.round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)   # final: 0-4 added 2026-11-14 (resumes)
    ap.add_argument("--mode", choices=["tune", "final", "report", "report_tune", "diag_tune", "diag_final", "select"], required=True)
    a = ap.parse_args()
    if a.mode.startswith("diag_"):
        run_diag(a.mode[5:])
    elif a.mode == "select":
        select()
    elif a.mode == "report":
        report("final")
    elif a.mode == "report_tune":
        report("tune")
    else:
        run_mode(a.mode, a.seeds)


if __name__ == "__main__":
    main()
