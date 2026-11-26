"""One training run = (dataset, split, optimizer, hyperparameters, seed). Writes one JSON file.

Splits
  UCI   tune      : train batches 1-3, validate on 4-5 (forward-chaining validation)
        main      : train 1-5, test 6, 7, 8, 9, 10 (each reported) -- the main result
        roll<k>   : train 1..k, test k+1 (k = 2..9)            -- rolling extra analysis
  Smell tune      : train sessions 1-4, validate on 5
        loso<h>   : train all sessions except h, test h (h = 1..6); loso6 = official split
Domains = UCI batches / SmellNet sessions (days). All optimizers use the same domain-homogeneous
minibatches (each batch comes from one domain), so SA-MSAM's per-domain EMAs cost nothing.

Usage: python -m src.train --dataset uci --split main --opt msam --hp '{"lr":0.03,"rho":0.3}' \
           --seed 0 --epochs 100 --out results/runs/x.json
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score

from src.data_smellnet import load_sessions, make_windows, substance_index
from src.data_twin import feature_columns as twin_columns, features as twin_features
from src.data_uci import feature_columns, load_uci
from src.models import build_model
from src.optim import SAM, SAMSAM, build_optimizer
from src.sharpness import all_sharpness

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SM_WIN, SM_STRIDE = 100, 50


# ---------------------------------------------------------------- data
def split_domains(dataset, split):
    """Return (train_domains, eval_domains, eval_role) where eval_role is 'val' or 'test'."""
    if dataset == "uci":
        if split == "tune": return [1, 2, 3], [4, 5], "val"
        if split == "tune2": return [1, 2], [3], "val"          # 2nd (earlier) forward fold, selection study
        if split == "main": return [1, 2, 3, 4, 5], [6, 7, 8, 9, 10], "test"
        if split.startswith("roll"):
            k = int(split[4:]); return list(range(1, k + 1)), [k + 1], "test"
    elif dataset in ("twin", "twinreg"):
        # Twin units have no temporal order (measured on interleaved days). Tuning: train units 1-3,
        # validate on unit 4 (unit 5 unused). Test: leave one unit out.
        if split == "tune": return [1, 2, 3], [4], "val"
        if split.startswith("louo"):
            h = int(split[4:]); return [u for u in range(1, 6) if u != h], [h], "test"
    else:
        if split == "tune": return [1, 2, 3, 4], [5], "val"
        if split == "tune2": return [1, 2, 3], [4], "val"       # 2nd (earlier) forward fold, selection study
        if split.startswith("loso"):
            h = int(split[4:]); return [s for s in range(1, 7) if s != h], [h], "test"
    raise ValueError(split)


def domain_standardize(X, dom):
    """Per-domain normalization: standardize each domain (batch/session) with its OWN feature
    statistics. For evaluation domains this uses their unlabeled inputs (transductive, like per-batch
    sensor recalibration); labels are never used. Applied on top of the usual train-scaler output."""
    X = X.copy()
    for k in np.unique(dom):
        m = dom == k
        flat = X[m].reshape(-1, X.shape[-1])
        X[m] = (X[m] - flat.mean(0)) / (flat.std(0) + 1e-8)
    return X


def ccpca_directions(X, y, dom, k):
    """Class-conditional drift directions (a generalisation of CC-PCA, Artursson et al. 2000), estimated
    from TRAINING data only: per-(domain, class) mean minus that class's overall mean leaves the
    domain-specific (drift) part; its top-k principal directions are returned (d x k).
    SmellNet windows (N, T, C) are reduced to per-window channel means first (directions live in C)."""
    F = X.mean(1) if X.ndim == 3 else X
    res = []
    for c in np.unique(y):
        mc = F[y == c].mean(0)
        for d in np.unique(dom[y == c]):
            res.append(F[(y == c) & (dom == d)].mean(0) - mc)
    _, _, vt = np.linalg.svd(np.array(res), full_matrices=False)
    return vt[:k].T


def project_out(X, P):
    return X - (X @ P) @ P.T          # works for (N, d) and (N, T, C) with P of shape (C, k)


def load_data(dataset, split, norm="train", norm_k=1):
    D = _load_data(dataset, split)
    if norm == "domain":
        D["Xtr"] = domain_standardize(D["Xtr"], D["dtr"]); D["Xev"] = domain_standardize(D["Xev"], D["dev"])
    if norm == "ccpca":
        P = ccpca_directions(D["Xtr"], D.get("gtr", D["ytr"]), D["dtr"], norm_k)   # twinreg: classes = gases
        D["Xtr"] = project_out(D["Xtr"], P); D["Xev"] = project_out(D["Xev"], P)
    return D


def _load_data(dataset, split):
    if dataset == "smellgru":                     # same raw SmellNet windows as "smell"; GRU instead of CNN
        return _load_data("smell", split)
    if dataset == "smellfeat":
        # SmellNet, same windows and splits, but each window summarized by the 30 window_features used by the
        # random-forest baseline (mean/std/min/max/slope per channel), train-only standardization -> MLP.
        # Separates "features vs model" in the RF-vs-CNN comparison (added 2026-11-13).
        from src.data_smellnet import window_features
        D = _load_data("smell", split)
        Ftr, Fev = window_features(D["Xtr"]), window_features(D["Xev"])
        mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-8
        return dict(D, Xtr=(Ftr - mu) / sd, Xev=(Fev - mu) / sd)
    tr_d, ev_d, role = split_domains(dataset, split)
    if dataset == "uci":
        df = load_uci(); F = feature_columns()
        tr = df[df.batch.isin(tr_d)]; ev = df[df.batch.isin(ev_d)]
        mu, sd = tr[F].mean().to_numpy(), tr[F].std().to_numpy() + 1e-8      # train-only scaler
        Xtr = (tr[F].to_numpy() - mu) / sd; Xev = (ev[F].to_numpy() - mu) / sd
        return dict(Xtr=Xtr, ytr=tr.label.to_numpy(), dtr=tr.batch.to_numpy(),
                    Xev=Xev, yev=ev.label.to_numpy(), dev=ev.batch.to_numpy(),
                    train_domains=tr_d, eval_domains=ev_d, role=role, n_cls=6)
    if dataset in ("twin", "twinreg"):
        df = twin_features(); F = twin_columns(df)
        tr = df[df.unit.isin(tr_d)]; ev = df[df.unit.isin(ev_d)]
        mu, sd = tr[F].mean().to_numpy(), tr[F].std().to_numpy() + 1e-8      # train-only scaler
        Xtr = (tr[F].to_numpy() - mu) / sd; Xev = (ev[F].to_numpy() - mu) / sd
        D = dict(dtr=tr.unit.to_numpy(), dev=ev.unit.to_numpy(), train_domains=tr_d, eval_domains=ev_d, role=role,
                 gtr=tr.gas.to_numpy(), gev=ev.gas.to_numpy())
        if dataset == "twin":                     # classify the gas (4 classes)
            return dict(D, Xtr=Xtr, Xev=Xev, ytr=tr.gas.to_numpy(), yev=ev.gas.to_numpy(), n_cls=4)
        # twinreg: regress log-concentration of a KNOWN gas (gas one-hot appended to the inputs);
        # target standardized with training statistics; RMSE/MAE are reported in log-concentration units
        oh = np.eye(4)
        lt, le = np.log(tr.conc.to_numpy(float)), np.log(ev.conc.to_numpy(float))
        y_mu, y_sd = lt.mean(), lt.std()
        return dict(D, Xtr=np.hstack([Xtr, oh[tr.gas]]), Xev=np.hstack([Xev, oh[ev.gas]]),
                    ytr=(lt - y_mu) / y_sd, yev=(le - y_mu) / y_sd, n_cls=1, task="reg", y_mu=y_mu, y_sd=y_sd)
    S = load_sessions(); cls = substance_index(S)

    def build(sessions):
        X, y, d = [], [], []
        for s in sessions:
            w = make_windows(s["data"], SM_WIN, SM_STRIDE)
            X.append(w); y += [cls[s["substance"]]] * len(w); d += [s["session"]] * len(w)
        return np.concatenate(X), np.array(y), np.array(d)
    Xtr, ytr, dtr = build([s for s in S if s["session"] in tr_d])
    Xev, yev, dev = build([s for s in S if s["session"] in ev_d])
    mu = Xtr.reshape(-1, Xtr.shape[-1]).mean(0); sd = Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-8
    return dict(Xtr=(Xtr - mu) / sd, ytr=ytr, dtr=dtr, Xev=(Xev - mu) / sd, yev=yev, dev=dev,
                train_domains=tr_d, eval_domains=ev_d, role=role, n_cls=len(cls))


def domain_batches(dom, bs, rng):
    """Domain-homogeneous minibatches: shuffle within each domain, chunk, then shuffle batch order."""
    batches = []
    for k in np.unique(dom):
        idx = rng.permutation(np.where(dom == k)[0])
        batches += [(int(k), idx[i:i + bs]) for i in range(0, len(idx), bs)]
    order = rng.permutation(len(batches))
    return [batches[i] for i in order]


def mixed_batches(dom, bs, rng):
    """Standard i.i.d. minibatches (ablation). Domain label = majority domain of the batch,
    used only by SA-MSAM (so SA-MSAM is not meaningful with this sampler)."""
    idx = rng.permutation(len(dom))
    out = []
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        vals, cnt = np.unique(dom[b], return_counts=True)
        out.append((int(vals[cnt.argmax()]), b))
    return out


# ---------------------------------------------------------------- eval
def _decide(o):
    """Class index for classifiers, the scalar output for regressors (output width 1)."""
    return o[:, 0] if o.shape[1] == 1 else o.argmax(1)


@torch.no_grad()
def predict(model, X, t=None, bs=4096):
    model.eval()
    if t is None:
        out = torch.cat([_decide(model(X[i:i + bs])) for i in range(0, len(X), bs)])
    else:
        out = torch.cat([_decide(model(X[i:i + bs], t[i:i + bs])) for i in range(0, len(X), bs)])
    model.train()
    return out.cpu().numpy()


def metrics(y, p, n_cls, y_sd=1.0):
    """Classification: accuracy / macro-F1 / balanced acc; score = macro-F1 over present classes.
    Regression (n_cls == 1, standardized targets): R2, and RMSE / MAE in original (log) units; score = R2."""
    if n_cls == 1:
        err = p - y
        r2 = 1.0 - float((err ** 2).sum()) / max(float(((y - y.mean()) ** 2).sum()), 1e-12)
        return dict(n=int(len(y)), r2=r2, rmse=float(np.sqrt((err ** 2).mean())) * y_sd,
                    mae=float(np.abs(err).mean()) * y_sd, score=r2)
    present = np.unique(y)
    f1p = float(f1_score(y, p, labels=present, average="macro", zero_division=0))
    return dict(n=int(len(y)), acc=float((y == p).mean()),
                macro_f1_present=f1p,
                macro_f1_all=float(f1_score(y, p, labels=np.arange(n_cls), average="macro", zero_division=0)),
                bal_acc=float(balanced_accuracy_score(y, p)), score=f1p)


def eval_all(model, D, Xev, n_cls, tev=None):
    p = predict(model, Xev, tev)
    ysd = D.get("y_sd", 1.0)
    res = {"pooled": metrics(D["yev"], p, n_cls, ysd)}
    for k in D["eval_domains"]:
        m = D["dev"] == k
        res[str(k)] = metrics(D["yev"][m], p[m], n_cls, ysd)
    keys = ("r2", "rmse", "mae", "score") if n_cls == 1 else ("acc", "macro_f1_present", "bal_acc", "score")
    res["mean_over_domains"] = {key: float(np.mean([res[str(k)][key] for k in D["eval_domains"]])) for key in keys}
    return res, p


# ---------------------------------------------------------------- train
def run(args, return_model=False):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    D = load_data(args.dataset, args.split, args.norm, args.norm_k)
    Xtr = torch.tensor(D["Xtr"], dtype=torch.float32, device=DEV)
    reg = D.get("task") == "reg"
    ytr = torch.tensor(D["ytr"], dtype=torch.float32 if reg else torch.long, device=DEV)
    if args.method in ("coral", "groupdro") or args.opt == "disam":
        assert args.sampler == "mixed", f"{args.method}/{args.opt} needs multi-domain batches: use --sampler mixed"
    Xev = torch.tensor(D["Xev"], dtype=torch.float32, device=DEV)
    dom_ids = {k: i for i, k in enumerate(D["train_domains"])}       # domain -> 0..K-1 (time order)
    mhp = json.loads(args.mhp)
    time_input = args.method == "gi"
    model = build_model(args.dataset, time_input=time_input).to(DEV)
    hp = json.loads(args.hp)
    opt = build_optimizer(args.opt, model.parameters(), hp, n_domains=len(dom_ids),
                          domain_order=list(range(len(dom_ids))), seed=args.seed)
    if reg:
        _mse = torch.nn.MSELoss()
        lossf = lambda o, y: _mse(o.squeeze(-1), y)           # noqa: E731
    else:
        lossf = torch.nn.CrossEntropyLoss()
    # forward+backward passes per step (GI: 3 forwards, backward through all three)
    passes = (2 if isinstance(opt, SAM) else 1) * (3 if args.method == "gi" else 1)
    # Time coordinate for GI: domain value scaled so that training domains span [0, 1]
    tmin, tmax = min(D["train_domains"]), max(D["train_domains"])
    tscale = lambda v: (float(v) - tmin) / max(1.0, tmax - tmin)  # noqa: E731
    dt_unit = 1.0 / max(1.0, tmax - tmin)                         # one domain step in scaled time
    K = len(dom_ids)
    dtr_idx = torch.tensor([dom_ids[v] for v in D["dtr"]], dtype=torch.long, device=DEV)  # per-sample 0..K-1
    q_dro = torch.ones(K, device=DEV) / K                          # GroupDRO domain weights
    lossf_none = ((lambda o, y: (o.squeeze(-1) - y) ** 2) if reg  # noqa: E731
                  else torch.nn.CrossEntropyLoss(reduction="none"))

    def fwd(x, t=None):
        return model(x, t) if time_input else model(x)

    def domain_losses(per_sample, db):
        present = torch.unique(db)
        return present, torch.stack([per_sample[db == d].mean() for d in present])

    def coral_penalty(f, db):
        """DomainBed CORAL: mean over domain pairs of mean-diff^2 + covariance-diff^2 (both averaged)."""
        present = torch.unique(db)
        stats = []
        for d in present:
            fd = f[db == d]
            if len(fd) < 2:
                continue
            mu = fd.mean(0, keepdim=True); c = fd - mu
            stats.append((mu, c.t() @ c / (len(fd) - 1)))
        pen, n = 0.0, 0
        for i in range(len(stats)):
            for j in range(i + 1, len(stats)):
                pen = pen + (stats[i][0] - stats[j][0]).pow(2).mean() + (stats[i][1] - stats[j][1]).pow(2).mean(); n += 1
        return pen / max(n, 1)

    def loss_fn(xb, yb, k, db=None, perturb=False):
        """Training loss for one batch. k = domain value of a domain-homogeneous batch; db = per-sample
        domain index (0..K-1), used by the multi-domain methods (CORAL, GroupDRO, DISAM)."""
        nonlocal q_dro
        if args.opt == "disam":                       # perturbation loss CE - lam*Var_d(L_d); descent: CE
            if not perturb:
                return lossf(model(xb), yb)
            per = lossf_none(model(xb), yb)
            _, ld = domain_losses(per, db)
            return per.mean() - json.loads(args.hp).get("lam", 0.1) * ld.var(unbiased=False)
        if args.method in ("erm", "swa"):
            return lossf(fwd(xb), yb)
        if args.method == "driftaug":                 # random gain/offset mimicking sensor drift
            a, s_ = mhp.get("gain", 0.2), mhp.get("offset", 0.2)
            if xb.ndim == 2:                          # UCI (16 sensors) / Twin (8): one gain per sensor
                S = 16 if args.dataset == "uci" else 8    # (8 features each); Twin's gas one-hot is untouched
                xs = xb[:, :S * 8]
                g = 1 + a * (2 * torch.rand(len(xb), S, 1, device=DEV) - 1)
                xa = (xs.reshape(len(xb), S, 8) * g).reshape(len(xb), -1) + s_ * torch.randn_like(xs)
                xa = torch.cat([xa, xb[:, S * 8:]], 1)
            else:                                     # SmellNet: one gain and offset per channel per window
                C = xb.shape[-1]
                g = 1 + a * (2 * torch.rand(len(xb), 1, C, device=DEV) - 1)
                xa = xb * g + s_ * torch.randn(len(xb), 1, C, device=DEV)
            return lossf(fwd(xa), yb)
        if args.method == "coral":
            f = model.features(xb)
            return lossf(model.head(f), yb) + mhp.get("lam", 1.0) * coral_penalty(f, db)
        if args.method == "groupdro":                 # online GroupDRO (Sagawa et al.), DomainBed form
            present, ld = domain_losses(lossf_none(model(xb), yb), db)
            with torch.no_grad():
                q_dro[present] *= torch.exp(mhp.get("eta", 0.01) * ld)
                q_dro /= q_dro.sum()
            return (q_dro[present] * ld).sum() / q_dro[present].sum()
        if args.method == "recency":                  # RIDER-style: up-weight recent domains
            g = mhp.get("gamma", 0.7)
            w = np.array([g ** (K - 1 - i) for i in range(K)]); w = w / w.mean()
            return float(w[dom_ids[k]]) * lossf(fwd(xb), yb)
        if args.method == "gi":                       # GI-style (finite-difference variant):
            t = torch.full((len(xb),), tscale(k), device=DEV)   # fit at true time t AND at earlier
            l0 = lossf(fwd(xb, t), yb)                           # times t - delta (worst of 2), so the
            dmax = mhp.get("delta", 1.0) * dt_unit               # model must extrapolate forward
            ls = [lossf(fwd(xb, t - f * dmax), yb) for f in (0.5, 1.0)]
            return l0 + mhp.get("lam", 1.0) * torch.stack(ls).max()
        raise ValueError(args.method)
    swa_start = mhp.get("swa_start", 0) if args.method == "swa" else 0
    swa_model = torch.optim.swa_utils.AveragedModel(model) if swa_start else None
    swa_on = False                                    # dense (every-step) averaging from epoch swa_start

    def eval_model():
        return swa_model.module if swa_on else model
    curve, n_steps, fb = [], 0, 0
    best = dict(score=-1, epoch=0)
    if DEV.type == "cuda":
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    tev = (torch.tensor([tscale(v) for v in D["dev"]], dtype=torch.float32, device=DEV) if time_input else None)
    t0 = time.perf_counter(); eval_time = 0.0
    diverged = False
    for ep in range(1, args.epochs + 1):
        ep_loss = []
        batches = (domain_batches(D["dtr"], args.bs, rng) if args.sampler == "domain"
                   else mixed_batches(D["dtr"], args.bs, rng))
        for k, idx in batches:
            xb, yb, db = Xtr[idx], ytr[idx], dtr_idx[idx]
            if isinstance(opt, SAM):
                def closure(perturb=True):
                    opt.zero_grad(); l = loss_fn(xb, yb, k, db, perturb); l.backward(); return l
                loss = opt.step(closure)
            else:
                opt.zero_grad(); loss = loss_fn(xb, yb, k, db); loss.backward()
                if isinstance(opt, SAMSAM):
                    opt.set_domain(dom_ids[k], loss.item())
                opt.step()
            ep_loss.append(loss.item()); n_steps += 1; fb += passes
            if swa_model is not None and ep >= swa_start:
                swa_model.update_parameters(model); swa_on = True
        if not np.isfinite(np.mean(ep_loss)):
            diverged = True; break
        if D["role"] == "val":                       # per-epoch validation only in tuning mode
            te = time.perf_counter()
            getattr(opt, "move_back", lambda: None)()
            r, _ = eval_all(eval_model(), D, Xev, D["n_cls"], tev)
            getattr(opt, "move_up", lambda: None)()
            score = r["mean_over_domains"]["score"]
            curve.append(dict(epoch=ep, train_loss=float(np.mean(ep_loss)), val_macro_f1=score))
            if score > best["score"]:
                best = dict(score=score, epoch=ep)
            eval_time += time.perf_counter() - te
        else:
            curve.append(dict(epoch=ep, train_loss=float(np.mean(ep_loss))))
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    train_time = time.perf_counter() - t0 - eval_time
    getattr(opt, "move_back", lambda: None)()        # evaluate at un-perturbed weights
    model = eval_model()                              # SWA: everything below uses the averaged weights
    out = dict(dataset=args.dataset, split=args.split, opt=args.opt, method=args.method, mhp=mhp, sampler=args.sampler, norm=args.norm, norm_k=args.norm_k, hp=hp, seed=args.seed, epochs=args.epochs,
               bs=args.bs, diverged=diverged, n_steps=n_steps, fwd_bwd_passes=fb,
               train_time_s=train_time, device=str(DEV),
               gpu=torch.cuda.get_device_name(0) if DEV.type == "cuda" else None,
               peak_mem_mb=torch.cuda.max_memory_allocated() / 2**20 if DEV.type == "cuda" else None,
               torch=torch.__version__, curve=curve, best_val=best if D["role"] == "val" else None,
               role=D["role"])
    if not diverged:
        res, pred = eval_all(model, D, Xev, D["n_cls"], tev)
        out["final_eval"] = res
        # collapse check: accuracy on (up to) 4096 training samples; a run that "trains" to near chance
        # without a non-finite loss is flagged collapsed (see analysis).
        n = min(len(Xtr), 4096)
        sel_c = np.random.default_rng(1).choice(len(Xtr), n, replace=False)
        ttr = (torch.tensor([tscale(v) for v in D["dtr"][sel_c]], dtype=torch.float32, device=DEV) if time_input else None)
        ptr = predict(model, Xtr[sel_c], ttr)
        out["train_acc"] = (metrics(D["ytr"][sel_c], ptr, 1)["r2"] if reg      # regression: train R2
                            else float((ptr == D["ytr"][sel_c]).mean()))
        out["collapsed"] = bool(out["train_acc"] < 0.5)
        if D["role"] == "test":
            out["test_pred"] = pred.tolist(); out["test_true"] = D["yev"].tolist(); out["test_dom"] = D["dev"].tolist()
        if args.sharpness and not time_input:
            n = min(len(Xtr), 4096)
            sel = torch.tensor(np.random.default_rng(0).choice(len(Xtr), n, replace=False), device=DEV)
            out["sharpness"] = all_sharpness(model, Xtr[sel], ytr[sel])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out))
    return (out, model, D) if return_model else out


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["uci", "smell", "twin", "twinreg", "smellfeat", "smellgru"], required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--opt", required=True)
    p.add_argument("--hp", default="{}")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--sharpness", action="store_true")
    p.add_argument("--method", choices=["erm", "recency", "gi", "coral", "groupdro", "driftaug", "swa"], default="erm")
    p.add_argument("--norm_k", type=int, default=1, help="number of drift directions removed by --norm ccpca")
    p.add_argument("--norm", choices=["train", "domain", "ccpca"], default="train",
                   help="train: one scaler fit on training data; domain: per-domain standardization")
    p.add_argument("--sampler", choices=["domain", "mixed"], default="domain")
    p.add_argument("--mhp", default="{}", help="method hyperparameters (JSON)")
    p.add_argument("--out", required=True)
    return p.parse_args(argv)


if __name__ == "__main__":
    o = run(parse())
    fe = o.get("final_eval", {})
    print(json.dumps(dict(opt=o["opt"], split=o["split"], time=round(o["train_time_s"], 1), passes=o["fwd_bwd_passes"],
                          best_val=o["best_val"], mean=fe.get("mean_over_domains"), sharp=o.get("sharpness"))))
