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
        if split == "main": return [1, 2, 3, 4, 5], [6, 7, 8, 9, 10], "test"
        if split.startswith("roll"):
            k = int(split[4:]); return list(range(1, k + 1)), [k + 1], "test"
    else:
        if split == "tune": return [1, 2, 3, 4], [5], "val"
        if split.startswith("loso"):
            h = int(split[4:]); return [s for s in range(1, 7) if s != h], [h], "test"
    raise ValueError(split)


def load_data(dataset, split):
    tr_d, ev_d, role = split_domains(dataset, split)
    if dataset == "uci":
        df = load_uci(); F = feature_columns()
        tr = df[df.batch.isin(tr_d)]; ev = df[df.batch.isin(ev_d)]
        mu, sd = tr[F].mean().to_numpy(), tr[F].std().to_numpy() + 1e-8      # train-only scaler
        Xtr = (tr[F].to_numpy() - mu) / sd; Xev = (ev[F].to_numpy() - mu) / sd
        return dict(Xtr=Xtr, ytr=tr.label.to_numpy(), dtr=tr.batch.to_numpy(),
                    Xev=Xev, yev=ev.label.to_numpy(), dev=ev.batch.to_numpy(),
                    train_domains=tr_d, eval_domains=ev_d, role=role, n_cls=6)
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
@torch.no_grad()
def predict(model, X, t=None, bs=4096):
    model.eval()
    if t is None:
        out = torch.cat([model(X[i:i + bs]).argmax(1) for i in range(0, len(X), bs)])
    else:
        out = torch.cat([model(X[i:i + bs], t[i:i + bs]).argmax(1) for i in range(0, len(X), bs)])
    model.train()
    return out.cpu().numpy()


def metrics(y, p, n_cls):
    present = np.unique(y)
    return dict(n=int(len(y)), acc=float((y == p).mean()),
                macro_f1_present=float(f1_score(y, p, labels=present, average="macro", zero_division=0)),
                macro_f1_all=float(f1_score(y, p, labels=np.arange(n_cls), average="macro", zero_division=0)),
                bal_acc=float(balanced_accuracy_score(y, p)))


def eval_all(model, D, Xev, n_cls, tev=None):
    p = predict(model, Xev, tev)
    res = {"pooled": metrics(D["yev"], p, n_cls)}
    for k in D["eval_domains"]:
        m = D["dev"] == k
        res[str(k)] = metrics(D["yev"][m], p[m], n_cls)
    res["mean_over_domains"] = {key: float(np.mean([res[str(k)][key] for k in D["eval_domains"]]))
                                for key in ("acc", "macro_f1_present", "bal_acc")}
    return res, p


# ---------------------------------------------------------------- train
def run(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    D = load_data(args.dataset, args.split)
    Xtr = torch.tensor(D["Xtr"], dtype=torch.float32, device=DEV)
    ytr = torch.tensor(D["ytr"], dtype=torch.long, device=DEV)
    Xev = torch.tensor(D["Xev"], dtype=torch.float32, device=DEV)
    dom_ids = {k: i for i, k in enumerate(D["train_domains"])}       # domain -> 0..K-1 (time order)
    mhp = json.loads(args.mhp)
    time_input = args.method == "gi"
    model = build_model(args.dataset, time_input=time_input).to(DEV)
    hp = json.loads(args.hp)
    opt = build_optimizer(args.opt, model.parameters(), hp, n_domains=len(dom_ids),
                          domain_order=list(range(len(dom_ids))), seed=args.seed)
    lossf = torch.nn.CrossEntropyLoss()
    # forward+backward passes per step (GI: 3 forwards, backward through all three)
    passes = (2 if isinstance(opt, SAM) else 1) * (3 if args.method == "gi" else 1)
    # Time coordinate for GI: domain value scaled so that training domains span [0, 1]
    tmin, tmax = min(D["train_domains"]), max(D["train_domains"])
    tscale = lambda v: (float(v) - tmin) / max(1.0, tmax - tmin)  # noqa: E731
    dt_unit = 1.0 / max(1.0, tmax - tmin)                         # one domain step in scaled time
    K = len(dom_ids)

    def fwd(x, t=None):
        return model(x, t) if time_input else model(x)

    def loss_fn(xb, yb, k):
        """Training loss for one domain-homogeneous batch from domain value k."""
        if args.method == "erm":
            return lossf(fwd(xb), yb)
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
            xb, yb = Xtr[idx], ytr[idx]
            if isinstance(opt, SAM):
                def closure():
                    opt.zero_grad(); l = loss_fn(xb, yb, k); l.backward(); return l
                loss = opt.step(closure)
            else:
                opt.zero_grad(); loss = loss_fn(xb, yb, k); loss.backward()
                if isinstance(opt, SAMSAM):
                    opt.set_domain(dom_ids[k], loss.item())
                opt.step()
            ep_loss.append(loss.item()); n_steps += 1; fb += passes
        if not np.isfinite(np.mean(ep_loss)):
            diverged = True; break
        if D["role"] == "val":                       # per-epoch validation only in tuning mode
            te = time.perf_counter()
            getattr(opt, "move_back", lambda: None)()
            r, _ = eval_all(model, D, Xev, D["n_cls"], tev)
            getattr(opt, "move_up", lambda: None)()
            score = r["mean_over_domains"]["macro_f1_present"]
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
    out = dict(dataset=args.dataset, split=args.split, opt=args.opt, method=args.method, mhp=mhp, sampler=args.sampler, hp=hp, seed=args.seed, epochs=args.epochs,
               bs=args.bs, diverged=diverged, n_steps=n_steps, fwd_bwd_passes=fb,
               train_time_s=train_time, device=str(DEV),
               gpu=torch.cuda.get_device_name(0) if DEV.type == "cuda" else None,
               peak_mem_mb=torch.cuda.max_memory_allocated() / 2**20 if DEV.type == "cuda" else None,
               torch=torch.__version__, curve=curve, best_val=best if D["role"] == "val" else None,
               role=D["role"])
    if not diverged:
        res, pred = eval_all(model, D, Xev, D["n_cls"], tev)
        out["final_eval"] = res
        if D["role"] == "test":
            out["test_pred"] = pred.tolist(); out["test_true"] = D["yev"].tolist(); out["test_dom"] = D["dev"].tolist()
        if args.sharpness and not time_input:
            n = min(len(Xtr), 4096)
            sel = torch.tensor(np.random.default_rng(0).choice(len(Xtr), n, replace=False), device=DEV)
            out["sharpness"] = all_sharpness(model, Xtr[sel], ytr[sel])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out))
    return out


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["uci", "smell"], required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--opt", required=True)
    p.add_argument("--hp", default="{}")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--sharpness", action="store_true")
    p.add_argument("--method", choices=["erm", "recency", "gi"], default="erm")
    p.add_argument("--sampler", choices=["domain", "mixed"], default="domain")
    p.add_argument("--mhp", default="{}", help="method hyperparameters (JSON)")
    p.add_argument("--out", required=True)
    return p.parse_args(argv)


if __name__ == "__main__":
    o = run(parse())
    fe = o.get("final_eval", {})
    print(json.dumps(dict(opt=o["opt"], split=o["split"], time=round(o["train_time_s"], 1), passes=o["fwd_bwd_passes"],
                          best_val=o["best_val"], mean=fe.get("mean_over_domains"), sharp=o.get("sharpness"))))
