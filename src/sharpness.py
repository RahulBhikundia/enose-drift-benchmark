"""Sharpness measures for a trained model (computed on the training loss, full batch).

  hessian_top_eig : top eigenvalue of the Hessian, by power iteration with Hessian-vector products
  random_sharp    : mean increase in loss under random weight perturbations of norm rho
  adv_sharp       : increase in loss under the one-step SAM perturbation of norm rho (worst-case proxy)
All are measured at the UN-perturbed weights (call optimizer.move_back() first).
"""
import torch
from torch.nn.utils import parameters_to_vector


def _loss(model, X, y, bs=4096):
    tot, n = 0.0, len(y)
    for i in range(0, n, bs):
        if y.dtype.is_floating_point:      # regression (Twin concentration): mean squared error
            tot = tot + torch.nn.functional.mse_loss(model(X[i:i + bs]).squeeze(-1), y[i:i + bs], reduction="sum")
        else:
            tot = tot + torch.nn.functional.cross_entropy(model(X[i:i + bs]), y[i:i + bs], reduction="sum")
    return tot / n


def hessian_top_eig(model, X, y, iters=30, tol=1e-4, seed=0):
    ps = [p for p in model.parameters() if p.requires_grad]
    g = torch.Generator(device="cpu").manual_seed(seed)
    v = [torch.randn(p.shape, generator=g).to(p.device) for p in ps]
    nrm = torch.sqrt(sum((x * x).sum() for x in v)); v = [x / nrm for x in v]
    loss = _loss(model, X, y)
    grads = torch.autograd.grad(loss, ps, create_graph=True)
    eig = None
    for _ in range(iters):
        hv = torch.autograd.grad(grads, ps, grad_outputs=v, retain_graph=True)
        new = sum((a * b).sum() for a, b in zip(hv, v)).item()
        nrm = torch.sqrt(sum((x * x).sum() for x in hv))
        v = [x / (nrm + 1e-12) for x in hv]
        if eig is not None and abs(new - eig) < tol * max(1.0, abs(eig)):
            eig = new; break
        eig = new
    return float(eig)


@torch.no_grad()
def random_sharp(model, X, y, rho=0.05, n=10, seed=0):
    ps = [p for p in model.parameters() if p.requires_grad]
    base = _loss(model, X, y).item()
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = []
    for _ in range(n):
        d = [torch.randn(p.shape, generator=g).to(p.device) for p in ps]
        nrm = torch.sqrt(sum((x * x).sum() for x in d))
        for p, x in zip(ps, d): p.add_(x * rho / nrm)
        out.append(_loss(model, X, y).item() - base)
        for p, x in zip(ps, d): p.sub_(x * rho / nrm)
    return float(sum(out) / n)


def adv_sharp(model, X, y, rho=0.05):
    ps = [p for p in model.parameters() if p.requires_grad]
    loss = _loss(model, X, y)
    gr = torch.autograd.grad(loss, ps)
    nrm = torch.sqrt(sum((x * x).sum() for x in gr))
    with torch.no_grad():
        for p, x in zip(ps, gr): p.add_(x * rho / nrm)
        up = _loss(model, X, y).item()
        for p, x in zip(ps, gr): p.sub_(x * rho / nrm)
    return float(up - loss.item())


def all_sharpness(model, X, y, rho=0.05):
    model.eval()
    return dict(hessian_top_eig=hessian_top_eig(model, X, y), random_sharp=random_sharp(model, X, y, rho),
                adv_sharp=adv_sharp(model, X, y, rho), param_norm=float(parameters_to_vector(model.parameters()).norm()))
