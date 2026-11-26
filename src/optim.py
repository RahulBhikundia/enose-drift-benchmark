"""Optimizers: SAM, MSAM, SA-MSAM (shift-aware MSAM) and a random-direction control.

Written from the equations in Foret et al. (SAM, ICLR 2021) and Becker et al. (MSAM, NeurIPS 2025).
The official MSAM code (GPL-3, github.com/MarlonBecker/MSAM) is NOT copied; tests/test_optim.py
checks numerically that our MSAM gives the same iterates as the official one.

MSAM recap (SGD-momentum version). The stored parameters are kept at the perturbed point
    theta_tilde = theta + e,   e = -rho * v / ||v||
At each step the loss gradient g is computed at theta_tilde (one forward/backward pass), then
    g     <- g + wd * theta_tilde          (weight decay at the perturbed point, as in the official code)
    v     <- mu * v + g
    theta <- theta - lr * v
    e     <- -rho * v / ||v||              (new perturbation)
IMPORTANT: call `move_back()` before evaluation and `move_up()` before resuming training,
otherwise models are evaluated at the perturbed point.

SA-MSAM changes only the perturbation direction:
    e = -rho * normalize( v/||v|| + beta * d/||d|| )
where d is a "shift direction" built from per-domain gradient EMAs. Training uses domain-homogeneous
minibatches, so the batch gradient is that domain's gradient and the EMAs cost nothing extra.
  variant "extrap":   d = m_newest - m_oldest        (time-ordered domains; extrapolates the trend)
  variant "disagree": d = m_hardest - mean_k m_k     (no trend assumed; hardest = highest EMA loss)
  variant "random":   d = fresh Gaussian vector each step (control: same norm budget, no information)
The sign of beta is a hyperparameter (beta < 0 flips d); the go/no-go on validation decides.
"""
from __future__ import annotations

import torch


def _global_norm(tensors) -> torch.Tensor:
    return torch.linalg.vector_norm(torch.stack(torch._foreach_norm(list(tensors))))


class MSAM(torch.optim.Optimizer):
    """Momentum-SAM on top of SGD with (heavy-ball) momentum. Single parameter group."""

    def __init__(self, params, lr=0.1, momentum=0.9, weight_decay=0.0, rho=0.3):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay, rho=rho))
        if len(self.param_groups) != 1:
            raise ValueError("MSAM supports a single parameter group")
        self._at_perturbed = True     # stored params include e (e = 0 at start)
        for p in self._params():
            st = self.state[p]
            st["v"] = torch.zeros_like(p)
            st["e"] = torch.zeros_like(p)

    def _params(self):
        return [p for p in self.param_groups[0]["params"] if p.requires_grad]

    # Direction hook overridden by SA-MSAM. Returns a list of tensors (unnormalised OK).
    def _direction(self, params, vs, vnorm):
        return torch._foreach_div(vs, vnorm + 1e-12)

    @torch.no_grad()
    def step(self, closure=None):
        # Vectorised with torch._foreach_* (like torch.optim.SGD's foreach path) so wall-clock
        # comparisons against SGD/AdamW are fair.
        assert self._at_perturbed, "call move_up() before training steps"
        g_ = self.param_groups[0]
        lr, mu, wd, rho = g_["lr"], g_["momentum"], g_["weight_decay"], g_["rho"]
        params = [p for p in self._params() if p.grad is not None]
        grads = [p.grad for p in params]
        vs = [self.state[p]["v"] for p in params]
        es = [self.state[p]["e"] for p in params]
        d_p = torch._foreach_add(grads, params, alpha=wd) if wd != 0 else grads  # wd at perturbed point
        torch._foreach_sub_(params, es)              # back to theta
        torch._foreach_mul_(vs, mu)
        torch._foreach_add_(vs, d_p)
        torch._foreach_add_(params, vs, alpha=-lr)   # SGD-momentum step
        vnorm = _global_norm(vs)
        dirs = self._direction(params, vs, vnorm)
        dnorm = _global_norm(dirs)
        new_e = torch._foreach_mul(dirs, -rho / (dnorm + 1e-12))
        torch._foreach_add_(params, new_e)           # move to the new perturbed point
        for p, e in zip(params, new_e):
            self.state[p]["e"] = e
        return None

    @torch.no_grad()
    def move_back(self):
        """Remove the perturbation (use before evaluation/checkpointing)."""
        if self._at_perturbed:
            for p in self._params():
                p.sub_(self.state[p]["e"])
            self._at_perturbed = False

    @torch.no_grad()
    def move_up(self):
        """Re-apply the perturbation (use before continuing training)."""
        if not self._at_perturbed:
            for p in self._params():
                p.add_(self.state[p]["e"])
            self._at_perturbed = True


class SAMSAM(MSAM):
    """Shift-Aware MSAM. Call `set_domain(k, loss)` before each `step()`; k indexes the domain of
    the (domain-homogeneous) minibatch whose gradient is in p.grad; `domain_order` gives the
    time order of the training domains (used by the 'extrap' variant)."""

    def __init__(self, params, n_domains: int, domain_order=None, variant="extrap", beta=1.0,
                 ema=0.9, warmup_steps=50, seed=0, **kw):
        super().__init__(params, **kw)
        assert variant in ("extrap", "disagree", "random")
        self.variant, self.beta, self.ema, self.warmup = variant, beta, ema, warmup_steps
        self.n_domains = n_domains
        self.order = list(domain_order) if domain_order is not None else list(range(n_domains))
        self._cur, self._cur_loss = None, None
        self._seen = [0] * n_domains
        self._loss_ema = [None] * n_domains
        self._step = 0
        self._gen = torch.Generator(device="cpu").manual_seed(seed)
        if variant != "random":
            for p in self._params():
                self.state[p]["m"] = torch.zeros((n_domains,) + tuple(p.shape), device=p.device, dtype=p.dtype)

    def set_domain(self, k: int, loss: float | None = None):
        self._cur, self._cur_loss = int(k), loss

    @torch.no_grad()
    def step(self, closure=None):
        k = self._cur
        if self.variant != "random":
            assert k is not None, "SA-MSAM: call set_domain(k) before step()"
            ps = [p for p in self._params() if p.grad is not None]
            ms = [self.state[p]["m"][k] for p in ps]
            gs = [p.grad for p in ps]
            if self._seen[k] == 0:
                torch._foreach_copy_(ms, gs)
            else:
                torch._foreach_mul_(ms, self.ema)
                torch._foreach_add_(ms, gs, alpha=1 - self.ema)
            self._seen[k] += 1
            if self._cur_loss is not None:
                le = self._loss_ema[k]
                self._loss_ema[k] = self._cur_loss if le is None else self.ema * le + (1 - self.ema) * self._cur_loss
        self._step += 1
        super().step()
        self._cur = None

    def _shift_dir(self, params):
        if self.variant == "random":
            return [torch.randn(p.shape, generator=self._gen).to(p.device, p.dtype) for p in params]
        seen = [i for i in range(self.n_domains) if self._seen[i] > 0]
        if len(seen) < 2:
            return None
        if self.variant == "extrap":
            ordered = [i for i in self.order if i in seen]
            new, old = ordered[-1], ordered[0]
            return [self.state[p]["m"][new] - self.state[p]["m"][old] for p in params]
        # 'disagree': hardest domain (highest loss EMA) minus mean over seen domains
        losses = {i: self._loss_ema[i] for i in seen if self._loss_ema[i] is not None}
        hard = max(losses, key=losses.get) if losses else seen[-1]
        return [self.state[p]["m"][hard] - self.state[p]["m"][seen].mean(0) for p in params]

    def _direction(self, params, vs, vnorm):
        base = torch._foreach_div(vs, vnorm + 1e-12)
        if self.beta == 0 or self._step < self.warmup:
            return base
        d = self._shift_dir(params)
        if d is None:
            return base
        dn = _global_norm(d)
        torch._foreach_add_(base, torch._foreach_mul(d, self.beta / (dn + 1e-12)))
        return base


class SAM:
    """Two-pass SAM wrapped around any base optimizer (SGD-momentum or AdamW).

    Usage per step:  loss = sam.step(closure)  where closure() zeroes grads, computes the loss,
    calls backward and returns the loss. Two forward/backward passes per step."""

    def __init__(self, params, base_optimizer: torch.optim.Optimizer, rho=0.05):
        self.params = [p for p in params if p.requires_grad]
        self.base, self.rho = base_optimizer, rho
        self.param_groups = base_optimizer.param_groups

    def zero_grad(self, set_to_none=True):
        self.base.zero_grad(set_to_none=set_to_none)

    def step(self, closure):
        loss = closure()                                  # pass 1: gradient at theta
        with torch.no_grad():
            ps = [p for p in self.params if p.grad is not None]
            n = _global_norm([p.grad for p in ps])
            es = []
            for p in ps:
                e = p.grad * (self.rho / (n + 1e-12))
                p.add_(e); es.append(e)
        closure()                                         # pass 2: gradient at theta + e
        with torch.no_grad():
            for p, e in zip(ps, es):
                p.sub_(e)
        self.base.step()
        return loss

    # no-ops so the training loop can treat all optimizers alike
    def move_back(self):
        pass

    def move_up(self):
        pass


def build_optimizer(name: str, params, hp: dict, n_domains: int = 1, domain_order=None, seed: int = 0):
    """Factory. name in: sgd, adamw, sam, msam, samsam_extrap, samsam_disagree, samsam_random."""
    params = list(params)
    lr, wd = hp["lr"], hp.get("weight_decay", 0.0)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=hp.get("momentum", 0.9), weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "sam":
        base = torch.optim.SGD(params, lr=lr, momentum=hp.get("momentum", 0.9), weight_decay=wd)
        return SAM(params, base, rho=hp["rho"])
    if name == "msam":
        return MSAM(params, lr=lr, momentum=hp.get("momentum", 0.9), weight_decay=wd, rho=hp["rho"])
    if name.startswith("samsam_"):
        return SAMSAM(params, n_domains=n_domains, domain_order=domain_order, variant=name.split("_", 1)[1],
                      beta=hp.get("beta", 1.0), ema=hp.get("ema", 0.9), warmup_steps=hp.get("warmup", 50),
                      seed=seed, lr=lr, momentum=hp.get("momentum", 0.9), weight_decay=wd, rho=hp["rho"])
    raise ValueError(name)
