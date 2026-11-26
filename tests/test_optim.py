"""Correctness tests for src/optim.py. Run: .venv/Scripts/python.exe -m pytest -q tests  (or python tests/test_optim.py)

1. Our MSAM gives the same iterates as the official MSAM implementation (external/msam_code).
2. SA-MSAM with beta = 0 is identical to MSAM.
3. Our SAM matches a hand-written reference for several steps.
4. move_back()/move_up() round-trip exactly, and move_back() leaves the un-perturbed weights.
5. SA-MSAM variants run, and the perturbation norm equals rho.
"""
import copy
import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.optim import MSAM, SAM, SAMSAM, _global_norm  # noqa: E402


def _official_msam_cls():
    spec = importlib.util.spec_from_file_location("official_msam", ROOT / "external" / "msam_code" / "optimizer" / "msam.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.MSAM


def _toy(seed=0):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(10, 16), torch.nn.Tanh(), torch.nn.Linear(16, 3)).double()
    X = torch.randn(64, 10, dtype=torch.float64); y = torch.randint(0, 3, (64,))
    return model, X, y


def _loss(model, X, y):
    return torch.nn.functional.cross_entropy(model(X), y)


def test_msam_matches_official():
    Official = _official_msam_cls()
    m1, X, y = _toy(); m2 = copy.deepcopy(m1)
    hp = dict(lr=0.05, momentum=0.9, weight_decay=1e-3, rho=0.3)
    o1 = MSAM(m1.parameters(), **hp)
    o2 = Official(m2.parameters(), nesterov=False, **hp)
    o2.move_up_to_momentumAscent()
    for t in range(30):
        idx = torch.randint(0, 64, (16,), generator=torch.Generator().manual_seed(t))
        for m, o in ((m1, o1), (m2, o2)):
            o.zero_grad(); _loss(m, X[idx], y[idx]).backward(); o.step()
    o1.move_back(); o2.move_back_from_momentumAscent()
    for a, b in zip(m1.parameters(), m2.parameters()):
        assert torch.allclose(a, b, atol=1e-10), (a - b).abs().max()


def test_samsam_beta0_equals_msam():
    m1, X, y = _toy(1); m2 = copy.deepcopy(m1)
    hp = dict(lr=0.05, momentum=0.9, weight_decay=0.0, rho=0.2)
    o1 = MSAM(m1.parameters(), **hp)
    o2 = SAMSAM(m2.parameters(), n_domains=3, variant="extrap", beta=0.0, warmup_steps=0, **hp)
    for t in range(25):
        idx = torch.arange(16) + 16 * (t % 3)
        o1.zero_grad(); _loss(m1, X[idx], y[idx]).backward(); o1.step()
        o2.zero_grad(); l2 = _loss(m2, X[idx], y[idx]); l2.backward(); o2.set_domain(t % 3, l2.item()); o2.step()
    for a, b in zip(m1.parameters(), m2.parameters()):
        assert torch.allclose(a, b, atol=1e-12)


def test_sam_matches_reference():
    m1, X, y = _toy(2); m2 = copy.deepcopy(m1)
    lr, mom, rho = 0.05, 0.9, 0.05
    base = torch.optim.SGD(m1.parameters(), lr=lr, momentum=mom)
    sam = SAM(m1.parameters(), base, rho=rho)
    bufs = [torch.zeros_like(p) for p in m2.parameters()]
    for t in range(10):
        def closure():
            sam.zero_grad(); l = _loss(m1, X, y); l.backward(); return l
        sam.step(closure)
        # reference
        ps = list(m2.parameters())
        g1 = torch.autograd.grad(_loss(m2, X, y), ps)
        n = _global_norm(g1)
        with torch.no_grad():
            for p, g in zip(ps, g1): p.add_(g * rho / (n + 1e-12))
        g2 = torch.autograd.grad(_loss(m2, X, y), ps)
        with torch.no_grad():
            for p, g in zip(ps, g1): p.sub_(g * rho / (n + 1e-12))
            for i, (p, g) in enumerate(zip(ps, g2)):
                bufs[i] = g.clone() if t == 0 else bufs[i].mul(mom).add(g)
                p.sub_(lr * bufs[i])
    for a, b in zip(m1.parameters(), m2.parameters()):
        assert torch.allclose(a, b, atol=1e-10)


def test_move_back_roundtrip_and_rho():
    for variant in ("extrap", "disagree", "random"):
        m, X, y = _toy(3)
        o = SAMSAM(m.parameters(), n_domains=3, variant=variant, beta=1.0, warmup_steps=2, lr=0.05, rho=0.3)
        for t in range(12):
            idx = torch.arange(16) + 16 * (t % 3)
            o.zero_grad(); l = _loss(m, X[idx], y[idx]); l.backward(); o.set_domain(t % 3, l.item()); o.step()
        e_norm = _global_norm([o.state[p]["e"] for p in m.parameters()])
        assert abs(e_norm.item() - 0.3) < 1e-9, (variant, e_norm)
        before = [p.detach().clone() for p in m.parameters()]
        o.move_back()
        theta = [p.detach().clone() for p in m.parameters()]
        for b, th, p in zip(before, theta, m.parameters()):
            assert torch.allclose(b - o.state[p]["e"], th)
        o.move_up()
        for b, p in zip(before, m.parameters()):
            assert torch.allclose(b, p)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
