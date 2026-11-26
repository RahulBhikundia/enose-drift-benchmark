"""Fixed model architectures (identical across optimizer conditions). No BatchNorm, because
BatchNorm statistics interact with SAM's second pass and MSAM's perturbed weights.

time_input=True (used only by the GI-style baseline) appends a scalar time coordinate t:
UCI MLP gets it as an extra input feature; the SmellNet CNN gets it as an extra constant channel.
"""
import torch
from torch import nn


class MLP(nn.Module):
    """UCI: 128 (+1 time) -> 256 -> 128 -> 6, ReLU."""

    def __init__(self, d_in=128, n_cls=6, hidden=(256, 128), time_input=False):
        super().__init__()
        self.time_input = time_input
        layers, d = [], d_in + int(time_input)
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, n_cls))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t=None):
        if self.time_input:
            x = torch.cat([x, t[:, None]], dim=1)
        return self.net(x)

    def features(self, x):
        """Penultimate representation (used by CORAL)."""
        return self.net[:-1](x)

    def head(self, h):
        return self.net[-1](h)


class CNN1D(nn.Module):
    """SmellNet: input (B, T, C=6) -> three 1-D conv layers -> global average pool -> linear."""

    def __init__(self, c_in=6, n_cls=50, width=64, time_input=False):
        super().__init__()
        self.time_input = time_input
        self.f = nn.Sequential(
            nn.Conv1d(c_in + int(time_input), width // 2, 5, padding=2), nn.ReLU(),
            nn.Conv1d(width // 2, width, 5, padding=2), nn.ReLU(),
            nn.Conv1d(width, width, 5, padding=2), nn.ReLU(),
        )
        self.head = nn.Linear(width, n_cls)

    def forward(self, x, t=None):
        if self.time_input:
            x = torch.cat([x, t[:, None, None].expand(-1, x.shape[1], 1)], dim=2)
        h = self.f(x.transpose(1, 2))          # (B, C, T)
        return self.head(h.mean(-1))

    def features(self, x):
        """Penultimate representation (used by CORAL)."""
        return self.f(x.transpose(1, 2)).mean(-1)


def build_model(dataset: str, time_input: bool = False):
    return MLP(time_input=time_input) if dataset == "uci" else CNN1D(time_input=time_input)
