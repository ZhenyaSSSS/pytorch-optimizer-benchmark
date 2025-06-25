"""Utility to visualise 2-D loss surface around a trained model.

We sample two random but orthonormal directions in parameter space (d₁,d₂),
and evaluate the loss on a grid (α,β) ↦ L(w + α·d₁ + β·d₂).
This is computationally heavy, so we restrict to small networks.
"""
from __future__ import annotations

from typing import Iterable, Tuple
import math
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt


@torch.no_grad()
def get_random_directions(params: Iterable[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    flat, shapes = _flatten(params)
    d1 = torch.randn_like(flat)
    d1 = d1 / d1.norm()
    # make d2 orthogonal to d1
    d2 = torch.randn_like(flat)
    d2 = d2 - torch.dot(d2, d1) * d1
    d2 = d2 / d2.norm()
    return d1, d2


def _flatten(params: Iterable[torch.Tensor]):
    shapes = [p.shape for p in params]
    flat = torch.cat([p.detach().view(-1) for p in params])
    return flat, shapes


def _inject(params: Iterable[torch.Tensor], flat: torch.Tensor, shapes):
    idx = 0
    for p, shp in zip(params, shapes):
        n = math.prod(shp)
        p.copy_(flat[idx: idx + n].view(shp))
        idx += n


@torch.no_grad()
def plot_loss_surface(model: nn.Module, loss_fn, dataloader, steps: int = 21, span: float = 1.0, device="cpu"):
    model.eval()
    flat, shapes = _flatten(model.parameters())
    d1, d2 = get_random_directions(model.parameters())

    alphas = np.linspace(-span, span, steps)
    betas = np.linspace(-span, span, steps)
    Z = np.zeros((steps, steps))
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            new_flat = flat + a * d1 + b * d2
            _inject(model.parameters(), new_flat, shapes)
            loss_sum = 0.0
            n_total = 0
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = loss_fn(out, y)
                loss_sum += loss.item() * x.size(0)
                n_total += x.size(0)
            Z[i, j] = loss_sum / n_total
    # restore original weights
    _inject(model.parameters(), flat, shapes)

    # plot -----------------------------------------------------------
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    A, B = np.meshgrid(alphas, betas)
    ax.plot_surface(A, B, Z, cmap="viridis", edgecolor="none")
    ax.set_xlabel("α")
    ax.set_ylabel("β")
    ax.set_zlabel("Loss")
    ax.set_title("Loss surface around model")
    plt.tight_layout()
    plt.show() 