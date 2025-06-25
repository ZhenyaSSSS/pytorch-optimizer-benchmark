"""optimizers/hatam.py
Heuristic Anisotropic Trajectory-Aware Minimisation (HATAM) optimiser for PyTorch.

HATAM can be seen as a lightweight 1×-cost variant of SAM that (1) analyses the
*trajectory* via the gradient difference g_t − g_{t-1} and (2) modulates an
anisotropic correction using the second moment from Adam (v_t).

The update rule is identical to AdamW except that the gradient fed into the
Adam update is `g_hatam = g + γ · S ⊙ c`, where
    c_t   = β_c · c_{t−1} + (1 − β_c) · (g_t − g_{t-1})   — EMA of gradient difference
    S     =  h / (mean(h) + ε)                             — anisotropic modulator
    h     =  √(v_prev_corr)                                — per-parameter std-dev

See accompanying paper for details.
"""
from __future__ import annotations

from typing import Iterable, Optional, Dict, Any, Tuple
import math

import torch
from torch import Tensor
from torch.optim import Optimizer


class HATAM(Optimizer):
    """PyTorch implementation of the HATAM optimiser.

    The constructor mirrors the signature of AdamW for familiarity.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        beta_c: float = 0.9,
        gamma: float = 0.1,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0.0:
            raise ValueError("Invalid learning rate")
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameters")
        if not 0.0 <= beta_c < 1.0:
            raise ValueError("Invalid beta_c")
        defaults = dict(
            lr=lr,
            betas=betas,
            beta_c=beta_c,
            gamma=gamma,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _get_state(state, name, shape, dtype, device):
        if name not in state:
            state[name] = torch.zeros(shape, dtype=dtype, device=device)
        return state[name]

    # ------------------------------------------------------------------ main
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            beta_c = group["beta_c"]
            gamma = group["gamma"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.detach()
                state = self.state[p]

                # step counter ------------------------------------------------
                if "step" not in state:
                    state["step"] = 0
                state["step"] += 1
                t = state["step"]

                # initialisation ---------------------------------------------
                m = self._get_state(state, "m", p.shape, p.dtype, p.device)
                v = self._get_state(state, "v", p.shape, p.dtype, p.device)
                c = self._get_state(state, "c", p.shape, p.dtype, p.device)
                g_prev = self._get_state(state, "g_prev", p.shape, p.dtype, p.device)

                # compute HATAM correction -----------------------------------
                # y_t = g_t - g_{t-1} аппроксимирует произведение Hessian * step
                y = grad - g_prev  
                # c_t = β_c * c_{t-1} + (1 - β_c) * y_t (EMA сглаживание)
                c.mul_(beta_c).add_(y, alpha=1 - beta_c)  

                # Используем v_{t-1} для анизотропной модуляции (избегаем обратной связи)
                v_prev_corr = v / (1 - beta2 ** (t - 1)) if t > 1 else v
                h = v_prev_corr.sqrt().add_(eps)  # per-parameter стандартное отклонение
                h_mean = h.mean()
                S = h / torch.clamp(h_mean, min=eps)  # относительный модулятор S

                # Итоговая коррекция: γ * S ⊙ c
                hatam_corr = gamma * S * c
                # Модифицированный градиент для Adam: g_hatam = g + γ * S ⊙ c
                g_hatam = grad + hatam_corr

                # AdamW moments ---------------------------------------------
                m.mul_(beta1).add_(g_hatam, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(g_hatam, g_hatam, value=1 - beta2)

                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)

                update = m_hat / (v_hat.sqrt().add_(eps))
                if wd != 0:
                    update += wd * p.data

                p.data.add_(update, alpha=-lr)

                # store current gradient for next step -----------------------
                g_prev.copy_(grad)

        return loss 