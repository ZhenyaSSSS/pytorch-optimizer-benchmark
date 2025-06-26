"""optimizers/a2sam.py
Accelerated Anisotropic Sharpness-Aware Minimization (A²SAM) implementation for PyTorch.

This implementation follows the formulation described in the original A²SAM
manuscript.  At a high-level it is a two-step procedure analogous to SAM, but it
(1) perturbs the weights along an *anisotropic* ellipsoid defined by a low-rank
approximation of the Hessian and (2) amortises the expensive Hessian update so
that in expectation the compute overhead is only ≈(M+k)/M back-ward passes.

Notation (see paper):
    ρ (rho)     – radius of the ellipsoid (scalar)
    α (alpha)   – strength of Hessian contribution when forming the metric  M = I + αH_k
    k           – target rank of the Hessian approximation (number of leading
                  eigen-pairs to keep)
    M_freq      – number of optimiser steps between full Hessian recalculation

Internally we wrap an arbitrary *base* optimiser (SGD, AdamW, …).  The public
API mimics torch.optim.Optimizer so that existing training loops can simply
replace their optimiser with `A2SAM(base_optimizer, **params)`.
"""
from __future__ import annotations

import math
import itertools
from typing import Iterable, List, Tuple, Dict, Any, Optional

import torch
from torch import nn
from torch import Tensor
from torch.optim import Optimizer
from torch.autograd import functional as F

# ----------------------------- helper utils ----------------------------------

def _flatten_params(params: Iterable[Tensor]) -> Tuple[Tensor, List[Tuple[int, ...]]]:
    """Flattens a list/iterable of *parameter* tensors into a 1-D vector.
    Returns both the flattened vector and the *shapes* of each tensor so that
    it can be unflattened later."""
    shapes: List[Tuple[int, ...]] = [p.shape for p in params]
    flat = torch.cat([p.data.view(-1) for p in params])
    return flat, shapes


def _unflatten_to(tgt: Tensor, shapes: List[Tuple[int, ...]]) -> List[Tensor]:
    """Splits 1-D tensor `tgt` back into a list of tensors with `shapes`."""
    views: List[Tensor] = []
    offset = 0
    for shp in shapes:
        n = math.prod(shp)
        views.append(tgt[offset : offset + n].view(shp))
        offset += n
    return views


def _power_iteration(
    hvp_fn,
    params: Iterable[Tensor],
    k: int,
    n_iters: int = 20,
    tol: float = 1e-4,
    device: Optional[torch.device] = None,
) -> Tuple[Tensor, Tensor]:
    """Approximate *k* leading (eigenvalue, eigenvector) pairs of the Hessian
    using stochastic power iteration with deflation.

    hvp_fn: callable(v) -> Hv (Hessian-vector product)
    params: list of network parameters (used for shapes & dtype)
    Returns:
        eigvals – (k,) tensor of eigenvalues  (sorted descending)
        eigvecs – (k, N) matrix; rows are eigenvectors (unit-norm)
    """
    flat, _ = _flatten_params(params)
    N = flat.numel()
    device = device or flat.device

    eigvals = torch.zeros(k, device=device)
    eigvecs = torch.zeros(k, N, device=device)

    def proj_orthonormal(v, basis):
        for b in basis:
            v -= (v @ b) * b
        return v

    for j in range(k):
        # initialise with random +/-1 Rademacher vector
        v = torch.randint_like(flat, high=2, low=0, dtype=torch.float32) * 2 - 1
        v = v / v.norm()
        last_ev = None
        for it in range(n_iters):
            if j:
                v = proj_orthonormal(v, eigvecs[:j])
                v = v / (v.norm() + 1e-12)
            Hv = hvp_fn(v)
            ev = torch.dot(v, Hv)
            v = Hv / (Hv.norm() + 1e-12)
            # check convergence
            if last_ev is not None and abs(ev - last_ev) < tol * abs(ev):
                break
            last_ev = ev
        eigvals[j] = ev.detach()
        eigvecs[j] = v.detach()
    return eigvals, eigvecs


# ----------------------------- main optimiser --------------------------------

class A2SAM(Optimizer):
    """PyTorch implementation of Accelerated Anisotropic Sharpness-Aware Minimisation."""

    def __init__(
        self,
        params: Iterable[Tensor],
        base_optimizer_cls,
        base_optimizer_kwargs: Optional[Dict[str, Any]] = None,
        rho: float = 0.05,
        alpha: float = 0.1,
        k: int = 1,
        hessian_update_freq: int = 30,
        power_iter_steps: int = 4,
        tol: float = 1e-4,
        eps: float = 1e-12,
    ) -> None:
        if rho <= 0.0:
            raise ValueError("rho must be positive")
        if alpha <= 0.0:
            raise ValueError("alpha must be positive")
        if k < 1:
            raise ValueError("k must be >=1")
        base_optimizer_kwargs = base_optimizer_kwargs or {}

        # We deliberately *do not* call super().__init__ with actual parameter
        # groups; instead we store them for state tracking, but delegate the
        # heavy-lifting to the wrapped optimiser.
        self.params = list(params)
        self.rho = rho
        self.alpha = alpha
        self.k = k
        self.hessian_update_freq = hessian_update_freq
        self.power_iter_steps = power_iter_steps
        self.tol = tol
        self.eps = eps

        self._step = 0
        # storage for Hessian eigenpairs
        self._eigvals: Optional[Tensor] = None  # shape (k,)
        self._eigvecs: Optional[Tensor] = None  # shape (k, N)

        # build wrapped optimiser
        self.base_optimizer: Optimizer = base_optimizer_cls(self.params, **base_optimizer_kwargs)

    # ------------------------------------------------------------------ utils

    def _gather_flat_grad(self) -> Tensor:
        grads = [p.grad.view(-1) if p.grad is not None else torch.zeros_like(p.data).view(-1) for p in self.params]
        return torch.cat(grads)

    # ------------------------------------------------------------------ main

    def _compute_eps(self) -> List[Tensor]:
        """Compute anisotropic perturbation ε* and return it split per parameter."""
        g_flat = self._gather_flat_grad()
        g_norm = g_flat.norm()
        if g_norm < self.eps:
            return [torch.zeros_like(p) for p in self.params]

        if self._eigvals is None or self._eigvecs is None:
            # fall back to isotropic SAM
            eps_flat = self.rho * g_flat / g_norm
        else:
            # Woodbury: (I + αVΛVᵀ)^(-1) g = g – αV (Λ / (1 + αΛ)) (Vᵀ g)
            V = self._eigvecs  # (k, N)
            Λ = self._eigvals  # (k,)
            g_proj = torch.mv(V, g_flat)  # (k,)
            # scale per-component
            coeffs = self.alpha * Λ / (1.0 + self.alpha * Λ)
            coeffs = coeffs * g_proj  # (k,)
            M_inv_g = g_flat - torch.matmul(V.t(), coeffs)
            eps_flat = self.rho * M_inv_g / (M_inv_g.norm() + self.eps)
        # unflatten back to parameter shapes
        _, shapes = _flatten_params(self.params)
        return _unflatten_to(eps_flat, shapes)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = True) -> None:
        eps = self._compute_eps()
        # climb to the *worst-case* point
        for p, e in zip(self.params, eps):
            p.add_(e)
            # save perturbation for later restoration
            state = self.base_optimizer.state[p]
            state["eps"] = e
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = True) -> None:
        # restore parameters & apply optimiser update
        for p in self.params:
            state = self.base_optimizer.state[p]
            e = state.pop("eps")
            p.sub_(e)
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def step(self, closure):
        """Perform *one* optimiser step.

        `closure` is a callable that reevaluates the model and returns the loss.
        This is identical to torch-opt style for SAM."""
        assert closure is not None, "A2SAM requires a closure"
        # ░ step 1 – forward/backward pass on clean weights
        loss = closure()
        # optionally update Hessian approx every M steps ---------------------------------
        if self.hessian_update_freq > 0 and self._step % self.hessian_update_freq == 0:
            self._update_hessian(closure, loss)
        # ░ step 2 – first ascent step to worst-case neighbourhood ------------------------
        self.first_step(zero_grad=True)
        # ░ step 3 – evaluate loss & grads at perturbed weights ---------------------------
        closure()  # backward on perturbed weights populates grads
        # ░ step 4 – descent step of base optimiser & restore -----------------------------
        self.second_step(zero_grad=True)
        self._step += 1
        return loss.detach()

    # ------------------------------------------------------------------ Hessian utils

    def _update_hessian(self, closure, loss):
        """Recompute top-k Hessian eigen-pairs via power iteration."""
        model = None  # we don't actually need the model here because we only need hvp
        # --- finite-difference Hessian-vector product to avoid double-backward issues
        # take a *single* gradient snapshot at current params --------------------------
        g0_flat = self._gather_flat_grad().detach()
        
        # ✅ ИСПРАВЛЕНИЕ: Сохраняем исходные градиенты для восстановления
        original_grads = [p.grad.clone() if p.grad is not None else None for p in self.params]

        def hvp_fn(v: Tensor) -> Tensor:
            epsilon = 1e-3 / (v.norm() + 1e-12)

            # 1) perturb params --------------------------------------------------
            _, shapes_ = _flatten_params(self.params)
            delta_list = _unflatten_to(epsilon * v, shapes_)
            for p, d in zip(self.params, delta_list):
                with torch.no_grad():
                    p.data.add_(d)

            # 2) compute grad at perturbed params --------------------------------
            self.zero_grad()
            closure()  # populates p.grad
            g1_flat = self._gather_flat_grad().detach()

            # 3) restore params ---------------------------------------------------
            for p, d in zip(self.params, delta_list):
                with torch.no_grad():
                    p.data.sub_(d)

            # 4) finite-difference approximation ----------------------------------
            hv_flat = (g1_flat - g0_flat) / epsilon
            return hv_flat

        print(f"[A2SAM] step {self._step}: recomputing top-{self.k} Hessian eigenvalues…", flush=True)
        eigvals, eigvecs = _power_iteration(
            hvp_fn,
            self.params,
            self.k,
            n_iters=self.power_iter_steps,
            tol=self.tol,
            device=g0_flat.device,
        )
        self._eigvals = eigvals
        self._eigvecs = eigvecs
        print(f"[A2SAM]   λ = {eigvals.cpu().numpy()}", flush=True)
        
        # ✅ ИСПРАВЛЕНИЕ: Восстанавливаем исходные градиенты
        for p, orig_grad in zip(self.params, original_grads):
            if orig_grad is not None:
                p.grad = orig_grad.clone()
            else:
                p.grad = None

    # ------------------------------------------------------------------ proxy API
    def zero_grad(self):  # delegation helper
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_() 