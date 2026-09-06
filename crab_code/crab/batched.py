"""Batched CROWN: bound propagation over many subdomains at once.

Why this exists
---------------
The per-domain implementation in crown.py issues thousands of tiny matmuls
from a Python loop.  On a GPU that is pathological: kernel-launch overhead
dominates, and an M4 will be no faster than a CPU.  Throughput comes from
treating B subdomains as one tensor and doing B bound computations in a
single set of kernels.  This is design principle P5, which the first
implementation did not honour.

Shapes
------
    x_lb, x_ub          (B, n_in)
    pre_lb[i], pre_ub[i](B, width_i)
    status[i]           (B, width_i)   int8 in {-1, 0, +1}
    C                   (m, width_k)   shared across the batch
    A                   (B, m, width)

Only the weights are shared; every domain carries its own intervals and split
pattern.  Written in NumPy with an explicit `xp` array-module hook so the same
code runs under torch on MPS/CUDA (see torch_backend.py).

Soundness note
--------------
This computes exactly the same quantity as crown.backward, and
tests/test_batched.py asserts bitwise-comparable agreement.  Batching is a
performance change, never a semantic one -- if the two ever disagree beyond
float tolerance, the batched path is wrong and must not be used.
"""
from __future__ import annotations

import numpy as np

ACTIVE, FREE, INACTIVE = 1, 0, -1
EPS = 1e-12


def relu_relax_batch(lb, ub, status, xp=np):
    """Vectorised relaxation coefficients.  All arrays (B, width)."""
    forced_on = (status == ACTIVE) | (lb >= 0)
    forced_off = (status == INACTIVE) | (ub <= 0)
    unstable = ~(forced_on | forced_off)

    d = xp.maximum(ub - lb, EPS)
    su_uns = ub / d
    iu_uns = -ub * lb / d
    sl_uns = (ub > -lb).astype(lb.dtype)          # adaptive CROWN lower slope

    one = xp.ones_like(lb)
    zero = xp.zeros_like(lb)
    slope_lo = xp.where(forced_on, one, xp.where(unstable, sl_uns, zero))
    slope_up = xp.where(forced_on, one, xp.where(unstable, su_uns, zero))
    icpt_up = xp.where(unstable, iu_uns, zero)
    return slope_lo, slope_up, icpt_up


def backward_batch(net, k, C, pre_lb, pre_ub, status, x_lb, x_ub, xp=np):
    """Lower bound on C @ zhat^(k) for every domain in the batch.

    Returns (lb (B, m), lambdas dict of (B, m, width)).
    """
    B = x_lb.shape[0]
    dt = x_lb.dtype
    W = [w.astype(dt) for w in net.W]
    b = [v.astype(dt) for v in net.b]

    A = xp.broadcast_to(C.astype(dt), (B,) + C.shape).copy()
    bias = A @ b[k - 1]
    A = A @ W[k - 1]
    lambdas = {}

    for i in range(k - 1, 0, -1):
        sl, su, iu = relu_relax_batch(pre_lb[i], pre_ub[i], status[i], xp)
        pos = xp.maximum(A, 0)
        neg = xp.minimum(A, 0)
        bias = bias + xp.einsum("bmw,bw->bm", neg, iu)
        A = pos * sl[:, None, :] + neg * su[:, None, :]
        lambdas[i] = A
        bias = bias + A @ b[i - 1]
        A = A @ W[i - 1]

    centre = 0.5 * (x_lb + x_ub)
    radius = 0.5 * (x_ub - x_lb)
    lb = bias + xp.einsum("bmw,bw->bm", A, centre) \
        - xp.einsum("bmw,bw->bm", xp.abs(A), radius)
    return lb, lambdas


def _clamp(lb, ub, st, xp=np):
    return (xp.where(st == ACTIVE, xp.maximum(lb, 0), lb),
            xp.where(st == INACTIVE, xp.minimum(ub, 0), ub))


def compute_bounds_batch(net, x_lb, x_ub, status, C_spec, pre_lb=None, pre_ub=None,
                         from_layer=1, xp=np, slack=0.0):
    """Batched intermediate + output bounds.

    `slack` is subtracted from the returned spec bounds.  It exists for
    reduced-precision backends: float32 can report a bound ABOVE the true
    value (measured: up to 1.8e-6, growing with depth), so a float32 GPU run
    must round outward before comparing against zero.  Leave it at 0.0 in
    float64.
    """
    L = net.L
    B = x_lb.shape[0]
    dt = x_lb.dtype
    pre_lb = [None] * (L + 1) if pre_lb is None else list(pre_lb)
    pre_ub = [None] * (L + 1) if pre_ub is None else list(pre_ub)

    for i in range(1, min(from_layer, L)):
        if pre_lb[i] is not None:
            pre_lb[i], pre_ub[i] = _clamp(pre_lb[i], pre_ub[i], status[i], xp)

    for k in range(from_layer, L):
        w = net.widths[k]
        # one pass for both directions: rows [I; -I] give lower and -upper
        eye = np.eye(w, dtype=dt)
        C = np.concatenate([eye, -eye], axis=0)
        out, _ = backward_batch(net, k, C, pre_lb, pre_ub, status, x_lb, x_ub, xp)
        lo, hi = out[:, :w], -out[:, w:]
        if pre_lb[k] is not None:
            lo = xp.maximum(lo, pre_lb[k])
            hi = xp.minimum(hi, pre_ub[k])
        lo, hi = _clamp(lo, hi, status[k], xp)
        pre_lb[k], pre_ub[k] = lo, hi

    spec, lambdas = backward_batch(net, L, C_spec, pre_lb, pre_ub,
                                   status, x_lb, x_ub, xp)
    infeasible = xp.zeros(B, dtype=bool)
    for k in range(1, L):
        infeasible = infeasible | (pre_lb[k] > pre_ub[k] + 1e-9).any(axis=1)
    return pre_lb, pre_ub, spec - slack, lambdas, infeasible


def stack_domains(domains, net, dtype=np.float64):
    """Pack a list of per-domain dicts into batched arrays."""
    B = len(domains)
    status = [None] + [np.stack([d["status"][i] for d in domains])
                       for i in range(1, net.L)]
    x_lb = np.stack([d["x_lb"] for d in domains]).astype(dtype)
    x_ub = np.stack([d["x_ub"] for d in domains]).astype(dtype)
    pre_lb = [None] + [np.stack([d["pre_lb"][i] for d in domains]).astype(dtype)
                       if domains[0]["pre_lb"][i] is not None else None
                       for i in range(1, net.L)]
    pre_ub = [None] + [np.stack([d["pre_ub"][i] for d in domains]).astype(dtype)
                       if domains[0]["pre_ub"][i] is not None else None
                       for i in range(1, net.L)]
    return x_lb, x_ub, status, pre_lb, pre_ub
