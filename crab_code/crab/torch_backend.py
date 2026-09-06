"""PyTorch backend — runs the batched bound propagation on Apple MPS or CUDA.

IMPORTANT: this file was written without a GPU available to test on.  Run
`python3 tests/test_torch_backend.py` on your machine BEFORE trusting any
result from it.  That script validates every bound against the NumPy
reference implementation and measures the float32 slack your hardware
actually needs.

Precision and soundness
-----------------------
MPS does not support float64.  Measured on CPU (tests/test_float32.py),
float32 reports a bound ABOVE the float64 value in 57-80% of cases, worst
observed 1.8e-6, and the drift grows with depth.  A bound that is too high
can certify an unsafe network, so float32 must round outward: subtract a
slack before comparing against zero.

The recommended architecture is therefore two-tier, not float32-everywhere:

    fast tier   float32 on MPS, with slack.  Prunes the easy majority.
    exact tier  float64 on CPU.  Re-checks every domain whose float32 bound
                lands within `slack` of zero.

Anything the fast tier prunes is pruned with margin and is safe.  Anything
borderline is escalated.  This keeps GPU throughput without ever certifying
on a float32 bound alone.
"""
from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError:  # keep the package importable without torch
    torch = None


def pick_device(prefer="auto"):
    """Return (device, dtype, note)."""
    if torch is None:
        raise ImportError("PyTorch not installed. pip install torch")
    if prefer in ("auto", "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), torch.float32, \
            "Apple MPS — float32 only, outward slack REQUIRED"
    if prefer in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda"), torch.float64, "CUDA — float64 available"
    return torch.device("cpu"), torch.float64, "CPU — float64, exact tier"


ACTIVE, FREE, INACTIVE = 1, 0, -1


def _relax(lb, ub, status, eps):
    forced_on = (status == ACTIVE) | (lb >= 0)
    forced_off = (status == INACTIVE) | (ub <= 0)
    unstable = ~(forced_on | forced_off)
    d = torch.clamp(ub - lb, min=eps)
    su_u = ub / d
    iu_u = -ub * lb / d
    sl_u = (ub > -lb).to(lb.dtype)
    one = torch.ones_like(lb)
    zero = torch.zeros_like(lb)
    slope_lo = torch.where(forced_on, one, torch.where(unstable, sl_u, zero))
    slope_up = torch.where(forced_on, one, torch.where(unstable, su_u, zero))
    icpt_up = torch.where(unstable, iu_u, zero)
    return slope_lo, slope_up, icpt_up


class TorchBounds:
    """Batched CROWN on a torch device.  Mirrors crab.batched exactly."""

    def __init__(self, net, device=None, dtype=None, prefer="auto"):
        dev, dt, note = pick_device(prefer)
        self.device = device or dev
        self.dtype = dtype or dt
        self.note = note
        self.eps = 1e-12 if self.dtype == torch.float64 else 1e-7
        self.W = [torch.tensor(w, device=self.device, dtype=self.dtype) for w in net.W]
        self.b = [torch.tensor(v, device=self.device, dtype=self.dtype) for v in net.b]
        self.net = net
        self.L = net.L
        self.widths = net.widths

    def backward(self, k, C, pre_lb, pre_ub, status, x_lb, x_ub):
        B = x_lb.shape[0]
        A = C.unsqueeze(0).expand(B, *C.shape).clone()
        bias = A @ self.b[k - 1]
        A = A @ self.W[k - 1]
        for i in range(k - 1, 0, -1):
            sl, su, iu = _relax(pre_lb[i], pre_ub[i], status[i], self.eps)
            pos = torch.clamp(A, min=0)
            neg = torch.clamp(A, max=0)
            bias = bias + torch.einsum("bmw,bw->bm", neg, iu)
            A = pos * sl.unsqueeze(1) + neg * su.unsqueeze(1)
            bias = bias + A @ self.b[i - 1]
            A = A @ self.W[i - 1]
        centre = 0.5 * (x_lb + x_ub)
        radius = 0.5 * (x_ub - x_lb)
        return bias + torch.einsum("bmw,bw->bm", A, centre) \
            - torch.einsum("bmw,bw->bm", A.abs(), radius)

    def compute(self, x_lb, x_ub, status, C_spec, slack=0.0):
        """Full bound computation.  Returns (pre_lb, pre_ub, spec_lb).

        `spec_lb` already has `slack` subtracted.  On float32 you MUST pass a
        positive slack (see calibrate_slack) or results are not sound.
        """
        L = self.L
        pre_lb = [None] * (L + 1)
        pre_ub = [None] * (L + 1)
        for k in range(1, L):
            w = self.widths[k]
            eye = torch.eye(w, device=self.device, dtype=self.dtype)
            C = torch.cat([eye, -eye], dim=0)
            out = self.backward(k, C, pre_lb, pre_ub, status, x_lb, x_ub)
            lo, hi = out[:, :w], -out[:, w:]
            lo = torch.where(status[k] == ACTIVE, torch.clamp(lo, min=0), lo)
            hi = torch.where(status[k] == INACTIVE, torch.clamp(hi, max=0), hi)
            pre_lb[k], pre_ub[k] = lo, hi
        spec = self.backward(L, C_spec, pre_lb, pre_ub, status, x_lb, x_ub)
        return pre_lb, pre_ub, spec - slack

    def to_device(self, x_lb, x_ub, status, C_spec):
        t = lambda a, d=None: torch.tensor(a, device=self.device,
                                           dtype=d or self.dtype)
        st = [None] + [torch.tensor(status[i], device=self.device,
                                    dtype=torch.int8) for i in range(1, self.L)]
        return t(x_lb), t(x_ub), st, t(C_spec)


def calibrate_slack(net, n_trials=200, depth_scale=True, seed=0):
    """Measure, on THIS machine, how far float32 can overestimate the bound.

    Returns a slack to subtract from every float32 bound.  Do not reuse a
    number measured elsewhere: it depends on network depth, width, and the
    backend's accumulation order.
    """
    from crab.batched import compute_bounds_batch

    rng = np.random.default_rng(seed)
    tb32 = TorchBounds(net, dtype=torch.float32)
    worst = 0.0
    for _ in range(n_trials):
        B = 16
        x0 = rng.normal(size=(B, net.n_in))
        eps = rng.uniform(0.1, 0.5, (B, 1))
        st = [None] + [np.zeros((B, w), dtype=np.int8) for w in net.hidden_sizes()]
        C = np.eye(net.n_out)[:1] - np.eye(net.n_out)[1:2]
        ref = compute_bounds_batch(net, (x0 - eps).astype(np.float64),
                                   (x0 + eps).astype(np.float64), st, C)[2]
        a, b, s, c = tb32.to_device(x0 - eps, x0 + eps, st, C)
        got = tb32.compute(a, b, s, c)[2].cpu().numpy().astype(np.float64)
        worst = max(worst, float((got - ref).max()))
    margin = 10.0 * max(worst, 1e-7)          # order-of-magnitude safety factor
    if depth_scale:
        margin *= max(1.0, (net.L - 1) / 4.0)
    return margin
