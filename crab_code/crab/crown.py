"""CROWN-style backward linear bound propagation with activation splits.

Soundness contract
------------------
`backward` returns a valid LOWER bound on  min_{x in D} C @ zhat^(k)(x)
where D is the input box intersected with the activation constraints in
`status`.  Every relaxation used below is an over-approximation of the ReLU
graph on the interval it is applied to, so the returned value is sound
regardless of how branching decisions were made.  Branching heuristics can
therefore never make a result unsound -- only slower.

Split encoding
--------------
`status[i]` is an int8 array over layer i's pre-activations:
    +1  neuron forced active   (domain restricted to zhat >= 0)
    -1  neuron forced inactive (domain restricted to zhat <= 0)
     0  free
A split additionally tightens the stored pre-activation interval
(lb := max(lb,0) or ub := min(ub,0)).  That is sound because the subdomain
only contains points satisfying the constraint.  It is also the mechanism
through which a split propagates to *other* neurons and can stabilise them --
the effect that feature F2 tries to predict.
"""
from __future__ import annotations

import numpy as np

ACTIVE, FREE, INACTIVE = 1, 0, -1


def relu_relax(lb, ub, status):
    """Per-neuron linear relaxation coefficients.

    Returns (slope_lo, slope_up, icpt_up).  The relaxation is
        slope_lo * zhat        <=  relu(zhat)  <=  slope_up * zhat + icpt_up
    """
    n = lb.shape[0]
    slope_lo = np.zeros(n)
    slope_up = np.zeros(n)
    icpt_up = np.zeros(n)

    forced_on = (status == ACTIVE) | (lb >= 0)
    forced_off = (status == INACTIVE) | (ub <= 0)
    unstable = ~(forced_on | forced_off)

    slope_lo[forced_on] = 1.0
    slope_up[forced_on] = 1.0
    # forced_off keeps zeros

    if unstable.any():
        u, l = ub[unstable], lb[unstable]
        d = np.maximum(u - l, 1e-12)
        slope_up[unstable] = u / d
        icpt_up[unstable] = -u * l / d
        # adaptive CROWN lower slope: pick the one with smaller relaxation area
        slope_lo[unstable] = (u > -l).astype(np.float64)
    return slope_lo, slope_up, icpt_up


def unstable_mask(lb, ub, status):
    return (status == FREE) & (lb < 0) & (ub > 0)


def backward(net, k, C, pre_lb, pre_ub, status, x_lb, x_ub, collect=False):
    """Lower bound on C @ zhat^(k) over the subdomain.

    C: (m, width_k).  Returns (lb (m,), lambdas dict or None, A_input (m,n_in)).
    `lambdas[i]` is the coefficient matrix on zhat^(i) -- the backward
    sensitivity used by BaBSR-style scores.
    """
    A = np.asarray(C, dtype=np.float64)
    bias = A @ net.b[k - 1]
    A = A @ net.W[k - 1]                      # coefficients on z^(k-1)
    lambdas = {} if collect else None

    for i in range(k - 1, 0, -1):
        sl, su, iu = relu_relax(pre_lb[i], pre_ub[i], status[i])
        pos = np.maximum(A, 0.0)
        neg = np.minimum(A, 0.0)
        bias = bias + neg @ iu                # only the upper relaxation has an intercept
        A = pos * sl + neg * su               # coefficients on zhat^(i)
        if collect:
            lambdas[i] = A.copy()
        bias = bias + A @ net.b[i - 1]
        A = A @ net.W[i - 1]                  # coefficients on z^(i-1)

    center = 0.5 * (x_lb + x_ub)
    radius = 0.5 * (x_ub - x_lb)
    lb = bias + A @ center - np.abs(A) @ radius
    return lb, lambdas, A


class BoundResult:
    __slots__ = ("pre_lb", "pre_ub", "spec_lb", "lambdas", "infeasible", "n_passes")

    def __init__(self, pre_lb, pre_ub, spec_lb, lambdas, infeasible, n_passes):
        self.pre_lb = pre_lb
        self.pre_ub = pre_ub
        self.spec_lb = spec_lb
        self.lambdas = lambdas
        self.infeasible = infeasible
        self.n_passes = n_passes


def _clamp(lb, ub, st):
    lb = np.where(st == ACTIVE, np.maximum(lb, 0.0), lb)
    ub = np.where(st == INACTIVE, np.minimum(ub, 0.0), ub)
    return lb, ub


def compute_bounds(net, x_lb, x_ub, status, C_spec,
                   pre_lb=None, pre_ub=None, from_layer=1, collect=True):
    """Full or incremental bound computation.

    `from_layer` enables warm starting: when a child domain is created by
    splitting a neuron in layer i, layers < i are unaffected, so their
    intermediate bounds are inherited verbatim.  This is the single biggest
    constant-factor saving in the whole solver and it is also why the
    per-layer branching cost differs by layer (see the cost model, section 3.6
    of the design note).
    """
    L = net.L
    pre_lb = [None] * (L + 1) if pre_lb is None else list(pre_lb)
    pre_ub = [None] * (L + 1) if pre_ub is None else list(pre_ub)
    n_passes = 0
    infeasible = False

    # layers before `from_layer` keep their inherited bounds, but a newly
    # applied split on those layers still tightens the stored interval
    for i in range(1, min(from_layer, L)):
        if pre_lb[i] is not None:
            pre_lb[i], pre_ub[i] = _clamp(pre_lb[i], pre_ub[i], status[i])

    for k in range(from_layer, L):
        w = net.widths[k]
        eye = np.eye(w)
        lo, _, _ = backward(net, k, eye, pre_lb, pre_ub, status, x_lb, x_ub)
        hi, _, _ = backward(net, k, -eye, pre_lb, pre_ub, status, x_lb, x_ub)
        hi = -hi
        n_passes += 2 * w
        # intersect with anything inherited (bounds can only get tighter)
        if pre_lb[k] is not None:
            lo = np.maximum(lo, pre_lb[k])
            hi = np.minimum(hi, pre_ub[k])
        lo, hi = _clamp(lo, hi, status[k])
        if np.any(lo > hi + 1e-9):
            infeasible = True
        pre_lb[k], pre_ub[k] = lo, hi

    spec_lb, lambdas, _ = backward(net, L, C_spec, pre_lb, pre_ub,
                                   status, x_lb, x_ub, collect=collect)
    n_passes += C_spec.shape[0]
    return BoundResult(pre_lb, pre_ub, spec_lb, lambdas, infeasible, n_passes)


def pgd_attack(net, x_lb, x_ub, C_spec, rng, n_restart=8, n_step=30):
    """Cheap falsification.  Returns (best_value, best_x).

    Minimises min_j (C_spec[j] . f(x)).  A negative value is a genuine
    counterexample (verified by a forward pass, so no soundness risk).
    """
    best_v, best_x = np.inf, None
    center = 0.5 * (x_lb + x_ub)
    radius = 0.5 * (x_ub - x_lb)
    for j in range(C_spec.shape[0]):
        c = C_spec[j]
        x = center + radius * rng.uniform(-1, 1, (n_restart, x_lb.shape[0]))
        step = radius * 0.35
        for _ in range(n_step):
            v, g = net.forward_with_grad(x, c)
            x = np.clip(x - step * np.sign(g), x_lb, x_ub)
            step *= 0.92
        v = net.forward(x) @ c
        i = int(np.argmin(v))
        if v[i] < best_v:
            best_v, best_x = float(v[i]), x[i].copy()
    return best_v, best_x
