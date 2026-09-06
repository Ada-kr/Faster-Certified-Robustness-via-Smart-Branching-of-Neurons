"""Benchmark suite.

No internet access in this container, so the models are trained on synthetic
tasks rather than MNIST/CIFAR.  What matters for a branching study is the
*structure* of the search problem -- number of unstable neurons, depth, and
the distribution of instance difficulty -- not the semantics of the data.
The instance-difficulty calibration below is the important part: gains from
branching concentrate in a middle band, so a benchmark that is all-easy or
all-hopeless cannot measure anything (design note, section 7).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .crown import compute_bounds, pgd_attack
from .network import train_mlp


@dataclass
class Instance:
    net: object
    x0: np.ndarray
    eps: float
    label: int
    C_spec: np.ndarray
    family: str
    root_lb: float
    n_unstable: int

    @property
    def x_lb(self):
        return self.x0 - self.eps

    @property
    def x_ub(self):
        return self.x0 + self.eps


def gaussian_task(rng, n_in, n_class, n=1200):
    centres = rng.normal(0, 1.6, (n_class, n_in))
    y = rng.integers(0, n_class, n)
    X = centres[y] + rng.normal(0, 0.75, (n, n_in))
    return X, y


def spec_matrix(net, x0, label):
    """Rows  e_label - e_other  -- all must be positive for robustness."""
    n_out = net.n_out
    rows = [np.eye(n_out)[label] - np.eye(n_out)[j]
            for j in range(n_out) if j != label]
    return np.stack(rows)


def _root_stats(net, x0, eps, C):
    status = [None] + [np.zeros(w, dtype=np.int8) for w in net.hidden_sizes()]
    res = compute_bounds(net, x0 - eps, x0 + eps, status, C)
    n_uns = sum(int(((res.pre_lb[i] < 0) & (res.pre_ub[i] > 0)).sum())
                for i in range(1, net.L))
    return float(res.spec_lb.min()), n_uns


def build_suite(seed=0, verbose=True):
    """Three model families spanning the easy / medium / hard regimes."""
    rng = np.random.default_rng(seed)
    families = [
        ("small",  6, [20, 20],     3),
        ("medium", 8, [30, 30],     4),
        ("wide",   8, [45, 35, 25], 4),
        ("deep",   8, [24, 24, 24, 24, 24], 4),
    ]
    instances = []
    for fam, n_in, hidden, n_class in families:
        X, y = gaussian_task(rng, n_in, n_class)
        net = train_mlp(rng, n_in, hidden, n_class, X, y, name=fam)
        acc = (net.forward(X).argmax(1) == y).mean()
        if verbose:
            print(f"  {fam:7s} in={n_in} hidden={hidden} acc={acc:.3f}")

        pool = np.flatnonzero(net.forward(X).argmax(1) == y)
        picks = rng.choice(pool, size=min(40, pool.size), replace=False)
        kept = 0
        for p in picks:
            x0, lab = X[p], int(y[p])
            C = spec_matrix(net, x0, lab)
            # calibrate eps: find the smallest eps whose root bound already
            # fails, then push slightly past it -- this is the band where
            # branching decisions actually determine the outcome
            eps = None
            for e in np.linspace(0.02, 0.55, 20):
                lb, _ = _root_stats(net, x0, e, C)
                if lb <= 0:
                    eps = e
                    break
            if eps is None:
                continue
            for mult in (1.0, 1.35):
                e = eps * mult
                lb, nu = _root_stats(net, x0, e, C)
                if lb > 0 or nu < 3:
                    continue
                instances.append(Instance(net, x0, float(e), lab, C, fam, lb, nu))
                kept += 1
        if verbose:
            print(f"          -> {kept} non-trivial instances")
    return instances


def calibrate_band(instances, lo=12, hi=150, node_cap=400, timeout=1.2, verbose=True):
    """Rescale each instance's epsilon until BaBSR solves it in [lo, hi] nodes.

    Instances outside that band cannot discriminate between branching
    heuristics: a 2-node instance is solved whatever you pick, and a hopeless
    one is unsolved whatever you pick.  Skipping this step was the single
    biggest methodological error of the development phase -- see the report.
    """
    from .bab import verify
    from .branching import BaBSR

    out = []
    for ins in instances:
        a, b = ins.eps * 0.6, ins.eps * 2.6
        for _ in range(7):
            mid = 0.5 * (a + b)
            r = verify(ins.net, ins.x0 - mid, ins.x0 + mid, ins.C_spec, BaBSR(),
                       max_nodes=node_cap, timeout=timeout, seed=0)
            if r.verdict == "verified" and r.nodes < lo:
                a = mid                       # too easy -> widen the ball
            elif r.verdict in ("timeout", "unknown") or r.nodes > hi:
                b = mid                       # too hard -> shrink it
            elif r.verdict == "falsified":
                b = mid                       # falsified instances terminate on
                                              # attack luck, not branching quality
            else:
                rl, nu = _root_stats(ins.net, ins.x0, mid, ins.C_spec)
                out.append(Instance(ins.net, ins.x0, float(mid), ins.label,
                                    ins.C_spec, ins.family + "-cal", rl, nu))
                break
    if verbose:
        print(f"  calibrated {len(out)}/{len(instances)} into the "
              f"[{lo},{hi}] node band")
    return out
