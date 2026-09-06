"""Soundness audit for the bound engine.

The contract: for every subdomain D and spec c, the returned lower bound must
satisfy  lb <= min_{x in D} c.f(x).  We check it against dense sampling,
including on subdomains produced by random activation splits (where sampling
is done by rejection so that the constraints actually hold).

A single violation invalidates every downstream experiment, so this runs first
and runs on a lot of random networks.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from crab.network import MLP
from crab.crown import compute_bounds, ACTIVE, INACTIVE, FREE, unstable_mask


def random_net(rng, widths):
    W = [rng.normal(0, 1.0 / np.sqrt(widths[i]), (widths[i + 1], widths[i]))
         for i in range(len(widths) - 1)]
    b = [rng.normal(0, 0.1, widths[i + 1]) for i in range(len(widths) - 1)]
    return MLP(W, b)


def sample_in_domain(net, x_lb, x_ub, status, rng, n=40000):
    """Rejection-sample points satisfying the activation constraints."""
    X = rng.uniform(x_lb, x_ub, (n, x_lb.shape[0]))
    h = X
    ok = np.ones(n, dtype=bool)
    for i in range(net.L - 1):
        pre = h @ net.W[i].T + net.b[i]
        st = status[i + 1]
        on, off = st == ACTIVE, st == INACTIVE
        if on.any():
            ok &= (pre[:, on] >= 0).all(1)
        if off.any():
            ok &= (pre[:, off] <= 0).all(1)
        h = np.maximum(pre, 0.0)
    return X[ok]


def run(n_nets=40, seed=0):
    rng = np.random.default_rng(seed)
    worst_slack = np.inf
    n_checked = 0
    for t in range(n_nets):
        widths = [rng.integers(2, 5), rng.integers(4, 9), rng.integers(4, 9), 3]
        widths = [int(w) for w in widths]
        net = random_net(rng, widths)
        c0 = rng.normal(size=(2, widths[-1]))
        x0 = rng.normal(size=widths[0])
        eps = float(rng.uniform(0.2, 1.2))
        x_lb, x_ub = x0 - eps, x0 + eps

        status = [None] + [np.zeros(w, dtype=np.int8) for w in widths[1:-1]]
        res = compute_bounds(net, x_lb, x_ub, status, c0)

        for trial in range(6):
            st = [None] + [s.copy() for s in status[1:]]
            if trial > 0:
                # random splits on currently unstable neurons
                for i in range(1, net.L):
                    um = unstable_mask(res.pre_lb[i], res.pre_ub[i], st[i])
                    idx = np.flatnonzero(um)
                    if len(idx):
                        pick = rng.choice(idx, size=min(len(idx), trial), replace=False)
                        st[i][pick] = rng.choice([ACTIVE, INACTIVE], size=len(pick))
            r = compute_bounds(net, x_lb, x_ub, st, c0)
            X = sample_in_domain(net, x_lb, x_ub, st, rng)
            if X.shape[0] < 50:
                continue
            emp = (net.forward(X) @ c0.T).min(0)
            slack = float((emp - r.spec_lb).min())
            n_checked += 1
            worst_slack = min(worst_slack, slack)
            if slack < -1e-7:
                print(f"UNSOUND net {t} trial {trial}: bound exceeds empirical min "
                      f"by {-slack:.3e}")
                print("  widths", widths, "eps", eps)
                return False
    print(f"checked {n_checked} (net, subdomain) pairs")
    print(f"minimum slack (empirical_min - lower_bound) = {worst_slack:.6f}")
    print("PASS: no bound ever exceeded the empirical minimum")
    return True


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
