"""Does float32 break soundness?

Apple's MPS backend does not support float64.  Any GPU port of this verifier
therefore runs the bound propagation in float32.  That is a soundness
question, not a speed question: if float32 rounding can push a lower bound
ABOVE the true minimum of the domain, the verifier can certify an unsafe
network, which is the one failure mode a verifier must never have.

This measures three things on identical networks and domains:
  1. how far float32 bounds drift from float64 bounds,
  2. whether the drift is ever in the unsafe direction (bound too high),
  3. how the drift scales with depth and with the number of applied splits,
     since error accumulates through the backward pass.

The output is the outward-rounding slack a float32 GPU port needs in order to
stay sound.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from crab.crown import compute_bounds, unstable_mask, ACTIVE, INACTIVE
from crab.network import MLP


def to_dtype(net, dt):
    return MLP([w.astype(dt) for w in net.W], [b.astype(dt) for b in net.b], net.name)


def random_net(rng, widths, dt=np.float64):
    W = [rng.normal(0, 1.0 / np.sqrt(widths[i]), (widths[i + 1], widths[i]))
         for i in range(len(widths) - 1)]
    b = [rng.normal(0, 0.1, widths[i + 1]) for i in range(len(widths) - 1)]
    return MLP([w.astype(dt) for w in W], [v.astype(dt) for v in b])


def run(seed=0):
    rng = np.random.default_rng(seed)
    by_depth = {}
    worst_unsafe = 0.0
    n = 0

    for depth in (2, 4, 6, 8):
        drifts, unsafe = [], []
        for _ in range(60):
            widths = [6] + [28] * depth + [4]
            net64 = random_net(rng, widths, np.float64)
            net32 = to_dtype(net64, np.float32)
            x0 = rng.normal(size=widths[0])
            eps = float(rng.uniform(0.15, 0.6))
            C = rng.normal(size=(3, widths[-1]))

            st = [None] + [np.zeros(w, dtype=np.int8) for w in widths[1:-1]]
            r64 = compute_bounds(net64, x0 - eps, x0 + eps, st, C)

            # apply a random batch of splits to emulate a deep BaB node,
            # where error has accumulated through many recomputations
            for i in range(1, net64.L):
                um = unstable_mask(r64.pre_lb[i], r64.pre_ub[i], st[i])
                idx = np.flatnonzero(um)
                if idx.size:
                    pick = rng.choice(idx, size=max(1, idx.size // 4), replace=False)
                    st[i][pick] = rng.choice([ACTIVE, INACTIVE], size=len(pick))

            a = compute_bounds(net64, x0 - eps, x0 + eps, st, C)
            b = compute_bounds(net32, (x0 - eps).astype(np.float32),
                               (x0 + eps).astype(np.float32), st,
                               C.astype(np.float32))
            d = b.spec_lb.astype(np.float64) - a.spec_lb   # >0 means f32 claims MORE
            drifts.append(np.abs(d).max())
            unsafe.append(d.max())
            worst_unsafe = max(worst_unsafe, float(d.max()))
            n += 1
        by_depth[depth] = (np.median(drifts), np.max(drifts),
                           np.mean(np.array(unsafe) > 0), np.max(unsafe))

    print("float32 vs float64 bound drift (negative drift is safe/conservative)")
    print(f"{'hidden layers':>14} {'median |drift|':>15} {'max |drift|':>13} "
          f"{'% too high':>11} {'worst too high':>15}")
    for d, (med, mx, frac, wu) in by_depth.items():
        print(f"{d:>14} {med:>15.2e} {mx:>13.2e} {100*frac:>10.1f}% {wu:>15.2e}")

    print(f"\nchecked {n} (network, subdomain) pairs")
    print(f"worst float32 overestimate of the bound: {worst_unsafe:.3e}")
    if worst_unsafe > 0:
        print("\nfloat32 CAN report a bound above the float64 value.")
        print(f"A sound float32 port must subtract a slack of at least "
              f"{worst_unsafe:.2e}\nfrom every bound before comparing against zero, "
              "and scale that slack\nwith depth.")
    else:
        print("\nNo overestimate observed, but absence of evidence at this scale is "
              "not\na soundness proof -- keep the slack.")
    return worst_unsafe


if __name__ == "__main__":
    run()
