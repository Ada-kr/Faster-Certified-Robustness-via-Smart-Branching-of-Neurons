"""Branch-and-bound verification loop.

Two correctness points worth stating explicitly, both learned the hard way:

1.  "falsified" is ONLY ever declared from a forward pass that actually
    evaluates negative.  A negative lower bound is never evidence of a
    counterexample, not even on a domain where every ReLU is stable: the
    bound minimises over the input box, whereas the domain is that box
    intersected with the activation constraints, so the bound can sit below
    the true minimum of the domain.
2.  When a domain has no unstable neurons left there is nothing to split in
    activation space, but the domain may still be unresolved.  We fall back
    to input-space branching (bisect the widest input dimension).  Since the
    unstable set strictly shrinks along any path and input boxes shrink
    geometrically, the procedure terminates and is complete up to the input
    split depth limit.

Instrumented to report the metrics that matter for a branching study: node
count (decision quality, backend-independent), backward passes
(implementation-independent work), and wall clock.
"""
from __future__ import annotations

import heapq
import time
from dataclasses import dataclass

import numpy as np

from .branching import gather_candidates
from .crown import ACTIVE, INACTIVE, compute_bounds, pgd_attack


@dataclass
class Domain:
    status: list
    pre_lb: list
    pre_ub: list
    x_lb: np.ndarray
    x_ub: np.ndarray
    lb: float
    depth: int
    in_depth: int = 0
    predicted: float = None


@dataclass
class Result:
    verdict: str
    nodes: int = 0
    input_splits: int = 0
    passes: int = 0
    filter_passes: int = 0
    filter_calls: int = 0
    seconds: float = 0.0
    depth_max: int = 0
    global_lb: float = -np.inf


MAX_INPUT_DEPTH = 12


def verify(net, x_lb, x_ub, C_spec, brancher, max_nodes=3000, timeout=60.0,
           attack_every=25, seed=0, layer_norm=True):
    rng = np.random.default_rng(seed)
    t0 = time.time()
    stats = {"filter_passes": 0, "filter_calls": 0}
    brancher.reset(net, C_spec)

    status = [None] + [np.zeros(w, dtype=np.int8) for w in net.hidden_sizes()]
    res = compute_bounds(net, x_lb, x_ub, status, C_spec)
    total_passes = res.n_passes
    root_lb = float(res.spec_lb.min())
    r = Result(verdict="timeout", global_lb=root_lb)

    if root_lb > 0:
        r.verdict, r.nodes, r.passes, r.seconds = "verified", 1, total_passes, time.time() - t0
        return r

    adv, _ = pgd_attack(net, x_lb, x_ub, C_spec, rng)
    if adv < 0:
        r.verdict, r.nodes, r.passes, r.seconds = "falsified", 1, total_passes, time.time() - t0
        return r

    root = Domain(status, res.pre_lb, res.pre_ub, x_lb, x_ub, root_lb, 0)
    counter = 0
    queue = [(root_lb, counter, root, res)]
    nodes = 0
    finished = True

    while queue:
        if nodes >= max_nodes or time.time() - t0 > timeout:
            r.verdict = "timeout"
            finished = False
            break
        lb, _, dom, dres = heapq.heappop(queue)
        r.global_lb = lb

        cand = gather_candidates(net, dres, dom.status, layer_norm=layer_norm)

        children = []
        if cand is None:
            # no activation split available -> bisect the widest input dimension
            if dom.in_depth >= MAX_INPUT_DEPTH:
                r.verdict = "unknown"
                finished = False
                break
            d = int(np.argmax(dom.x_ub - dom.x_lb))
            mid = 0.5 * (dom.x_lb[d] + dom.x_ub[d])
            r.input_splits += 1
            for lo_hi in (0, 1):
                nl, nu = dom.x_lb.copy(), dom.x_ub.copy()
                if lo_hi == 0:
                    nu[d] = mid
                else:
                    nl[d] = mid
                st = [None] + [s.copy() for s in dom.status[1:]]
                cres = compute_bounds(net, nl, nu, st, C_spec)
                total_passes += cres.n_passes
                children.append((st, cres, nl, nu, dom.depth, dom.in_depth + 1, None))
        else:
            ctx = {"net": net, "domain": dom, "res": dres, "cand": cand,
                   "C_spec": C_spec, "depth": dom.depth, "stats": stats}
            layer, idx, predicted = brancher.select(ctx)
            nodes += 1
            r.depth_max = max(r.depth_max, dom.depth + 1)
            for val in (INACTIVE, ACTIVE):
                st = [None] + [s.copy() for s in dom.status[1:]]
                st[layer][idx] = val
                cres = compute_bounds(net, dom.x_lb, dom.x_ub, st, C_spec,
                                      pre_lb=dom.pre_lb, pre_ub=dom.pre_ub,
                                      from_layer=layer)
                total_passes += cres.n_passes
                if cres.infeasible:
                    continue
                brancher.observe((layer, idx), predicted,
                                 float(cres.spec_lb.min()) - dom.lb)
                children.append((st, cres, dom.x_lb, dom.x_ub,
                                 dom.depth + 1, dom.in_depth, predicted))

        for st, cres, nl, nu, dep, idep, pred in children:
            if cres.infeasible:
                continue
            clb = float(cres.spec_lb.min())
            if clb > 0:
                continue
            counter += 1
            heapq.heappush(queue, (clb, counter,
                                   Domain(st, cres.pre_lb, cres.pre_ub, nl, nu,
                                          clb, dep, idep, pred), cres))

        if attack_every and nodes and nodes % attack_every == 0:
            adv, _ = pgd_attack(net, x_lb, x_ub, C_spec, rng, n_restart=4, n_step=20)
            if adv < 0:
                r.verdict = "falsified"
                finished = False
                break

    if finished:
        r.verdict = "verified"
    r.nodes = nodes
    r.passes = total_passes + stats["filter_passes"]
    r.filter_passes = stats["filter_passes"]
    r.filter_calls = stats["filter_calls"]
    r.seconds = time.time() - t0
    return r
