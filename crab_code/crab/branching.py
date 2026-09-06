"""Branching heuristics.

Every heuristic implements `select(ctx) -> (layer, index)`.  All of them are
sound by construction: the choice only affects which case split is performed,
never the validity of any bound.

CRAB is implemented as a single class whose components are individually
switchable, so each row of the ablation table corresponds to one flag.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .crown import compute_bounds, unstable_mask, ACTIVE, INACTIVE

EPS = 1e-12


@dataclass
class Candidates:
    layer: np.ndarray      # (n,) int
    index: np.ndarray      # (n,) int
    lb: np.ndarray         # (n,) pre-activation lower bound
    ub: np.ndarray         # (n,) pre-activation upper bound
    lam: np.ndarray        # (n,) |backward sensitivity| on the worst spec row

    def __len__(self):
        return self.layer.shape[0]


def gather_candidates(net, res, status, layer_norm=False):
    """Collect all currently unstable neurons with their sensitivities."""
    worst = int(np.argmin(res.spec_lb))
    ls, ix, lbs, ubs, lams = [], [], [], [], []
    for i in range(1, net.L):
        um = unstable_mask(res.pre_lb[i], res.pre_ub[i], status[i])
        idx = np.flatnonzero(um)
        if idx.size == 0:
            continue
        lam = np.abs(res.lambdas[i][worst, idx])
        if layer_norm:
            # fixes the systematic cross-layer scale mismatch (design note 1.3)
            scale = np.abs(res.lambdas[i][worst]).mean() + EPS
            lam = lam / scale
        ls.append(np.full(idx.size, i))
        ix.append(idx)
        lbs.append(res.pre_lb[i][idx])
        ubs.append(res.pre_ub[i][idx])
        lams.append(lam)
    if not ls:
        return None
    return Candidates(np.concatenate(ls), np.concatenate(ix),
                      np.concatenate(lbs), np.concatenate(ubs),
                      np.concatenate(lams))


def relaxation_gap(lb, ub):
    """Maximum vertical gap of the triangle relaxation: -u*l/(u-l)."""
    return -ub * lb / np.maximum(ub - lb, EPS)


def f1_scores(cand):
    """Analytic one-step estimate: sensitivity x relaxation gap."""
    return cand.lam * relaxation_gap(cand.lb, cand.ub)


def branch_share_estimates(cand, base):
    """Asymmetric per-branch gain estimates.

    Forcing a neuron inactive discards the share u/(u-l) of its pre-activation
    range; forcing it active discards -l/(u-l).  Gain is taken proportional to
    the share removed.  This makes the two branch estimates differ, which is
    what gives the product rule something to discriminate on -- a symmetric
    estimate would make the product rule equivalent to squaring F1.
    """
    d = np.maximum(cand.ub - cand.lb, EPS)
    return base * (cand.ub / d), base * (-cand.lb / d)


def implied_stabilisation(net, res, status, cand):
    """Feature F2: how many downstream neurons would this split stabilise.

    Interval-arithmetic estimate, one layer deep.  Forcing neuron j of layer i
    inactive removes its post-activation range [0, u_j] from the input of
    layer i+1, shrinking every downstream radius by |W[i+1][m,j]| * u_j/2.
    A downstream neuron stabilises when its remaining radius no longer spans
    zero, i.e. radius - reduction < |centre|.

    This is deliberately a loose estimate: it is a ranking feature, not a
    bound.  Using exact propagation here would cost as much as strong
    branching and defeat the purpose.
    """
    out = np.zeros(len(cand), dtype=np.float64)
    for i in np.unique(cand.layer):
        i = int(i)
        if i + 1 > net.L - 1:
            continue                       # no unstable neurons past the last hidden layer
        sel = cand.layer == i
        j = cand.index[sel]
        lo, hi = res.pre_lb[i + 1], res.pre_ub[i + 1]
        rad = 0.5 * (hi - lo)
        ctr = 0.5 * (hi + lo)
        um = unstable_mask(lo, hi, status[i + 1])
        if not um.any():
            continue
        # (n_downstream_unstable, n_candidates_in_layer_i)
        red = net.absW[i][np.ix_(um, j)] * (0.5 * res.pre_ub[i][j])[None, :]
        stabilised = (rad[um][:, None] - red) < np.abs(ctr[um])[:, None]
        out[sel] = stabilised.sum(0)
    return out


class Brancher:
    name = "base"

    def reset(self, net, C_spec):
        pass

    def observe(self, key, predicted, realised):
        pass

    def select(self, ctx):
        raise NotImplementedError


class RandomBranching(Brancher):
    name = "random"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def select(self, ctx):
        cand = ctx["cand"]
        i = int(self.rng.integers(len(cand)))
        return int(cand.layer[i]), int(cand.index[i]), None


class BaBSR(Brancher):
    """Sensitivity x relaxation-gap score, no filtering.  The classic baseline."""
    name = "babsr"

    def select(self, ctx):
        cand = ctx["cand"]
        s = f1_scores(cand)
        i = int(np.argmax(s))
        return int(cand.layer[i]), int(cand.index[i]), float(s[i])


class FSB(Brancher):
    """Filtered smart branching: shortlist by F1, then evaluate for real.

    Fixed budget k at every node -- this is precisely the uniform effort
    allocation that CRAB replaces with an adaptive one.
    """
    name = "fsb"

    def __init__(self, k=6):
        self.k = k

    def select(self, ctx):
        cand = ctx["cand"]
        s = f1_scores(cand)
        top = np.argsort(-s)[: self.k]
        best, best_key = -np.inf, None
        for t in top:
            lo, hi = _evaluate_split(ctx, int(cand.layer[t]), int(cand.index[t]))
            sc = max(lo, EPS) * max(hi, EPS)          # product rule
            if sc > best:
                best, best_key = sc, (int(cand.layer[t]), int(cand.index[t]))
        return best_key[0], best_key[1], best


def _evaluate_split(ctx, layer, idx):
    """Actually compute both child bounds.  Returns the two realised gains."""
    net, dom, C = ctx["net"], ctx["domain"], ctx["C_spec"]
    parent = dom.lb
    gains = []
    for val in (INACTIVE, ACTIVE):
        st = [None] + [s.copy() for s in dom.status[1:]]
        st[layer][idx] = val
        r = compute_bounds(net, dom.x_lb, dom.x_ub, st, C,
                           pre_lb=dom.pre_lb, pre_ub=dom.pre_ub,
                           from_layer=layer, collect=False)
        ctx["stats"]["filter_passes"] += r.n_passes
        gains.append(np.inf if r.infeasible else float(r.spec_lb.min()) - parent)
    return gains[0], gains[1]


@dataclass
class CrabConfig:
    layer_norm: bool = True
    product_rule: bool = True
    use_f2: bool = True
    f2_weight: float = 0.5           # multiplicative strength of the F2 term
    pseudocost: bool = True
    pc_kappa: float = 4.0            # shrinkage strength (neuron -> layer -> global)
    pc_reliable: int = 4             # observations before a neuron is trusted
    adaptive_filter: bool = True
    margin_tau: float = 0.25         # skip filtering when top-1 dominates by this
    depth_d0: int = 10               # shallow nodes get the full budget
    filter_k: int = 6                # budget at shallow depth
    filter_k_deep: int = 3           # reduced budget below d0 (never zero)
    cost_gamma: float = 0.5          # score / cost^gamma
    name: str = "crab"


class CRAB(Brancher):
    def __init__(self, cfg: CrabConfig = None):
        self.cfg = cfg or CrabConfig()
        self.name = self.cfg.name

    def reset(self, net, C_spec):
        self.pc = {}                                  # (layer, idx) -> [sum_ratio, n]
        self.pc_layer = {}                            # layer -> [sum_ratio, n]
        self.pc_global = [0.0, 0]
        # analytic cost model: passes needed to recompute after splitting layer i
        self.cost = {}
        for i in range(1, net.L):
            self.cost[i] = sum(2 * net.widths[k] for k in range(i, net.L)) \
                + C_spec.shape[0]
        m = min(self.cost.values())
        self.cost = {k: v / m for k, v in self.cost.items()}

    # ---- pseudocost with hierarchical empirical-Bayes shrinkage ----
    def _rho(self, layers, idxs):
        cfg = self.cfg
        g = self.pc_global[0] / self.pc_global[1] if self.pc_global[1] else 1.0
        out = np.empty(len(layers))
        n_obs = np.zeros(len(layers))
        for t, (l, j) in enumerate(zip(layers, idxs)):
            ls, ln = self.pc_layer.get(int(l), (0.0, 0))
            lay = ls / ln if ln else g
            s, n = self.pc.get((int(l), int(j)), (0.0, 0))
            out[t] = (s + cfg.pc_kappa * lay) / (n + cfg.pc_kappa)
            n_obs[t] = n
        return out, n_obs

    def observe(self, key, predicted, realised):
        if predicted is None or predicted <= EPS or not np.isfinite(realised):
            return
        ratio = float(np.clip(realised / predicted, 0.0, 20.0))
        s, n = self.pc.get(key, (0.0, 0))
        self.pc[key] = (s + ratio, n + 1)
        s, n = self.pc_layer.get(key[0], (0.0, 0))
        self.pc_layer[key[0]] = (s + ratio, n + 1)
        self.pc_global[0] += ratio
        self.pc_global[1] += 1

    def select(self, ctx):
        cfg = self.cfg
        cand = ctx["cand"]
        base = f1_scores(cand)

        if cfg.use_f2:
            f2 = implied_stabilisation(ctx["net"], ctx["res"], ctx["domain"].status, cand)
            # F2 is normalised WITHIN each layer.  It is undefined for the last
            # hidden layer (no downstream ReLU exists to stabilise), so a raw
            # multiplicative term would silently act as an "early layer" flag
            # and distort the cross-layer ranking -- measured, not assumed.
            mult = np.ones_like(base)
            for L in np.unique(cand.layer):
                m = cand.layer == L
                hi = f2[m].max()
                if hi > 0:
                    mult[m] = 1.0 + cfg.f2_weight * (f2[m] / hi)
            base = base * mult

        if cfg.product_rule:
            d_off, d_on = branch_share_estimates(cand, base)
        else:
            d_off = d_on = base

        if cfg.pseudocost:
            rho, n_obs = self._rho(cand.layer, cand.index)
            d_off, d_on = d_off * rho, d_on * rho
        else:
            n_obs = np.zeros(len(cand))

        score = (np.maximum(d_off, EPS) * np.maximum(d_on, EPS)) if cfg.product_rule \
            else np.maximum(d_off, EPS)

        if cfg.cost_gamma > 0:
            c = np.array([self.cost[int(l)] for l in cand.layer])
            score = score / np.power(c, cfg.cost_gamma)

        order = np.argsort(-score)
        best = int(order[0])
        pred = float(d_off[best])

        if cfg.adaptive_filter and len(cand) > 1:
            s1, s2 = score[order[0]], score[order[1]]
            margin = (s1 - s2) / max(abs(s1), EPS)
            unreliable = n_obs[order[0]] < cfg.pc_reliable
            if margin <= cfg.margin_tau and unreliable:
                ctx["stats"]["filter_calls"] += 1
                # depth reduces the budget but never switches filtering off:
                # measurement showed a hard depth cutoff disables filtering on
                # the majority of nodes, because these trees are deep
                k = cfg.filter_k if ctx["depth"] <= cfg.depth_d0 else cfg.filter_k_deep
                bestv = -np.inf
                for t in order[:k]:
                    lo, hi = _evaluate_split(ctx, int(cand.layer[t]), int(cand.index[t]))
                    sc = max(lo, EPS) * max(hi, EPS)
                    if sc > bestv:
                        bestv, best, pred = sc, int(t), max(lo, EPS)

        return int(cand.layer[best]), int(cand.index[best]), pred


BRANCHERS = {
    "random": lambda: RandomBranching(),
    "babsr": lambda: BaBSR(),
    "fsb": lambda: FSB(k=6),
    "crab": lambda: CRAB(CrabConfig()),
}


class AdaptiveFSB(Brancher):
    """Non-uniform allocation of a fixed evaluation budget.

    The measured result that motivates this: shortlist *ordering* barely
    matters once filtering is on, but the *number* of candidates evaluated
    matters a lot.  So the question is not "filter or not" but "where to spend
    a fixed total budget".  Nodes where the top-1 F1 score clearly dominates
    are easy decisions and get k_lo; nodes where the top scores are bunched
    are genuinely uncertain and get k_hi.

    `evals` records candidates actually evaluated so the comparison against
    uniform FSB can be made at equal cost rather than equal k.
    """
    def __init__(self, k_lo=2, k_hi=12, tau=0.25):
        self.k_lo, self.k_hi, self.tau = k_lo, k_hi, tau
        self.name = f"afsb-{k_lo}/{k_hi}"

    def reset(self, net, C_spec):
        self.evals = 0
        self.calls = 0

    def select(self, ctx):
        cand = ctx["cand"]
        s = f1_scores(cand)
        order = np.argsort(-s)
        if len(cand) > 1:
            s1, s2 = s[order[0]], s[order[1]]
            margin = (s1 - s2) / max(abs(s1), EPS)
            k = self.k_hi if margin <= self.tau else self.k_lo
        else:
            k = 1
        k = min(k, len(cand))
        self.evals += k
        self.calls += 1
        best, best_key = -np.inf, None
        for t in order[:k]:
            lo, hi = _evaluate_split(ctx, int(cand.layer[t]), int(cand.index[t]))
            sc = max(lo, EPS) * max(hi, EPS)
            if sc > best:
                best, best_key = sc, (int(cand.layer[t]), int(cand.index[t]))
        return best_key[0], best_key[1], best
