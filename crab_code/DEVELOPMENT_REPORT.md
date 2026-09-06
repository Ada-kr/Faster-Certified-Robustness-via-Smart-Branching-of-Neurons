# CRAB — Development Phase Report

### What the implementation and experiments did to the design document

*Companion to the CRAB design note. Read that first; this document records what survived contact with data.*

---

## Summary

Five of the seven components proposed in the design note were tested and **four were refuted**. One survived and is strong. The most consequential finding was methodological rather than algorithmic: the first benchmark I built was structurally incapable of measuring branching quality at all, and every conclusion drawn from it was worthless. Rebuilding it turned *p* ≈ 0.6 null results into *p* < 10⁻¹⁶ effects.

| Component (design note §) | Verdict | Evidence |
|---|---|---|
| Pseudocost calibration (§3.4) | **Survives — strong** | −44.5% nodes at *p* = 2×10⁻¹⁶, and cheaper (1.66M → 1.22M passes) |
| Product rule (§3.1) | Negligible | No measurable effect in isolation |
| Layer normalisation (§3.2) | Negligible | Identical node counts to baseline |
| **F2 implied stabilisation (§3.3)** | **Refuted** | −2.5% nodes at *p* = 0.126 on deep nets, the regime built to favour it |
| Adaptive filter budget (§3.5) | **Refuted** | +7% nodes vs uniform at matched cost, *p* = 0.24 |
| **Cost normalisation (§3.6)** | **Refuted — harmful** | γ > 0 costs 10 instances and doubles nodes at every value tested |
| Submodular set selection (§3.7) | Not tested | Requires multi-neuron splitting, not implemented |
| Dual-mode falsification (§3.8) | Not tested | Deferred |

---

## 1. What was built

A self-contained branch-and-bound verifier in NumPy (~700 lines), because the container has no GPU, no PyTorch, one CPU and 3 GB of RAM.

- `crab/crown.py` — CROWN backward linear bound propagation with activation-split constraints, warm-started incremental recomputation (a split at layer *i* leaves layers < *i* untouched), infeasibility detection, and a PGD falsifier.
- `crab/bab.py` — priority-queue BaB loop with activation branching, input-space bisection fallback, and instrumentation for nodes, backward passes, filtering cost and depth.
- `crab/branching.py` — `RandomBranching`, `BaBSR`, `FSB`, `AdaptiveFSB`, and a `CRAB` class whose every component sits behind a config flag, so each ablation row is exactly one flag.
- `crab/benchmarks.py` — synthetic task generation, NumPy SGD training, and per-instance ε calibration.
- `crab/experiment.py`, `tests/test_bounds.py` — ablation harness and soundness audit.

**Scope limit, stated plainly.** These are node-count claims about decision quality on small synthetic networks. They are not wall-clock claims and not claims about real networks. Node count is the right primary metric here precisely because the backend is a single-threaded NumPy interpreter — timing would measure the backend, not the decisions.

---

## 2. A soundness bug in my own solver

BaBSR and FSB returned **contradictory verdicts on the same instance** during the first smoke test — one "verified", one "falsified". Since branching cannot affect soundness, this had to be a solver bug.

Cause: I declared `falsified` whenever a domain had no unstable neurons remaining. The reasoning was that the network is linear there, so the bound is exact. It is not. The bound minimises over the **input box**, whereas the domain is that box **intersected with the activation constraints**. The bound can therefore sit below the true minimum over the domain, and a negative value proves nothing.

Fix, in two parts:
1. `falsified` is now declared **only** from a forward pass that evaluates negative. A bound is never evidence of a counterexample.
2. A domain with no unstable neurons falls back to **input-space bisection** of the widest input dimension. This also makes the solver genuinely complete rather than silently incomplete.

The bound engine itself passed its audit unmodified: 103 (network, subdomain) pairs, minimum slack `empirical_min − lower_bound` = 0.0032, never negative. Splits were sampled randomly and the check used rejection sampling so the activation constraints actually held.

The lesson generalises: the contradiction was only visible because two heuristics were run on the same instances and compared. A single-heuristic evaluation would have shipped the bug.

---

## 3. The benchmark was the bottleneck

The first suite gave a clean null: uniform vs adaptive allocation, n = 228, **122 vs 122 solved, zero discordant pairs, *p* = 0.64**. That looks like a decisive refutation. It was not — it was uninformative, and checking why was the most valuable step in the whole phase.

Difficulty distribution of that suite:

| band | count | share of solved |
|---|---|---|
| unsolved (hopeless at cap) | 106 / 228 | — |
| trivial (≤ 2 nodes) | 77 | 63% |
| easy (3–10 nodes) | 40 | 33% |
| **middle band (11–50)** | **5** | **4%** |
| hard (> 50) | 0 | 0% |

Only 2% of instances were in the band where a branching decision can change the outcome. The design note had *predicted* that gains concentrate in a middle band; the ε-calibration then failed to produce one, and I did not check until a null result forced me to.

**Fix.** Per-instance bisection on ε, targeting instances that BaBSR solves in 12–150 nodes. 221 calibrated instances in 151 s. Every subsequent comparison became sharply significant.

This is worth stating as a general point: in verification benchmarking, an easy instance and a hopeless instance are equally uninformative, and a suite that is mostly either measures nothing regardless of sample size. Statistical power came from **instance composition**, not from *n*.

---

## 4. Results on the calibrated suite (n = 221)

### 4.1 Scoring, no filtering

| configuration | solved | nodes | cost (M passes) |
|---|---|---|---|
| BaBSR baseline | 221 | 6,641 | 1.66 |
| **+ pseudocost calibration** | 220 | **4,063** | **1.22** |
| + cost normalisation (γ=0.5) | 211 | 8,453 | 1.66 |

Pseudocost vs BaBSR: **−44.5% nodes, *p* = 2.0×10⁻¹⁶**, at **26% lower cost**. It dominates on both axes rather than trading one for the other, which is the property you want and rarely get.

Why it works, and why it is safe: it learns a *ratio* (realised gain / predicted gain) rather than an absolute gain, so the statistic is scale-free across tree depths; and hierarchical shrinkage neuron → layer → global means an unseen neuron inherits its layer's correction. When no signal exists it converges to 1 and the method degenerates exactly to BaBSR.

### 4.2 Filtering budget, and whether calibration stacks with it

| k | FSB nodes | FSB cost | + pseudocost nodes | + pseudocost cost | *p* |
|---|---|---|---|---|---|
| 3 | 2,385 | 1.83M | 2,258 | 1.66M | 2.3×10⁻⁴ |
| 5 | 1,843 | 2.04M | 1,800 | 1.93M | 8.9×10⁻² |
| 7 | 1,592 | 2.36M | 1,569 | 2.28M | 5.0×10⁻² |
| 10 | 1,120 | 2.47M | 1,099 | 2.42M | 5.0×10⁻² |

Pseudocost improves on FSB at **every** budget, on both nodes and cost — but the gain decays monotonically, from −5.3% nodes at k=3 to −1.9% at k=10.

The interpretation is clean and is the most useful thing learned in this phase: **calibration substitutes for filtering budget.** A larger shortlist re-ranks candidates by direct evaluation anyway, so shortlist quality matters less the more you can afford to evaluate. Calibration therefore pays most in the cheap regime — which is precisely the regime that matters for wall clock on a GPU verifier, where per-node cost is the binding constraint.

### 4.3 Refutations

**F2 implied stabilisation.** Inert on shallow nets. Diagnosis: nonzero for 38.5% of layer-1 candidates and **0% of last-hidden-layer candidates by construction**, since no downstream ReLU layer exists there to stabilise. It was functioning as a disguised "early layer" indicator. I fixed it (within-layer normalisation, so it breaks ties instead of shifting cross-layer ranking) and rebuilt the benchmark with five hidden layers so stabilisation could cascade. Result on 80 deep instances: 40 vs 39 solved, −2.5% nodes, **Wilcoxon *p* = 0.126**. Refuted in the regime designed to favour it.

**Adaptive filter budget.** Three measurements, converging on a null. Initially the gate fired on only 3.8% of nodes despite 52% having a small enough score margin — an absolute depth cutoff `d0 = 10` was blocking them, because these trees spend most of their time deeper than that. That was a real design error ("shallow decisions have more leverage" implemented as a hard skip rather than a reduced budget). After replacing it with a decayed budget, the matched-cost control settled it: `afsb-2/12` and `fsb-k10` cost within 1.5% of each other and adaptive was **7% worse** on nodes, *p* = 0.24. Budget *size* dominates; allocation strategy is noise.

**Cost normalisation.** Not mistuned — harmful at every positive value:

| γ | solved | nodes |
|---|---|---|
| 0.00 | 220 | 4,063 |
| 0.25 | 210 | 8,262 |
| 0.50 | 211 | 8,453 |
| 1.00 | 210 | 9,921 |

The error in §3.6 was conceptual. Early-layer splits cost more to recompute *and* are disproportionately more valuable in deep networks, because their effect cascades through more layers. Cost and value are **positively correlated**, so dividing by cost discards the correlation along with the cost. A cost model is only useful when cost is independent of value, and here it plainly is not.

---

## 5. Revised proposal

The contribution is now much narrower than the design note claimed, and correspondingly better supported.

**Keep:** online hierarchical pseudocost calibration as a multiplicative correction to the analytic branching score. Cheap, dominates on nodes and cost simultaneously, degrades gracefully to the baseline, no distribution shift because it is fit per instance. Most valuable at low filtering budgets.

**Drop:** F2, cost normalisation, adaptive budget allocation, and the framing of layer normalisation and the product rule as contributions.

**Reframe:** the §3.5 story was "spend effort where the decision is uncertain". The data says the useful version is "calibration and budget are substitutes — calibrate when you cannot afford to filter". That is a smaller claim and a defensible one.

**Honest framing for a paper.** Pseudocost and reliability branching are transplants from the MILP literature. The contribution here is the *adaptation* — ratio-valued corrections rather than absolute pseudocosts, hierarchical shrinkage for cold start in short searches, and the measured substitution curve against filtering budget. Claiming more than that would not survive review, and the F2 result is a good illustration of why: it was the most novel-sounding component and it did not work.

---

## 6. What to do next

1. **Port pseudocost calibration into α,β-CROWN** and re-test on VNN-COMP. Everything above is node counts on synthetic NumPy models; none of it is a wall-clock claim about real networks. This is the only step that can convert these results into a real finding.
2. **Re-run the substitution curve on GPU**, where filtering is batched and relatively far cheaper than here. The k at which calibration stops paying will move, and where it lands determines whether the method is useful or a curiosity.
3. **Adopt the ε-bisection calibration as standard practice** for any branching benchmark. It cost 151 s and was the difference between measuring nothing and measuring effects at *p* < 10⁻¹⁶.
4. **Test the two untested components** (submodular multi-neuron selection, dual-mode falsification) only after 1–2, and with the prior that the base rate for these ideas surviving is now measured at roughly one in five.

---

## Appendix: reproduction

```
python3 tests/test_bounds.py        # soundness audit of the bound engine
python3 -m crab.experiment          # ablation grid
```

Metrics reported: instances solved, total BaB nodes, backward passes (cost), Wilcoxon signed-rank on jointly solved instances, McNemar exact test on discordant solve/fail pairs.
