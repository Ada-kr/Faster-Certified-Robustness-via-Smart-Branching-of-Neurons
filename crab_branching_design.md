# Cost-Aware Reliability-Adaptive Branching (CRAB)

### A design document for faster certified robustness via smarter neuron branching

*Technical design note — treat "CRAB" as a working handle, not a claim to a brand.*

---

## 0. How I am framing the problem

Before proposing anything, it is worth being precise about what is actually being optimized, because most weak proposals in this area optimize the wrong quantity.

Branch-and-bound (BaB) verification of a ReLU network is a **sequential decision process**. At each node of the search tree we hold a subdomain `D` — the input region plus a set of already-fixed ReLU activation patterns. We compute a sound lower bound `l(D)` on the verification objective (typically `min f_true(x) - f_other(x)` over the domain). If `l(D) > 0`, the domain is pruned. If an attack finds `x ∈ D` with a negative objective, the property is falsified and we stop globally. Otherwise we choose an unstable neuron `j` and replace `D` with two children, `D ∧ (ẑ_j ≤ 0)` and `D ∧ (ẑ_j ≥ 0)`.

The only free choice in this loop is **which `j` to pick** (and, secondarily, which domain to expand next). Everything else — bound tightness, GPU throughput, attack strength — is fixed infrastructure.

The objective is **not** "maximize the bound improvement at this node". It is:

> minimize the expected wall-clock time until every leaf is either pruned or falsified

This distinction drives almost every design decision below. Three consequences follow immediately:

1. **Bound improvement is a proxy, not the goal.** A split that improves the bound by 0.3 but costs 4× more to evaluate is worse than one improving it by 0.15 cheaply.
2. **The value of a decision is depth-dependent.** A mistake at the root can cost `2^k` nodes. A mistake at depth 25 costs a handful. Effort spent on the decision should be allocated accordingly — current heuristics spend the *same* effort at every depth, which is an obvious misallocation.
3. **Soundness is decoupled from the heuristic.** Any exhaustive, disjoint case split is sound regardless of how `j` is chosen. This gives enormous design freedom: we can be as aggressive, learned, or stochastic as we like without ever risking a false "verified". Only *completeness* imposes a constraint, and a weak one (see §3.9).

---

## 1. Diagnosis: where the current heuristics actually lose

A solution should be motivated by specific, nameable failure modes rather than a general wish for "better scores". Here are the ones I consider load-bearing.

### 1.1 The analytic score is structurally myopic

BaBSR-style scoring estimates, for unstable neuron `j` with pre-activation bounds `[l_j, u_j]`, the bound gain from removing its CROWN relaxation. The upper relaxation line has slope `u_j/(u_j - l_j)` and intercept `-u_j l_j/(u_j - l_j)`; splitting eliminates that intercept. Weighting by the backward sensitivity coefficient `Λ_j` reaching that neuron gives

```
F1(j) ≈ |Λ_j| · ( u_j · (-l_j) / (u_j - l_j) )
```

This is a good, cheap signal. But it measures the *immediate, local, first-order* gain at this node only. It is blind to the thing that actually collapses search trees: a split that tightens intermediate bounds enough to **stabilize other neurons downstream**, permanently removing them from the candidate set in the entire subtree. That effect is second-order in the score but first-order in tree size.

### 1.2 The estimate is inconsistent with what is later computed

The score is computed with the parent's `α` (relaxation slopes) and no `β` (Lagrangian multipliers for the split constraints). The child bound is then computed *after* re-optimizing `α` and `β`. So the ranking is produced by a different function than the one that determines the outcome. FSB partially repairs this by running a short bound computation on a shortlist, which is why FSB beats BaBSR — but it repairs it uniformly, at fixed cost, for every node.

### 1.3 Scores are not comparable across layers

`|Λ_j|` magnitudes differ systematically by depth because of how backward coefficients accumulate. Without per-layer normalization, heuristics develop a persistent layer bias. This is a calibration bug that looks like a modelling choice.

### 1.4 Nothing is learned during the search

Every node's decision is made from scratch. A MILP solver would never do this: it maintains **pseudocosts** — running averages of realized per-unit objective gain for each variable — and after a few observations trusts them over any static estimate. In verification the same neuron index recurs across thousands of nodes in one instance. That is a free, high-quality, self-correcting signal that is largely left on the table.

### 1.5 The sum/max aggregation rule is the wrong one

A domain is pruned only when **both** children close. Aggregating the two child gains by sum or max rewards lopsided splits — one child closes, one stays open, and you have made no structural progress. MILP practice settled this decades ago in favour of the **product rule**, `max(Δ⁻, ε) · max(Δ⁺, ε)`, which strongly penalizes lopsidedness.

### 1.6 Batched selection ignores interaction between splits

Modern α,β-CROWN expands thousands of domains per GPU batch and can split more than one neuron at a time. Selection is effectively independent top-`k`. But the bound improvements from a set of splits are **not additive** — two neurons with heavily overlapping downstream influence largely duplicate each other's effect. Independent top-`k` therefore systematically picks redundant sets.

### 1.7 Verification-oriented scoring on falsifiable instances

If the property is actually false, every second spent branching to tighten bounds is wasted; we want to branch so as to *isolate the counterexample*. A single score function cannot serve both regimes, and the solver typically does not know which regime it is in.

### 1.8 Learned branchers suffer distribution shift

GNN branchers trained to imitate strong branching work well in-distribution and degrade on new architectures, new ε, or new benchmarks. In a competition or deployment setting this fragility is disqualifying. Any learned component must either be trained *online, per instance* or be a small correction on top of a robust analytic prior.

---

## 2. Design principles

From the diagnosis, five principles:

- **P1 — Calibrate, do not replace.** The analytic score carries real signal. Learn a *multiplicative correction* on it rather than a new score from scratch. Correction models are low-variance and degrade gracefully.
- **P2 — Spend effort where the decision is uncertain and consequential.** Adaptive budget, not fixed budget.
- **P3 — Score per unit of cost, not per unit of bound.** Wall clock is the metric.
- **P4 — Select sets, not elements.** Batched branching is a submodular set-selection problem.
- **P5 — Everything must vectorize.** α,β-CROWN's speed comes from treating thousands of domains as tensors. Any per-domain adaptive logic that introduces a Python loop or a GPU→CPU sync will lose more time than the better decisions gain. This is a hard architectural constraint, and it is where most clever branching ideas die in practice.

---

## 3. The proposed method

CRAB replaces the branching module of α,β-CROWN. Nine components; each is separately ablatable.

### 3.1 Aggregation: product rule with a floor

For candidate `j`, with estimated child gains `Δ⁻_j, Δ⁺_j` relative to the parent bound:

```
S(j) = max(Δ⁻_j, ε) · max(Δ⁺_j, ε),    ε ≈ 10⁻⁶ · gap(D)
```

**Why:** a domain closes only when both children close. The product is the standard, well-validated aggregation for exactly this situation. The floor `ε` keeps the score from collapsing to zero when one side is trivially satisfied.

### 3.2 Feature F1 — improved analytic estimate

As in §1.1, but with two fixes:

- **Per-layer normalization.** Divide `|Λ_j|` by the layer's mean absolute sensitivity, so cross-layer comparison is meaningful (fixes §1.3).
- **Warm-started slopes.** Compute `Λ` using the parent's *optimized* `α`, not the initial CROWN slopes, so the estimate is closer to what the child solve will actually see (partially fixes §1.2).

Cost: one backward pass, already available as a by-product of the parent's bound computation. Effectively free.

### 3.3 Feature F2 — implied stabilization mass

This is the component I expect to matter most, because it targets tree size directly rather than node-local bound gain.

Fixing neuron `j` inactive sets its post-activation output to the constant 0, removing its entire contribution to the pre-activation range of every downstream neuron. Under interval arithmetic, downstream neuron `m` loses radius

```
r_{m←j} = |W_{mj}| · u_j / 2      (inactive branch)
```

and analogously for the active branch. Define

```
F2(j) = # { m downstream : m is currently unstable
                           and radius(m) - r_{m←j} < |center(m)| }
```

i.e. the number of neurons that would flip from unstable to stable as a direct consequence of this split. Neurons that flip are removed from the branching candidate set **for the whole subtree** — the payoff compounds.

**Why interval arithmetic when the solver uses CROWN?** Because F2 is a *ranking feature*, not a bound. Interval arithmetic is a conservative under-estimate of the real tightening, it is monotone in the right direction, and it costs one sparse `|W|` matrix–vector product per layer, which is a single fused GPU kernel. Using exact CROWN propagation here would cost as much as strong branching and defeat the purpose.

Practical note: precompute `|W|` row sums once per network. F2 for all candidates in a batch is a few matmuls.

### 3.4 Online calibration by hierarchical pseudocosts

For each neuron `j` maintain a **correction factor**, not a raw pseudocost:

```
ρ_j = running mean of ( realized Δ / predicted Δ )   over observed splits of j
n_j = number of observations
```

Learning a ratio rather than an absolute gain makes the statistic scale-free across tree depths, which is the standard reason naive pseudocosts underperform in verification (gains shrink with depth, so raw averages are dominated by shallow nodes).

Cold start is handled by **empirical-Bayes shrinkage** up a three-level hierarchy — neuron → layer → global:

```
ρ̂_j = (n_j · ρ_j + κ · ρ_layer(j)) / (n_j + κ),    κ ≈ 4
```

so a never-branched neuron inherits its layer's correction, and a layer with no data inherits the global one. The corrected score is `ρ̂_j · S(j)`.

**Why this and not a pretrained model:** it is fit online, per instance, so there is no distribution shift (§1.8); it is nearly free; it strictly improves with search time; and if the correction signal is absent it degenerates to `ρ̂ ≈ 1`, recovering the analytic heuristic exactly. It cannot do worse by much, which is the property you want from a search heuristic.

### 3.5 Uncertainty-gated strong branching (reliability branching)

FSB's insight — verify a shortlist with real bound computations — is correct but applied at constant cost. CRAB makes the budget adaptive along two axes:

**Axis 1 — decision uncertainty.** Compute the normalized margin between the top two candidates:

```
margin = (S₍₁₎ - S₍₂₎) / S₍₁₎
```

If `margin > τ` (top candidate clearly dominant), skip filtering entirely and take the argmax. If `margin ≤ τ`, filter the top `k` candidates with a short bound computation (a few `α,β` iterations). Set `k` on a small ladder driven by the margin, e.g. `k ∈ {0, 3, 8}`.

**Axis 2 — depth decay.** The leverage of a decision falls roughly geometrically with depth. Allocate

```
k_effective = k · 1[depth ≤ d₀]
```

with `d₀` around 8–12, tuned per benchmark family. Deep nodes get pure scoring; shallow nodes get careful evaluation.

**Reliability gate.** Skip strong branching for candidates whose `n_j ≥ η` (η ≈ 4): their calibration is already trustworthy. This is Achterberg's reliability branching, transplanted. The result is that strong branching effort decays automatically as the search accumulates information — early nodes are expensive and careful, later nodes are cheap and fast.

**Vectorization (P5).** All of this must be expressed as masked tensor operations over the batch: compute margins for all domains at once, build a boolean mask of domains needing filtering, gather those rows into a single sub-batch, run one filtered bound pass, scatter results back. No per-domain branching in Python. This is the single most important implementation detail in the entire proposal.

### 3.6 Cost normalization

Splitting a neuron in layer `i` forces recomputation of intermediate bounds for layers `i+1 … L`. Early-layer splits are therefore more expensive *and* typically more impactful. Estimate per-layer cost `c_i` once per network by direct timing during a short profiling phase, then rank by

```
S_final(j) = ρ̂_j · S(j) / c_{layer(j)}^γ,    γ ∈ [0, 1]
```

`γ` interpolates between pure bound-gain ranking (`γ=0`) and pure gain-per-second (`γ=1`). I would start at `γ = 0.5` and tune. **Why not `γ=1` immediately:** cost is measured with noise, and over-penalizing early layers can suppress exactly the splits that produce the largest stabilization cascades.

### 3.7 Batched selection as submodular maximization

When branching `p > 1` neurons per domain (α,β-CROWN supports multi-neuron splits, which is how it keeps large GPU batches busy), do not take independent top-`p`. Total improvement from a set is approximately submodular: overlapping downstream influence means marginal gains diminish.

Greedy selection with an overlap discount:

```
select j₁ = argmax S_final(j)
for t = 2..p:
    j_t = argmax_j  S_final(j) · (1 - max_{s<t} overlap(j, j_s))
```

where `overlap(j, j')` is a cheap similarity between downstream influence supports — cosine similarity of the sparsified `|W|`-propagated influence vectors already computed for F2. Greedy on a submodular objective carries the standard `(1 - 1/e)` guarantee, which is a reasonable justification even though our objective is only approximately submodular.

### 3.8 Dual-mode scoring: verification vs falsification

Maintain a running estimate of `P(unsafe)` for the current instance from cheap signals: best attack loss found by PGD, the trend of the global lower bound, and the fraction of domains whose bound is worsening. Blend two scores:

```
S_mixed(j) = (1 - π) · S_verify(j) + π · S_falsify(j),   π = P̂(unsafe)
```

`S_falsify(j)` ranks neurons by how decisively the split *separates the best known adversarial candidate* from the rest of the domain — concretely, the normalized distance of `ẑ_j(x_adv)` from the interval center `(l_j + u_j)/2`. Splitting there drives the search rapidly into the sub-region containing the attack, where a genuine counterexample is confirmed and the whole instance terminates.

**Why this matters:** on falsifiable instances the entire tree is wasted work; the only useful action is to reach a counterexample fast. Benchmarks contain a meaningful fraction of such instances, and time saved there converts directly into timeout budget available elsewhere.

### 3.9 Completeness safeguard

Any exhaustive case split preserves soundness unconditionally. Completeness requires that no unstable neuron is starved forever. Guarantee it cheaply: if a domain's depth exceeds `d_max`, fall back to round-robin selection over its remaining unstable neurons in index order. Since the unstable set is finite and strictly shrinks along any path, every branch terminates in a fully-determined linear region. This costs nothing in practice (it almost never fires) and makes the completeness argument one line long.

---

## 4. Algorithm

```
Input: network f, input region C, property spec, timeout T
Precompute: |W| row sums; per-layer cost estimates c_i; layer sensitivity norms

Initialize: D₀ = C; global pseudocost tables ρ (neuron / layer / global)
Run incomplete pass (α-CROWN) + PGD attack. Return if resolved.

while queue nonempty and time < T:
    B ← pop batch of worst-bound domains                       # tensorized

    # --- scoring, fully batched ---
    F1 ← normalized analytic gains from parent backward pass    # free
    F2 ← implied-stabilization counts via |W| matvecs           # 1-2 kernels
    S  ← product_rule(F1, F2 blended)                           # §3.1-3.3
    S  ← S * shrunk_pseudocost(ρ)                               # §3.4
    S  ← S / c_layer ** γ                                       # §3.6

    # --- adaptive filtering, masked, no python loop ---
    m ← normalized_margin(S)
    mask ← (m ≤ τ) & (depth ≤ d₀) & (n_j < η)                   # §3.5
    if mask.any():
        sub ← gather(B, mask)
        Δ̂  ← short α,β bound pass on top-k candidates of sub
        S[mask] ← product_rule(Δ̂)
    if depth > d_max: S ← round_robin_override(S)               # §3.9

    # --- set selection ---
    J ← greedy_submodular_topk(S, overlap, p)                   # §3.7

    children ← split(B, J); bounds ← alpha_beta_crown(children)
    update ρ with realized/predicted ratios                     # §3.4
    prune children with bound > 0
    attack survivors; update P̂(unsafe)                          # §3.8
    push survivors
```

---

## 5. Overhead budget

The proposal is only viable if the added scoring cost is small relative to the bound computations it replaces. Target: **branching overhead ≤ 15% of solve time**, measured, with the ablation harness reporting it per benchmark.

| Component | Added cost per node | Notes |
|---|---|---|
| F1 normalization | ~0 | reuses parent backward pass |
| F2 stabilization count | 1–2 sparse matvecs | fused kernel, batched |
| Pseudocost lookup + update | ~0 | tensor gather/scatter |
| Adaptive filtering | **dominant term** | but strictly ≤ FSB's fixed cost, and decays with depth |
| Submodular selection | `p` × cosine sims | `p` is small (2–4) |
| Dual-mode blending | ~0 | attack already computed |

The key economic argument: CRAB's filtering cost is *upper-bounded* by FSB's, because the gates only ever remove work relative to FSB's always-on shortlist evaluation. So the comparison against the strongest baseline is structurally favourable — we should match FSB's decisions where they matter and skip the cost where they don't.

---

## 6. Evaluation protocol

Weak evaluation is the most common way good branching ideas fail to convince. The protocol should be fixed before any results are looked at.

**Benchmarks.** VNN-COMP suites, spanning regimes deliberately:
- MNIST / CIFAR-10 fully-connected and ConvSmall (fast, overhead-sensitive — tests whether CRAB's extra machinery hurts on easy instances)
- CIFAR-10 ConvBig / Wide / ResNet (the medium band where gains should concentrate)
- oval21 (the canonical BaB benchmark)
- Certifiably-trained models (SABR / MTL-IBP style) — few unstable neurons, different regime
- ACAS Xu (low input dimension; input splitting competitive — tests the fallback)

**Baselines.** BaBSR; FSB; the GNN brancher; stock α,β-CROWN at its competition settings. Same hardware, same timeout, same attack budget, same `α,β` iteration counts. Only the branching module differs.

**Metrics.**
1. **Instances verified within timeout** — headline.
2. **PAR-2 score** (unsolved instances counted at 2× timeout) — borrowed from SAT competitions; avoids the distortion of averaging only over solved instances.
3. **Median time on jointly solved instances** — paired, robust.
4. **Number of branches / subdomains** — isolates *decision quality* from *implementation speed*. Reporting only time makes it impossible to tell whether an improvement is scientific or engineering.
5. **Branching overhead fraction** — guards §5.
6. **Falsification time on unsafe instances** — isolates §3.8.

**Statistics.** Per-instance paired comparison, Wilcoxon signed-rank on times, bootstrap CI on verified counts. Cactus/survival plots rather than bar charts. Multiple seeds where attacks introduce randomness; report variance.

**Soundness audit.** Every "verified" result cross-checked with a strong attack (AutoAttack-class, 10× the normal budget). A single unsound result invalidates the method — this check is not optional.

**Ablations.** One row per component removed, on a fixed medium-difficulty subset:

| Configuration | Purpose |
|---|---|
| Full CRAB | — |
| − F2 (stabilization) | is the structural feature the main driver? |
| − pseudocost calibration | does online learning pay? |
| − adaptive gating (fixed `k`, i.e. FSB-like) | is the adaptivity or just the filtering doing the work? |
| − cost normalization (`γ=0`) | does wall-clock weighting help? |
| − submodular selection (independent top-`p`) | does set-awareness matter? |
| − dual-mode (`π=0`) | falsification contribution |
| product → sum aggregation | validates §3.1 |

---

## 7. Expected outcome, stated honestly

I would predict gains **concentrated in a middle band** of difficulty, and I would say so in the paper before showing numbers:

- **Easy instances** — resolved by the incomplete pass or a handful of splits. No heuristic helps; CRAB must simply not *hurt* (overhead guard, §5).
- **Hopeless instances** — the tree is astronomically large under any policy. No heuristic helps.
- **Medium instances** — where the tree is large but not unbounded, branching quality is the binding constraint, and a 2–5× reduction in nodes flips timeouts into solves.

A realistic target is **+5% to +15% instances verified** over FSB on hard benchmarks at fixed timeout, with larger relative gains in node count than in wall-clock (because some of the node savings is spent on better decisions). Any claim of an order-of-magnitude improvement across the board should be treated as a bug until proven otherwise — most likely an unsound bound or a mis-specified baseline.

**Falsification of the hypothesis.** If ablating F2 changes nothing, the "implied stabilization" thesis is wrong and the gains are just better-tuned FSB. That is worth knowing and worth reporting.

---

## 8. Risks and limitations

| Risk | Mitigation |
|---|---|
| Scoring overhead exceeds decision gains on small nets | overhead guard; auto-disable F2/filtering when the estimated tree is small |
| Pseudocosts never warm up (short searches) | hierarchical shrinkage (§3.4) degenerates safely to the analytic score |
| Adaptive logic breaks GPU batching | P5 — masked tensor ops only; profile for hidden `.item()` syncs |
| Interval-arithmetic F2 too loose on deep nets | evaluate a one-layer CROWN refinement of F2 as a variant |
| Hyperparameters (`τ, k, d₀, η, γ, κ`) overfit to the tuning set | tune on a held-out split; report sensitivity curves; prefer a single default configuration across all benchmarks |
| Non-determinism from attacks confounds comparisons | fixed seeds; report across seeds |

**On novelty.** Being candid about provenance strengthens rather than weakens the work: the product rule, pseudocosts, and reliability branching are transplants from the MILP literature (Achterberg and successors); the shortlist-filtering idea is FSB's. The genuinely new contributions are (i) the implied-stabilization feature F2 as a first-class branching signal, (ii) hierarchical multiplicative calibration of the analytic score as an online, shift-free alternative to pretrained branchers, (iii) uncertainty-and-depth-gated effort allocation, and (iv) submodular set selection for batched multi-neuron splitting. That is a defensible contribution set; overclaiming the transplanted parts is the fastest way to lose a reviewer.

---

## 9. Staged plan

| Stage | Duration | Deliverable | Go/no-go |
|---|---|---|---|
| 1. Harness | 3 wks | Reproduce BaBSR/FSB baselines in α,β-CROWN; instrument node counts, overhead, per-layer costs | Baselines within noise of published numbers |
| 2. Aggregation + normalization (§3.1–3.2) | 2 wks | Cheapest possible wins | Any consistent node reduction |
| 3. F2 stabilization feature (§3.3) | 4 wks | Batched GPU implementation + ablation | F2 ablation shows a real effect |
| 4. Pseudocost calibration (§3.4) | 3 wks | Hierarchical shrinkage, online update | Calibration converges; no regression on short searches |
| 5. Adaptive gating (§3.5–3.6) | 4 wks | Fully vectorized masked filtering | Overhead ≤ 15%; beats fixed-`k` FSB |
| 6. Set selection + dual-mode (§3.7–3.8) | 4 wks | Multi-neuron and falsification paths | Gains on unsafe instances specifically |
| 7. Full evaluation + write-up | 4 wks | VNN-COMP-scale results, ablations, soundness audit | — |

Roughly six months of focused work. Stages 2–4 are independently publishable-as-a-component if the later stages disappoint, which is a deliberate hedge in the plan structure.

---

## 10. One-paragraph summary

Current branching heuristics for neural network verification rank neurons by a myopic, uncalibrated, cost-blind estimate of one-step bound improvement, aggregate the two child gains in a way that rewards lopsided splits, spend identical effort on decisions of wildly different leverage, and select batches of splits as if their effects were additive. CRAB addresses each of these directly: the product rule for aggregation; a downstream *implied-stabilization* feature that scores splits by how many neurons they permanently remove from the problem; online hierarchical calibration that learns a multiplicative correction to the analytic score during the search itself and therefore never suffers distribution shift; effort allocation gated jointly on decision uncertainty and depth, so expensive evaluation is spent only where it is both uncertain and consequential; cost normalization so the objective is gain-per-second rather than gain-per-node; greedy submodular selection for batched multi-neuron splitting; and a falsification-aware mode for instances where the property is actually false. Every component is separately ablatable, all of them are sound by construction, and the whole design is constrained to remain expressible as batched tensor operations so it does not forfeit the GPU throughput that makes modern verifiers fast in the first place.
