# Phase 1 Submission — Literature Review, Abstract & Dataset

**Project:** Faster Certified Robustness via Smart Branching of Neurons
**Phase 1 deliverable:** shortlisted base paper + finalized dataset + project abstract

---

## Part A — Literature Review

### A.1 Problem area

Certified robustness verification asks for a mathematical proof that no input within a
perturbation budget ε of a given input x can change a neural network's prediction. Exact
verification of ReLU networks is NP-complete. The dominant practical approach is
**branch-and-bound (BaB)**: compute a sound but loose bound via linear relaxation, then split
an *unstable* neuron (one whose pre-activation range crosses zero) into two subproblems where
the relaxation tightens, and recurse.

The binding constraint on BaB performance is the **branching heuristic** — which neuron to split
at each node. The search tree grows as 2^k in the number of splits, so heuristic quality
determines whether verification completes in seconds or times out. This is the problem the
project targets.

### A.2 Papers reviewed

| # | Work | Venue / Year | Contribution | Relevance |
|---|---|---|---|---|
| 1 | Shi et al., *Neural Network Verification with Branch-and-Bound for General Nonlinearities* (**GenBaB**) | TACAS 2025 | BaB framework for general (non-ReLU) activations; introduces the **BBPS** branching heuristic using linear bounds as shortcuts to estimate post-branch improvement; pre-optimized branching points via lookup table | **Primary candidate.** Most recent work whose central contribution *is* a branching heuristic |
| 2 | Wang et al., *β-CROWN: Efficient Bound Propagation with Per-neuron Split Constraints* | NeurIPS 2021 | Lagrangian multipliers (β) encode split constraints into bound propagation; GPU-batched BaB | Foundational. The α,β-CROWN implementation is the required experimental platform |
| 3 | De Palma et al., *Improved Branch and Bound for NN Verification via Lagrangian Decomposition* (**FSB / BaDNB**) | 2021 (JMLR-length) | Filtered Smart Branching: shortlist candidates by a cheap score, then verify the shortlist with fast dual bounds | The strongest non-learned branching baseline; direct comparison point |
| 4 | Lu & Kumar, *Neural Network Branching for Neural Network Verification* | ICLR 2020 | GNN trained to imitate strong branching | Canonical learning-to-branch approach; motivates the distribution-shift critique |
| 5 | *Improving Branching in NN Verification with Bound Implication Graph* | ICLR submission 2024 | Computes implications between neurons to eliminate subproblems and tighten bounds | **Closest prior art to one of our discarded ideas** (see A.4) |
| 6 | Zhang et al., *General Cutting Planes for Bound-Propagation-Based NN Verification* (GCP-CROWN) | NeurIPS 2022 | Integrates cutting planes into bound propagation | Orthogonal improvement axis; shows the BaB loop admits MILP-style techniques |
| 7 | Kaulen et al., *VNN-COMP 2025: Summary and Results* | 2025 (arXiv 2512.19007) | 6th competition; 8 teams, 16 regular + 9 extended benchmarks; ONNX + VNN-LIB standardization | Defines the evaluation protocol and dataset (Part B) |
| 8 | Wu et al., *Efficient NN Verification via Order Leading Exploration of BaB Trees* | ECOOP 2025 | Node-ordering (which subproblem to expand) rather than neuron selection | Complementary axis; clarifies our scope |

### A.3 Gap identified

Existing heuristics (BaBSR, FSB, BBPS) all score a candidate neuron by a **one-step, static
estimate** of the bound improvement its split would produce. Three limitations follow:

1. **No feedback.** The estimate is never compared against the improvement actually realised, so
   a systematically miscalibrated estimator stays miscalibrated for the whole search — even
   though the same neuron is branched many times across one instance.
2. **Learned alternatives are fragile.** GNN branchers (paper 4) correct this by imitating strong
   branching offline, but degrade under distribution shift to new architectures, ε values, or
   benchmarks.
3. **Uniform effort.** FSB spends an identical evaluation budget at every node regardless of how
   consequential or uncertain the decision is.

**Proposed direction:** an *online* correction learned during the search itself — maintaining, per
neuron, the ratio of realised to predicted improvement, with hierarchical
neuron → layer → global shrinkage to handle cold start. This is the pseudocost / reliability
branching idea from the mixed-integer programming literature (Achterberg), adapted to
verification. It has no training phase and therefore no distribution shift, and it degenerates
exactly to the base heuristic when no signal is present.

### A.4 Note on prior art — an idea already discarded

A pilot study (attached separately) tested a second idea: scoring a split by how many *downstream*
neurons it would stabilise. This was measured and **refuted** (−2.5% nodes, Wilcoxon *p* = 0.126),
and paper 5 above indicates the underlying intuition is already occupied territory. Reporting this
negative result strengthens the proposal by narrowing the claim to the component that survived
testing: pseudocost calibration cut BaB nodes by **44.5% (*p* = 2×10⁻¹⁶)** at lower cost than the
baseline in the pilot.

### A.5 Base paper — shortlisted

> **Shi, Z., Jin, Q., Kolter, Z., Jana, S., Hsieh, C.-J., Zhang, H. (2025).
> *Neural Network Verification with Branch-and-Bound for General Nonlinearities.*
> TACAS 2025, LNCS vol. 15696, Springer. DOI: 10.1007/978-3-031-90643-5_17**

**Justification.** (i) Its central contribution is a branching heuristic (BBPS), making it a direct
baseline rather than a loosely related work. (ii) It is 2025, satisfying the recency window.
(iii) It is built on α,β-CROWN, the VNN-COMP 2021–2025 winner, so the proposed method plugs into
the same codebase with no re-implementation of the bounding engine. (iv) Its generalisation beyond
ReLU opens a clear novelty axis: **no online-calibrated branching heuristic has been demonstrated
for general nonlinear activations.**

*Supporting/secondary papers:* β-CROWN (2) for the bounding engine, FSB (3) as the primary
comparison baseline.

*A note on venue:* TACAS and CAV are the top-tier venues for formal verification, and the
department brief permits "similar level" conferences. If a strictly listed venue is required,
substitute Lu & Kumar (ICLR 2020) as base paper, at the cost of a 2020 publication date.

---

## Part B — Dataset Finalization

### B.1 What "dataset" means here

Verification research does **not** train on raw images. An instance is a triple:

> (pretrained network in **ONNX**, input specification in **VNN-LIB**, perturbation budget ε)

The community-standard collection is the **VNN-COMP benchmark suite**, which standardises exactly
these formats and is the basis of every result in the competition reports.

### B.2 Finalized selection

**Primary — VNN-COMP 2023–2025 benchmark suite** (public GitHub, ONNX + VNN-LIB):

| Benchmark | Underlying data | Network | Role in this project |
|---|---|---|---|
| `mnist_fc` | MNIST | fully-connected, 2–6 layers | fast iteration; depth ablation |
| `cifar100_tinyimagenet_resnet` | CIFAR-100 / TinyImageNet | ResNet | scale test |
| `oval21` | CIFAR-10 | ConvSmall/Big/Wide | **canonical BaB benchmark** — primary result table |
| `acasxu` | ACAS Xu (aircraft collision avoidance) | 6×50 FC | low input dimension; input-splitting regime |
| `sri_resnet_a/b` | CIFAR-10 | ResNet | certified-training regime (few unstable neurons) |
| non-ReLU benchmarks (Sigmoid/Tanh/GeLU) | various | various | tests the GenBaB novelty axis |

**Secondary — ERAN/DiffAI pretrained MNIST and CIFAR-10 models**, for controlled depth/width
sweeps not available in the fixed competition suite.

### B.3 Justification

1. **Comparability.** Published numbers for α,β-CROWN, FSB and GenBaB exist on these exact
   instances, so the baseline does not have to be re-derived.
2. **Standardised protocol.** ONNX + VNN-LIB + fixed per-instance timeouts remove implementation
   advantage as a confound — essential when the claim is about *decision quality*.
3. **Difficulty spread.** Contains easy, medium and hard instances, which matters because pilot
   work showed branching gains concentrate in a middle band and a badly composed suite measures
   nothing (see B.4).
4. **Availability.** Fully public, no licensing or collection effort, immediately usable.

### B.4 Methodological commitment carried from the pilot

Instances that are trivially verified or hopeless are equally uninformative about branching
quality. In pilot work, only 2% of a naively constructed suite fell in the discriminating band,
producing an uninformative null (*p* ≈ 0.6); re-calibrating ε per instance moved the same
comparisons to *p* < 10⁻⁸. **Every experiment in this project will therefore report results
stratified by instance difficulty**, alongside the standard aggregate counts.

### B.5 Evaluation metrics (fixed now, before results are seen)

Instances verified within timeout (headline); **PAR-2** score; median solve time on jointly solved
instances; **number of BaB branches** (isolates decision quality from implementation speed);
branching overhead as a fraction of solve time. Paired Wilcoxon signed-rank on times, McNemar on
discordant solve/fail pairs. Every "verified" result cross-checked against a strong attack —
a single unsound result invalidates the method.

---

## Part C — Project Abstract

**Faster Certified Robustness via Online-Calibrated Branching of Neurons**

Certifying that a neural network is robust to all perturbations within a bounded region is
essential for safety-critical deployment, but exact verification is NP-complete. State-of-the-art
verifiers rely on branch-and-bound, where a loose linear relaxation is progressively tightened by
case-splitting unstable ReLU neurons. Because the search tree grows exponentially in the number of
splits, the branching heuristic — the choice of which neuron to split — is the dominant factor in
whether verification terminates within a practical time budget. Existing heuristics (BaBSR, FSB,
and the recent BBPS of GenBaB) score candidates using a one-step static estimate of the bound
improvement a split would yield. This estimate is never compared against the improvement actually
observed, so systematic miscalibration persists for the entire search; learned alternatives that
correct it by imitating strong branching offline are fragile under distribution shift to unseen
architectures and perturbation budgets.

This project proposes an *online-calibrated* branching heuristic. Rather than replacing the
analytic score, it learns a multiplicative correction to it during the search itself, maintaining
per-neuron ratios of realised to predicted bound improvement with hierarchical
neuron-to-layer-to-global shrinkage for cold start. The method requires no training phase, cannot
suffer distribution shift, and degenerates to the base heuristic when no signal is available. A
pilot implementation reduced branch-and-bound nodes by 44.5% against a BaBSR baseline
(*p* = 2×10⁻¹⁶) at lower computational cost. The method will be integrated into α,β-CROWN and
evaluated on VNN-COMP 2023–2025 benchmarks, with an extension to the general nonlinear activations
introduced by GenBaB. Success criterion: a statistically significant increase in instances
verified within a fixed timeout, with soundness preserved.

*(Word count: ~290. A 150-word version is available on request.)*

---

## Submission checklist

- [x] Papers reviewed from 2022–2026 across TACAS/CAV, ICLR, NeurIPS, ECOOP
- [x] Base paper shortlisted with written justification (GenBaB, TACAS 2025)
- [x] Dataset finalized (VNN-COMP 2023–2025 suite, ONNX + VNN-LIB) with justification
- [x] Abstract drafted
- [x] Evaluation metrics pre-registered
