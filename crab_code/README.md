# CRAB — branching heuristics for neural network verification

NumPy branch-and-bound verifier for certified robustness, with an optional
PyTorch backend for Apple MPS / CUDA.

## Install

    python3 -m pip install numpy scipy matplotlib
    python3 -m pip install torch          # optional, for the GPU backend

## Run, in this order

    python3 tests/test_bounds.py          # 1. bound soundness (always first)
    python3 tests/test_batched.py         # 2. batched == per-domain
    python3 tests/test_float32.py         # 3. float32 precision risk
    python3 tests/test_torch_backend.py   # 4. GPU validation (needs torch)
    python3 -m crab.experiment --mode band --nodes 400 --timeout 2

`--mode band` calibrates epsilon per instance so the suite lands in the
12-150 node range where branching decisions actually change the outcome.
`--mode quick` skips that; it is a smoke test, not a result.

## Apple Silicon (M-series) notes

MPS does not support float64.  float32 can report a bound ABOVE the true
value (measured: 57-80% of cases, worst 1.8e-6, growing with depth), and a
bound that is too high can certify an unsafe network.  So the GPU tier is
not a drop-in.  Use two tiers:

  fast   float32 on MPS, with an outward slack from calibrate_slack().
         Prunes the easy majority with margin.
  exact  float64 on CPU.  Re-checks any domain whose float32 bound lands
         within the slack of zero.

Run tests/test_torch_backend.py first — it validates the port against the
NumPy reference and calibrates the slack on your hardware.  Do not reuse a
slack measured on someone else's machine.

## Layout

  crab/crown.py         per-domain CROWN bound propagation + PGD attack
  crab/batched.py       batched CROWN over many subdomains (GPU-shaped)
  crab/torch_backend.py MPS/CUDA backend, float32 slack handling
  crab/bab.py           branch-and-bound loop with input-split fallback
  crab/branching.py     Random / BaBSR / FSB / AdaptiveFSB / CRAB
  crab/benchmarks.py    synthetic tasks, training, epsilon calibration
  crab/experiment.py    ablation harness, PAR-2, paired statistics

## Findings

See DEVELOPMENT_REPORT.md.  Headline: pseudocost calibration cuts nodes 44.5%
(p=2e-16) at lower cost.  The implied-stabilisation feature, cost
normalisation and adaptive budget allocation were all refuted.
