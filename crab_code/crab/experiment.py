"""Ablation harness.

Reports the metrics fixed in the design note before any results were seen:
verified count, PAR-2 (unsolved charged at 2x the node budget), median nodes
on jointly solved instances, and branching overhead.  Node counts are the
primary signal here because the backend is NumPy on one CPU -- wall clock
would mostly measure the backend, not the decisions.
"""
from __future__ import annotations

import json
import time
from dataclasses import replace

import numpy as np

from .bab import verify
from .benchmarks import build_suite
from .branching import CRAB, FSB, BaBSR, CrabConfig, RandomBranching


def configs():
    """Ablation grid.  Each CRAB row disables exactly one component."""
    base = CrabConfig()
    grid = {
        "random": lambda: RandomBranching(seed=1),
        "babsr": lambda: BaBSR(),
        "fsb-k6": lambda: FSB(k=6),
        "crab-full": lambda: CRAB(replace(base, name="crab-full")),
        "crab-noF2": lambda: CRAB(replace(base, use_f2=False, name="crab-noF2")),
        "crab-noPC": lambda: CRAB(replace(base, pseudocost=False, name="crab-noPC")),
        "crab-noFilter": lambda: CRAB(replace(base, adaptive_filter=False,
                                              name="crab-noFilter")),
        "crab-noCost": lambda: CRAB(replace(base, cost_gamma=0.0, name="crab-noCost")),
        "crab-noProd": lambda: CRAB(replace(base, product_rule=False, name="crab-noProd")),
        "crab-noLayerNorm": lambda: CRAB(replace(base, layer_norm=False,
                                                 name="crab-noLayerNorm")),
    }
    return grid


def run(instances, grid, max_nodes=2000, timeout=30.0, verbose=True):
    rows = {}
    for name, make in grid.items():
        t0 = time.time()
        recs = []
        for k, ins in enumerate(instances):
            ln = not name.endswith("noLayerNorm")
            r = verify(ins.net, ins.x_lb, ins.x_ub, ins.C_spec, make(),
                       max_nodes=max_nodes, timeout=timeout, seed=k, layer_norm=ln)
            recs.append({"verdict": r.verdict, "nodes": r.nodes, "passes": r.passes,
                         "seconds": r.seconds, "family": ins.family,
                         "filter_calls": r.filter_calls,
                         "filter_passes": r.filter_passes,
                         "input_splits": r.input_splits})
        rows[name] = recs
        if verbose:
            solved = sum(1 for x in recs if x["verdict"] in ("verified", "falsified"))
            print(f"  {name:18s} solved {solved:3d}/{len(instances)}  "
                  f"({time.time() - t0:.1f}s)")
    return rows


def summarise(rows, instances, max_nodes=2000):
    names = list(rows)
    n = len(instances)
    solved_sets = {k: {i for i, x in enumerate(v)
                       if x["verdict"] in ("verified", "falsified")}
                   for k, v in rows.items()}
    common = set.intersection(*solved_sets.values()) if solved_sets else set()

    table = []
    for k in names:
        recs = rows[k]
        solved = solved_sets[k]
        par2 = np.mean([recs[i]["nodes"] if i in solved else 2 * max_nodes
                        for i in range(n)])
        med_common = np.median([recs[i]["nodes"] for i in common]) if common else float("nan")
        tot_common = sum(recs[i]["nodes"] for i in common)
        ovh = np.mean([recs[i]["filter_passes"] / max(recs[i]["passes"], 1)
                       for i in range(n)])
        table.append({"name": k, "solved": len(solved), "par2_nodes": par2,
                      "median_nodes_common": med_common,
                      "total_nodes_common": tot_common,
                      "overhead_frac": ovh,
                      "mean_seconds": np.mean([r["seconds"] for r in recs])})
    return table, common


def paired_test(rows, a, b, common):
    """Wilcoxon signed-rank on node counts over jointly solved instances."""
    from scipy.stats import wilcoxon
    xa = np.array([rows[a][i]["nodes"] for i in sorted(common)], float)
    xb = np.array([rows[b][i]["nodes"] for i in sorted(common)], float)
    d = xa - xb
    if np.allclose(d, 0):
        return 1.0, 0.0
    stat, p = wilcoxon(xa, xb)
    return float(p), float(np.median(d))


def print_table(table, common_n):
    hdr = f"{'config':20s} {'solved':>7s} {'PAR2':>9s} {'med(common)':>12s} " \
          f"{'tot(common)':>12s} {'overhead':>9s} {'sec':>7s}"
    print(hdr)
    print("-" * len(hdr))
    for r in table:
        print(f"{r['name']:20s} {r['solved']:7d} {r['par2_nodes']:9.1f} "
              f"{r['median_nodes_common']:12.1f} {r['total_nodes_common']:12d} "
              f"{100*r['overhead_frac']:8.1f}% {r['mean_seconds']:7.2f}")
    print(f"\n(common = {common_n} instances solved by every configuration)")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="CRAB branching ablation harness")
    ap.add_argument("--mode", choices=["quick", "band", "full"], default="quick",
                    help="quick: ~2 min smoke ablation. band: calibrated "
                         "middle-band suite (the one the report uses). "
                         "full: every config on the raw suite (slow).")
    ap.add_argument("--nodes", type=int, default=400, help="node cap per instance")
    ap.add_argument("--timeout", type=float, default=2.0, help="seconds per instance")
    ap.add_argument("--limit", type=int, default=0, help="cap instance count (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("building benchmark suite...")
    inst = build_suite(seed=args.seed)
    if args.mode == "band":
        from .benchmarks import calibrate_band
        inst = calibrate_band(inst)
    if args.limit and args.limit < len(inst):
        # sample across families -- a prefix would take only the easiest family
        rs = np.random.default_rng(args.seed)
        inst = [inst[i] for i in sorted(rs.choice(len(inst), args.limit, replace=False))]
    print(f"{len(inst)} instances\n")

    grid = configs()
    if args.mode == "quick":
        keep = ("babsr", "fsb-k6", "crab-full", "crab-noPC", "crab-noF2")
        grid = {k: v for k, v in grid.items() if k in keep}

    print("running ablation grid...")
    rows = run(inst, grid, max_nodes=args.nodes, timeout=args.timeout)
    table, common = summarise(rows, inst, max_nodes=args.nodes)
    print()
    print_table(table, len(common))
    names = list(rows)
    if "babsr" in names:
        print("\npaired vs babsr (Wilcoxon on nodes, jointly solved):")
        for n in names:
            if n == "babsr":
                continue
            p, med = paired_test(rows, "babsr", n, common)
            print(f"  {n:18s} median node diff {med:+7.1f}   p = {p:.3g}")
    with open("results_ablation.json", "w") as f:
        json.dump({"rows": rows, "table": table}, f, indent=1, default=float)
    print("\nwrote results_ablation.json")


if __name__ == "__main__":
    main()
