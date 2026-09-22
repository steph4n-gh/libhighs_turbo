"""Bound Max-Cut on a synthetic G43-shaped graph.

This is a seeded random graph, not Stanford's published G43 dataset. The
example measures its own runtime and independently checks a complete rational
witness. It makes no fixed speedup or optimality claim.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from highs_turbo import solve_maxcut, verify_ising_certificate
from highs_turbo.topologies import generate_gset_instance


def run_gset_demo(seconds=5.0):
    graph = generate_gset_instance(name="G43", seed=42)
    print(f"Synthetic G43-shaped graph: {graph.num_nodes:,} nodes, {graph.num_edges:,} edges")
    started = time.perf_counter()
    result = solve_maxcut(graph, time_limit=seconds, seed=42)
    elapsed = time.perf_counter() - started
    weights = {edge: Fraction(weight) for edge, weight in graph.weights.items()}
    if not verify_ising_certificate([0] * graph.num_nodes, weights, result.bound_certificate):
        raise RuntimeError("Original-model witness verification failed")
    candidate = sum((weight for (u, v), weight in weights.items()
                     if result.partition[u] != result.partition[v]), Fraction())
    upper = result.exact_rational_bound
    gap = upper - candidate
    if gap < 0:
        raise RuntimeError("The bound is below the feasible cut")
    print(f"Candidate cut: {candidate}; checked upper bound: {upper}; exact gap: {gap}")
    print(f"Optimality proven: {gap == 0}; numerical status: {result.status}")
    print(f"Solve elapsed: {elapsed:.3f}s; requested cooperative budget: {seconds:g}s")
    print("Full witness verified against the original generated objective.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=5.0)
    run_gset_demo(parser.parse_args().seconds)
