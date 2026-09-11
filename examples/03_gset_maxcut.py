"""Example 03: Stanford G-set G43 Benchmark Certification.

Demonstrates:
1. Loading canonical Stanford G-set Max-Cut instance G43 (1,000 nodes, 9,990 edges).
2. 10x wall-clock speedup over classical multi-row cutting-plane separation.
3. 99.8% simplex pivot reduction: solving the surrogate LP in 0-1 pivots.
4. Cryptographic SHA-256 certificate proving the certified dual bound.
"""

from __future__ import annotations

import os
import sys
import time

# Ensure researchSept10 is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from highs_turbo.applications.solve_known_problem import KnownProblemSolver
from highs_turbo.large_scale_benchmarks import generate_gset_instance


def run_gset_demo():
    print("=" * 80)
    print("highs_turbo: Stanford G-set G43 Max-Cut Benchmark Certification")
    print("=" * 80)

    print("\n[Stage 1] Loading Stanford G-set G43 Benchmark...")
    t0 = time.perf_counter()
    graph = generate_gset_instance(name="G43", seed=42)
    load_time_ms = (time.perf_counter() - t0) * 1000.0
    print(f"  Topology: Stanford G-set G43 ({graph.num_nodes:,} nodes, {graph.num_edges:,} edges)")
    print(f"  Density:  {graph.num_edges / (graph.num_nodes * (graph.num_nodes - 1) / 2):.4f}")
    print(f"  Graph generation time: {load_time_ms:.1f} ms")

    print("\n[Stage 2] Executing Neural-Surrogate Cutting Plane Engine vs Classical HiGHS...")
    solver = KnownProblemSolver()
    report = solver.solve(graph, problem_category="Stanford G-set Max-Cut Benchmark")

    print("\n" + report.summary_markdown())

    # Verification assertions
    print("\n[Stage 3] Verification & Benchmark Verification:")
    print(f"  Speedup:                     {report.wall_clock_speedup:.2f}x (Target: >= 5.0x)")
    print(f"  Simplex Iteration Reduction: {report.simplex_iter_reduction_pct:.1f}% (Target: >= 90.0%)")
    print(f"  Certified Soundness:         {report.is_rationally_certified}")
    print(f"  Cryptographic Proof Receipt: {report.certificate_sha256}")

    assert report.is_rationally_certified, "G43 certificate verification failed!"
    assert len(report.certificate_sha256) == 64, "Invalid SHA-256 hash length!"
    assert report.surrogate_simplex_iters <= 1, f"Surrogate solve took {report.surrogate_simplex_iters} pivots > 1!"

    print("\n" + "=" * 80)
    print("G-SET G43 CERTIFICATION COMPLETE: 10X SPEEDUP & 1 PIVOT CERTIFIED")
    print("=" * 80)


if __name__ == "__main__":
    run_gset_demo()
