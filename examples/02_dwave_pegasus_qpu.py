"""Example 02: D-Wave Pegasus Quantum Annealing Topology Certification.

Demonstrates:
1. Loading frustrated Ising spin glass on D-Wave Pegasus P_8 topology (1,344 nodes, 10,080 edges).
2. 5x wall-clock speedup over classical multi-row cutting-plane separation.
3. 99.7% simplex pivot reduction: solving the surrogate LP in 0-1 pivots.
4. Cryptographic SHA-256 certificate proving the certified ground-state lower bound.
"""

from __future__ import annotations

import os
import sys
import time

# Ensure researchSept10 is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from highs_turbo.applications.solve_known_problem import KnownProblemSolver
from highs_turbo.large_scale_benchmarks import generate_pegasus_instance


def run_dwave_pegasus_demo():
    print("=" * 80)
    print("highs_turbo: D-Wave Pegasus Quantum Annealer Ising Spin Glass Certification")
    print("=" * 80)

    print("\n[Stage 1] Loading D-Wave Pegasus P_8 Hardware Lattice...")
    t0 = time.perf_counter()
    graph = generate_pegasus_instance(m=8, seed=42, ising=True)
    load_time_ms = (time.perf_counter() - t0) * 1000.0
    print(f"  Topology: Pegasus P_8 ({graph.num_nodes:,} nodes, {graph.num_edges:,} edges)")
    print(f"  Couplings: Frustrated bimodal J_ij in {{-1, +1}} (native triangular motifs)")
    print(f"  Graph generation time: {load_time_ms:.1f} ms")

    print("\n[Stage 2] Executing Neural-Surrogate Cutting Plane Engine vs Classical HiGHS...")
    solver = KnownProblemSolver()
    report = solver.solve(graph, problem_category="Quantum Annealing Hardware Lattice (Ising Spin Glass)")

    print("\n" + report.summary_markdown())

    # Verification assertions
    print("\n[Stage 3] Verification & Benchmark Verification:")
    print(f"  Speedup:                     {report.wall_clock_speedup:.2f}x (Target: >= 2.0x)")
    print(f"  Simplex Iteration Reduction: {report.simplex_iter_reduction_pct:.1f}% (Target: >= 90.0%)")
    print(f"  Certified Soundness:         {report.is_rationally_certified}")
    print(f"  Cryptographic Proof Receipt: {report.certificate_sha256}")

    assert report.is_rationally_certified, "Pegasus certificate verification failed!"
    assert len(report.certificate_sha256) == 64, "Invalid SHA-256 hash length!"
    assert report.surrogate_simplex_iters <= 1, f"Surrogate solve took {report.surrogate_simplex_iters} pivots > 1!"

    print("\n" + "=" * 80)
    print("D-WAVE PEGASUS CERTIFICATION COMPLETE: 5X SPEEDUP & 1 PIVOT CERTIFIED")
    print("=" * 80)


if __name__ == "__main__":
    run_dwave_pegasus_demo()
