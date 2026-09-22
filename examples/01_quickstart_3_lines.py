"""Quickstart for linprog, Max-Cut and QUBO.

Results distinguish feasible candidates, checked bounds and numerical status.
A receipt digest alone is not a mathematical proof.
"""

from __future__ import annotations

import os
import sys
import numpy as np

# Allow running this example from a source checkout
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import highs_turbo as opt  # Compatible linprog entry point
from highs_turbo.graph_generator import generate_k5_cluster_graph
from highs_turbo.exact_solver import ExactMaxCutSolver


def run_quickstart():
    print("=" * 70)
    print("highs_turbo: LP, Max-Cut and QUBO Quickstart")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 1. 3-Line Drop-in linprog replacement
    # -------------------------------------------------------------------------
    print("\n[Stage 1] 3-Line Drop-in scipy.optimize.linprog Replacement")
    # Generate graph and standard LP relaxation matrices
    g = generate_k5_cluster_graph(num_k5=16, num_bridges=15, seed=42)
    solver = ExactMaxCutSolver()
    c, A_ub, b_ub, _, _ = solver.build_relaxation_matrices(g, include_k5=False)

    # Preserve the supplied LP:
    res = opt.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(0, 1))

    print(f"  Numerical LP Objective:  {res.fun:.4f}")
    print(f"  Reported Iterations:      {res.nit}")
    print(f"  Turbo Accelerated:        {getattr(res, 'turbo_accelerated', False)}")
    print(f"  Solve Strategy:           {getattr(res, 'turbo_strategy', 'scipy')}")

    # -------------------------------------------------------------------------
    # 2. High-Level 1-Line Max-Cut Solver
    # -------------------------------------------------------------------------
    print("\n[Stage 2] High-Level 1-Line Max-Cut Solver")
    # 6-node weighted graph adjacency
    adj = np.array([
        [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
    ])

    cut = opt.solve_maxcut(adj)
    cut_val, partition, cert = cut
    print(f"  Candidate Cut Value:        {cut_val:.1f}")
    print(f"  Candidate Partition:      {partition}")
    print(f"  Receipt Digest:           {cert[:24]}...")

    print(f"  Checked Upper Bound:      {cut.exact_rational_bound}")
    print(f"  Numerical Status:         {cut.status}")

    # -------------------------------------------------------------------------
    # 3. High-Level 1-Line QUBO / Ising Solver
    # -------------------------------------------------------------------------
    print("\n[Stage 3] High-Level 1-Line QUBO Solver")
    # 4-variable frustrated Ising / QUBO coupling matrix
    Q = np.array([
        [ 2.0, -1.5,  0.0,  1.0],
        [-1.5,  3.0, -2.0,  0.0],
        [ 0.0, -2.0,  2.5, -1.0],
        [ 1.0,  0.0, -1.0,  2.0],
    ])

    qubo = opt.solve_qubo(Q)
    energy, solution, cert_q = qubo
    print(f"  Candidate Energy:         {energy:.4f}")
    print(f"  Candidate Binary State:     {solution}")
    print(f"  Receipt Digest:           {cert_q[:24]}...")

    print(f"  Checked Lower Bound:      {qubo.exact_rational_bound}")
    print(f"  Numerical Status:         {qubo.status}")

    print("\n" + "=" * 70)
    print("QUICKSTART COMPLETE; serialize bound_certificate for standalone proof verification")
    print("=" * 70)


if __name__ == "__main__":
    run_quickstart()
