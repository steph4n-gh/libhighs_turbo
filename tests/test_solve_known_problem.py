"""Automated tests for the Known Problem Solver Application.

Verifies:
1. Ground-state energy certification on D-Wave Pegasus & Chimera Ising spin glasses.
2. G-set Max-Cut benchmark (G43, G11) speedup (>= 2x) and simplex iteration reduction (>= 70%).
3. Biological regulatory network (E. coli sign-inversion motifs).
4. 1-row surrogate LP solve in exactly 0-1 simplex pivots.
5. Exact rational verification and cryptographic SHA-256 certificate generation.
6. Zero shortcuts: rejection of adversarial mutated cuts.
"""

from __future__ import annotations

import json
import os
import tempfile
import pytest
import numpy as np

from highs_turbo.applications.solve_known_problem import (
    KnownProblemSolver,
    ProblemSolutionReport,
    solve_known_problem,
)
from highs_turbo.compiled_engine import COMPILED_ENGINE_AVAILABLE


@pytest.mark.skipif(not COMPILED_ENGINE_AVAILABLE, reason="Compiled engine not available")
def test_solve_dwave_pegasus_ising_ground_state():
    """Verify Pegasus P_4 and P_8 frustrated Ising spin glass ground-state certification."""
    solver = KnownProblemSolver()
    g = solver.load_dwave_pegasus_ising(m=4, seed=42)
    assert g.num_nodes == 288
    assert g.num_edges > 2000

    rep: ProblemSolutionReport = solver.solve(g, problem_category="Pegasus P_4 Ising Spin Glass")

    # 1. Mathematical Soundness & Certification
    assert rep.is_rationally_certified, f"Verification failed: {rep.verification_status}"
    assert len(rep.certificate_sha256) == 64
    assert rep.verification_status in ("CERTIFIED_VALID_CYCLE_CONIC_COMBINATION", "CERTIFIED_VALID_CONIC_COMBINATION")

    # 2. Ground-state energy bounds: surrogate bound must be >= unconstrained lower bound
    assert rep.ising_ground_state_surrogate_bound is not None
    assert rep.ising_ground_state_base_bound is not None
    assert rep.ising_ground_state_surrogate_bound >= rep.ising_ground_state_base_bound

    # 3. 1-Row Surrogate LP solves in 1 simplex pivot
    assert rep.surrogate_simplex_iters <= 1
    assert rep.simplex_iter_reduction_pct >= 70.0
    assert rep.surrogate_constraint_nonzeros > 0


@pytest.mark.skipif(not COMPILED_ENGINE_AVAILABLE, reason="Compiled engine not available")
def test_solve_dwave_chimera_ising_logical_cliques():
    """Verify Chimera C_{6,6,4} with embedded logical K5 cliques."""
    solver = KnownProblemSolver()
    g = solver.load_dwave_chimera_ising(m=6, n=6, t=4, seed=42, embedded_cliques=10)
    assert g.num_nodes == 288
    assert g.num_edges > 800

    rep: ProblemSolutionReport = solver.solve(g, problem_category="Chimera C_{6,6,4} Logical Cliques")

    assert rep.is_rationally_certified
    assert len(rep.certificate_sha256) == 64
    assert rep.surrogate_simplex_iters <= 1
    assert rep.std_num_constraints_added >= 1
    assert rep.surrogate_constraint_nonzeros > 0


@pytest.mark.skipif(not COMPILED_ENGINE_AVAILABLE, reason="Compiled engine not available")
def test_solve_gset_g43_dense_instance():
    """Verify Stanford G-set G43 (1,000 nodes, 9,990 edges) benchmark."""
    rep = solve_known_problem(problem="g43", size=0)

    assert rep.num_nodes == 1000
    assert rep.num_edges == 9990
    assert rep.is_rationally_certified
    assert len(rep.certificate_sha256) == 64

    # Speedup >= 2.0x and iteration reduction >= 70%
    assert rep.wall_clock_speedup >= 2.0, f"Speedup {rep.wall_clock_speedup:.2f}x < 2.0x target"
    assert rep.simplex_iter_reduction_pct >= 70.0, f"Iteration reduction {rep.simplex_iter_reduction_pct:.1f}% < 70%"
    assert rep.surrogate_simplex_iters <= 1
    assert rep.separated_cut_count > 0


@pytest.mark.skipif(not COMPILED_ENGINE_AVAILABLE, reason="Compiled engine not available")
def test_solve_biological_regulatory_network():
    """Verify E. coli transcriptional regulation interaction network."""
    rep = solve_known_problem(problem="biological", size=0)

    assert rep.num_nodes == 500
    assert rep.num_edges > 1000
    assert rep.is_rationally_certified
    assert len(rep.certificate_sha256) == 64
    assert rep.surrogate_simplex_iters <= 1
    assert rep.simplex_iter_reduction_pct >= 70.0


@pytest.mark.skipif(not COMPILED_ENGINE_AVAILABLE, reason="Compiled engine not available")
def test_json_receipt_serialization():
    """Verify cryptographic receipt serialization to JSON matches dataclass."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        rep = solve_known_problem(problem="pegasus", size=4, output_json=tmp_path)
        assert os.path.exists(tmp_path)

        with open(tmp_path, "r") as f:
            data = json.load(f)

        assert data["instance_name"] == rep.instance_name
        assert data["certificate_sha256"] == rep.certificate_sha256
        assert data["is_rationally_certified"] is True
        assert data["surrogate_simplex_iters"] == rep.surrogate_simplex_iters
        assert "exact_rational_rhs" in data
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@pytest.mark.skipif(not COMPILED_ENGINE_AVAILABLE, reason="Compiled engine not available")
def test_adversarial_tamper_rejection():
    """Verify that tampering with candidate multipliers causes rational rejection."""
    solver = KnownProblemSolver()
    g = solver.load_dwave_chimera_ising(m=4, n=4, t=4, seed=42, embedded_cliques=5)

    # Hostile adversarial multipliers: negative multiplier
    hostile_mults = {(0, 1, 2, 3, 4): -1.5}
    cert = solver.verifier.verify_clique_conic_combination(g, hostile_mults)

    assert not cert.is_valid
    assert cert.status == "REJECTED_NEGATIVE_MULTIPLIER"


@pytest.mark.skipif(not COMPILED_ENGINE_AVAILABLE, reason="Compiled engine not available")
def test_solve_edwards_anderson_ising_ground_state():
    """Verify Edwards-Anderson frustrated 2D lattice Ising spin glass ground-state certification."""
    solver = KnownProblemSolver()
    g = solver.load_edwards_anderson_ising(l=8, dim=2, seed=42)
    assert g.num_nodes == 64
    assert len(g.edges) == 128

    rep = solver.solve(g, problem_category="Edwards-Anderson 8x8 2D Ising Spin Glass")

    assert rep.is_rationally_certified
    assert len(rep.certificate_sha256) == 64
    assert rep.surrogate_simplex_iters <= 1
    assert rep.ising_ground_state_surrogate_bound is not None
    assert rep.ising_ground_state_base_bound is not None
    assert rep.ising_ground_state_surrogate_bound >= rep.ising_ground_state_base_bound
    assert rep.base_objective >= rep.surrogate_objective


@pytest.mark.skipif(not COMPILED_ENGINE_AVAILABLE, reason="Compiled engine not available")
def test_solve_senate_polarization_network():
    """Verify Senate voting polarization network with bipartisan cuts."""
    solver = KnownProblemSolver()
    g = solver.load_senate_polarization_network(num_senators=100, seed=42)
    assert g.num_nodes == 100
    assert len(g.edges) > 500

    rep = solver.solve(g, problem_category="US Senate Polarization Network")

    assert rep.is_rationally_certified
    assert len(rep.certificate_sha256) == 64
    assert rep.surrogate_simplex_iters <= 1
    assert rep.base_objective >= rep.surrogate_objective


def test_mathematical_gap_and_objective_consistency():
    """Verify consistent positive Max-Cut relaxation objectives and mathematical gap calculations."""
    solver = KnownProblemSolver()
    g = solver.load_dwave_pegasus_ising(m=4, seed=42)
    rep = solver.solve(g)

    # Base objective must be strictly positive (Max-Cut upper bound)
    assert rep.base_objective > 0.0
    assert rep.surrogate_objective > 0.0
    # Adding surrogate cut tightens the upper bound (must not exceed base)
    assert rep.surrogate_objective <= rep.base_objective + 1e-6
    # Ising ground state energy lower bound tightens (must not be lower than base)
    assert rep.ising_ground_state_surrogate_bound >= rep.ising_ground_state_base_bound - 1e-6


def test_edge_case_empty_and_acyclic_graphs():
    """Verify robust handling of empty graphs and acyclic tree structures with zero cuts."""
    from highs_turbo.graph_generator import GraphInstance

    solver = KnownProblemSolver()

    # 1. Empty graph
    g_empty = GraphInstance("empty_graph", 0, [], {})
    rep_empty = solver.solve(g_empty)
    assert rep_empty.solver_message == "SUCCESS"
    assert rep_empty.separated_cut_count == 0
    assert rep_empty.surrogate_simplex_iters == 0

    # 2. Acyclic tree graph (bipartite, no odd or even cycles to separate)
    edges = [(i, i + 1) for i in range(9)]
    weights = {e: 1.0 for e in edges}
    g_tree = GraphInstance("tree_graph", 10, edges, weights)
    rep_tree = solver.solve(g_tree)

    assert rep_tree.solver_message == "SUCCESS"
    assert rep_tree.separated_cut_count == 0
    assert rep_tree.base_objective == 9.0
    assert rep_tree.surrogate_objective == 9.0
    assert rep_tree.bound_tightness_gap_closed_pct == 0.0


@pytest.mark.skipif(not COMPILED_ENGINE_AVAILABLE, reason="Compiled engine not available")
def test_adversarial_even_f_cycle_rejection():
    """Verify that verifier rejects cycle cuts with even |F| (violating Chvatal parity)."""
    from highs_turbo.compiled_engine import CompiledBitGraph
    from highs_turbo.graph_generator import GraphInstance

    g = GraphInstance(
        "c6",
        6,
        [(i, (i + 1) % 6) for i in range(6)],
        {(min(i, (i + 1) % 6), max(i, (i + 1) % 6)): 1.0 for i in range(6)},
    )
    cbg = CompiledBitGraph.from_graph_instance(g)
    solver = KnownProblemSolver()

    # Even |F| = 4 on a 6-cycle
    even_f = [(0, 1), (1, 2), (2, 3), (3, 4)]
    cert = solver.verifier.verify_cycle_conic_combination(cbg, [(list(range(6)), even_f, 1.0)])
    assert not cert.is_valid
    assert cert.status == "REJECTED_EVEN_F_CARDINALITY"


def test_sparse_csr_matrix_contract():
    """Verify pure-NumPy csr_matrix constructor signatures, transpose, and indexing."""
    import scipy.sparse as sp

    # 1. COO construction (data, (row, col))
    A = sp.csr_matrix(([10.0, 20.0, 30.0], ([0, 1, 0], [0, 1, 2])), shape=(2, 3))
    assert A.shape == (2, 3)
    assert A.nnz == 3
    assert A[0, 0] == 10.0
    assert A[0, 2] == 30.0
    assert A[1, 1] == 20.0
    assert A[1, 0] == 0.0

    # 2. Transpose
    AT = A.T
    assert AT.shape == (3, 2)
    assert AT[0, 0] == 10.0
    assert AT[2, 0] == 30.0
    assert AT[1, 1] == 20.0

    # 3. Vector multiplication
    x = np.array([1.0, 2.0, 3.0])
    y = A @ x
    assert np.allclose(y, [100.0, 40.0])

    # 4. Row slicing
    row1 = A[1]
    assert row1.shape == (1, 3)
    assert row1[0, 1] == 20.0


def test_knapsack_infeasible_rejection():
    """Verify linprog detects infeasible knapsack inequalities."""
    from scipy.optimize import linprog

    # Minimizing x1 + x2 s.t. x1 + x2 <= -1 with 0 <= x <= 1 is impossible
    res = linprog([1.0, 1.0], A_ub=[[1.0, 1.0]], b_ub=[-1.0], bounds=[(0.0, 1.0), (0.0, 1.0)])
    assert not res.success
    assert res.status == 2
    assert "infeasible" in res.message.lower()

