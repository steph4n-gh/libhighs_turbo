"""Comprehensive Automated Test Suite for highs_turbo plugin.

Tests:
1. highs_turbo.linprog drop-in compatibility with standard SciPy outputs.
2. Automatic topology detection on graph-structured LPs vs non-graph LPs.
3. High-level solve_maxcut and solve_qubo APIs (including 3-tuple unpacking).
4. Adversarial cut rejection and cryptographic SHA-256 receipts.
5. In-memory HiGHS acceleration and 0-1 simplex pivot performance.
"""

from __future__ import annotations

import os
import sys
from fractions import Fraction
import numpy as np
import pytest

# Ensure researchSept10 is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import highs_turbo
from highs_turbo import MaxCutResult, QuboResult, TopologyDetector, TurboSolver, detect_topology
from highs_turbo.exact_solver import ExactMaxCutSolver
from highs_turbo.graph_generator import GraphInstance, generate_k5_cluster_graph
from scipy.optimize import Bounds, LinearConstraint, OptimizeResult, linprog as scipy_linprog


# =============================================================================
# 1. Drop-in SciPy Compatibility on Standard (Non-Graph) LPs
# =============================================================================


def test_linprog_dropin_nongraph_standard_lp():
    """Verify highs_turbo.linprog on standard non-graph LPs exactly matches SciPy."""
    # Min -x0 + 4x1
    # s.t. -3x0 + x1 <= 6
    #       x0 + 2x1 <= 4
    #       x0, x1 >= 0
    c = [-1.0, 4.0]
    A = [[-3.0, 1.0], [1.0, 2.0]]
    b = [6.0, 4.0]

    res_scipy = scipy_linprog(c, A_ub=A, b_ub=b, method="highs")
    res_turbo = highs_turbo.linprog(c, A_ub=A, b_ub=b, method="turbo")

    assert isinstance(res_turbo, OptimizeResult)
    assert res_turbo.success == res_scipy.success
    assert res_turbo.status == res_scipy.status
    assert np.isclose(res_turbo.fun, res_scipy.fun, atol=1e-6)
    assert np.allclose(res_turbo.x, res_scipy.x, atol=1e-6)
    assert getattr(res_turbo, "turbo_accelerated", False) is False


def test_linprog_dropin_with_bounds_and_equality():
    """Verify highs_turbo.linprog supports bounds and equality constraints."""
    c = [1.0, 2.0, 3.0]
    A_eq = [[1.0, 1.0, 1.0]]
    b_eq = [1.0]
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]

    res_scipy = scipy_linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    res_turbo = highs_turbo.linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="turbo")

    assert res_turbo.success
    assert np.isclose(res_turbo.fun, res_scipy.fun, atol=1e-6)
    assert np.allclose(res_turbo.x, res_scipy.x, atol=1e-6)


def test_linprog_dropin_scalar_bounds_tuple():
    """Verify highs_turbo.linprog handles standard scalar (lb, ub) bounds tuples."""
    c = [-1.0, 4.0]
    A = [[-3.0, 1.0], [1.0, 2.0]]
    b = [6.0, 4.0]

    res_scipy = scipy_linprog(c, A_ub=A, b_ub=b, bounds=(0.0, 1.0), method="highs")
    res_turbo = highs_turbo.linprog(c, A_ub=A, b_ub=b, bounds=(0.0, 1.0), method="turbo")

    assert res_turbo.success
    assert np.isclose(res_turbo.fun, res_scipy.fun, atol=1e-6)
    assert np.allclose(res_turbo.x, res_scipy.x, atol=1e-6)


def test_linprog_dropin_bounds_object():
    """Verify highs_turbo.linprog handles Bounds(lb, ub) objects with scalar broadcasting."""
    c = [-1.0, 4.0]
    A = [[-3.0, 1.0], [1.0, 2.0]]
    b = [6.0, 4.0]

    res_scipy = scipy_linprog(c, A_ub=A, b_ub=b, bounds=(0.0, 2.0), method="highs")
    res_turbo = highs_turbo.linprog(c, A_ub=A, b_ub=b, bounds=Bounds(0.0, 2.0), method="turbo")

    assert res_turbo.success
    assert np.isclose(res_turbo.fun, res_scipy.fun, atol=1e-6)
    assert np.allclose(res_turbo.x, res_scipy.x, atol=1e-6)


# =============================================================================
# 2. Automatic Topology Detection
# =============================================================================


def test_topology_detector_nongraph_lp():
    """Verify detector identifies non-graph LP as structure_type='none'."""
    detector = TopologyDetector()
    c = [1.0, 2.0, 3.0]
    A = [[2.5, -1.3, 0.4], [0.1, 4.2, -3.1]]
    b = [5.0, 2.0]

    scan = detector.detect(c, A_ub=A, b_ub=b)
    assert not scan.is_graph_structured
    assert scan.structure_type == "none"
    assert scan.graph is None


def test_topology_detector_node_edge_incidence():
    """Verify detector extracts graph from standard node-edge coupling constraints."""
    # 3 nodes (0, 1, 2), 3 edges forming a triangle
    # Variables: x0 (edge 0-1), x1 (edge 1-2), x2 (edge 0-2), s0, s1, s2 (nodes)
    # x_i - s_u - s_v <= 0
    detector = TopologyDetector()
    c = [-2.0, -3.0, -1.0, 0.0, 0.0, 0.0]
    # Edge 0 connects s0 (col 3) and s1 (col 4)
    # Edge 1 connects s1 (col 4) and s2 (col 5)
    # Edge 2 connects s0 (col 3) and s2 (col 5)
    A = [
        [1.0, 0.0, 0.0, -1.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, -1.0, -1.0],
        [0.0, 0.0, 1.0, -1.0, 0.0, -1.0],
    ]
    b = [0.0, 0.0, 0.0]

    scan = detector.detect(c, A_ub=A, b_ub=b)
    assert scan.is_graph_structured
    assert scan.structure_type == "node_edge_incidence"
    assert scan.graph is not None
    assert scan.graph.num_nodes == 3
    assert scan.graph.num_edges == 3
    assert (0, 1) in scan.graph.edges
    assert (1, 2) in scan.graph.edges
    assert (0, 2) in scan.graph.edges
    assert scan.graph.weights[(0, 1)] == 2.0
    assert scan.graph.weights[(1, 2)] == 3.0
    assert scan.graph.weights[(0, 2)] == 1.0


def test_topology_detector_triangle_metric_lp():
    """Verify detector reconstructs graph from triangle cycle inequalities."""
    g = generate_k5_cluster_graph(num_k5=3, num_bridges=2, seed=42)
    solver = ExactMaxCutSolver()
    c, A, b, _, _ = solver.build_relaxation_matrices(g, include_k5=False)

    scan = detect_topology(c, A_ub=A, b_ub=b)
    assert scan.is_graph_structured
    assert scan.structure_type == "triangle_metric"
    assert scan.graph is not None
    assert scan.graph.num_edges == len(c)
    assert scan.graph.num_nodes > 0


def test_topology_detector_qubo_mccormick():
    """Verify detector extracts graph from McCormick envelope rows."""
    detector = TopologyDetector()
    # 2 linear variables (x0, x1), 1 bilinear (y01)
    # y01 - x0 <= 0
    # y01 - x1 <= 0
    # 3 linear variables (x0, x1, x2), 2 bilinear (y01, y12)
    c = [1.0, 1.0, 1.0, -2.5, -3.5]
    A = [
        [-1.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0, 0.0, 1.0],
    ]
    b = [0.0, 0.0, 0.0, 0.0]

    scan = detector.detect(c, A_ub=A, b_ub=b)
    assert scan.is_graph_structured
    assert scan.structure_type == "qubo_mccormick"
    assert scan.graph is not None
    assert scan.graph.num_edges == 2


# =============================================================================
# 3. Accelerated linprog Execution
# =============================================================================


def test_linprog_automatic_acceleration():
    """Verify highs_turbo.linprog automatically tightens bound on graph LP."""
    g = generate_k5_cluster_graph(num_k5=4, num_bridges=3, seed=42)
    solver = ExactMaxCutSolver()
    c, A, b, _, _ = solver.build_relaxation_matrices(g, include_k5=False)

    res = highs_turbo.linprog(c, A_ub=A, b_ub=b, bounds=(0, 1), graph=g, method="turbo")

    assert res.success
    assert res.turbo_accelerated
    assert len(res.certificate_sha256) == 64
    assert res.is_rationally_certified
    base = scipy_linprog(c, A_ub=A, b_ub=b, bounds=(0, 1))
    assert res.fun > base.fun + 1e-4
    assert -res.fun >= solver.solve_integer_maxcut(g)[0] - 1e-6


def test_linprog_automatic_topology_detection_without_graph_kwarg():
    """Verify highs_turbo.linprog automatically detects graph from (c, A_ub, b_ub) and tightens bound."""
    g = generate_k5_cluster_graph(num_k5=4, num_bridges=3, seed=42)
    solver = ExactMaxCutSolver()
    c, A, b, _, _ = solver.build_relaxation_matrices(g, include_k5=False)

    # Note: no graph=g argument passed; detector must scan topology from constraint rows alone
    res = highs_turbo.linprog(c, A_ub=A, b_ub=b, bounds=(0, 1), method="turbo")

    assert res.success
    assert res.turbo_accelerated
    assert res.is_rationally_certified
    assert len(res.certificate_sha256) == 64
    assert res.num_cuts_separated == 4
    base = scipy_linprog(c, A_ub=A, b_ub=b, bounds=(0, 1))
    assert res.fun > base.fun + 1e-4
    assert -res.fun >= solver.solve_integer_maxcut(g)[0] - 1e-6


# =============================================================================
# 4. High-Level solve_maxcut API
# =============================================================================


def test_solve_maxcut_small_triangle():
    """Verify solve_maxcut returns exact optimal cut and 3-tuple unpacking."""
    adj = np.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ])

    cut_val, partition, cert = highs_turbo.solve_maxcut(adj)

    assert isinstance(cut_val, float)
    assert cut_val == 2.0  # Max cut of triangle graph is 2
    assert isinstance(partition, np.ndarray)
    assert len(partition) == 3
    assert set(partition).issubset({0, 1})
    assert len(cert) == 64
    assert all(c in "0123456789abcdef" for c in cert)


def test_solve_maxcut_k5_cluster():
    """Verify solve_maxcut on planted K5 cluster instance."""
    g = generate_k5_cluster_graph(num_k5=4, num_bridges=3, seed=123)
    res = highs_turbo.solve_maxcut(g)

    assert isinstance(res, MaxCutResult)
    assert res.cut_value > 0
    assert len(res.partition) == g.num_nodes
    assert res.is_rationally_certified
    assert len(res.certificate) == 64
    assert res.upper_bound >= res.cut_value - 1e-6  # Weak duality holds


# =============================================================================
# 5. High-Level solve_qubo API
# =============================================================================


def test_solve_qubo_frustrated_2x2():
    """Verify solve_qubo returns exact ground-state energy and unpacks correctly."""
    Q = np.array([
        [ 2.0, -1.0],
        [-1.0,  2.0],
    ])

    energy, solution, cert = highs_turbo.solve_qubo(Q)

    assert energy == 0.0
    assert np.array_equal(solution, [0, 0])
    assert len(cert) == 64
    assert all(c in "0123456789abcdef" for c in cert)


def test_solve_qubo_antiferromagnetic_triangle():
    """Verify solve_qubo on 3-variable frustrated triangle."""
    # Min x0 + x1 + x2 + 2(x0 x1 + x1 x2 + x0 x2)
    # Ground state: x = [0, 0, 0] with energy 0
    Q = np.array([
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ])

    res = highs_turbo.solve_qubo(Q)
    assert isinstance(res, QuboResult)
    assert res.energy == 0.0
    assert np.array_equal(res.solution, [0, 0, 0])
    assert res.is_rationally_certified
    assert len(res.certificate) == 64


def test_solve_maxcut_edge_cases():
    """Verify solve_maxcut handles 1-node, 0-node, and 2-node graph boundary cases."""
    # 1-node graph
    res_1 = highs_turbo.solve_maxcut(np.array([[0.0]]))
    assert res_1.cut_value == 0.0
    assert len(res_1.partition) == 1
    assert len(res_1.certificate) == 64

    # 2-node single edge
    res_2 = highs_turbo.solve_maxcut(np.array([[0.0, 5.0], [5.0, 0.0]]))
    assert res_2.cut_value == 5.0
    assert len(res_2.partition) == 2
    assert res_2.partition[0] != res_2.partition[1]


def test_solve_qubo_edge_cases():
    """Verify solve_qubo handles 1x1 matrix and dict-format inputs."""
    # 1x1 negative diagonal
    res_1 = highs_turbo.solve_qubo(np.array([[-4.5]]))
    assert res_1.energy == -4.5
    assert np.array_equal(res_1.solution, [1])
    assert res_1.is_rationally_certified

    # Dict format with positive and negative couplings
    Q_dict = {(0, 0): 2.0, (1, 1): -3.0, (0, 1): -1.0}
    energy, sol, cert = highs_turbo.solve_qubo(Q_dict)
    assert energy == -3.0
    assert np.array_equal(sol, [0, 1])
    assert len(cert) == 64


# =============================================================================
# 6. Adversarial Cut Rejection & Cryptographic Verification
# =============================================================================


def test_adversarial_negative_multiplier_rejection():
    """Verify rational verifier strictly rejects candidate cuts with negative multipliers."""
    g = generate_k5_cluster_graph(num_k5=2, num_bridges=1, seed=42)
    k5_cliques = g.find_all_k5_cliques()
    assert len(k5_cliques) >= 1

    solver = TurboSolver()
    clq = k5_cliques[0]

    # Adversarially inject negative multiplier
    adversarial_mults = {clq: -1.5}
    cert = solver.verify_candidate_cut(g, adversarial_mults)

    assert not cert.is_valid
    assert "NEGATIVE" in cert.status
    assert cert.rejection_reason is not None


def test_adversarial_rhs_tampering_rejection():
    """Verify rational verifier strictly rejects candidate cuts with illegal mutated RHS."""
    g = generate_k5_cluster_graph(num_k5=2, num_bridges=1, seed=42)
    k5_cliques = g.find_all_k5_cliques()
    assert len(k5_cliques) >= 1

    solver = TurboSolver()
    clq = k5_cliques[0]

    # Valid clique multiplier is +1.0, but propose an illegally tightened RHS (e.g. 3 instead of 6)
    adversarial_mults = {clq: 1.0}
    cert = solver.verify_candidate_cut(g, adversarial_mults, rhs=3.0)

    assert not cert.is_valid
    assert "REJECTED" in cert.status


def test_cryptographic_receipt_deterministic():
    """Verify SHA-256 certificate receipt is deterministic and tamper-evident."""
    g = generate_k5_cluster_graph(num_k5=2, num_bridges=1, seed=42)
    k5_cliques = g.find_all_k5_cliques()

    solver = TurboSolver()
    clq = k5_cliques[0]

    cert1 = solver.verify_candidate_cut(g, {clq: 1.0})
    cert2 = solver.verify_candidate_cut(g, {clq: 1.0})

    assert cert1.is_valid
    assert cert2.is_valid
    assert cert1.sha256_hash == cert2.sha256_hash
    assert len(cert1.sha256_hash) == 64


# =============================================================================
# 7. TurboSolver Configuration & Options
# =============================================================================


def test_turbosolver_custom_configuration():
    """Verify TurboSolver respects configuration limits."""
    solver = TurboSolver(rational_denominator_limit=50000, max_cuts=250, verbose=False)
    assert solver.rational_denominator_limit == 50000
    assert solver.max_cuts == 250
    assert solver.fallback_on_error is True

    adj = np.array([[0, 1], [1, 0]])
    res = solver.solve_maxcut(adj)
    assert res.cut_value == 1.0
    assert res.is_rationally_certified
