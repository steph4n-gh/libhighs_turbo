"""Tier 2: Boundaries and Corner Cases E2E Tests.

Tests extreme graph topologies and boundary conditions per requirements R1-R5:
- Empty graphs (0 nodes, 0 edges) and single-isolated-node graphs
- Single-edge graphs
- Triangle graphs (K3)
- Complete graph K5
- Disjoint components (isolated nodes, multiple disconnected subgraphs)
- Trees and bipartite graphs with 0 triangles (paths, stars, K_{4,4}, K_{5,5})
- Large dense cliques (K6, K7, K8)
- Boundary thresholds for cut violation (tight equality, epsilon margins)
- Near-zero, exact zero, and micro-negative multipliers

Coverage threshold: >= 5 boundary/corner cases per feature (total >= 20 tests).
"""

from __future__ import annotations

import itertools
from fractions import Fraction
import numpy as np
import pytest

from highs_turbo.exact_solver import ExactMaxCutSolver
from highs_turbo.graph_generator import GraphInstance, generate_k5_cluster_graph
from highs_turbo.rational_verifier import RationalCutVerifier
from tests.e2e.topologies import BitParallelGraph


@pytest.fixture
def solver():
    return ExactMaxCutSolver()


@pytest.fixture
def verifier():
    return RationalCutVerifier()


# =====================================================================
# Boundary 1: Empty & Trivial Graphs
# =====================================================================

def test_boundary_empty_graph(solver, verifier):
    """Verify empty graph (0 nodes, 0 edges) behavior across solver, verifier, and bit-graph."""
    g = GraphInstance(name="empty_0", num_nodes=0, edges=[])
    bg = BitParallelGraph.from_graph_instance(g)

    assert bg.num_nodes == 0
    assert bg.words_per_row == 0
    assert bg.find_triangles() == []
    assert bg.find_k5_cliques() == []

    cert = verifier.verify_clique_conic_combination(g, {})
    assert cert.is_valid
    assert cert.num_active_supports == 0
    assert cert.exact_rhs == Fraction(0, 1)


def test_boundary_single_isolated_vertex(solver, verifier):
    """Verify single isolated vertex (1 node, 0 edges) handles relaxation and verification."""
    g = GraphInstance(name="single_node", num_nodes=1, edges=[])
    bg = BitParallelGraph.from_graph_instance(g)

    assert bg.num_nodes == 1
    assert bg.words_per_row == 1
    assert bg.common_neighbors(0, 0) == 0
    assert bg.find_triangles() == []
    assert bg.find_k5_cliques() == []

    # Verifier handles empty cut on single node
    cert = verifier.verify_clique_conic_combination(g, {})
    assert cert.is_valid
    assert cert.exact_rhs == Fraction(0, 1)


# =====================================================================
# Boundary 2: Single-Edge Graph
# =====================================================================

def test_boundary_single_edge_graph(solver, verifier):
    """Verify single-edge graph has exact integer Max-Cut = 1.0 and identical LP bounds."""
    g = GraphInstance(name="single_edge", num_nodes=2, edges=[(0, 1)])
    bg = BitParallelGraph.from_graph_instance(g)

    assert bg.has_edge(0, 1)
    assert bg.common_neighbors(0, 1) == 0
    assert bg.find_triangles() == []
    assert bg.find_k5_cliques() == []

    val, _ = solver.solve_integer_maxcut(g)
    assert abs(val - 1.0) < 1e-6

    sol_cycle = solver.solve_cycle_relaxation(g)
    assert abs(sol_cycle.objective_value - 1.0) < 1e-6


def test_boundary_single_edge_verifier_out_of_bounds(verifier):
    """Verify attempting to verify a K5 on a single-edge graph fails cleanly."""
    g = GraphInstance(name="single_edge", num_nodes=2, edges=[(0, 1)])
    cert = verifier.verify_clique_conic_combination(g, {(0, 1, 2, 3, 4): 1.0})
    assert not cert.is_valid
    assert cert.status == "REJECTED_OUT_OF_BOUNDS_NODE"


# =====================================================================
# Boundary 3: Triangle Graph (K3)
# =====================================================================

def test_boundary_triangle_graph_properties(solver, verifier):
    """Verify triangle graph K3 Max-Cut = 2 and cycle inequality verification."""
    edges = [(0, 1), (1, 2), (0, 2)]
    g = GraphInstance(name="k3", num_nodes=3, edges=edges)
    bg = BitParallelGraph.from_graph_instance(g)

    assert bg.find_triangles() == [(0, 1, 2)]
    assert bg.count_triangles() == 1
    assert bg.common_neighbors(0, 1) == 1

    ip_val, _ = solver.solve_integer_maxcut(g)
    assert int(round(ip_val)) == 2

    # Cycle inequality: x_01 + x_12 + x_02 <= 2
    candidate_cycles = [
        ([0, 1, 2], [(0, 1), (1, 2), (0, 2)], Fraction(1, 1)),
    ]
    cert = verifier.verify_cycle_conic_combination(g, candidate_cycles)
    assert cert.is_valid
    assert cert.exact_rhs == Fraction(2, 1)


# =====================================================================
# Boundary 4: Complete Graph K5
# =====================================================================

def test_boundary_complete_graph_k5_tightness(solver, verifier):
    """Verify complete graph K5 facet tightness and cycle gap closure."""
    edges = list(itertools.combinations(range(5), 2))
    g = GraphInstance(name="k5", num_nodes=5, edges=edges)
    bg = BitParallelGraph.from_graph_instance(g)

    assert len(bg.find_k5_cliques()) == 1
    assert bg.count_triangles() == 10

    sol_cycle = solver.solve_cycle_relaxation(g)
    # Cycle bound is 10 * 2/3 = 20/3
    assert abs(sol_cycle.objective_value - 20.0 / 3.0) < 1e-4

    sol_k5 = solver.solve_k5_relaxation(g)
    assert abs(sol_k5.objective_value - 6.0) < 1e-4


def test_boundary_complete_graph_k5_missing_edge(verifier):
    """Verify removing a single edge from K5 causes verification rejection."""
    edges = list(itertools.combinations(range(5), 2))
    edges.remove((0, 1))  # Remove edge (0, 1)
    g = GraphInstance(name="k5_minus_1", num_nodes=5, edges=edges)

    cert = verifier.verify_clique_conic_combination(g, {(0, 1, 2, 3, 4): 1.0})
    assert not cert.is_valid
    assert cert.status == "REJECTED_NON_CLIQUE_SUPPORT"


# =====================================================================
# Boundary 5: Disjoint Components
# =====================================================================

def test_boundary_disjoint_k5_and_isolated_nodes(solver):
    """Verify disconnected graph with 1 K5 and 20 isolated vertices."""
    k5_edges = list(itertools.combinations(range(5), 2))
    num_nodes = 25  # nodes 0..4 in K5, nodes 5..24 isolated
    g = GraphInstance(name="k5_isolated_20", num_nodes=num_nodes, edges=k5_edges)

    ip_val, _ = solver.solve_integer_maxcut(g)
    assert int(round(ip_val)) == 6

    sol_k5 = solver.solve_k5_relaxation(g)
    assert abs(sol_k5.objective_value - 6.0) < 1e-4


def test_boundary_disjoint_forest_and_k5(solver):
    """Verify disconnected graph with 1 K5, 2 separate trees, and 3 isolated edges."""
    k5_edges = list(itertools.combinations(range(5), 2))
    tree1 = [(5, 6), (6, 7), (6, 8)]  # 3 edges
    tree2 = [(9, 10), (10, 11), (11, 12)]  # 3 edges
    edges_isolated = [(13, 14), (15, 16), (17, 18)]  # 3 edges
    all_edges = k5_edges + tree1 + tree2 + edges_isolated

    g = GraphInstance(name="k5_trees_edges", num_nodes=20, edges=all_edges)
    ip_val, _ = solver.solve_integer_maxcut(g)
    # K5 gives 6, tree1 gives 3, tree2 gives 3, isolated gives 3 => 6 + 3 + 3 + 3 = 15
    assert int(round(ip_val)) == 15

    sol_k5 = solver.solve_k5_relaxation(g)
    assert abs(sol_k5.objective_value - 15.0) < 1e-4


def test_boundary_completely_disconnected_nodes(solver):
    """Verify graph with 50 nodes and 0 edges, and disconnected graph with 1 edge + 48 isolated nodes."""
    g_empty = GraphInstance(name="disconnected_50", num_nodes=50, edges=[])
    bg = BitParallelGraph.from_graph_instance(g_empty)
    assert bg.num_nodes == 50
    assert bg.words_per_row == 1
    assert bg.find_triangles() == []
    assert bg.find_k5_cliques() == []

    # 1 edge and 48 isolated nodes (50 nodes total)
    g_one_edge = GraphInstance(name="one_edge_48_isolated", num_nodes=50, edges=[(0, 1)])
    val, part = solver.solve_integer_maxcut(g_one_edge)
    assert abs(val - 1.0) < 1e-6
    assert len(part) == 50


# =====================================================================
# Boundary 6: Trees and Bipartite Graphs (0 Triangles)
# =====================================================================

def test_boundary_star_graph(solver):
    """Verify star graph S_10 (bipartite, 0 triangles, integrality gap = 0)."""
    edges = [(0, i) for i in range(1, 10)]
    g = GraphInstance(name="star_10", num_nodes=10, edges=edges)
    bg = BitParallelGraph.from_graph_instance(g)

    assert bg.find_triangles() == []
    assert bg.find_k5_cliques() == []

    ip_val, _ = solver.solve_integer_maxcut(g)
    assert int(round(ip_val)) == 9

    sol_cycle = solver.solve_cycle_relaxation(g)
    assert abs(sol_cycle.objective_value - 9.0) < 1e-6


def test_boundary_path_graph_long(solver):
    """Verify long path graph P_40 (bipartite, 0 triangles)."""
    edges = [(i, i + 1) for i in range(39)]
    g = GraphInstance(name="path_40", num_nodes=40, edges=edges)

    ip_val, _ = solver.solve_integer_maxcut(g)
    assert int(round(ip_val)) == 39

    sol_cycle = solver.solve_cycle_relaxation(g)
    assert abs(sol_cycle.objective_value - 39.0) < 1e-6


def test_boundary_complete_bipartite_k44(solver):
    """Verify complete bipartite graph K_{4,4} (16 edges, 0 triangles)."""
    edges = [(u, v) for u in range(4) for v in range(4, 8)]
    g = GraphInstance(name="k4_4", num_nodes=8, edges=edges)
    bg = BitParallelGraph.from_graph_instance(g)

    assert bg.find_triangles() == []
    assert bg.find_k5_cliques() == []

    ip_val, _ = solver.solve_integer_maxcut(g)
    assert int(round(ip_val)) == 16

    sol_cycle = solver.solve_cycle_relaxation(g)
    assert abs(sol_cycle.objective_value - 16.0) < 1e-6


def test_boundary_complete_bipartite_k55(solver):
    """Verify complete bipartite graph K_{5,5} (25 edges, 0 triangles)."""
    edges = [(u, v) for u in range(5) for v in range(5, 10)]
    g = GraphInstance(name="k5_5", num_nodes=10, edges=edges)

    ip_val, _ = solver.solve_integer_maxcut(g)
    assert int(round(ip_val)) == 25

    sol_cycle = solver.solve_cycle_relaxation(g)
    assert abs(sol_cycle.objective_value - 25.0) < 1e-6


# =====================================================================
# Boundary 7: Large Dense Cliques (K6, K7, K8)
# =====================================================================

def test_boundary_dense_clique_k6(solver, verifier):
    """Verify complete graph K6 contains exactly 6 K5 subgraphs and closes gap."""
    edges = list(itertools.combinations(range(6), 2))
    g = GraphInstance(name="k6", num_nodes=6, edges=edges)
    bg = BitParallelGraph.from_graph_instance(g)

    k5_list = bg.find_k5_cliques()
    assert len(k5_list) == 6  # (6 choose 5) = 6

    # Max-Cut of K6 is floor(6^2 / 4) = 9
    ip_val, _ = solver.solve_integer_maxcut(g)
    assert int(round(ip_val)) == 9

    sol_cycle = solver.solve_cycle_relaxation(g)
    # Cycle bound on K6: 15 * 2/3 = 10.0
    assert abs(sol_cycle.objective_value - 10.0) < 1e-4

    sol_k5 = solver.solve_k5_relaxation(g)
    assert abs(sol_k5.objective_value - 9.0) < 1e-4


def test_boundary_dense_clique_k7(solver):
    """Verify complete graph K7 contains exactly 21 K5 subgraphs."""
    edges = list(itertools.combinations(range(7), 2))
    g = GraphInstance(name="k7", num_nodes=7, edges=edges)
    bg = BitParallelGraph.from_graph_instance(g)

    k5_list = bg.find_k5_cliques()
    assert len(k5_list) == 21  # (7 choose 5) = 21

    # Max-Cut of K7 is floor(7 * 7 / 4) = 12
    ip_val, _ = solver.solve_integer_maxcut(g)
    assert int(round(ip_val)) == 12


def test_boundary_dense_clique_k8(solver):
    """Verify complete graph K8 contains exactly 56 K5 subgraphs."""
    edges = list(itertools.combinations(range(8), 2))
    g = GraphInstance(name="k8", num_nodes=8, edges=edges)
    bg = BitParallelGraph.from_graph_instance(g)

    k5_list = bg.find_k5_cliques()
    assert len(k5_list) == 56  # (8 choose 5) = 56

    # Max-Cut of K8 is floor(8 * 8 / 4) = 16
    ip_val, _ = solver.solve_integer_maxcut(g)
    assert int(round(ip_val)) == 16


# =====================================================================
# Boundary 8: Boundary Thresholds for Cut Violation
# =====================================================================

def test_boundary_threshold_exact_equality(verifier):
    """Verify candidate RHS matching exact sound bound passes."""
    edges = list(itertools.combinations(range(5), 2))
    g = GraphInstance(name="k5", num_nodes=5, edges=edges)
    k5 = (0, 1, 2, 3, 4)

    # Exact RHS is 6.0. Claim 6.0 exactly.
    cert = verifier.verify_clique_conic_combination(g, {k5: 1.0}, candidate_rhs=6.0)
    assert cert.is_valid
    assert cert.exact_rhs == Fraction(6, 1)


def test_boundary_threshold_tolerance_epsilon(verifier):
    """Verify tolerance threshold: claim lower than sound bound by more than tol is rejected."""
    edges = list(itertools.combinations(range(5), 2))
    g = GraphInstance(name="k5", num_nodes=5, edges=edges)
    k5 = (0, 1, 2, 3, 4)

    # Sound RHS is 6.0. Claim 5.9999 (deficit of 1e-4 > tol 1e-6)
    cert_reject = verifier.verify_clique_conic_combination(g, {k5: 1.0}, candidate_rhs=5.9999, tol=1e-6)
    assert not cert_reject.is_valid
    assert cert_reject.status == "REJECTED_UNSOUND_RHS_DEFICIT"

    # Deficit within 1e-7 < tol 1e-6 should pass
    cert_pass = verifier.verify_clique_conic_combination(g, {k5: 1.0}, candidate_rhs=6.0 - 1e-7, tol=1e-6)
    assert cert_pass.is_valid


# =====================================================================
# Boundary 9: Near-Zero and Negative Zero Multipliers
# =====================================================================

def test_boundary_near_zero_positive_multiplier(verifier):
    """Verify tiny positive multiplier (1e-12) is safely converted and certified."""
    edges = list(itertools.combinations(range(5), 2))
    g = GraphInstance(name="k5", num_nodes=5, edges=edges)
    k5 = (0, 1, 2, 3, 4)

    # 1e-12 limit_denominator should produce 0 or positive fraction
    cert = verifier.verify_clique_conic_combination(g, {k5: 1e-12})
    assert cert.is_valid


def test_boundary_strictly_negative_micro_multiplier(verifier):
    """Verify micro-negative multiplier (-1e-15) is rejected fail-closed."""
    edges = list(itertools.combinations(range(5), 2))
    g = GraphInstance(name="k5", num_nodes=5, edges=edges)
    k5 = (0, 1, 2, 3, 4)

    cert = verifier.verify_clique_conic_combination(g, {k5: -1e-15})
    assert not cert.is_valid
    assert cert.status == "REJECTED_NEGATIVE_MULTIPLIER"


def test_boundary_all_zero_multipliers(verifier):
    """Verify all-zero multipliers dictionary produces zero active supports."""
    edges = list(itertools.combinations(range(5), 2))
    g = GraphInstance(name="k5", num_nodes=5, edges=edges)
    k5 = (0, 1, 2, 3, 4)

    cert = verifier.verify_clique_conic_combination(g, {k5: 0.0})
    assert cert.is_valid
    assert cert.num_active_supports == 0
    assert cert.exact_rhs == Fraction(0, 1)


def test_boundary_mixed_zero_and_positive_multipliers(verifier):
    """Verify filtering of zero multipliers when mixed with positive multipliers."""
    g = generate_k5_cluster_graph(num_k5=2, num_bridges=1, seed=42)
    c1, c2 = g.find_all_k5_cliques()

    cert = verifier.verify_clique_conic_combination(g, {c1: 1.0, c2: 0.0})
    assert cert.is_valid
    assert cert.num_active_supports == 1
    assert cert.exact_rhs == Fraction(6, 1)
