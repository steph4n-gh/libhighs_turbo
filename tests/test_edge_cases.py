"""Edge case and boundary value tests for neural surrogate engine."""

from fractions import Fraction
import numpy as np
import pytest
import torch

from highs_turbo.graph_generator import GraphInstance, generate_ab_pair
from highs_turbo.exact_solver import ExactMaxCutSolver
from highs_turbo.rational_verifier import RationalCutVerifier
from highs_turbo.surrogate_model import (
    EdgeEquivariantSurrogateGNN,
    extract_topological_features,
)


def test_empty_and_trivial_graphs():
    # Empty graph
    g_empty = GraphInstance(name="empty", num_nodes=0, edges=[])
    assert g_empty.num_nodes == 0
    assert g_empty.num_edges == 0
    assert g_empty.find_all_k5_cliques() == []
    assert g_empty.find_triangles() == []
    nf0, ef0, _ = extract_topological_features(g_empty)
    assert nf0.shape == (0, 5)
    assert ef0.shape == (0, 8)

    # Single isolated vertex
    g_single = GraphInstance(name="single", num_nodes=1, edges=[])
    assert g_single.num_nodes == 1
    assert g_single.num_edges == 0
    assert g_single.find_all_k5_cliques() == []
    assert g_single.find_triangles() == []
    nf1, ef1, _ = extract_topological_features(g_single)
    assert nf1.shape == (1, 5)
    assert ef1.shape == (0, 8)


def test_triangle_free_bipartite_graph():
    # Complete bipartite graph K_{3,3}: no triangles, no K5
    # Max-Cut of K_{3,3} is 9 (all edges).
    edges = []
    for u in range(3):
        for v in range(3, 6):
            edges.append((u, v))
    g = GraphInstance(name="k3_3", num_nodes=6, edges=edges)
    assert g.find_triangles() == []
    assert g.find_all_k5_cliques() == []

    solver = ExactMaxCutSolver()
    ip_val, _ = solver.solve_integer_maxcut(g)
    assert int(round(ip_val)) == 9

    sol_cycle = solver.solve_cycle_relaxation(g)
    assert abs(sol_cycle.objective_value - 9.0) < 1e-4


def test_verifier_tiny_negative_multiplier():
    verifier = RationalCutVerifier()
    # A graph with 1 K5
    import itertools
    edges = list(itertools.combinations(range(5), 2))
    g = GraphInstance(name="k5", num_nodes=5, edges=edges)
    k5 = (0, 1, 2, 3, 4)

    # Multiplier is -1e-12 (strictly negative)
    cert = verifier.verify_clique_conic_combination(g, {k5: -1e-12})
    # Must reject negative multipliers fail-closed
    assert not cert.is_valid
    assert cert.status == "REJECTED_NEGATIVE_MULTIPLIER"


def test_verifier_all_zero_multipliers():
    verifier = RationalCutVerifier()
    import itertools
    edges = list(itertools.combinations(range(5), 2))
    g = GraphInstance(name="k5", num_nodes=5, edges=edges)
    k5 = (0, 1, 2, 3, 4)

    # Multiplier is 0.0
    cert = verifier.verify_clique_conic_combination(g, {k5: 0.0})
    assert cert.is_valid
    assert cert.num_active_supports == 0
    assert cert.exact_rhs == Fraction(0, 1)


def test_gnn_multiple_random_permutations():
    # Test equivariance across 10 random permutations
    from highs_turbo.graph_generator import generate_k5_cluster_graph
    g = generate_k5_cluster_graph(num_k5=3, num_bridges=2, seed=55)
    model = EdgeEquivariantSurrogateGNN(hidden_dim=16, num_layers=2)
    model.eval()

    with torch.no_grad():
        out_orig = model(g)
        rhs_orig = out_orig["candidate_rhs"]
        mults_orig = out_orig["k5_multipliers"]

        for seed in range(10):
            _, g_perm, pi, _ = generate_ab_pair(g, seed=seed)
            out_perm = model(g_perm)
            rhs_perm = out_perm["candidate_rhs"]
            assert abs(rhs_orig - rhs_perm) < 1e-4

            mults_perm = out_perm["k5_multipliers"]
            for clq, val in mults_orig.items():
                perm_clq = tuple(sorted([pi[v] for v in clq]))
                assert perm_clq in mults_perm
                assert abs(val - mults_perm[perm_clq]) < 1e-4


def test_disconnected_graph_with_k5_and_isolated_nodes():
    import itertools
    # 5 nodes form a K5, 2 nodes form an isolated edge, 3 nodes are completely isolated
    k5_edges = list(itertools.combinations(range(5), 2))
    edges = k5_edges + [(5, 6)]
    num_nodes = 10  # 0..4 (K5), 5..6 (edge), 7..9 (isolated)

    g = GraphInstance(name="disconnected_test", num_nodes=num_nodes, edges=edges)
    assert len(g.find_all_k5_cliques()) == 1

    solver = ExactMaxCutSolver()
    ip_val, _ = solver.solve_integer_maxcut(g)
    # K5 MaxCut = 6, isolated edge = 1 => 7
    assert int(round(ip_val)) == 7

    sol_k5 = solver.solve_k5_relaxation(g)
    assert abs(sol_k5.objective_value - 7.0) < 1e-4

    center = solver.compute_face_invariant_dual_center(g)
    assert len(center) == 1
    assert abs(center[(0, 1, 2, 3, 4)] - 1.0) < 1e-4
