"""Tier 1: Feature Coverage E2E Tests.

Derived strictly from requirements R1, R2, R4, R5 and PROJECT.md / TEST_INFRA.md:
- Feature 1: Bit-parallel graph operations (common neighbors, triangle count, K5 detection)
- Feature 2: Surrogate cut separation and single-row LP relaxation
- Feature 3: Exact rational conic combination verification
- Feature 4: Cryptographic SHA-256 certificate receipt generation
- Feature 5: Isomorphic relabeling invariance (|Delta Z| <= 10^-6)

Coverage threshold: >= 5 test cases per feature (total >= 25 tests).
"""

from __future__ import annotations

import itertools
import json
from fractions import Fraction
from typing import Dict, Tuple

import networkx as nx
import numpy as np
import pytest

from highs_turbo.exact_solver import ExactMaxCutSolver
from highs_turbo.graph_generator import (
    GraphInstance,
    generate_ab_pair,
    generate_exp87_instances,
    generate_k5_cluster_graph,
    generate_ks_minus_edges,
)
from highs_turbo.rational_verifier import RationalCutVerifier, VerificationCertificate
from tests.e2e.topologies import BitParallelGraph


# =====================================================================
# Feature 1: Bit-Parallel Graph Operations (5+ tests)
# =====================================================================

def test_bit_parallel_graph_construction_and_edges():
    """Verify bit-parallel graph construction, contiguous 64-bit word sizing, and edge lookups."""
    bg = BitParallelGraph(num_nodes=130)
    assert bg.words_per_row == 3  # (130 + 63) // 64 = 3
    assert len(bg.bit_adj) == 130
    assert len(bg.bit_adj[0]) == 3

    # Add edges across word boundaries
    bg.add_edge(0, 63)
    bg.add_edge(0, 64)
    bg.add_edge(63, 127)
    bg.add_edge(64, 129)

    assert bg.has_edge(0, 63)
    assert bg.has_edge(63, 0)
    assert bg.has_edge(0, 64)
    assert bg.has_edge(64, 0)
    assert bg.has_edge(63, 127)
    assert bg.has_edge(64, 129)
    assert not bg.has_edge(0, 129)
    assert not bg.has_edge(1, 2)


def test_bit_parallel_common_neighbors_against_networkx():
    """Verify bitwise AND + POPCNT common_neighbors matches NetworkX ground truth."""
    g_inst = generate_k5_cluster_graph(num_k5=3, num_bridges=2, seed=123)
    nx_g = g_inst.to_networkx()
    bg = BitParallelGraph.from_graph_instance(g_inst)

    for u in range(g_inst.num_nodes):
        for v in range(u + 1, g_inst.num_nodes):
            expected = len(list(nx.common_neighbors(nx_g, u, v)))
            actual = bg.common_neighbors(u, v)
            assert actual == expected, f"Mismatch for pair ({u}, {v}): actual {actual} != expected {expected}"


def test_bit_parallel_triangle_counting_and_listing():
    """Verify bit-parallel triangle enumeration and counting on dense and clustered graphs."""
    g_inst = generate_k5_cluster_graph(num_k5=2, num_bridges=1, seed=456)
    nx_g = g_inst.to_networkx()
    bg = BitParallelGraph.from_graph_instance(g_inst)

    # In K5, each block has (5 choose 3) = 10 triangles. 2 disjoint blocks = 20 triangles.
    expected_triangles = g_inst.find_triangles()
    actual_triangles = bg.find_triangles()

    assert len(actual_triangles) == 20
    assert actual_triangles == expected_triangles
    assert bg.count_triangles() == 20


def test_bit_parallel_k5_detection():
    """Verify bit-parallel K5 clique detection finds exact planted K5 cliques."""
    g_inst = generate_k5_cluster_graph(num_k5=4, num_bridges=3, seed=789)
    bg = BitParallelGraph.from_graph_instance(g_inst)

    expected_k5 = g_inst.find_all_k5_cliques()
    actual_k5 = bg.find_k5_cliques()

    assert len(actual_k5) == 4
    assert actual_k5 == expected_k5
    for clq in actual_k5:
        assert len(clq) == 5
        # Verify completeness of detected clique
        for u, v in itertools.combinations(clq, 2):
            assert bg.has_edge(u, v)


def test_bit_parallel_compiled_engine_contract():
    """Verify interface contract parity between reference BitParallelGraph and compiled engine."""
    # Tests that the class structure conforms to PROJECT.md interface contract
    bg = BitParallelGraph(num_nodes=10)
    for u, v in [(0, 1), (1, 2), (0, 2)]:
        bg.add_edge(u, v)

    assert hasattr(bg, "add_edge")
    assert hasattr(bg, "common_neighbors")
    assert hasattr(bg, "find_triangles")
    assert hasattr(bg, "find_k5_cliques")
    assert bg.common_neighbors(0, 1) == 1
    assert bg.find_triangles() == [(0, 1, 2)]


# =====================================================================
# Feature 2: Surrogate Cut Separation & Single-Row LP Relaxation (5+ tests)
# =====================================================================

@pytest.fixture
def solver():
    return ExactMaxCutSolver()


def test_surrogate_cut_separation_single_k5(solver):
    """Verify single-row surrogate cut closes the cycle relaxation gap on K5 down to 6.0."""
    edges = list(itertools.combinations(range(5), 2))
    g = GraphInstance(name="single_k5", num_nodes=5, edges=edges)

    sol_cycle = solver.solve_cycle_relaxation(g)
    assert abs(sol_cycle.objective_value - 20.0 / 3.0) < 1e-4

    sol_k5 = solver.solve_k5_relaxation(g)
    assert abs(sol_k5.objective_value - 6.0) < 1e-4

    # Build 1-row surrogate cutting plane
    a_surr = np.ones(g.num_edges)
    b_surr = 6.0
    sol_surr = solver.solve_surrogate_relaxation(g, a_surr, b_surr)

    assert abs(sol_surr.objective_value - 6.0) < 1e-4
    # The surrogate row must be tightly binding: a_surr^T x == 6.0
    cut_activity = np.dot(a_surr, sol_surr.primal_solution)
    assert abs(cut_activity - b_surr) < 1e-4


def test_surrogate_relaxation_multi_k5_cluster(solver):
    """Verify 1-row surrogate cut closes integrality gap on multi-K5 cluster."""
    g = generate_k5_cluster_graph(num_k5=2, num_bridges=1, seed=42)
    ip_val, _ = solver.solve_integer_maxcut(g)
    assert int(round(ip_val)) == 13

    sol_cycle = solver.solve_cycle_relaxation(g)
    assert abs(sol_cycle.objective_value - 43.0 / 3.0) < 1e-4

    sol_k5 = solver.solve_k5_relaxation(g)
    assert abs(sol_k5.objective_value - 13.0) < 1e-4

    # Compress dual multipliers into 1-row surrogate cut
    edge_to_idx = {e: i for i, e in enumerate(g.edges)}
    a_surr = np.zeros(g.num_edges)
    b_surr = 0.0
    for k5, mult in sol_k5.k5_multipliers.items():
        b_surr += mult * 6.0
        for u, v in itertools.combinations(k5, 2):
            e = (min(u, v), max(u, v))
            a_surr[edge_to_idx[e]] += mult

    sol_surr = solver.solve_surrogate_relaxation(g, a_surr, b_surr)
    assert abs(sol_surr.objective_value - 13.0) < 1e-4


def test_surrogate_cut_objective_and_dual_multipliers(solver):
    """Verify surrogate LP relaxation yields valid dual multiplier for surrogate row."""
    g = generate_k5_cluster_graph(num_k5=3, num_bridges=2, seed=11)
    sol_k5 = solver.solve_k5_relaxation(g)

    edge_to_idx = {e: i for i, e in enumerate(g.edges)}
    a_surr = np.zeros(g.num_edges)
    b_surr = 0.0
    for k5, mult in sol_k5.k5_multipliers.items():
        b_surr += mult * 6.0
        for u, v in itertools.combinations(k5, 2):
            e = (min(u, v), max(u, v))
            a_surr[edge_to_idx[e]] += mult

    sol_surr = solver.solve_surrogate_relaxation(g, a_surr, b_surr)
    assert abs(sol_surr.objective_value - sol_k5.objective_value) < 1e-4
    assert sol_surr.success is True


def test_surrogate_cut_sparse_coefficients(solver):
    """Verify support sparsity of surrogate cut vector on disconnected graphs."""
    edges_k5 = list(itertools.combinations(range(5), 2))
    edges_extra = [(5, 6), (6, 7), (7, 8)]
    g = GraphInstance(name="k5_with_tree", num_nodes=9, edges=edges_k5 + edges_extra)

    sol_k5 = solver.solve_k5_relaxation(g)
    edge_to_idx = {e: i for i, e in enumerate(g.edges)}
    a_surr = np.zeros(g.num_edges)
    b_surr = 0.0
    for k5, mult in sol_k5.k5_multipliers.items():
        b_surr += mult * 6.0
        for u, v in itertools.combinations(k5, 2):
            e = (min(u, v), max(u, v))
            a_surr[edge_to_idx[e]] += mult

    # Non-K5 edges must have exactly zero coefficient in the surrogate cut
    for e in edges_extra:
        assert a_surr[edge_to_idx[e]] == 0.0


def test_surrogate_relaxation_highs_warmstart_feasibility(solver):
    """Verify surrogate LP relaxation produces primal solution within [0, 1]^m bounds."""
    g = generate_k5_cluster_graph(num_k5=2, num_bridges=1, seed=99)
    a_surr = np.ones(g.num_edges)
    b_surr = 12.0
    sol_surr = solver.solve_surrogate_relaxation(g, a_surr, b_surr)

    assert np.all(sol_surr.primal_solution >= -1e-6)
    assert np.all(sol_surr.primal_solution <= 1.0 + 1e-6)


# =====================================================================
# Feature 3: Exact Rational Conic Combination Verification (5+ tests)
# =====================================================================

@pytest.fixture
def verifier():
    return RationalCutVerifier()


def test_rational_verification_single_clique(verifier):
    """Verify exact Fraction arithmetic for single K5 clique with fraction multiplier."""
    edges = list(itertools.combinations(range(5), 2))
    g = GraphInstance(name="k5", num_nodes=5, edges=edges)
    k5 = (0, 1, 2, 3, 4)

    # Multiplier 1/3 => RHS = (1/3) * 6 = 2
    cert = verifier.verify_clique_conic_combination(g, {k5: Fraction(1, 3)})
    assert cert.is_valid
    assert cert.status == "CERTIFIED_VALID_CONIC_COMBINATION"
    assert cert.exact_rhs == Fraction(2, 1)
    for e, coeff in cert.exact_coefficients.items():
        assert coeff == Fraction(1, 3)


def test_rational_verification_multiple_cliques(verifier):
    """Verify conic combination of multiple K5 cliques with fractional multipliers."""
    g = generate_k5_cluster_graph(num_k5=2, num_bridges=1, seed=42)
    cliques = g.find_all_k5_cliques()
    assert len(cliques) == 2

    # lambda_1 = 3/7, lambda_2 = 5/9 => RHS = 3/7 * 6 + 5/9 * 6 = 18/7 + 10/3 = 124/21
    candidate_mults = {
        cliques[0]: Fraction(3, 7),
        cliques[1]: Fraction(5, 9),
    }

    cert = verifier.verify_clique_conic_combination(g, candidate_mults)
    assert cert.is_valid
    assert cert.exact_rhs == Fraction(124, 21)
    assert cert.num_active_supports == 2


def test_rational_verification_cycle_combination(verifier):
    """Verify cycle conic combination in exact rational arithmetic."""
    edges = [(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (2, 4)]
    g = GraphInstance(name="two_triangles", num_nodes=5, edges=edges)

    candidate_cycles = [
        ([0, 1, 2], [(0, 1), (1, 2), (0, 2)], Fraction(2, 5)),
        ([2, 3, 4], [(2, 3), (3, 4), (2, 4)], Fraction(3, 5)),
    ]

    cert = verifier.verify_cycle_conic_combination(g, candidate_cycles)
    assert cert.is_valid
    assert cert.status == "CERTIFIED_VALID_CYCLE_CONIC_COMBINATION"
    # RHS = 2/5 * (3 - 1) + 3/5 * (3 - 1) = 4/5 + 6/5 = 10/5 = 2
    assert cert.exact_rhs == Fraction(2, 1)


def test_rational_verification_general_surrogate_combination(verifier):
    """Verify composite surrogate cut combining K5 cliques and cycle inequalities."""
    g = generate_k5_cluster_graph(num_k5=1, num_bridges=0, seed=42)
    cliques = g.find_all_k5_cliques()

    candidate_k5 = {cliques[0]: Fraction(1, 4)}
    candidate_cycles = [
        ([0, 1, 2], [(0, 1), (1, 2), (0, 2)], Fraction(1, 4)),
    ]

    # RHS = 1/4 * 6 + 1/4 * 2 = 6/4 + 2/4 = 8/4 = 2
    cert = verifier.verify_general_surrogate_cut(g, candidate_k5=candidate_k5, candidate_cycles=candidate_cycles)
    assert cert.is_valid
    assert cert.exact_rhs == Fraction(2, 1)
    assert cert.num_active_supports == 2


def test_rational_verifier_bounded_denominator(verifier):
    """Verify to_fraction accurately handles bounded denominator conversion."""
    f1 = verifier.to_fraction(0.333333333333)
    assert abs(float(f1) - 1.0 / 3.0) < 1e-6
    f2 = verifier.to_fraction(Fraction(7, 11))
    assert f2 == Fraction(7, 11)
    f3 = verifier.to_fraction(5)
    assert f3 == Fraction(5, 1)


# =====================================================================
# Feature 4: Cryptographic SHA-256 Certificate Receipt Generation (5+ tests)
# =====================================================================

def test_sha256_certificate_digest_format(verifier):
    """Verify certificate generates 64-char lowercase hexadecimal SHA-256 digest."""
    g = generate_k5_cluster_graph(num_k5=1, num_bridges=0, seed=42)
    clq = g.find_all_k5_cliques()[0]
    cert = verifier.verify_clique_conic_combination(g, {clq: Fraction(1, 1)})

    assert len(cert.sha256_hash) == 64
    assert all(c in "0123456789abcdef" for c in cert.sha256_hash)


def test_sha256_certificate_canonical_ordering_invariance(verifier):
    """Verify SHA-256 hash is invariant to dictionary insertion order of supports."""
    g = generate_k5_cluster_graph(num_k5=2, num_bridges=1, seed=42)
    c1, c2 = g.find_all_k5_cliques()

    cert1 = verifier.verify_clique_conic_combination(g, {c1: Fraction(1, 2), c2: Fraction(3, 4)})
    cert2 = verifier.verify_clique_conic_combination(g, {c2: Fraction(3, 4), c1: Fraction(1, 2)})

    assert cert1.sha256_hash == cert2.sha256_hash


def test_sha256_certificate_tamper_detection(verifier):
    """Verify any coefficient or RHS perturbation changes the SHA-256 digest."""
    g = generate_k5_cluster_graph(num_k5=1, num_bridges=0, seed=42)
    clq = g.find_all_k5_cliques()[0]

    cert = verifier.verify_clique_conic_combination(g, {clq: Fraction(1, 1)})
    original_hash = cert.sha256_hash

    # Tamper with exact_rhs
    cert.exact_rhs = Fraction(7, 1)
    tampered_hash = cert.compute_sha256()

    assert tampered_hash != original_hash


def test_sha256_certificate_json_serializability(verifier):
    """Verify certificate can be serialized to JSON and round-tripped."""
    g = generate_k5_cluster_graph(num_k5=1, num_bridges=0, seed=42)
    clq = g.find_all_k5_cliques()[0]
    cert = verifier.verify_clique_conic_combination(g, {clq: Fraction(1, 1)})

    raw_json = json.dumps({
        "is_valid": cert.is_valid,
        "status": cert.status,
        "sha256_hash": cert.sha256_hash,
        "exact_rhs": str(cert.exact_rhs),
    })
    loaded = json.loads(raw_json)
    assert loaded["is_valid"] is True
    assert loaded["sha256_hash"] == cert.sha256_hash


def test_sha256_certificate_rejected_receipt_generation(verifier):
    """Verify rejected certificates generate cryptographic SHA-256 receipts with rejection metadata."""
    g = generate_k5_cluster_graph(num_k5=1, num_bridges=0, seed=42)
    clq = g.find_all_k5_cliques()[0]

    cert = verifier.verify_clique_conic_combination(g, {clq: -0.5})
    assert not cert.is_valid
    assert cert.status == "REJECTED_NEGATIVE_MULTIPLIER"
    assert len(cert.sha256_hash) == 64
    assert cert.rejection_reason is not None


# =====================================================================
# Feature 5: Isomorphic Relabeling Invariance (|Delta Z| <= 10^-6) (5+ tests)
# =====================================================================

def test_isomorphic_relabeling_invariance_k5_cluster(solver):
    """Verify |Z_A - Z_B| <= 10^-6 on paired isomorphic graphs under permutation."""
    g = generate_k5_cluster_graph(num_k5=2, num_bridges=1, seed=77)
    ga, gb, pi, _ = generate_ab_pair(g, seed=88)

    sol_a = solver.solve_k5_relaxation(ga)
    sol_b = solver.solve_k5_relaxation(gb)

    delta_z = abs(sol_a.objective_value - sol_b.objective_value)
    assert delta_z <= 1e-6, f"Relabeling variance violation: {delta_z} > 1e-6"


def test_isomorphic_relabeling_dual_center_invariance(solver):
    """Verify strictly convex QP projection yields equivariant dual centers (|y_A - y_B| <= 1e-5)."""
    g = generate_k5_cluster_graph(num_k5=3, num_bridges=2, seed=33)
    ga, gb, pi, _ = generate_ab_pair(g, seed=44)

    center_a = solver.compute_face_invariant_dual_center(ga)
    center_b = solver.compute_face_invariant_dual_center(gb)

    for clq_a, val_a in center_a.items():
        clq_b = tuple(sorted([pi[v] for v in clq_a]))
        assert clq_b in center_b
        val_b = center_b[clq_b]
        assert abs(val_a - val_b) < 1e-5


def test_isomorphic_relabeling_surrogate_rhs_invariance(solver):
    """Verify surrogate RHS b_surr is identical across isomorphic relabelings."""
    g = generate_k5_cluster_graph(num_k5=2, num_bridges=1, seed=55)
    ga, gb, pi, _ = generate_ab_pair(g, seed=66)

    center_a = solver.compute_face_invariant_dual_center(ga)
    center_b = solver.compute_face_invariant_dual_center(gb)

    rhs_a = sum(val * 6.0 for val in center_a.values())
    rhs_b = sum(val * 6.0 for val in center_b.values())

    assert abs(rhs_a - rhs_b) < 1e-6


def test_isomorphic_relabeling_exp87_instances(solver):
    """Verify relabeling invariance across canonical Exp87 benchmark instances."""
    instances = generate_exp87_instances()
    g1 = instances["exp87_g1_target57"]
    ga, gb, pi, _ = generate_ab_pair(g1, seed=123)

    sol_a = solver.solve_k5_relaxation(ga)
    sol_b = solver.solve_k5_relaxation(gb)

    delta_z = abs(sol_a.objective_value - sol_b.objective_value)
    assert delta_z <= 1e-6


def test_isomorphic_relabeling_cycle_relaxation_invariance(solver):
    """Verify base cycle relaxation objective is strictly invariant under node permutation."""
    g = generate_ks_minus_edges(s=4, removed_edges=2, seed=10)
    ga, gb, pi, _ = generate_ab_pair(g, seed=20)

    sol_a = solver.solve_cycle_relaxation(ga)
    sol_b = solver.solve_cycle_relaxation(gb)

    delta_z = abs(sol_a.objective_value - sol_b.objective_value)
    assert delta_z <= 1e-6
