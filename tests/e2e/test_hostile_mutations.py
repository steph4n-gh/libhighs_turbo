"""8-Vector Hostile Mutation Suite.

Requirement R4 and Acceptance Criteria:
- 0% invalid cuts admitted into solver state; 100% of adversarial/mutated cuts rejected.
- Exhaustive verification across all 8 attack vectors:
  1. Negative multipliers (lambda < 0)
  2. Missing edges in support (support edges not present in E(G))
  3. Support size violation (|K| != 5)
  4. Out-of-bounds node index (v < 0 or v >= n)
  5. RHS claim deficit (b_claim < b_exact)
  6. Odd cycle parity violation (|F| even)
  7. Exact precision overflow attempt (extreme numerators/denominators)
  8. Dimension mismatch between graph and cut vector (len(a) != |E(G)|)
"""

from __future__ import annotations

import itertools
import math
from fractions import Fraction
import numpy as np
import pytest

from highs_turbo.graph_generator import GraphInstance, generate_k5_cluster_graph
from highs_turbo.rational_verifier import RationalCutVerifier


@pytest.fixture
def verifier():
    return RationalCutVerifier()


@pytest.fixture
def test_graph():
    return generate_k5_cluster_graph(num_k5=2, num_bridges=1, seed=42)


# =====================================================================
# Vector 1: Negative Multipliers (lambda < 0)
# =====================================================================

@pytest.mark.parametrize("bad_mult", [-1.0, -0.0001, -1e-15, Fraction(-3, 4), -1000.0])
def test_hostile_vector1_negative_multiplier_clique(verifier, test_graph, bad_mult):
    """Vector 1: Reject negative multipliers in clique conic combination."""
    cliques = test_graph.find_all_k5_cliques()
    candidate_mults = {cliques[0]: bad_mult, cliques[1]: 1.0}

    cert = verifier.verify_clique_conic_combination(test_graph, candidate_mults)
    assert not cert.is_valid
    assert cert.status == "REJECTED_NEGATIVE_MULTIPLIER"
    assert len(cert.sha256_hash) == 64


@pytest.mark.parametrize("bad_mult", [-0.5, Fraction(-1, 3), -1e-12])
def test_hostile_vector1_negative_multiplier_cycle(verifier, test_graph, bad_mult):
    """Vector 1: Reject negative multipliers in cycle conic combination."""
    candidate_cycles = [
        ([0, 1, 2], [(0, 1), (1, 2), (0, 2)], bad_mult),
    ]
    cert = verifier.verify_cycle_conic_combination(test_graph, candidate_cycles)
    assert not cert.is_valid
    assert cert.status == "REJECTED_NEGATIVE_MULTIPLIER"


# =====================================================================
# Vector 2: Missing Edges in Support
# =====================================================================

def test_hostile_vector2_missing_edges_in_clique_support(verifier, test_graph):
    """Vector 2: Reject candidate clique support when edges are missing in E(G)."""
    # Nodes 0, 1, 2 are in block 1; nodes 5, 6 are in block 2 (no edges between them)
    disconnected_support = (0, 1, 2, 5, 6)
    candidate_mults = {disconnected_support: 1.0}

    cert = verifier.verify_clique_conic_combination(test_graph, candidate_mults)
    assert not cert.is_valid
    assert cert.status == "REJECTED_NON_CLIQUE_SUPPORT"


def test_hostile_vector2_missing_edge_in_cycle_support(verifier, test_graph):
    """Vector 2: Reject candidate cycle when an edge is missing from graph."""
    # (0, 5) does not exist in graph
    fake_cycle = ([0, 5, 6], [(0, 5), (5, 6), (0, 6)], 1.0)
    cert = verifier.verify_cycle_conic_combination(test_graph, [fake_cycle])
    assert not cert.is_valid
    assert cert.status == "REJECTED_NON_GRAPH_EDGE"


# =====================================================================
# Vector 3: Support Size Violation (|K| != 5)
# =====================================================================

@pytest.mark.parametrize("bad_size_nodes", [
    (0,),
    (0, 1),
    (0, 1, 2),
    (0, 1, 2, 3),
    (0, 1, 2, 3, 4, 5),
    (0, 1, 2, 3, 4, 5, 6),
])
def test_hostile_vector3_support_size_violation_clique(verifier, test_graph, bad_size_nodes):
    """Vector 3: Reject clique supports with size != 5."""
    cert = verifier.verify_clique_conic_combination(test_graph, {bad_size_nodes: 1.0})
    assert not cert.is_valid
    assert cert.status == "REJECTED_INVALID_SUPPORT_SIZE"


@pytest.mark.parametrize("bad_cycle_nodes", [[0], [0, 1]])
def test_hostile_vector3_cycle_length_too_small(verifier, test_graph, bad_cycle_nodes):
    """Vector 3: Reject cycles with length < 3."""
    cert = verifier.verify_cycle_conic_combination(test_graph, [(bad_cycle_nodes, [], 1.0)])
    assert not cert.is_valid
    assert cert.status == "REJECTED_INVALID_CYCLE_LENGTH"


# =====================================================================
# Vector 4: Out-of-Bounds Node Index
# =====================================================================

@pytest.mark.parametrize("oob_support", [
    (-1, 0, 1, 2, 3),
    (0, 1, 2, 3, 10),  # graph has 10 nodes (0..9)
    (0, 1, 2, 3, 999),
    (-100, -50, 0, 1, 2),
])
def test_hostile_vector4_out_of_bounds_node(verifier, test_graph, oob_support):
    """Vector 4: Reject supports containing node indices outside [0, num_nodes - 1]."""
    cert = verifier.verify_clique_conic_combination(test_graph, {oob_support: 1.0})
    assert not cert.is_valid
    assert cert.status == "REJECTED_OUT_OF_BOUNDS_NODE"


# =====================================================================
# Vector 5: RHS Claim Deficit (b_claim < b_exact)
# =====================================================================

@pytest.mark.parametrize("deficit", [0.001, 0.1, 1.0, 5.0])
def test_hostile_vector5_rhs_claim_deficit_clique(verifier, test_graph, deficit):
    """Vector 5: Reject claimed RHS lower than exact mathematically sound conic bound."""
    cliques = test_graph.find_all_k5_cliques()
    candidate_mults = {cliques[0]: 1.0}
    sound_rhs = 6.0
    unsound_claimed_rhs = sound_rhs - deficit

    cert = verifier.verify_clique_conic_combination(test_graph, candidate_mults, candidate_rhs=unsound_claimed_rhs)
    assert not cert.is_valid
    assert cert.status == "REJECTED_UNSOUND_RHS_DEFICIT"


def test_hostile_vector5_rhs_claim_deficit_cycle(verifier, test_graph):
    """Vector 5: Reject cycle inequality claiming RHS tighter than |F| - 1."""
    # Triangle on (0, 1, 2) with all 3 edges in F. Sound RHS = 3 - 1 = 2.
    candidate_cycles = [
        ([0, 1, 2], [(0, 1), (1, 2), (0, 2)], Fraction(1, 1)),
    ]
    # Claim tighter unsound RHS 1.5
    cert = verifier.verify_cycle_conic_combination(test_graph, candidate_cycles, candidate_rhs=1.5)
    assert not cert.is_valid
    assert cert.status == "REJECTED_UNSOUND_RHS_DEFICIT"


# =====================================================================
# Vector 6: Odd Cycle Parity Violation (|F| even)
# =====================================================================

@pytest.mark.parametrize("even_f_edges", [
    [],  # |F| = 0
    [(0, 1), (1, 2)],  # |F| = 2
])
def test_hostile_vector6_even_f_cardinality(verifier, test_graph, even_f_edges):
    """Vector 6: Reject cycle inequalities where |F| is even."""
    cycle_nodes = [0, 1, 2]
    candidate_cycles = [
        (cycle_nodes, even_f_edges, Fraction(1, 1)),
    ]
    cert = verifier.verify_cycle_conic_combination(test_graph, candidate_cycles)
    assert not cert.is_valid
    assert cert.status == "REJECTED_EVEN_F_CARDINALITY"


def test_hostile_vector6_f_edge_not_in_cycle(verifier, test_graph):
    """Vector 6: Reject cycle inequalities where edge in F is not in the cycle edges."""
    cycle_nodes = [0, 1, 2]
    # (3, 4) is not in cycle (0, 1, 2)
    bad_f_edges = [(0, 1), (1, 2), (3, 4)]
    candidate_cycles = [
        (cycle_nodes, bad_f_edges, Fraction(1, 1)),
    ]
    cert = verifier.verify_cycle_conic_combination(test_graph, candidate_cycles)
    assert not cert.is_valid
    assert cert.status == "REJECTED_F_EDGE_NOT_IN_CYCLE"


# =====================================================================
# Vector 7: Exact Precision Overflow Attempt
# =====================================================================

def test_hostile_vector7_extreme_fractions_stability(verifier, test_graph):
    """Vector 7: Verify rational verifier handles extreme fractions without overflow or precision loss."""
    cliques = test_graph.find_all_k5_cliques()
    # 2^128 order fraction
    huge_num = 1 << 128
    huge_den = (1 << 128) + 1
    extreme_frac = Fraction(huge_num, huge_den)

    cert = verifier.verify_clique_conic_combination(test_graph, {cliques[0]: extreme_frac})
    assert cert.is_valid
    assert cert.exact_rhs == extreme_frac * 6
    # Denominator limit handling
    limited_frac = verifier.to_fraction(float(extreme_frac))
    assert isinstance(limited_frac, Fraction)


def test_hostile_vector7_subnormal_floating_point(verifier, test_graph):
    """Vector 7: Verify subnormal floating point numbers do not silently corrupt verifier."""
    cliques = test_graph.find_all_k5_cliques()
    subnormal_val = 1e-308

    cert = verifier.verify_clique_conic_combination(test_graph, {cliques[0]: subnormal_val})
    assert cert.is_valid


# =====================================================================
# Vector 8: Dimension Mismatch Between Graph and Cut Vector
# =====================================================================

@pytest.mark.parametrize("bad_dim_delta", [-5, -1, 1, 10])
def test_hostile_vector8_dimension_mismatch(verifier, test_graph, bad_dim_delta):
    """Vector 8: Reject cut polytope soundness checks when coefficient length != num_edges."""
    m = test_graph.num_edges
    bad_coeffs = np.ones(m + bad_dim_delta)

    cert = verifier.verify_cut_polytope_soundness(test_graph, bad_coeffs, rhs=10.0)
    assert not cert.is_valid
    assert cert.status == "REJECTED_DIMENSION_MISMATCH"


# =====================================================================
# Compound Adversarial Attack & 100% Rejection Audit
# =====================================================================

def test_hostile_compound_adversarial_sweep(verifier, test_graph):
    """Verify 100% rejection rate across a sweep of 20 randomized hostile mutations."""
    cliques = test_graph.find_all_k5_cliques()
    rejections = 0
    total_attacks = 20

    mutations = [
        # Vector 1: Negative multipliers
        lambda: verifier.verify_clique_conic_combination(test_graph, {cliques[0]: -0.1}),
        lambda: verifier.verify_clique_conic_combination(test_graph, {cliques[0]: -100.0}),
        lambda: verifier.verify_cycle_conic_combination(test_graph, [([0, 1, 2], [(0, 1), (1, 2), (0, 2)], -0.5)]),
        # Vector 2: Missing edges
        lambda: verifier.verify_clique_conic_combination(test_graph, {(0, 1, 2, 7, 8): 1.0}),
        lambda: verifier.verify_cycle_conic_combination(test_graph, [([0, 7, 8], [(0, 7), (7, 8), (0, 8)], 1.0)]),
        # Vector 3: Bad support sizes
        lambda: verifier.verify_clique_conic_combination(test_graph, {(0, 1, 2, 3): 1.0}),
        lambda: verifier.verify_clique_conic_combination(test_graph, {(0, 1, 2, 3, 4, 5): 1.0}),
        lambda: verifier.verify_cycle_conic_combination(test_graph, [([0, 1], [(0, 1)], 1.0)]),
        # Vector 4: Out-of-bounds nodes
        lambda: verifier.verify_clique_conic_combination(test_graph, {(-1, 0, 1, 2, 3): 1.0}),
        lambda: verifier.verify_clique_conic_combination(test_graph, {(0, 1, 2, 3, 50): 1.0}),
        # Vector 5: RHS claim deficits
        lambda: verifier.verify_clique_conic_combination(test_graph, {cliques[0]: 1.0}, candidate_rhs=5.0),
        lambda: verifier.verify_clique_conic_combination(test_graph, {cliques[0]: 2.0}, candidate_rhs=11.0),
        lambda: verifier.verify_cycle_conic_combination(test_graph, [([0, 1, 2], [(0, 1), (1, 2), (0, 2)], 1.0)], candidate_rhs=1.0),
        # Vector 6: Even cycle parity
        lambda: verifier.verify_cycle_conic_combination(test_graph, [([0, 1, 2], [(0, 1), (1, 2)], 1.0)]),
        lambda: verifier.verify_cycle_conic_combination(test_graph, [([0, 1, 2], [], 1.0)]),
        lambda: verifier.verify_cycle_conic_combination(test_graph, [([0, 1, 2], [(0, 1), (1, 2), (8, 9)], 1.0)]),
        # Vector 8: Dimension mismatch
        lambda: verifier.verify_cut_polytope_soundness(test_graph, np.ones(test_graph.num_edges + 3), rhs=6.0),
        lambda: verifier.verify_cut_polytope_soundness(test_graph, np.ones(test_graph.num_edges - 2), rhs=6.0),
        lambda: verifier.verify_cut_polytope_soundness(test_graph, np.array([]), rhs=6.0),
        # Cut polytope violation (unsound RHS on valid cut)
        lambda: verifier.verify_cut_polytope_soundness(test_graph, np.ones(test_graph.num_edges), rhs=1.0),
    ]

    for attack_fn in mutations:
        cert = attack_fn()
        assert not cert.is_valid, f"Security breach! Adversarial mutation was accepted: {cert}"
        rejections += 1

    assert rejections == total_attacks
    # Acceptance Criteria: 0% invalid cuts admitted into solver state; 100% rejected.
    rejection_rate = (rejections / total_attacks) * 100.0
    assert rejection_rate == 100.0
