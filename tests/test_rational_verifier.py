"""Automated tests for rational verifier and hostile mutation testing."""

from fractions import Fraction
import numpy as np
import pytest

from highs_turbo.graph_generator import generate_k5_cluster_graph
from highs_turbo.rational_verifier import RationalCutVerifier


@pytest.fixture
def verifier():
    return RationalCutVerifier()


def test_valid_clique_conic_combination(verifier):
    g = generate_k5_cluster_graph(num_k5=2, num_bridges=1, seed=42)
    cliques = g.find_all_k5_cliques()
    assert len(cliques) == 2

    # Give positive valid multipliers
    candidate_mults = {
        cliques[0]: Fraction(1, 2),
        cliques[1]: Fraction(3, 4),
    }

    cert = verifier.verify_clique_conic_combination(g, candidate_mults)
    assert cert.is_valid
    assert cert.status == "CERTIFIED_VALID_CONIC_COMBINATION"
    assert cert.num_active_supports == 2
    # RHS = 1/2 * 6 + 3/4 * 6 = 3 + 4.5 = 7.5 = 15/2
    assert cert.exact_rhs == Fraction(15, 2)
    assert len(cert.sha256_hash) == 64


def test_hostile_mutation_negative_multiplier(verifier):
    g = generate_k5_cluster_graph(num_k5=2, num_bridges=1, seed=42)
    cliques = g.find_all_k5_cliques()

    # Adversarial injection: negative multiplier
    candidate_mults = {
        cliques[0]: Fraction(-1, 2),
        cliques[1]: Fraction(1, 1),
    }

    cert = verifier.verify_clique_conic_combination(g, candidate_mults)
    assert not cert.is_valid
    assert cert.status == "REJECTED_NEGATIVE_MULTIPLIER"


def test_hostile_mutation_non_clique_support(verifier):
    g = generate_k5_cluster_graph(num_k5=2, num_bridges=1, seed=42)

    # Support with nodes from two different K5 clusters that do not form a clique
    fake_clique = (0, 1, 2, 6, 7)  # (6, 7) is in second block, no edges to 0, 1, 2
    candidate_mults = {fake_clique: Fraction(1, 1)}

    cert = verifier.verify_clique_conic_combination(g, candidate_mults)
    assert not cert.is_valid
    assert cert.status == "REJECTED_NON_CLIQUE_SUPPORT"


def test_hostile_mutation_invalid_support_size(verifier):
    g = generate_k5_cluster_graph(num_k5=2, num_bridges=1, seed=42)

    candidate_mults = {(0, 1, 2, 3): Fraction(1, 1)}  # size 4, not 5

    cert = verifier.verify_clique_conic_combination(g, candidate_mults)
    assert not cert.is_valid
    assert cert.status == "REJECTED_INVALID_SUPPORT_SIZE"


def test_hostile_mutation_rhs_deficit(verifier):
    g = generate_k5_cluster_graph(num_k5=2, num_bridges=1, seed=42)
    cliques = g.find_all_k5_cliques()

    candidate_mults = {cliques[0]: 1.0}
    # Sound RHS is 6.0. Claim tighter RHS 5.0
    cert = verifier.verify_clique_conic_combination(g, candidate_mults, candidate_rhs=5.0)
    assert not cert.is_valid
    assert cert.status == "REJECTED_UNSOUND_RHS_DEFICIT"


def test_cut_polytope_soundness_verification(verifier):
    g = generate_k5_cluster_graph(num_k5=1, num_bridges=0, seed=42)
    # Single K5: all 10 edge coefficients 1.0, RHS 6.0
    coeffs = np.ones(g.num_edges)
    cert = verifier.verify_cut_polytope_soundness(g, coeffs, rhs=6.0)
    assert cert.is_valid
    assert cert.status == "CERTIFIED_VALID_CUT_POLYTOPE"

    # Hostile cut with RHS 5.0 (violated by any cut of size 6)
    cert_invalid = verifier.verify_cut_polytope_soundness(g, coeffs, rhs=5.0)
    assert not cert_invalid.is_valid
    assert cert_invalid.status == "REJECTED_CUT_POLYTOPE_VIOLATION"


def test_verify_cycle_conic_combination(verifier):
    g = generate_k5_cluster_graph(num_k5=1, num_bridges=0, seed=42)
    # Triangle on nodes 0, 1, 2 with all 3 edges in F: x_01 + x_12 + x_02 <= 2
    candidate_cycles = [
        ([0, 1, 2], [(0, 1), (1, 2), (0, 2)], Fraction(1, 1)),
    ]
    cert = verifier.verify_cycle_conic_combination(g, candidate_cycles)
    assert cert.is_valid
    assert cert.status == "CERTIFIED_VALID_CYCLE_CONIC_COMBINATION"
    assert cert.exact_rhs == Fraction(2, 1)

    # Hostile: Even |F| cardinality
    bad_f = [
        ([0, 1, 2], [(0, 1), (1, 2)], Fraction(1, 1)),
    ]
    cert_even = verifier.verify_cycle_conic_combination(g, bad_f)
    assert not cert_even.is_valid
    assert cert_even.status == "REJECTED_EVEN_F_CARDINALITY"

    # Hostile: Negative multiplier
    cert_neg = verifier.verify_cycle_conic_combination(g, [([0, 1, 2], [(0, 1), (1, 2), (0, 2)], -1.0)])
    assert not cert_neg.is_valid
    assert cert_neg.status == "REJECTED_NEGATIVE_MULTIPLIER"


def test_verify_general_surrogate_cut(verifier):
    g = generate_k5_cluster_graph(num_k5=2, num_bridges=1, seed=42)
    cliques = g.find_all_k5_cliques()

    candidate_k5 = {cliques[0]: Fraction(1, 1)}
    candidate_cycles = [
        ([0, 1, 2], [(0, 1), (1, 2), (0, 2)], Fraction(1, 2)),
    ]

    # Combined RHS: 1 * 6 + 1/2 * 2 = 6 + 1 = 7
    cert = verifier.verify_general_surrogate_cut(g, candidate_k5=candidate_k5, candidate_cycles=candidate_cycles)
    assert cert.is_valid
    assert cert.status == "CERTIFIED_VALID_SURROGATE_CUT"
    assert cert.exact_rhs == Fraction(7, 1)
    assert cert.num_active_supports == 2

    # Claim tighter unsound RHS 6.0
    cert_deficit = verifier.verify_general_surrogate_cut(g, candidate_k5=candidate_k5, candidate_cycles=candidate_cycles, candidate_rhs=6.0)
    assert not cert_deficit.is_valid
    assert cert_deficit.status == "REJECTED_UNSOUND_RHS_DEFICIT"
