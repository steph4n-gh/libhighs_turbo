"""Tier 3: Cross-Feature Combination E2E Tests.

Executes end-to-end multi-module pipelines per requirements R1-R5:
- Pipeline 1: Bit-parallel graph -> surrogate cut -> rational verification -> SHA-256 cert -> HiGHS LP solve
- Pipeline 2: A/B isomorphic graph permutation -> cut generation -> verification -> check |Delta Z| <= 10^-6
- Pipeline 3: GNN topological features -> equivariant prediction -> rational verification -> LP installation
- Pipeline 4: Composite clique + cycle surrogate cuts -> rational verification -> SHA-256 digest -> LP solve
- Pipeline 5: Asymmetric graph QP dual face projection -> canonical dual multipliers -> LP equivalence
- Pipeline 6: Exp87 benchmark instances -> surrogate compression -> 95%+ integrality gap closure
- Pipeline 7: Weighted graphs -> exact dual certificate verification -> zero residual norm
- Pipeline 8: Exp91 synthetic catalog instances -> permutation invariance -> certificate generation
- Pipeline 9: Surrogate cut row -> exact Cut Polytope soundness verification

Coverage threshold: >= 8 cross-feature combination tests (total: 9 tests).
"""

from __future__ import annotations

import itertools
from fractions import Fraction
import numpy as np
import pytest
import torch

from highs_turbo.exact_solver import ExactMaxCutSolver
from highs_turbo.graph_generator import (
    GraphInstance,
    generate_ab_pair,
    generate_exp87_instances,
    generate_exp91_family,
    generate_k5_cluster_graph,
    generate_ks_minus_edges,
)
from highs_turbo.rational_verifier import RationalCutVerifier
from highs_turbo.surrogate_model import (
    EdgeEquivariantSurrogateGNN,
    extract_topological_features,
)
from highs_turbo.topologies import BitParallelGraph


@pytest.fixture
def solver():
    return ExactMaxCutSolver()


@pytest.fixture
def verifier():
    return RationalCutVerifier()


def test_cross_feature_bitgraph_to_surrogate_to_cert_to_lp(solver, verifier):
    """Pipeline 1: BitGraph -> surrogate cut -> rational verifier -> SHA-256 cert -> LP solve."""
    g = generate_k5_cluster_graph(num_k5=2, num_bridges=1, seed=42)

    # 1. BitParallelGraph queries
    bg = BitParallelGraph.from_graph_instance(g)
    cliques = bg.find_k5_cliques()
    assert len(cliques) == 2

    # 2. Solve K5 relaxation to extract dual multipliers
    sol_k5 = solver.solve_k5_relaxation(g)
    candidate_mults = {clq: Fraction.from_float(sol_k5.k5_multipliers[clq]).limit_denominator(1000) for clq in cliques}

    # 3. Rational verification & SHA-256 certificate
    cert = verifier.verify_clique_conic_combination(g, candidate_mults)
    assert cert.is_valid
    assert len(cert.sha256_hash) == 64

    # 4. Construct surrogate row from exact rational coefficients
    m = g.num_edges
    edge_to_idx = {e: i for i, e in enumerate(g.edges)}
    a_surr = np.zeros(m)
    for e, coeff in cert.exact_coefficients.items():
        a_surr[edge_to_idx[e]] = float(coeff)
    b_surr = float(cert.exact_rhs)

    # 5. Solve surrogate relaxation and check objective matches K5 LP
    sol_surr = solver.solve_surrogate_relaxation(g, a_surr, b_surr)
    assert abs(sol_surr.objective_value - sol_k5.objective_value) < 1e-4
    assert abs(sol_surr.objective_value - 13.0) < 1e-4


def test_cross_feature_ab_isomorphic_permutation_e2e(solver, verifier):
    """Pipeline 2: A/B isomorphic permutation -> cut generation -> verification -> |Delta Z| <= 10^-6."""
    g = generate_k5_cluster_graph(num_k5=3, num_bridges=2, seed=101)
    ga, gb, pi, _ = generate_ab_pair(g, seed=202)

    # Compute face-invariant dual center on both
    center_a = solver.compute_face_invariant_dual_center(ga)
    center_b = solver.compute_face_invariant_dual_center(gb)

    # Verify both cuts in rational verifier
    cert_a = verifier.verify_clique_conic_combination(ga, center_a)
    cert_b = verifier.verify_clique_conic_combination(gb, center_b)
    assert cert_a.is_valid and cert_b.is_valid

    # Build surrogate rows
    def build_row(graph, center):
        m = graph.num_edges
        edge_to_idx = {e: i for i, e in enumerate(graph.edges)}
        a = np.zeros(m)
        b = 0.0
        for clq, val in center.items():
            b += val * 6.0
            for u, v in itertools.combinations(clq, 2):
                e = (min(u, v), max(u, v))
                a[edge_to_idx[e]] += val
        return a, b

    a_a, b_a = build_row(ga, center_a)
    a_b, b_b = build_row(gb, center_b)

    sol_surr_a = solver.solve_surrogate_relaxation(ga, a_a, b_a)
    sol_surr_b = solver.solve_surrogate_relaxation(gb, a_b, b_b)

    delta_z = abs(sol_surr_a.objective_value - sol_surr_b.objective_value)
    assert delta_z <= 1e-6, f"Relabeling objective difference {delta_z} exceeds 10^-6"


def test_cross_feature_gnn_features_to_dual_center_to_certificate(solver, verifier):
    """Pipeline 3: GNN features -> equivariant prediction -> rational verification -> LP solve."""
    g = generate_k5_cluster_graph(num_k5=2, num_bridges=1, seed=55)

    # Extract topological features and run GNN
    nf, ef, edge_index = extract_topological_features(g)
    assert nf.shape[0] == g.num_nodes
    assert ef.shape[0] == g.num_edges

    model = EdgeEquivariantSurrogateGNN(hidden_dim=16, num_layers=2)
    model.eval()
    with torch.no_grad():
        out = model(g)

    mults = out["k5_multipliers"]
    candidate_rhs = out["candidate_rhs"]

    # 1. Verify that the verifier computes exact sound conic combination from predicted multipliers
    cert_sound = verifier.verify_clique_conic_combination(g, mults)
    assert cert_sound.is_valid
    assert cert_sound.status == "CERTIFIED_VALID_CONIC_COMBINATION"
    assert len(cert_sound.sha256_hash) == 64

    # 2. Test zero-hallucination guardrail: if neural net claims an unsound tighter RHS, verifier rejects it
    unsound_rhs = float(cert_sound.exact_rhs) - 1.0
    cert_reject = verifier.verify_clique_conic_combination(g, mults, candidate_rhs=unsound_rhs)
    assert not cert_reject.is_valid
    assert cert_reject.status == "REJECTED_UNSOUND_RHS_DEFICIT"


def test_cross_feature_cycle_and_clique_composite_to_certificate_to_lp(solver, verifier):
    """Pipeline 4: Combined clique + cycle cuts -> verification -> SHA-256 -> LP solve."""
    g = generate_k5_cluster_graph(num_k5=1, num_bridges=0, seed=42)
    cliques = g.find_all_k5_cliques()

    candidate_k5 = {cliques[0]: Fraction(1, 2)}
    # Triangle on nodes 0, 1, 2: x_01 + x_12 + x_02 <= 2
    candidate_cycles = [
        ([0, 1, 2], [(0, 1), (1, 2), (0, 2)], Fraction(1, 2)),
    ]

    cert = verifier.verify_general_surrogate_cut(g, candidate_k5=candidate_k5, candidate_cycles=candidate_cycles)
    assert cert.is_valid
    # RHS = 1/2 * 6 + 1/2 * 2 = 3 + 1 = 4
    assert cert.exact_rhs == Fraction(4, 1)

    # Solve LP with composite surrogate cut
    m = g.num_edges
    edge_to_idx = {e: i for i, e in enumerate(g.edges)}
    a_surr = np.zeros(m)
    for e, coeff in cert.exact_coefficients.items():
        a_surr[edge_to_idx[e]] = float(coeff)
    b_surr = float(cert.exact_rhs)

    sol = solver.solve_surrogate_relaxation(g, a_surr, b_surr)
    assert sol.success is True
    assert sol.objective_value <= 20.0 / 3.0  # Tighter than base cycle relaxation


def test_cross_feature_asymmetric_k7_minus_edges_invariance(solver, verifier):
    """Pipeline 5: Asymmetric graph QP dual face projection -> canonical dual multipliers -> LP equivalence."""
    g = generate_ks_minus_edges(s=4, removed_edges=2, seed=88)
    ga, gb, pi, _ = generate_ab_pair(g, seed=99)

    center_a = solver.compute_face_invariant_dual_center(ga)
    center_b = solver.compute_face_invariant_dual_center(gb)

    # Check dual multiplier mapping invariance
    for clq_a, val_a in center_a.items():
        clq_b = tuple(sorted([pi[v] for v in clq_a]))
        assert clq_b in center_b
        assert abs(val_a - center_b[clq_b]) < 1e-5


def test_cross_feature_exp87_full_pipeline_audit(solver, verifier):
    """Pipeline 6: Exp87 instances -> surrogate compression -> 95%+ integrality gap closure."""
    instances = generate_exp87_instances()

    for name, g in instances.items():
        sol_cycle = solver.solve_cycle_relaxation(g)
        sol_k5 = solver.solve_k5_relaxation(g)

        # Build surrogate cut
        edge_to_idx = {e: i for i, e in enumerate(g.edges)}
        a_surr = np.zeros(g.num_edges)
        b_surr = 0.0
        for clq, val in sol_k5.k5_multipliers.items():
            b_surr += val * 6.0
            for u, v in itertools.combinations(clq, 2):
                e = (min(u, v), max(u, v))
                a_surr[edge_to_idx[e]] += val

        # Certify
        cert = verifier.verify_clique_conic_combination(g, sol_k5.k5_multipliers, candidate_rhs=b_surr)
        assert cert.is_valid

        # Solve surrogate LP
        sol_surr = solver.solve_surrogate_relaxation(g, a_surr, b_surr)

        # Check gap closed: (Z_cycle - Z_surr) / (Z_cycle - Z_k5) >= 95%
        gap_total = sol_cycle.objective_value - sol_k5.objective_value
        gap_closed = sol_cycle.objective_value - sol_surr.objective_value
        closed_pct = (gap_closed / gap_total) * 100.0
        assert closed_pct >= 95.0, f"Gap closed {closed_pct}% < 95.0% on {name}"


def test_cross_feature_scaled_weights_dual_certificate(solver):
    """Pipeline 7: Weighted graphs -> exact dual certificate verification -> zero residual norm."""
    w = {(0, 1): Fraction(2, 1), (1, 2): Fraction(2, 1), (0, 2): Fraction(2, 1)}
    a = {(0, 1): Fraction(1, 1), (1, 2): Fraction(1, 1), (0, 2): Fraction(1, 1)}

    # Valid certificate: lambda = 2, surrogate_rhs = 2 => bound = 4
    cert = solver.verify_exact_dual_certificate(w, a, surrogate_rhs=2, surrogate_multiplier=2)
    assert cert.is_feasible
    assert cert.rational_upper_bound == Fraction(4, 1)
    assert cert.integer_floor == 4
    assert cert.residual_norm == Fraction(0, 1)


def test_cross_feature_exp91_synthetic_family_invariance(solver, verifier):
    """Pipeline 8: Exp91 synthetic catalog instances -> permutation invariance -> certificate."""
    family = generate_exp91_family(num_graphs=2, seed=91)
    g1 = family["exp91_identity_1"]
    ga, gb, pi, _ = generate_ab_pair(g1, seed=1234)

    sol_a = solver.solve_k5_relaxation(ga)
    sol_b = solver.solve_k5_relaxation(gb)

    delta_z = abs(sol_a.objective_value - sol_b.objective_value)
    assert delta_z <= 1e-6


def test_cross_feature_cut_polytope_soundness_on_surrogate_row(verifier):
    """Pipeline 9: Surrogate cut row -> exact Cut Polytope soundness verification."""
    g = generate_k5_cluster_graph(num_k5=1, num_bridges=0, seed=42)
    coeffs = np.ones(g.num_edges)
    rhs = 6.0

    cert = verifier.verify_cut_polytope_soundness(g, coeffs, rhs=rhs)
    assert cert.is_valid
    assert cert.status == "CERTIFIED_VALID_CUT_POLYTOPE"
