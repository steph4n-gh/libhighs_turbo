"""Automated tests for exact solver module."""

import itertools
from fractions import Fraction
import numpy as np
import pytest

from highs_turbo.exact_solver import ExactMaxCutSolver
from highs_turbo.graph_generator import (
    GraphInstance,
    generate_ab_pair,
    generate_k5_cluster_graph,
)


@pytest.fixture
def solver():
    return ExactMaxCutSolver()


def test_integer_maxcut_k5(solver):
    # K5 has 5 nodes and 10 edges. Max-Cut is 6.
    edges = list(itertools.combinations(range(5), 2))
    g = GraphInstance(name="k5", num_nodes=5, edges=edges)
    val, part = solver.solve_integer_maxcut(g)
    assert int(round(val)) == 6
    # Cut partition should split into 2 and 3 nodes: 2 * 3 = 6 edges
    assert sum(part) in (2, 3)


def test_cycle_relaxation_vs_k5(solver):
    # On K5:
    # Cycle relaxation allows x_e = 2/3, so objective = 10 * 2/3 = 6.6667
    # Full K5 relaxation adds sum x_e <= 6, so objective = 6.0
    edges = list(itertools.combinations(range(5), 2))
    g = GraphInstance(name="k5", num_nodes=5, edges=edges)

    sol_cycle = solver.solve_cycle_relaxation(g)
    assert abs(sol_cycle.objective_value - 20.0 / 3.0) < 1e-4

    sol_k5 = solver.solve_k5_relaxation(g)
    assert abs(sol_k5.objective_value - 6.0) < 1e-4
    # The K5 constraint should have dual multiplier approx 1.0
    assert len(sol_k5.k5_multipliers) == 1
    mult = next(iter(sol_k5.k5_multipliers.values()))
    assert abs(mult - 1.0) < 1e-4


def test_surrogate_relaxation_single_row(solver):
    # Test 1-row surrogate cutting plane on two K5 blocks with 1 bridge
    g = generate_k5_cluster_graph(num_k5=2, num_bridges=1, seed=42)
    # Integer Max-Cut is 2 * 6 + 1 = 13
    ip_val, _ = solver.solve_integer_maxcut(g)
    assert int(round(ip_val)) == 13

    sol_cycle = solver.solve_cycle_relaxation(g)
    # Cycle LP is 2 * (20/3) + 1 = 43/3 = 14.3333
    assert abs(sol_cycle.objective_value - 43.0 / 3.0) < 1e-4

    sol_k5 = solver.solve_k5_relaxation(g)
    assert abs(sol_k5.objective_value - 13.0) < 1e-4

    # Build surrogate row
    m = g.num_edges
    edge_to_idx = {e: i for i, e in enumerate(g.edges)}
    a_surr = np.zeros(m)
    b_surr = 0.0
    for k5, mult in sol_k5.k5_multipliers.items():
        b_surr += mult * 6.0
        for u, v in itertools.combinations(k5, 2):
            e = (min(u, v), max(u, v))
            a_surr[edge_to_idx[e]] += mult

    sol_surr = solver.solve_surrogate_relaxation(g, a_surr, b_surr)
    assert abs(sol_surr.objective_value - 13.0) < 1e-4


def test_face_invariant_dual_center(solver):
    g = generate_k5_cluster_graph(num_k5=3, num_bridges=2, seed=77)
    ga, gb, pi, pi_inv = generate_ab_pair(g, seed=88)

    center_a = solver.compute_face_invariant_dual_center(ga)
    center_b = solver.compute_face_invariant_dual_center(gb)

    # Invariance check: mapped cliques should have identical center values
    for clq_a, val_a in center_a.items():
        clq_b = tuple(sorted([pi[v] for v in clq_a]))
        assert clq_b in center_b
        val_b = center_b[clq_b]
        assert abs(val_a - val_b) < 1e-5


def test_face_invariant_dual_center_asymmetric(solver):
    # Test on K7 minus 2 edges (asymmetric clique orbit)
    from highs_turbo.graph_generator import generate_ks_minus_edges
    g = generate_ks_minus_edges(s=4, removed_edges=2, seed=42)
    ga, gb, pi, pi_inv = generate_ab_pair(g, seed=99)

    sol_k5 = solver.solve_k5_relaxation(g)
    center_a = solver.compute_face_invariant_dual_center(ga)
    center_b = solver.compute_face_invariant_dual_center(gb)

    # 1. Permutation invariance:
    for clq_a, val_a in center_a.items():
        clq_b = tuple(sorted([pi[v] for v in clq_a]))
        assert clq_b in center_b
        val_b = center_b[clq_b]
        assert abs(val_a - val_b) < 1e-5

    # 2. Mathematical optimality: surrogate cut built from canonical duals matches Full K5 LP bound
    edge_to_idx = {e: i for i, e in enumerate(ga.edges)}
    a_surr = np.zeros(ga.num_edges)
    b_surr = 0.0
    for clq, val in center_a.items():
        b_surr += val * 6.0
        for u, v in itertools.combinations(clq, 2):
            e = (min(u, v), max(u, v))
            a_surr[edge_to_idx[e]] += val

    sol_surr = solver.solve_surrogate_relaxation(ga, a_surr, b_surr)
    assert abs(sol_surr.objective_value - sol_k5.objective_value) < 1e-4


def test_verify_exact_dual_certificate(solver):
    w = {(0, 1): Fraction(1, 1), (1, 2): Fraction(1, 1), (0, 2): Fraction(1, 1)}
    a = {(0, 1): Fraction(1, 1), (1, 2): Fraction(1, 1), (0, 2): Fraction(1, 1)}

    # Valid certificate: lambda = 1, b = 2 => bound = 2, floor = 2
    cert = solver.verify_exact_dual_certificate(w, a, surrogate_rhs=2, surrogate_multiplier=1)
    assert cert.is_feasible
    assert cert.rational_upper_bound == Fraction(2, 1)
    assert cert.integer_floor == 2
    assert cert.residual_norm == Fraction(0, 1)

    # Hostile test: negative multiplier
    cert_neg = solver.verify_exact_dual_certificate(w, a, surrogate_rhs=2, surrogate_multiplier=-1)
    assert not cert_neg.is_feasible

    # Hostile test: user-provided negative slack
    bad_slack = {(0, 1): Fraction(-1, 2), (1, 2): Fraction(0, 1), (0, 2): Fraction(0, 1)}
    cert_slack = solver.verify_exact_dual_certificate(w, a, surrogate_rhs=2, surrogate_multiplier=1, slack_bounds=bad_slack)
    assert not cert_slack.is_feasible

    # Insufficient slack / dual feasibility deficit:
    zero_slack = {(0, 1): Fraction(0, 1), (1, 2): Fraction(0, 1), (0, 2): Fraction(0, 1)}
    a_low = {(0, 1): Fraction(1, 2), (1, 2): Fraction(1, 2), (0, 2): Fraction(1, 2)}
    cert_deficit = solver.verify_exact_dual_certificate(w, a_low, surrogate_rhs=2, surrogate_multiplier=1, slack_bounds=zero_slack)
    assert not cert_deficit.is_feasible
    assert cert_deficit.residual_norm > Fraction(0, 1)
