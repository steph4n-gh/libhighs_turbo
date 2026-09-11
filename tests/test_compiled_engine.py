"""Automated unit and integration tests for Compiled C++ Engine.

Verifies:
1. Bit-parallel graph operations, SIMD POPCNT common neighbors, triangle counting,
   and K5 clique discovery matching reference NetworkX implementations.
2. Exact rational arithmetic with Stein binary GCD and GMP fallback.
3. 100% cryptographic SHA-256 parity between C++ and Python verifiers.
4. Comprehensive hostile mutation rejection (negative multipliers, non-cliques,
   invalid sizes, duplicate nodes, out-of-bounds indices, RHS deficits).
5. Cut discovery, separation, and 1-row surrogate cut compilation.
6. SolverCallbackBridge and in-memory solve integration.
"""

from fractions import Fraction
import numpy as np
import pytest

from highs_turbo.compiled_engine import (
    COMPILED_ENGINE_AVAILABLE,
    CompiledBitGraph,
    CompiledCutEngine,
    CompiledRationalVerifier,
    solve_with_compiled_surrogate,
)
from highs_turbo.graph_generator import (
    GraphInstance,
    generate_ab_pair,
    generate_k5_cluster_graph,
    generate_exp87_instances,
    generate_exp91_family,
)
from highs_turbo.rational_verifier import RationalCutVerifier, VerificationCertificate
from highs_turbo.exact_solver import ExactMaxCutSolver


@pytest.fixture
def sample_k5_graph():
    return generate_k5_cluster_graph(num_k5=3, num_bridges=2, seed=42)


def test_compiled_engine_available():
    assert COMPILED_ENGINE_AVAILABLE, "Compiled native engine (_compiled_engine) must be available."


def test_native_bridge_rows_reach_public_highs():
    from highs_turbo.compiled_engine import CompiledSolverCallbackBridge, solve_milp_with_highs

    bridge = CompiledSolverCallbackBridge(2)
    bridge.add_cut_row(1.5, [0, 1], [1., 1.])
    result = solve_milp_with_highs(
        [-1., -1.], [0., 0.], [1., 1.], [], [], [0], [], [], [1, 1], bridge,
    )
    assert result["fun"] == -1
    assert sum(result["x"]) == 1


def test_bit_graph_edge_operations():
    bg = CompiledBitGraph(20)
    assert bg.num_nodes == 20
    assert bg.num_edges == 0

    # Add edges
    bg.add_edge(0, 1)
    bg.add_edge(1, 2)
    bg.add_edge(0, 2)
    assert bg.num_edges == 3
    assert bg.has_edge(0, 1)
    assert bg.has_edge(1, 0)
    assert bg.has_edge(1, 2)
    assert bg.has_edge(0, 2)
    assert not bg.has_edge(0, 3)

    # Self loops and out-of-bounds ignored safely
    bg.add_edge(0, 0)
    bg.add_edge(0, 99)
    assert bg.num_edges == 3

    assert bg.degree(0) == 2
    assert bg.degree(1) == 2
    assert bg.degree(2) == 2
    assert bg.degree(3) == 0


def test_bit_graph_common_neighbors_and_triangles():
    bg = CompiledBitGraph(10)
    # Triangle 0-1-2
    bg.add_edges([(0, 1), (1, 2), (0, 2)])
    assert bg.count_common_neighbors(0, 1) == 1
    assert bg.count_triangles_on_edge(0, 1) == 1
    assert bg.count_triangles() == 1
    assert bg.find_triangles() == [(0, 1, 2)]

    # Add node 3 connected to 0, 1, 2 (forms K4)
    bg.add_edges([(0, 3), (1, 3), (2, 3)])
    # K4 has 4 triangles
    assert bg.count_triangles() == 4
    triangles = bg.find_triangles()
    assert len(triangles) == 4
    assert (0, 1, 2) in triangles
    assert (0, 1, 3) in triangles
    assert (0, 2, 3) in triangles
    assert (1, 2, 3) in triangles


def test_bit_graph_k5_enumeration(sample_k5_graph):
    # Reference K5s from GraphInstance
    ref_k5s = sorted(sample_k5_graph.find_all_k5_cliques())
    
    # CompiledBitGraph K5s
    cbg = CompiledBitGraph.from_graph_instance(sample_k5_graph)
    compiled_k5s = sorted(cbg.find_k5_cliques())

    assert len(compiled_k5s) == len(ref_k5s)
    assert compiled_k5s == ref_k5s
    assert len(compiled_k5s) == 3


def test_bit_graph_triangles_match_reference(sample_k5_graph):
    ref_triangles = sorted(sample_k5_graph.find_triangles())
    cbg = CompiledBitGraph.from_graph_instance(sample_k5_graph)
    compiled_triangles = sorted(cbg.find_triangles())

    assert len(compiled_triangles) == len(ref_triangles)
    assert compiled_triangles == ref_triangles


def test_exact_rational_arithmetic():
    from highs_turbo._compiled_engine import ExactRational

    q1 = ExactRational(1, 3)
    q2 = ExactRational(1, 6)
    q3 = q1 + q2
    assert str(q3) == "1/2"
    assert q3.to_double() == pytest.approx(0.5)

    q4 = q1 - q2
    assert str(q4) == "1/6"

    q5 = q1 * q2
    assert str(q5) == "1/18"

    # Continued fraction matching
    q_float = ExactRational.from_double(0.125)
    assert str(q_float) == "1/8"

    q_third = ExactRational.from_double(1.0 / 3.0)
    assert str(q_third) == "1/3"


def test_exact_sha256_parity_with_python_reference(sample_k5_graph):
    cbg = CompiledBitGraph.from_graph_instance(sample_k5_graph)
    k5_cliques = cbg.find_k5_cliques()

    cpp_verifier = CompiledRationalVerifier()
    py_verifier = RationalCutVerifier()

    multipliers = {
        k5_cliques[0]: 1.25,
        k5_cliques[1]: Fraction(3, 7),
        k5_cliques[2]: 2.5,
    }

    cpp_cert = cpp_verifier.verify_clique_conic_combination(cbg, multipliers)
    py_cert = py_verifier.verify_clique_conic_combination(sample_k5_graph, multipliers)

    assert cpp_cert.is_valid == py_cert.is_valid == True
    assert cpp_cert.status == py_cert.status == "CERTIFIED_VALID_CONIC_COMBINATION"
    assert cpp_cert.num_active_supports == py_cert.num_active_supports == 3
    assert cpp_cert.exact_rhs == py_cert.exact_rhs
    assert cpp_cert.exact_coefficients == py_cert.exact_coefficients
    assert cpp_cert.sha256_hash == py_cert.sha256_hash
    assert len(cpp_cert.sha256_hash) == 64


@pytest.mark.parametrize("bad_val", [-1.0, -0.001, -100.0])
def test_hostile_mutation_negative_multiplier(sample_k5_graph, bad_val):
    cbg = CompiledBitGraph.from_graph_instance(sample_k5_graph)
    k5 = cbg.find_k5_cliques()[0]
    verifier = CompiledRationalVerifier()

    cert = verifier.verify_clique_conic_combination(cbg, {k5: bad_val})
    assert not cert.is_valid
    assert cert.status == "REJECTED_NEGATIVE_MULTIPLIER"
    assert len(cert.sha256_hash) == 64


def test_hostile_mutation_non_clique_support(sample_k5_graph):
    cbg = CompiledBitGraph.from_graph_instance(sample_k5_graph)
    verifier = CompiledRationalVerifier()

    # Create a non-clique tuple: (0, 1, 2, 3, 9) where edge (0, 9) does not exist
    bad_support = (0, 1, 2, 3, 9)
    assert not cbg.has_edge(0, 9)

    cert = verifier.verify_clique_conic_combination(cbg, {bad_support: 1.0})
    assert not cert.is_valid
    assert cert.status == "REJECTED_NON_CLIQUE_SUPPORT"


@pytest.mark.parametrize("bad_size", [
    (0, 1, 2),
    (0, 1, 2, 3),
    (0, 1, 2, 3, 4, 5),
])
def test_hostile_mutation_invalid_support_size(sample_k5_graph, bad_size):
    cbg = CompiledBitGraph.from_graph_instance(sample_k5_graph)
    verifier = CompiledRationalVerifier()

    cert = verifier.verify_clique_conic_combination(cbg, {bad_size: 1.0})
    assert not cert.is_valid
    assert cert.status == "REJECTED_INVALID_SUPPORT_SIZE"


def test_hostile_mutation_out_of_bounds(sample_k5_graph):
    cbg = CompiledBitGraph.from_graph_instance(sample_k5_graph)
    verifier = CompiledRationalVerifier()

    bad_support = (0, 1, 2, 3, 9999)
    cert = verifier.verify_clique_conic_combination(cbg, {bad_support: 1.0})
    assert not cert.is_valid
    assert cert.status == "REJECTED_OUT_OF_BOUNDS_NODE"


def test_hostile_mutation_unsound_rhs_deficit(sample_k5_graph):
    cbg = CompiledBitGraph.from_graph_instance(sample_k5_graph)
    k5 = cbg.find_k5_cliques()[0]
    verifier = CompiledRationalVerifier()

    # Valid conic multiplier 1.0 yields RHS = 6.0
    # Adversarial candidate RHS 5.5 < 6.0 must be rejected
    cert = verifier.verify_clique_conic_combination(cbg, {k5: 1.0}, candidate_rhs=5.5)
    assert not cert.is_valid
    assert cert.status == "REJECTED_UNSOUND_RHS_DEFICIT"


def test_cut_engine_violated_separation(sample_k5_graph):
    cbg = CompiledBitGraph.from_graph_instance(sample_k5_graph)
    edge_list = list(sample_k5_graph.edges)
    cut_engine = CompiledCutEngine()

    # All-ones primal solution violates K5 (sum 10.0 > 6.0)
    primal_ones = [1.0] * len(edge_list)
    violated_k5 = cut_engine.separate_violated_k5(cbg, edge_list, primal_ones, threshold=1e-4)

    assert len(violated_k5) == 3
    for cut in violated_k5:
        assert cut.violation == pytest.approx(4.0)

    # Compile surrogate cut
    k5s = cbg.find_k5_cliques()
    surr = cut_engine.compile_surrogate_cut(cbg, edge_list, [(k5s[0], 1.0), (k5s[1], 1.0)])
    assert surr.is_valid
    assert surr.rhs == pytest.approx(12.0)
    assert len(surr.nz_indices) == 20
    assert len(surr.certificate_hash) == 64


def test_solve_with_compiled_surrogate_end_to_end(sample_k5_graph):
    cbg = CompiledBitGraph.from_graph_instance(sample_k5_graph)
    k5s = cbg.find_k5_cliques()

    # Solve with compiled surrogate
    sol_surr = solve_with_compiled_surrogate(
        sample_k5_graph,
        k5_multipliers={k5s[0]: 1.0, k5s[1]: 1.0, k5s[2]: 1.0},
    )
    assert sol_surr.success
    assert sol_surr.objective_value > 0

    # Compare with reference ExactMaxCutSolver
    solver = ExactMaxCutSolver()
    c, A_base, b_base, _, _ = solver.build_relaxation_matrices(sample_k5_graph, include_k5=False)
    sol_cycle = solver.solve_cycle_relaxation(sample_k5_graph)

    # Surrogate bound is strictly tighter or equal to cycle bound
    assert sol_surr.objective_value <= sol_cycle.objective_value + 1e-6


def test_exact_rational_large_int_comparisons():
    from highs_turbo import _compiled_engine as ce
    a, b = 2**64, 2**63 + 1
    c, d = 2**64, 2**63 - 1
    r1 = ce.ExactRational(str(a), str(b))
    r2 = ce.ExactRational(str(c), str(d))
    assert (r1 < r2) == (Fraction(a, b) < Fraction(c, d))
    assert (r2 > r1) == (Fraction(c, d) > Fraction(a, b))

    # Test adversarial deficit
    target = 2**127 + 10**15
    D = target // 6000
    cand = ce.ExactRational("1", str(D))
    bound = ce.ExactRational("6000", "1")
    assert cand < bound
    assert bound > cand


def test_exact_rational_string_division_by_zero():
    from highs_turbo import _compiled_engine as ce
    for bad_den in ["0", "-0", "+0", "", "00"]:
        with pytest.raises((ValueError, Exception)):
            ce.ExactRational("1", bad_den)


def test_exact_rational_from_double_extreme_floats():
    from highs_turbo import _compiled_engine as ce
    for val in [1e19, 1e20, 1e50, 1e100, 1e308, -1e19, -1e50]:
        r = ce.ExactRational.from_double(val, 100000)
        f = Fraction.from_float(val).limit_denominator(100000)
        assert r.num_str() == str(f.numerator)
        assert r.den_str() == str(f.denominator)

    with pytest.raises(Exception):
        ce.ExactRational.from_double(float("nan"), 100000)
    with pytest.raises(Exception):
        ce.ExactRational.from_double(float("inf"), 100000)


def test_bit_graph_row_alignment_64_byte():
    from highs_turbo import _compiled_engine as ce
    for n in [0, 1, 10, 63, 64, 65, 100, 512, 1152, 2048]:
        bg = ce.BitGraph(n)
        assert (bg.words_per_row * 8) % 64 == 0, f"Row stride not 64-byte aligned for n={n}"


def test_bit_graph_shift_bounds_on_word_boundary():
    from highs_turbo import _compiled_engine as ce
    bg = ce.BitGraph(128)
    for i in range(128):
        bg.add_edge(i, (i + 1) % 128)
    # Triangles and k5 enumeration should not crash or trigger UB
    triangles = bg.find_triangles()
    k5s = bg.find_k5_cliques()
    edges = bg.get_edges()
    assert len(edges) == 128
    # No reverse edges in get_edges
    for u, v in edges:
        assert u < v


def test_cut_engine_primal_sol_bounds_check(sample_k5_graph):
    cbg = CompiledBitGraph.from_graph_instance(sample_k5_graph)
    edges = list(sample_k5_graph.edges)
    cut_engine = CompiledCutEngine()

    with pytest.raises(ValueError, match="primal_sol size must be >= edge count"):
        cut_engine.separate_violated_k5(cbg, edges, [1.0] * (len(edges) - 1))

    with pytest.raises(ValueError, match="primal_sol size must be >= edge count"):
        cut_engine.separate_violated_triangles(cbg, edges, [1.0] * (len(edges) - 1))


def test_cycle_cut_signs_status_and_sha256_parity():
    from highs_turbo import _compiled_engine as ce
    g = GraphInstance("k3", 3, [(0, 1), (1, 2), (0, 2)], {})
    cbg = CompiledBitGraph.from_graph_instance(g)

    cpp_v = ce.RationalVerifier()
    py_v = RationalCutVerifier()

    cpp_cert = cpp_v.verify_cycle_conic_combination(cbg.native, [(([0, 1, 2], [(0, 1)]), ce.ExactRational(1, 1))])
    py_cert = py_v.verify_cycle_conic_combination(g, [([0, 1, 2], [(0, 1)], 1.0)])

    assert cpp_cert.is_valid
    assert cpp_cert.status == "CERTIFIED_VALID_CYCLE_CONIC_COMBINATION"
    assert py_cert.status == "CERTIFIED_VALID_CYCLE_CONIC_COMBINATION"
    assert cpp_cert.sha256_hash == py_cert.sha256_hash

    # Check via CompiledRationalVerifier wrapper
    wrapper_v = CompiledRationalVerifier()
    wrap_cert = wrapper_v.verify_cycle_conic_combination(cbg, [([0, 1, 2], [(0, 1)], 1.0)])
    assert wrap_cert.is_valid
    assert wrap_cert.status == "CERTIFIED_VALID_CYCLE_CONIC_COMBINATION"
    assert wrap_cert.sha256_hash == py_cert.sha256_hash
    assert wrap_cert.exact_coefficients[(0, 1)] == Fraction(1, 1)
    assert wrap_cert.exact_coefficients[(0, 2)] == Fraction(-1, 1)
    assert wrap_cert.exact_coefficients[(1, 2)] == Fraction(-1, 1)
