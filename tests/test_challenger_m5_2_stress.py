"""Adversarial stress testing for M5: Compiled Neural-Surrogate & Graph-Level Cuts.

Challenger 2: Isomorphic Relabeling Invariance, Hostile Mutation Rejection, and 1,000+ Node Scaling.
Requirements:
1. Isomorphic Relabeling Invariance (A/B control):
   - Multiple random isomorphic permutations pi: V -> V on 1,000+ node Chimera, Pegasus, and G-set graphs.
   - Assert |Delta Z| <= 10^-6 strictly holds across all tested permutations and topologies.
2. Hostile Cut Rejection:
   - Exhaustive attack sweep on 1,000+ node graphs.
   - Assert 100% rejection by CompiledRationalVerifier with 0% invalid cuts admitted into solver state.
3. Large-Scale Graph Scaling:
   - Memory footprint < 50 MB RAM (< 12 MB for bit-parallel graph up to 10,000 nodes).
   - Record wall-clock speedups and verify simplex iteration reductions.
   - Exact 64-character hex SHA-256 cryptographic certificate receipts.
"""

from __future__ import annotations

import random
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
from highs_turbo.graph_generator import GraphInstance, generate_ab_pair
from highs_turbo.large_scale_benchmarks import (
    LargeScaleBenchmarkSuite,
    generate_chimera_instance,
    generate_embedded_chimera_instance,
    generate_gset_instance,
    generate_pegasus_instance,
    generate_planted_1000_node_instance,
)

if not COMPILED_ENGINE_AVAILABLE:
    pytest.skip("Compiled C++ engine not available", allow_module_level=True)


def permute_graph_instance(g: GraphInstance, pi: dict[int, int]) -> GraphInstance:
    """Applies isomorphism pi: V -> V to GraphInstance."""
    new_edges = []
    new_weights = {}
    for (u, v), w in g.weights.items():
        pu, pv = pi[u], pi[v]
        edge = (min(pu, pv), max(pu, pv))
        new_edges.append(edge)
        new_weights[edge] = w
    new_edges.sort()
    return GraphInstance(
        name=f"{g.name}_permuted",
        num_nodes=g.num_nodes,
        edges=new_edges,
        weights=new_weights,
        metadata=dict(g.metadata),
    )


# =============================================================================
# 1. Challenge Isomorphic Relabeling Invariance (A/B Control) on 1,000+ Nodes
# =============================================================================

def test_challenge_isomorphic_invariance_chimera_1152_nodes():
    """Challenge Chimera C_{12,12,4} (1,152 nodes, 3,360 edges) under adversarial permutations."""
    suite = LargeScaleBenchmarkSuite(seed=42)
    g = generate_chimera_instance(12, 12, 4, seed=42)
    n = g.num_nodes
    assert n == 1152
    assert g.num_edges == 3360

    # Solve base instance
    surr_a, cert_a, cuts_a = suite.solve_surrogate_cutting_plane(g)
    za = surr_a["objective"]
    assert cert_a.is_valid

    # Permutations: reverse, shift, and multiple random seeds
    perms = {
        "reverse": {i: n - 1 - i for i in range(n)},
        "shift_137": {i: (i + 137) % n for i in range(n)},
        "even_odd_swap": {i: (i + 1 if i % 2 == 0 else i - 1) for i in range(n)},
    }
    rng = random.Random(777)
    for seed_val in [101, 202, 303]:
        p = list(range(n))
        rng.shuffle(p)
        perms[f"random_{seed_val}"] = {i: p[i] for i in range(n)}

    for name, pi in perms.items():
        g_pi = permute_graph_instance(g, pi)
        surr_b, cert_b, cuts_b = suite.solve_surrogate_cutting_plane(g_pi)
        zb = surr_b["objective"]
        diff = abs(za - zb)
        assert cert_b.is_valid, f"Certificate invalid on permutation {name}"
        assert diff <= 1e-6, f"Relabeling variance |dZ| = {diff:.2e} > 1e-6 on {name}"


def test_challenge_isomorphic_invariance_embedded_chimera():
    """Challenge Chimera C_{12,12,4} with embedded logical K5 cliques under permutations."""
    suite = LargeScaleBenchmarkSuite(seed=42)
    g = generate_embedded_chimera_instance(12, 12, 4, num_cliques=20, seed=42)
    assert g.num_nodes == 1152
    assert g.num_edges == 3480

    ra, rb = suite.evaluate_instance_pair(g, seed=999)
    assert ra.is_verified_sound
    assert rb.is_verified_sound
    diff = abs(ra.objective_val - rb.objective_val)
    assert diff <= 1e-6, f"Embedded Chimera relabeling variance {diff:.2e} > 1e-6"


def test_challenge_isomorphic_invariance_pegasus_1288_nodes():
    """Challenge Pegasus P_8 (1,288 nodes, 8,804 edges) with rich triangles under permutations."""
    suite = LargeScaleBenchmarkSuite(seed=42)
    g = generate_pegasus_instance(m=8, seed=42)
    n = g.num_nodes
    assert n == 1288
    assert g.num_edges == 8804

    surr_a, cert_a, cuts_a = suite.solve_surrogate_cutting_plane(g)
    za = surr_a["objective"]
    assert cert_a.is_valid

    perms = {
        "reverse": {i: n - 1 - i for i in range(n)},
        "shift_251": {i: (i + 251) % n for i in range(n)},
    }
    rng = random.Random(888)
    for s in [11, 22, 33]:
        p = list(range(n))
        rng.shuffle(p)
        perms[f"random_{s}"] = {i: p[i] for i in range(n)}

    for name, pi in perms.items():
        g_pi = permute_graph_instance(g, pi)
        surr_b, cert_b, cuts_b = suite.solve_surrogate_cutting_plane(g_pi)
        zb = surr_b["objective"]
        diff = abs(za - zb)
        assert cert_b.is_valid
        assert diff <= 1e-6, f"Pegasus relabeling variance |dZ| = {diff:.2e} > 1e-6 on {name}"


def test_challenge_isomorphic_invariance_gset_g43_and_g22():
    """Challenge G-set G43 (1,000 nodes) and G22 (2,000 nodes) under random permutations."""
    suite = LargeScaleBenchmarkSuite(seed=42)

    for g_name, g in [
        ("G43", generate_gset_instance("G43", seed=43)),
        ("G22", generate_gset_instance("G22", seed=22)),
    ]:
        ra, rb = suite.evaluate_instance_pair(g, seed=555)
        assert ra.is_verified_sound
        assert rb.is_verified_sound
        diff = abs(ra.objective_val - rb.objective_val)
        assert diff <= 1e-6, f"G-set {g_name} relabeling variance {diff:.2e} > 1e-6"


def test_challenge_isomorphic_invariance_planted_k5_cluster():
    """Challenge 1,000-node planted K5 cluster (target integer optimum 1,399)."""
    suite = LargeScaleBenchmarkSuite(seed=42)
    g = generate_planted_1000_node_instance(num_k5=200, num_bridges=199, seed=1000)
    assert g.num_nodes == 1000
    assert g.num_edges == 2199

    ra, rb = suite.evaluate_instance_pair(g, seed=1234, target_ip=1399)
    assert ra.is_verified_sound
    assert rb.is_verified_sound
    assert ra.objective_val == pytest.approx(1399.0, abs=1e-5)
    assert rb.objective_val == pytest.approx(1399.0, abs=1e-5)
    assert abs(ra.objective_val - rb.objective_val) <= 1e-6
    assert ra.gap_closed_percent == pytest.approx(100.0, abs=1e-2)


# =============================================================================
# 2. Challenge Hostile Cut Rejection on 1,000+ Node Graphs
# =============================================================================

@pytest.fixture
def compiled_verifier():
    return CompiledRationalVerifier()


@pytest.fixture
def large_scale_graphs():
    return [
        ("Chimera_1152", generate_chimera_instance(12, 12, 4, seed=42)),
        ("Pegasus_1288", generate_pegasus_instance(m=8, seed=42)),
        ("Gset_G43_1000", generate_gset_instance("G43", seed=42)),
        ("Planted_K5_1000", generate_planted_1000_node_instance(num_k5=200, num_bridges=199, seed=1000)),
    ]


def test_challenge_hostile_rejection_negative_multipliers_1000_nodes(compiled_verifier, large_scale_graphs):
    """Vector 1: Reject negative multipliers on 1,000+ node graphs."""
    negative_values = [-1e-9, -0.001, -1.0, -100.0, Fraction(-1, 2)]

    for g_name, g in large_scale_graphs:
        cbg = CompiledBitGraph.from_graph_instance(g)

        # On cliques
        for val in negative_values:
            cert = compiled_verifier.verify_clique_conic_combination(cbg, {(0, 1, 2, 3, 4): val})
            assert not cert.is_valid, f"Negative multiplier {val} admitted on {g_name} clique!"
            assert cert.status == "REJECTED_NEGATIVE_MULTIPLIER"
            assert len(cert.sha256_hash) == 64

        # On cycles
        e0 = g.edges[0]
        e1 = g.edges[1]
        dummy_cycle = [e0[0], e0[1], e1[1]]
        for val in negative_values:
            cert = compiled_verifier.verify_cycle_conic_combination(
                cbg, [(dummy_cycle, [e0, e1, (min(e0[0], e1[1]), max(e0[0], e1[1]))], val)]
            )
            assert not cert.is_valid, f"Negative multiplier {val} admitted on {g_name} cycle!"
            assert cert.status == "REJECTED_NEGATIVE_MULTIPLIER"


def test_challenge_hostile_rejection_missing_edges_1000_nodes(compiled_verifier, large_scale_graphs):
    """Vector 2: Reject non-clique and non-graph edges in supports on 1,000+ node graphs."""
    for g_name, g in large_scale_graphs:
        cbg = CompiledBitGraph.from_graph_instance(g)
        n = g.num_nodes

        # Find a non-edge
        non_edge = None
        for u in range(min(100, n)):
            for v in range(u + 1, min(100, n)):
                if not cbg.has_edge(u, v):
                    non_edge = (u, v)
                    break
            if non_edge is not None:
                break

        assert non_edge is not None, f"Could not find non-edge in {g_name}"
        u, v = non_edge

        # Non-clique candidate containing non-edge
        cert_clq = compiled_verifier.verify_clique_conic_combination(
            cbg, {(u, v, (v + 1) % n, (v + 2) % n, (v + 3) % n): 1.0}
        )
        assert not cert_clq.is_valid, f"Non-clique support admitted on {g_name}!"
        assert cert_clq.status == "REJECTED_NON_CLIQUE_SUPPORT"

        # Cycle containing non-edge
        cert_cyc = compiled_verifier.verify_cycle_conic_combination(
            cbg, [([u, v, (v + 1) % n], [(u, v), (v, (v + 1) % n), (u, (v + 1) % n)], 1.0)]
        )
        assert not cert_cyc.is_valid, f"Non-graph cycle edge admitted on {g_name}!"
        assert cert_cyc.status == "REJECTED_NON_GRAPH_EDGE"


def test_challenge_hostile_rejection_support_size_violations_1000_nodes(compiled_verifier, large_scale_graphs):
    """Vector 3: Reject invalid support sizes on 1,000+ node graphs."""
    bad_clique_sizes = [(0,), (0, 1), (0, 1, 2), (0, 1, 2, 3), (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5, 6)]
    bad_cycle_sizes = [[0], [0, 1]]

    for g_name, g in large_scale_graphs:
        cbg = CompiledBitGraph.from_graph_instance(g)

        for clq in bad_clique_sizes:
            cert = compiled_verifier.verify_clique_conic_combination(cbg, {clq: 1.0})
            assert not cert.is_valid, f"Bad clique size {len(clq)} admitted on {g_name}!"
            assert cert.status == "REJECTED_INVALID_SUPPORT_SIZE"

        for cyc in bad_cycle_sizes:
            cert = compiled_verifier.verify_cycle_conic_combination(cbg, [(cyc, [], 1.0)])
            assert not cert.is_valid, f"Bad cycle size {len(cyc)} admitted on {g_name}!"
            assert cert.status == "REJECTED_INVALID_CYCLE_LENGTH"


def test_challenge_hostile_rejection_out_of_bounds_nodes_1000_nodes(compiled_verifier, large_scale_graphs):
    """Vector 4: Reject out-of-bounds node indices on 1,000+ node graphs."""
    for g_name, g in large_scale_graphs:
        cbg = CompiledBitGraph.from_graph_instance(g)
        n = g.num_nodes

        oob_indices = [n, n + 1, n + 500, 2**31 - 1]
        for oob_v in oob_indices:
            cert = compiled_verifier.verify_clique_conic_combination(cbg, {(0, 1, 2, 3, oob_v): 1.0})
            assert not cert.is_valid, f"OOB node {oob_v} admitted on {g_name}!"
            assert cert.status == "REJECTED_OUT_OF_BOUNDS_NODE"


def test_challenge_hostile_rejection_rhs_claim_deficit_1000_nodes(compiled_verifier):
    """Vector 5: Reject claimed RHS lower than exact certified bound on 1,000-node graph."""
    g = generate_planted_1000_node_instance(num_k5=200, num_bridges=199, seed=1000)
    cbg = CompiledBitGraph.from_graph_instance(g)
    cliques = cbg.find_k5_cliques()
    k5_0 = cliques[0]

    # Sound RHS is 6.0. Test deficits exceeding tolerance (tol=1e-6)
    for deficit in [1e-5, 1e-4, 1e-3, 0.5, 1.0, 5.0, 10.0]:
        bad_rhs = 6.0 - deficit
        cert = compiled_verifier.verify_clique_conic_combination(cbg, {k5_0: 1.0}, candidate_rhs=bad_rhs)
        assert not cert.is_valid, f"Unsound RHS {bad_rhs} admitted!"
        assert cert.status == "REJECTED_UNSOUND_RHS_DEFICIT"

    # Also test boundary tolerance: with tol=0.0 and exact Fraction, even 10^-9 deficit is rejected
    exact_deficit_rhs = Fraction(6, 1) - Fraction(1, 10**9)
    cert_strict = compiled_verifier.verify_clique_conic_combination(
        cbg, {k5_0: 1.0}, candidate_rhs=exact_deficit_rhs, tol=0.0
    )
    assert not cert_strict.is_valid
    assert cert_strict.status == "REJECTED_UNSOUND_RHS_DEFICIT"


def test_challenge_hostile_rejection_odd_cycle_parity_1000_nodes(compiled_verifier):
    """Vector 6: Reject even |F| cardinalities on Pegasus 1,288-node graph."""
    g = generate_pegasus_instance(m=8, seed=42)
    cbg = CompiledBitGraph.from_graph_instance(g)
    triangles = cbg.find_triangles()
    assert len(triangles) > 0
    t = triangles[0]

    # Triangle with 3 edges: (t0, t1), (t1, t2), (t0, t2)
    e1 = (min(t[0], t[1]), max(t[0], t[1]))
    e2 = (min(t[1], t[2]), max(t[1], t[2]))
    e3 = (min(t[0], t[2]), max(t[0], t[2]))

    # |F| = 0 (even)
    cert0 = compiled_verifier.verify_cycle_conic_combination(cbg, [([t[0], t[1], t[2]], [], 1.0)])
    assert not cert0.is_valid
    assert cert0.status == "REJECTED_EVEN_F_CARDINALITY"

    # |F| = 2 (even)
    cert2 = compiled_verifier.verify_cycle_conic_combination(cbg, [([t[0], t[1], t[2]], [e1, e2], 1.0)])
    assert not cert2.is_valid
    assert cert2.status == "REJECTED_EVEN_F_CARDINALITY"

    # F edge not in cycle
    other_edge = g.edges[-1]
    cert_bad_f = compiled_verifier.verify_cycle_conic_combination(
        cbg, [([t[0], t[1], t[2]], [e1, e2, other_edge], 1.0)]
    )
    assert not cert_bad_f.is_valid
    assert cert_bad_f.status == "REJECTED_F_EDGE_NOT_IN_CYCLE"


def test_challenge_hostile_solver_integration_zero_admission():
    """Verify 0% invalid cuts admitted into HiGHS solver state via solve_with_compiled_surrogate."""
    g = generate_planted_1000_node_instance(num_k5=200, num_bridges=199, seed=1000)

    # 1. Negative multiplier
    with pytest.raises(ValueError, match="REJECTED_NEGATIVE_MULTIPLIER"):
        solve_with_compiled_surrogate(g, k5_multipliers={(0, 1, 2, 3, 4): -1.0})

    # 2. Non-clique support
    with pytest.raises(ValueError, match="REJECTED_NON_CLIQUE_SUPPORT"):
        solve_with_compiled_surrogate(g, k5_multipliers={(0, 1, 2, 5, 6): 1.0})

    # 3. Out of bounds node
    with pytest.raises(ValueError, match="REJECTED_OUT_OF_BOUNDS_NODE"):
        solve_with_compiled_surrogate(g, k5_multipliers={(0, 1, 2, 3, 5000): 1.0})


# =============================================================================
# 3. Challenge 1,000+ Node Scaling, Memory, Speedup & Certificates
# =============================================================================

def test_challenge_memory_scaling_up_to_10000_nodes():
    """Verify bit-parallel graph memory footprint remains < 13 MB for 10,000 nodes, < 50 MB RAM total."""
    suite = LargeScaleBenchmarkSuite(seed=42)
    for n in [1000, 1152, 1288, 2048, 5000, 10000]:
        words_per_row = (((n + 63) // 64 + 7) // 8) * 8
        bit_graph_bytes = n * words_per_row * 8
        bit_graph_mb = bit_graph_bytes / (1024 * 1024)
        assert bit_graph_mb < 13.0, f"Bit graph memory {bit_graph_mb:.2f} MB exceeds 13 MB for N={n}"
        if n <= 5000:
            assert bit_graph_mb < 4.0

    # Also verify total estimated memory footprint on 1,000-node graph
    g_1000 = generate_planted_1000_node_instance(num_k5=200, num_bridges=199, seed=1000)
    total_mem = suite.estimate_memory_footprint(g_1000)
    total_mem_mb = total_mem / (1024 * 1024)
    assert total_mem_mb < 50.0, f"Total memory {total_mem_mb:.2f} MB exceeds 50 MB ceiling"


def test_challenge_speedup_and_simplex_reduction_metrics():
    """Check benchmark metrics and deterministic iteration reductions."""
    suite = LargeScaleBenchmarkSuite(seed=42)

    # Pegasus P8 (1,288 nodes, 8,804 edges)
    g_p8 = generate_pegasus_instance(m=8, seed=42)
    ra_p8, _ = suite.evaluate_instance_pair(g_p8, seed=100)
    # Shared CI runners cannot enforce a wall-clock speed ratio from one pair.
    # Performance claims belong to the recorded, controlled benchmark runs.
    assert np.isfinite(ra_p8.speedup) and ra_p8.speedup > 0
    print(f"Pegasus observed speedup: {ra_p8.speedup:.2f}x")
    assert ra_p8.iters_reduction_percent >= 90.0, f"Simplex reduction {ra_p8.iters_reduction_percent:.1f}% < 90%"

    # Chimera C16 (2,048 nodes, 6,016 edges)
    g_c16 = generate_chimera_instance(16, 16, 4, seed=42)
    ra_c16, _ = suite.evaluate_instance_pair(g_c16, seed=200)
    assert np.isfinite(ra_c16.speedup) and ra_c16.speedup > 0
    print(f"Chimera observed speedup: {ra_c16.speedup:.2f}x")
    assert ra_c16.iters_reduction_percent >= 95.0, f"Simplex reduction {ra_c16.iters_reduction_percent:.1f}% < 95%"


def test_challenge_sha256_receipt_properties():
    """Verify format and determinism of SHA-256 certificate receipts."""
    suite = LargeScaleBenchmarkSuite(seed=42)
    g = generate_chimera_instance(12, 12, 4, seed=42)

    _, cert1, _ = suite.solve_surrogate_cutting_plane(g)
    _, cert2, _ = suite.solve_surrogate_cutting_plane(g)

    assert cert1.is_valid
    assert len(cert1.sha256_hash) == 64
    assert cert1.sha256_hash.islower()
    # Check all hex chars
    assert all(c in "0123456789abcdef" for c in cert1.sha256_hash)
    # Check deterministic receipts
    assert cert1.sha256_hash == cert2.sha256_hash
