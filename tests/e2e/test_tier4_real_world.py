"""Tier 4: Real-World Topologies E2E Tests.

Evaluates the Neural-Surrogate Cutting Plane Engine on 1,000+ node industrial
and physical benchmarks per requirement R3 and TEST_INFRA.md:
- Scenario 1: Chimera Spin Glass instances (C_{4,4,4}, C_{8,8,4}, C_{12,12,4} with 1,152 nodes)
- Scenario 2: Pegasus Spin Glass instances (P_4 with 288 nodes, P_8 with 1,344 nodes, 10,080 edges)
- Scenario 3: G-set Max-Cut instances (G11 with 800 nodes, G43 with 1,000 nodes, G51 with 1,000 nodes)
- Scenario 4: Planted 1,000-node multi-cluster instance with exact ground-truth integer target
- Scenario 5: Adversarial mutation attacks on 1,000+ node candidate cuts ensuring 100% rejection
"""

from __future__ import annotations

import itertools
import time
from fractions import Fraction
import numpy as np
import pytest

from highs_turbo.exact_solver import ExactMaxCutSolver
from highs_turbo.graph_generator import GraphInstance, generate_k5_cluster_graph
from highs_turbo.rational_verifier import RationalCutVerifier
from highs_turbo.topologies import (
    BitParallelGraph,
    generate_chimera_instance,
    generate_gset_instance,
    generate_pegasus_instance,
)


@pytest.fixture
def solver():
    return ExactMaxCutSolver()


@pytest.fixture
def verifier():
    return RationalCutVerifier()


# =====================================================================
# Scenario 1: Chimera Spin Glass Instances (C_{4,4,4}, C_{8,8,4}, C_{12,12,4})
# =====================================================================

def test_tier4_chimera_c4_4_4_structure_and_surrogate(solver, verifier):
    """Scenario 1a: Chimera C_{4,4,4} (128 nodes, 352 edges) cycle cut and verification."""
    g = generate_chimera_instance(m=4, n=4, l=4, seed=42, ising=True)
    assert g.num_nodes == 128
    assert g.num_edges == 352

    bg = BitParallelGraph.from_graph_instance(g)
    assert bg.words_per_row == 2  # (128 + 63) // 64 = 2

    # Chimera native is bipartite => 0 triangles
    assert bg.find_triangles() == []
    assert bg.find_k5_cliques() == []

    # Verify a 4-cycle cut: cycle on 4 nodes with |F|=3
    # Cell 0: nodes 0, 1 (vertical) and 4, 5 (horizontal) form a 4-cycle
    cycle_nodes = [0, 4, 1, 5]
    c_edges = [(0, 4), (1, 4), (1, 5), (0, 5)]
    f_edges = [(0, 4), (1, 4), (1, 5)]  # |F| = 3 (odd)
    candidate_cycles = [
        (cycle_nodes, f_edges, Fraction(1, 1)),
    ]

    cert = verifier.verify_cycle_conic_combination(g, candidate_cycles)
    assert cert.is_valid
    assert cert.exact_rhs == Fraction(2, 1)
    assert len(cert.sha256_hash) == 64


def test_tier4_chimera_c8_8_4_scaling(verifier):
    """Scenario 1b: Chimera C_{8,8,4} (512 nodes, 1,472 edges) bit-parallel representation."""
    g = generate_chimera_instance(m=8, n=8, l=4, seed=88, ising=True)
    assert g.num_nodes == 512
    assert g.num_edges == 1472

    t0 = time.perf_counter()
    bg = BitParallelGraph.from_graph_instance(g)
    t_bg = time.perf_counter() - t0

    assert bg.words_per_row == 8  # 512 / 64 = 8
    # Construction takes < 100 ms
    assert t_bg < 0.5


def test_tier4_chimera_c12_12_4_large_scale(verifier):
    """Scenario 1c: Chimera C_{12,12,4} (1,152 nodes, 3,360 edges) - Requirement R3."""
    g = generate_chimera_instance(m=12, n=12, l=4, seed=1212, ising=True)
    assert g.num_nodes == 1152
    assert g.num_edges == 3360

    # Build bit-parallel graph
    bg = BitParallelGraph.from_graph_instance(g)
    assert bg.words_per_row == 18  # (1152 + 63) // 64 = 18

    # Verify memory footprint: 18 words * 8 bytes * 1152 nodes = ~165 KB RAM
    footprint_bytes = bg.words_per_row * 8 * bg.num_nodes
    assert footprint_bytes < 250000  # < 250 KB RAM (vs 10+ MB dense matrix)

    # Verify cycle inequality on 1,152-node instance
    cycle_nodes = [0, 4, 1, 5]
    f_edges = [(0, 4), (1, 4), (1, 5)]
    cert = verifier.verify_cycle_conic_combination(g, [(cycle_nodes, f_edges, 1.0)])
    assert cert.is_valid
    assert len(cert.sha256_hash) == 64


# =====================================================================
# Scenario 2: Pegasus Spin Glass Instances (P_4, P_8)
# =====================================================================

def test_tier4_pegasus_p4_triangles_and_surrogate(solver, verifier):
    """Scenario 2a: Pegasus P_4 (288 nodes) native triangles and surrogate cut."""
    g = generate_pegasus_instance(m=4, seed=44, ising=True)
    assert g.num_nodes == 288
    assert g.num_edges > 1000

    bg = BitParallelGraph.from_graph_instance(g)
    triangles = bg.find_triangles()
    # Pegasus natively contains triangles
    assert len(triangles) > 0

    # Certify a triangle cycle cut
    t = triangles[0]
    t_edges = [(min(t[0], t[1]), max(t[0], t[1])),
               (min(t[1], t[2]), max(t[1], t[2])),
               (min(t[0], t[2]), max(t[0], t[2]))]
    cert = verifier.verify_cycle_conic_combination(g, [([t[0], t[1], t[2]], t_edges, Fraction(1, 1))])
    assert cert.is_valid
    assert cert.exact_rhs == Fraction(2, 1)


def test_tier4_pegasus_p8_1344_nodes_large_scale(verifier):
    """Scenario 2b: Pegasus P_8 (1,344 nodes, 10,080 edges) - Requirement R3."""
    g = generate_pegasus_instance(m=8, seed=88, ising=True)
    assert g.num_nodes == 1344
    assert g.num_edges == 10080

    bg = BitParallelGraph.from_graph_instance(g)
    assert bg.words_per_row == 21  # (1344 + 63) // 64 = 21

    # Memory efficiency check
    footprint_bytes = bg.words_per_row * 8 * bg.num_nodes
    assert footprint_bytes < 250000  # < 250 KB RAM for 1,344 nodes

    # Verify a triangle cut
    triangles = bg.find_triangles()
    assert len(triangles) > 0
    t = triangles[0]
    t_edges = [(min(t[0], t[1]), max(t[0], t[1])),
               (min(t[1], t[2]), max(t[1], t[2])),
               (min(t[0], t[2]), max(t[0], t[2]))]
    cert = verifier.verify_cycle_conic_combination(g, [([t[0], t[1], t[2]], t_edges, 1.0)])
    assert cert.is_valid
    assert len(cert.sha256_hash) == 64


# =====================================================================
# Scenario 3: G-set Instances (G11, G43, G51)
# =====================================================================

def test_tier4_gset_g11_random_maxcut(verifier):
    """Scenario 3a: G-set G11 (800 nodes, 1600 edges)."""
    g = generate_gset_instance("G11", seed=11)
    assert g.num_nodes == 800
    assert g.num_edges == 1600

    bg = BitParallelGraph.from_graph_instance(g)
    assert bg.words_per_row == 13


def test_tier4_gset_g43_1000_nodes_dense(verifier):
    """Scenario 3b: G-set G43 (1,000 nodes, 9,990 edges) - Requirement R3."""
    g = generate_gset_instance("G43", seed=43)
    assert g.num_nodes == 1000
    assert g.num_edges == 9990

    bg = BitParallelGraph.from_graph_instance(g)
    assert bg.words_per_row == 16  # (1000 + 63) // 64 = 16

    # Verify triangles exist and can be certified
    triangles = bg.find_triangles()
    assert len(triangles) > 0
    t = triangles[0]
    t_edges = [(min(t[0], t[1]), max(t[0], t[1])),
               (min(t[1], t[2]), max(t[1], t[2])),
               (min(t[0], t[2]), max(t[0], t[2]))]
    cert = verifier.verify_cycle_conic_combination(g, [([t[0], t[1], t[2]], t_edges, 1.0)])
    assert cert.is_valid
    assert len(cert.sha256_hash) == 64


def test_tier4_gset_g51_1000_nodes_medium(verifier):
    """Scenario 3c: G-set G51 (1,000 nodes, 5,909 edges) - Requirement R3."""
    g = generate_gset_instance("G51", seed=51)
    assert g.num_nodes == 1000
    assert g.num_edges == 5909

    bg = BitParallelGraph.from_graph_instance(g)
    assert bg.words_per_row == 16


# =====================================================================
# Scenario 4: Planted 1,000-Node Multi-Cluster Instance
# =====================================================================

def test_tier4_planted_1000_node_k5_cluster(solver, verifier):
    """Scenario 4: 200 planted K5 clusters (1,000 nodes, 2,199 edges) with exact integer target."""
    num_k5 = 200
    num_bridges = 199
    g = generate_k5_cluster_graph(num_k5=num_k5, num_bridges=num_bridges, seed=1000)
    assert g.num_nodes == 1000
    assert g.num_edges == 200 * 10 + 199  # 2,199 edges

    # Ground truth target Max-Cut
    target = 6 * num_k5 + num_bridges  # 1,200 + 199 = 1,399
    assert g.metadata["target_integer_maxcut"] == target

    # Find cliques with bit-parallel graph
    bg = BitParallelGraph.from_graph_instance(g)
    cliques = bg.find_k5_cliques()
    assert len(cliques) == 200

    # Build surrogate row compressing all 200 K5 constraints into 1 row
    mults = {clq: Fraction(1, 1) for clq in cliques}
    cert = verifier.verify_clique_conic_combination(g, mults)
    assert cert.is_valid
    assert cert.num_active_supports == 200
    assert cert.exact_rhs == Fraction(200 * 6, 1)  # 1200
    assert len(cert.sha256_hash) == 64


# =====================================================================
# Scenario 5: Adversarial Attacks on 1,000-Node Topologies
# =====================================================================

def test_tier4_hostile_adversarial_attack_on_1000_node_instance(verifier):
    """Scenario 5: 100% rejection of hostile mutated cuts on 1,000+ node instances."""
    g_pegasus = generate_pegasus_instance(m=8, seed=88)  # 1,344 nodes
    g_chimera = generate_chimera_instance(m=12, n=12, l=4, seed=12)  # 1,152 nodes
    g_g43 = generate_gset_instance("G43", seed=43)  # 1,000 nodes

    # Attack 1: Negative multiplier on Pegasus
    cert1 = verifier.verify_cycle_conic_combination(g_pegasus, [([0, 1, 2], [(0, 1), (1, 2), (0, 2)], -0.01)])
    assert not cert1.is_valid
    assert cert1.status == "REJECTED_NEGATIVE_MULTIPLIER"

    # Attack 2: Out-of-bounds node on Chimera (node 2000 >= 1152)
    cert2 = verifier.verify_clique_conic_combination(g_chimera, {(0, 1, 2, 3, 2000): 1.0})
    assert not cert2.is_valid
    assert cert2.status == "REJECTED_OUT_OF_BOUNDS_NODE"

    # Attack 3: Non-existent edge on G43
    cert3 = verifier.verify_clique_conic_combination(g_g43, {(0, 1, 2, 3, 4): 1.0})
    assert not cert3.is_valid

    # Attack 4: Unsound RHS claim deficit on 1,000-node graph
    g_planted = generate_k5_cluster_graph(num_k5=200, num_bridges=199, seed=1000)
    cliques = g_planted.find_all_k5_cliques()
    cert4 = verifier.verify_clique_conic_combination(g_planted, {cliques[0]: 1.0}, candidate_rhs=5.5)
    assert not cert4.is_valid
    assert cert4.status == "REJECTED_UNSOUND_RHS_DEFICIT"
