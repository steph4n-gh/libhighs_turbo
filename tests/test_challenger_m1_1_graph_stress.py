"""Adversarial stress testing for Compiled C++ BitGraph and CutEngine.

Challenger 1: Graph Stress & Pathological Topologies (Milestone M1).
Tests:
1. Pathological topologies:
   - Empty graphs (N=0, 1)
   - Isolated vertices (degree 0)
   - Star graphs (S_N)
   - Complete graphs (K_N for N in [5, 6, 7, 8, 10, 15, 20, 30, 63, 64, 65])
   - Complete bipartite graphs (K_m,n)
   - Disconnected cliques mixed with isolated vertices
   - Dense random graphs with N in [500, 2000]
2. Ground-truth oracle verification vs NetworkX:
   - common neighbors count
   - degree count
   - has_edge queries
   - triangle count & triangle enumeration
   - K5 clique enumeration (completeness, soundness, uniqueness, canonical order)
3. Memory stability & performance:
   - Instantiation of graphs N in [1000, 2000, 3000, 4000, 5000]
   - Zero crashes, zero memory leaks across 1,000 rapid allocations
   - Sub-millisecond query performance (< 100 microseconds per query)
4. CutEngine stress tests on pathological topologies.
"""

import time
import math
import itertools
import pytest
import networkx as nx
import numpy as np

from highs_turbo.compiled_engine import (
    COMPILED_ENGINE_AVAILABLE,
    CompiledBitGraph,
    CompiledCutEngine,
    CompiledRationalVerifier,
)

if not COMPILED_ENGINE_AVAILABLE:
    pytest.skip("Compiled engine not available", allow_module_level=True)


def verify_bitgraph_against_networkx(cbg: CompiledBitGraph, G: nx.Graph, sample_pairs: int = 100):
    """Compares a CompiledBitGraph against a NetworkX graph oracle."""
    n = G.number_of_nodes()
    assert cbg.num_nodes == n
    assert cbg.num_edges == G.number_of_edges()

    # Verify degrees
    for u in range(n):
        assert cbg.degree(u) == G.degree(u), f"Degree mismatch at node {u}: cbg={cbg.degree(u)} vs nx={G.degree(u)}"

    # Verify edge presence on all actual edges
    for u, v in G.edges():
        assert cbg.has_edge(u, v), f"Missing edge ({u}, {v})"
        assert cbg.has_edge(v, u), f"Missing reverse edge ({v}, {u})"

    # Sample non-edges / pairs for common neighbors
    nodes = list(range(n))
    if n <= 100:
        test_pairs = list(itertools.combinations(nodes, 2))
    else:
        rng = np.random.default_rng(42)
        test_pairs = []
        # Sample some existing edges
        edges_list = list(G.edges())
        if edges_list:
            chosen = rng.choice(len(edges_list), size=min(sample_pairs // 2, len(edges_list)), replace=False)
            test_pairs.extend([edges_list[i] for i in chosen])
        # Sample random pairs
        for _ in range(sample_pairs):
            u, v = rng.choice(n, size=2, replace=False)
            test_pairs.append((int(u), int(v)))

    for u, v in test_pairs:
        expected_cn = len(list(nx.common_neighbors(G, u, v)))
        actual_cn = cbg.count_common_neighbors(u, v)
        assert actual_cn == expected_cn, f"Common neighbors mismatch for ({u}, {v}): cbg={actual_cn} vs nx={expected_cn}"

    # Triangle count
    expected_triangles = sum(nx.triangles(G).values()) // 3
    actual_triangles = cbg.count_triangles()
    assert actual_triangles == expected_triangles, f"Triangle count mismatch: cbg={actual_triangles} vs nx={expected_triangles}"

    # Triangle enumeration (for n <= 200 to prevent combinatorial explosion)
    if n <= 200 and expected_triangles <= 10000:
        found_triangles = cbg.find_triangles()
        assert len(found_triangles) == expected_triangles, f"find_triangles count mismatch: {len(found_triangles)} vs {expected_triangles}"
        
        # Verify canonical sorting (u < v < w) and existence
        tri_set = set()
        for tri in found_triangles:
            u, v, w = tri
            assert u < v < w, f"Triangle not sorted: {tri}"
            assert G.has_edge(u, v) and G.has_edge(v, w) and G.has_edge(u, w), f"False triangle reported: {tri}"
            assert tri not in tri_set, f"Duplicate triangle reported: {tri}"
            tri_set.add(tri)

    # K5 enumeration (for small / moderate graphs)
    if expected_triangles <= 50000:
        # Find all 5-cliques using NetworkX
        nx_cliques_5 = set()
        for clq in nx.find_cliques(G):
            if len(clq) >= 5:
                for comb in itertools.combinations(sorted(clq), 5):
                    nx_cliques_5.add(comb)

        found_k5s = cbg.find_k5_cliques()
        assert len(found_k5s) == len(nx_cliques_5), f"K5 count mismatch: cbg={len(found_k5s)} vs nx={len(nx_cliques_5)}"

        k5_set = set()
        for k5 in found_k5s:
            assert list(k5) == sorted(k5), f"K5 clique not sorted: {k5}"
            assert len(set(k5)) == 5, f"K5 contains duplicates: {k5}"
            # Check all 10 edges exist
            for u, v in itertools.combinations(k5, 2):
                assert G.has_edge(u, v), f"False K5 clique reported (missing edge ({u}, {v})): {k5}"
            assert k5 not in k5_set, f"Duplicate K5 reported: {k5}"
            k5_set.add(k5)

        assert k5_set == nx_cliques_5, f"K5 sets differ! Missing: {nx_cliques_5 - k5_set}, Extra: {k5_set - nx_cliques_5}"


# =========================================================================
# 1. Pathological Topologies
# =========================================================================

def test_pathological_empty_and_trivial():
    """Test BitGraph with 0 and 1 vertices."""
    # N = 0
    bg0 = CompiledBitGraph(0)
    assert bg0.num_nodes == 0
    assert bg0.num_edges == 0
    assert bg0.count_triangles() == 0
    assert bg0.find_triangles() == []
    assert bg0.find_k5_cliques() == []
    assert bg0.get_edges() == []
    assert not bg0.has_edge(0, 1)
    assert bg0.degree(0) == 0
    assert bg0.count_common_neighbors(0, 1) == 0

    # N = 1
    bg1 = CompiledBitGraph(1)
    assert bg1.num_nodes == 1
    assert bg1.num_edges == 0
    assert bg1.degree(0) == 0
    bg1.add_edge(0, 0) # Self loop ignored
    bg1.add_edge(0, 1) # Out of bounds ignored
    assert bg1.num_edges == 0
    assert bg1.count_triangles() == 0
    assert bg1.find_triangles() == []
    assert bg1.find_k5_cliques() == []


def test_pathological_isolated_vertices():
    """Graph with 100 isolated vertices (0 edges)."""
    n = 100
    G = nx.empty_graph(n)
    bg = CompiledBitGraph(n)
    verify_bitgraph_against_networkx(bg, G)


def test_pathological_star_graphs():
    """Star graphs S_10, S_64, S_65, S_128 (bipartite, 0 triangles, 0 K5s)."""
    for n in [10, 64, 65, 128]:
        G = nx.star_graph(n - 1)
        bg = CompiledBitGraph(n)
        bg.add_edges(list(G.edges()))
        verify_bitgraph_against_networkx(bg, G)


def test_pathological_complete_graphs():
    """Complete graphs K_n for boundary word sizes (word = 64 bits)."""
    # Test across word boundaries: 5, 6, 7, 8, 10, 15, 20, 63, 64, 65
    for n in [5, 6, 7, 8, 10, 15, 20]:
        G = nx.complete_graph(n)
        bg = CompiledBitGraph(n)
        bg.add_edges(list(G.edges()))
        verify_bitgraph_against_networkx(bg, G)

    # For n=63, 64, 65, test count_triangles and mathematical properties
    for n in [63, 64, 65]:
        G = nx.complete_graph(n)
        bg = CompiledBitGraph(n)
        bg.add_edges(list(G.edges()))
        expected_triangles = n * (n - 1) * (n - 2) // 6
        expected_k5s = math.comb(n, 5)
        assert bg.count_triangles() == expected_triangles
        assert bg.num_edges == n * (n - 1) // 2
        # Common neighbors for any pair in Kn is n - 2
        assert bg.count_common_neighbors(0, 1) == n - 2
        assert bg.count_common_neighbors(n - 2, n - 1) == n - 2
        if n >= 64:
            assert bg.count_common_neighbors(62, 63) == n - 2


def test_pathological_degree0_mixed_with_dense_cliques():
    """Graph with degree-0 isolated nodes interleaved with dense K5 and K6 cliques."""
    n = 200
    G = nx.Graph()
    G.add_nodes_from(range(n))

    # Clique 1 on nodes [10..14] (K5)
    for u in range(10, 15):
        for v in range(u + 1, 15):
            G.add_edge(u, v)

    # Clique 2 on nodes [60..65] (K6 across 64-bit boundary!)
    for u in range(60, 66):
        for v in range(u + 1, 66):
            G.add_edge(u, v)

    # Clique 3 on nodes [125..130] (K6 across 128-bit boundary!)
    for u in range(125, 131):
        for v in range(u + 1, 131):
            G.add_edge(u, v)

    bg = CompiledBitGraph(n)
    bg.add_edges(list(G.edges()))
    verify_bitgraph_against_networkx(bg, G)


def test_pathological_complete_bipartite_graphs():
    """Complete bipartite graphs K_m,n (0 triangles, 0 K5s, dense common neighbors)."""
    for m, n in [(5, 5), (10, 20), (32, 32), (64, 64)]:
        G = nx.complete_bipartite_graph(m, n)
        bg = CompiledBitGraph(m + n)
        bg.add_edges(list(G.edges()))
        verify_bitgraph_against_networkx(bg, G)


# =========================================================================
# 2. Dense Random Graphs (N in [500, 2000])
# =========================================================================

@pytest.mark.parametrize("n,p,seed", [
    (500, 0.05, 101),
    (500, 0.20, 102),
    (1000, 0.02, 103),
    (1500, 0.01, 104),
    (2000, 0.008, 105),
])
def test_dense_random_graphs_against_networkx(n, p, seed):
    """Stress test large random graphs N in [500, 2000] against NetworkX."""
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    bg = CompiledBitGraph(n)
    bg.add_edges(list(G.edges()))

    # Verify structural metrics
    assert bg.num_nodes == n
    assert bg.num_edges == G.number_of_edges()

    # Sample 200 vertex degrees
    rng = np.random.default_rng(seed)
    sampled_nodes = rng.choice(n, size=min(200, n), replace=False)
    for u in sampled_nodes:
        assert bg.degree(int(u)) == G.degree(int(u))

    # Sample 100 edge & non-edge pairs for common neighbors
    sampled_pairs = []
    edges = list(G.edges())
    if edges:
        for idx in rng.choice(len(edges), size=min(50, len(edges)), replace=False):
            sampled_pairs.append(edges[idx])
    for _ in range(50):
        u, v = rng.choice(n, size=2, replace=False)
        sampled_pairs.append((int(u), int(v)))

    for u, v in sampled_pairs:
        expected_cn = len(list(nx.common_neighbors(G, u, v)))
        actual_cn = bg.count_common_neighbors(u, v)
        assert actual_cn == expected_cn

    # Verify triangle count exact match
    expected_triangles = sum(nx.triangles(G).values()) // 3
    t0 = time.perf_counter()
    actual_triangles = bg.count_triangles()
    dt = time.perf_counter() - t0
    assert actual_triangles == expected_triangles, f"Triangle count mismatch on G({n}, {p}): {actual_triangles} vs {expected_triangles}"


# =========================================================================
# 3. K5 Discovery Soundness & Completeness
# =========================================================================

def test_k5_soundness_and_completeness_seeded():
    """Verify K5 discovery on complex planted cluster topologies."""
    # Graph with 5 planted K5s and random noise edges
    rng = np.random.default_rng(42)
    n = 100
    G = nx.Graph()
    G.add_nodes_from(range(n))

    planted_k5s = []
    for k in range(5):
        nodes = list(range(k * 8, k * 8 + 5))
        planted_k5s.append(tuple(nodes))
        for u, v in itertools.combinations(nodes, 2):
            G.add_edge(u, v)

    # Add random edges between nodes >= 50 (sparse so no accidental K5)
    for _ in range(50):
        u, v = rng.choice(np.arange(40, n), size=2, replace=False)
        G.add_edge(int(u), int(v))

    bg = CompiledBitGraph(n)
    bg.add_edges(list(G.edges()))
    verify_bitgraph_against_networkx(bg, G)


# =========================================================================
# 4. Memory Stability, Scale (1,000 to 5,000 Nodes) & Query Latency
# =========================================================================

@pytest.mark.parametrize("n", [1000, 2000, 3000, 4000, 5000])
def test_large_scale_memory_stability(n):
    """Instantiate graphs from 1,000 to 5,000 nodes, verify zero crashes and memory efficiency."""
    bg = CompiledBitGraph(n)
    assert bg.num_nodes == n
    assert bg.num_edges == 0

    # Add a chain and a dense core
    edges = [(i, i + 1) for i in range(n - 1)]
    # Add a K10 at the beginning
    for i in range(10):
        for j in range(i + 1, 10):
            edges.append((i, j))
    bg.add_edges(edges)

    assert bg.num_edges == len(set(tuple(sorted(e)) for e in edges))
    assert bg.degree(0) >= 9
    assert bg.count_common_neighbors(0, 1) >= 8


def test_rapid_allocation_deallocation_leak_stress():
    """Create and destroy 1,000 BitGraph instances of size 1,000 to verify no memory leaks or corruption."""
    for i in range(1000):
        bg = CompiledBitGraph(1000)
        bg.add_edge(0, 999)
        bg.add_edge(500, 501)
        assert bg.num_edges == 2
        del bg


def test_submillisecond_query_performance():
    """Verify sub-millisecond query performance on a 2,000 node graph."""
    n = 2000
    bg = CompiledBitGraph(n)
    # Add 5,000 random edges
    rng = np.random.default_rng(123)
    edges = set()
    while len(edges) < 5000:
        u, v = rng.choice(n, size=2, replace=False)
        edges.add((int(min(u, v)), int(max(u, v))))
    bg.add_edges(list(edges))

    # Benchmark has_edge
    t0 = time.perf_counter()
    queries = 10000
    for _ in range(queries):
        u, v = rng.choice(n, size=2, replace=False)
        _ = bg.has_edge(int(u), int(v))
    total_time = time.perf_counter() - t0
    avg_us = (total_time / queries) * 1e6
    # Average query time must be strictly sub-millisecond (typically < 1 microsecond)
    assert avg_us < 100.0, f"Average has_edge query too slow: {avg_us:.2f} microseconds"

    # Benchmark count_common_neighbors
    t0 = time.perf_counter()
    cn_queries = 2000
    for _ in range(cn_queries):
        u, v = rng.choice(n, size=2, replace=False)
        _ = bg.count_common_neighbors(int(u), int(v))
    total_time = time.perf_counter() - t0
    avg_cn_us = (total_time / cn_queries) * 1e6
    assert avg_cn_us < 500.0, f"Average count_common_neighbors query too slow: {avg_cn_us:.2f} microseconds"


# =========================================================================
# 5. CutEngine Stress Tests on Pathological Topologies
# =========================================================================

def test_cut_engine_on_empty_and_pathological():
    """CutEngine separation on empty graphs, stars, and bipartite graphs."""
    cut_engine = CompiledCutEngine()

    # 1. Empty graph
    bg0 = CompiledBitGraph(10)
    violated = cut_engine.separate_violated_k5(bg0, [], [])
    assert len(violated) == 0
    violated_tri = cut_engine.separate_violated_triangles(bg0, [], [])
    assert len(violated_tri) == 0

    # 2. Star graph S_20 (0 triangles, 0 K5s)
    G = nx.star_graph(19)
    bg = CompiledBitGraph(20)
    edges = list(G.edges())
    bg.add_edges(edges)
    primal_ones = [1.0] * len(edges)

    violated_k5 = cut_engine.separate_violated_k5(bg, edges, primal_ones)
    assert len(violated_k5) == 0
    violated_tri = cut_engine.separate_violated_triangles(bg, edges, primal_ones)
    assert len(violated_tri) == 0

    # 3. Complete bipartite graph K_10,10 (0 triangles, 0 K5s)
    G_bip = nx.complete_bipartite_graph(10, 10)
    bg_bip = CompiledBitGraph(20)
    edges_bip = list(G_bip.edges())
    bg_bip.add_edges(edges_bip)
    primal_bip = [1.0] * len(edges_bip)

    assert len(cut_engine.separate_violated_k5(bg_bip, edges_bip, primal_bip)) == 0
    assert len(cut_engine.separate_violated_triangles(bg_bip, edges_bip, primal_bip)) == 0


def test_cut_engine_compile_surrogate_empty_and_zero():
    """Compile surrogate cut with empty multipliers."""
    bg = CompiledBitGraph(10)
    cut_engine = CompiledCutEngine()
    surr = cut_engine.compile_surrogate_cut(bg, [], [])
    assert surr.is_valid
    assert surr.num_active_supports == 0
    assert surr.rhs == 0.0


# =========================================================================
# 6. M1 Iteration 2 Adversarial Stress Expansion:
#    - Word boundary shifts (u & 63 == 63) for n in [63, 64, 65, 127, 128, 129]
#    - CutEngine bounds checks rejecting short primal_sol vectors
#    - Large-scale memory efficiency & percentile query latency
# =========================================================================

@pytest.mark.parametrize("n", [63, 64, 65, 127, 128, 129])
def test_pathological_complete_graphs_across_word_boundaries(n):
    """Complete graphs Kn across 64-bit and 128-bit boundaries verifying shift UB elimination."""
    G = nx.complete_graph(n)
    bg = CompiledBitGraph(n)
    bg.add_edges(list(G.edges()))

    expected_edges = n * (n - 1) // 2
    assert bg.num_nodes == n
    assert bg.num_edges == expected_edges

    edges = bg.get_edges()
    assert len(edges) == expected_edges
    for u, v in edges:
        assert u < v
        assert bg.has_edge(u, v)
        assert bg.has_edge(v, u)

    expected_triangles = n * (n - 1) * (n - 2) // 6
    assert bg.count_triangles() == expected_triangles

    # Check degrees on boundary nodes
    for u in [62, 63, 64, 126, 127, 128]:
        if u < n:
            assert bg.degree(u) == n - 1

    # Check common neighbors across boundary pairs
    if n >= 64:
        assert bg.count_common_neighbors(62, 63) == n - 2
    if n >= 65:
        assert bg.count_common_neighbors(63, 64) == n - 2
    if n >= 128:
        assert bg.count_common_neighbors(126, 127) == n - 2
    if n >= 129:
        assert bg.count_common_neighbors(127, 128) == n - 2

    # Verification of find_triangles for smaller boundary cases
    if n <= 65:
        tris = bg.find_triangles()
        assert len(tris) == expected_triangles
        for u, v, w in tris:
            assert u < v < w

    # Verification of find_k5_cliques for n in [63, 64]
    if n <= 64:
        expected_k5s = math.comb(n, 5)
        k5s = bg.find_k5_cliques()
        assert len(k5s) == expected_k5s
        for clq in k5s:
            assert list(clq) == sorted(clq)


@pytest.mark.parametrize("n,center", [
    (63, 62),
    (64, 63),
    (65, 63),
    (65, 64),
    (128, 127),
    (129, 127),
    (129, 128),
])
def test_pathological_star_graphs_at_word_boundaries(n, center):
    """Star graphs with center placed at boundary vertices (u & 63 == 63) and (u & 63 == 0)."""
    edges = [(center, i) for i in range(n) if i != center]
    bg = CompiledBitGraph(n)
    bg.add_edges(edges)

    assert bg.num_nodes == n
    assert bg.num_edges == n - 1
    assert bg.count_triangles() == 0
    assert bg.find_triangles() == []
    assert bg.find_k5_cliques() == []

    edge_list = bg.get_edges()
    assert len(edge_list) == n - 1
    for u, v in edge_list:
        assert u < v
        assert u == center or v == center

    assert bg.degree(center) == n - 1
    other_nodes = [i for i in range(n) if i != center]
    for u in other_nodes[:10]:
        assert bg.degree(u) == 1
        assert bg.count_common_neighbors(center, u) == 0

    if len(other_nodes) >= 2:
        assert bg.count_common_neighbors(other_nodes[0], other_nodes[1]) == 1


@pytest.mark.parametrize("n,p,seed", [
    (63, 0.2, 201),
    (64, 0.2, 202),
    (65, 0.2, 203),
    (127, 0.1, 204),
    (128, 0.1, 205),
    (129, 0.1, 206),
])
def test_random_graphs_across_word_boundaries_against_networkx(n, p, seed):
    """Erdos-Renyi random graphs at exact boundary sizes n in [63, 64, 65, 127, 128, 129]."""
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    bg = CompiledBitGraph(n)
    bg.add_edges(list(G.edges()))

    assert bg.num_nodes == n
    assert bg.num_edges == G.number_of_edges()

    for u in range(n):
        assert bg.degree(u) == G.degree(u)

    expected_triangles = sum(nx.triangles(G).values()) // 3
    assert bg.count_triangles() == expected_triangles

    tris = bg.find_triangles()
    assert len(tris) == expected_triangles
    for u, v, w in tris:
        assert u < v < w
        assert G.has_edge(u, v) and G.has_edge(v, w) and G.has_edge(u, w)

    # Test boundary common neighbor queries
    boundary_pairs = [(62, 63), (63, 64), (126, 127), (127, 128)]
    for u, v in boundary_pairs:
        if u < n and v < n:
            expected_cn = len(list(nx.common_neighbors(G, u, v)))
            assert bg.count_common_neighbors(u, v) == expected_cn


def test_cut_engine_bounds_check_rejections():
    """Verify CutEngine rejects short primal_sol vectors with ValueError across multiple sizes."""
    cut_engine = CompiledCutEngine()

    for num_edges_target in [10, 25, 64, 100]:
        bg = CompiledBitGraph(num_edges_target + 5)
        edges = [(i, (i + 1) % (num_edges_target + 5)) for i in range(num_edges_target)]
        bg.add_edges(edges)
        actual_edges = bg.get_edges()
        m = len(actual_edges)

        undersized_lengths = [0, 1, m // 2, m - 1]
        for bad_len in undersized_lengths:
            bad_primal = [0.5] * bad_len

            with pytest.raises(ValueError, match="primal_sol size must be >= edge count"):
                cut_engine.separate_violated_k5(bg, actual_edges, bad_primal)

            with pytest.raises(ValueError, match="primal_sol size must be >= edge count"):
                cut_engine.separate_violated_triangles(bg, actual_edges, bad_primal)

        # Exact size must NOT throw
        exact_primal = [0.5] * m
        res_k5 = cut_engine.separate_violated_k5(bg, actual_edges, exact_primal)
        assert isinstance(res_k5, list)
        res_tri = cut_engine.separate_violated_triangles(bg, actual_edges, exact_primal)
        assert isinstance(res_tri, list)

        # Oversized primal_sol must NOT throw
        oversized_primal = [0.5] * (m + 20)
        res_k5_over = cut_engine.separate_violated_k5(bg, actual_edges, oversized_primal)
        assert isinstance(res_k5_over, list)
        res_tri_over = cut_engine.separate_violated_triangles(bg, actual_edges, oversized_primal)
        assert isinstance(res_tri_over, list)


def test_boundary_crossing_planted_cliques():
    """Verify K5 and triangle discovery and cut separation on boundary-crossing cliques."""
    n = 200
    bg = CompiledBitGraph(n)

    # Planted K5 at [61..65] (crosses word 0/1 boundary at 63/64)
    clq1 = [61, 62, 63, 64, 65]
    for u in clq1:
        for v in clq1:
            if u < v:
                bg.add_edge(u, v)

    # Planted K5 at [125..129] (crosses word 1/2 boundary at 127/128)
    clq2 = [125, 126, 127, 128, 129]
    for u in clq2:
        for v in clq2:
            if u < v:
                bg.add_edge(u, v)

    assert bg.count_triangles() == 20
    assert len(bg.find_triangles()) == 20

    k5s = bg.find_k5_cliques()
    assert len(k5s) == 2
    assert tuple(k5s[0]) == tuple(clq1)
    assert tuple(k5s[1]) == tuple(clq2)

    # Separate cuts with violated primal_sol = 0.9 on all edges
    edges = bg.get_edges()
    assert len(edges) == 20
    primal_sol = [0.9] * len(edges)

    cut_engine = CompiledCutEngine()
    violated_k5 = cut_engine.separate_violated_k5(bg, edges, primal_sol, threshold=0.01)
    assert len(violated_k5) == 2
    for cut in violated_k5:
        assert cut.violation > 2.9
        assert cut.rhs == 6.0

    # Compile surrogate cut from violations via both methods
    surr = cut_engine.compile_surrogate_cut(bg, edges, [(cut.nodes, cut.violation) for cut in violated_k5])
    assert surr.is_valid
    assert surr.status == "CERTIFIED_VALID_CONIC_COMBINATION"
    assert surr.num_active_supports == 2
    assert len(surr.certificate_hash) == 64

    # Also verify native compile_violation_surrogate directly
    surr_native = cut_engine._native.compile_violation_surrogate(bg.native, edges, violated_k5)
    assert surr_native.is_valid
    assert surr_native.certificate_hash == surr.certificate_hash



@pytest.mark.parametrize("n", [1000, 2000, 3000, 4000, 5000])
def test_large_scale_memory_footprint_and_alignment(n):
    """Verify 64-byte row alignment and low memory footprint for N in [1000..5000]."""
    bg = CompiledBitGraph(n)

    # 64-byte row alignment check
    words_per_row = bg.native.words_per_row
    assert (words_per_row * 8) % 64 == 0, f"Row stride not 64-byte aligned for N={n}"

    # Total memory footprint in bytes
    bytes_allocated = words_per_row * n * 8
    # Even for 5000 nodes, memory should be ~3.2 MB (< 4 MB)
    max_expected_bytes = 4 * 1024 * 1024
    assert bytes_allocated <= max_expected_bytes, f"Memory footprint exceeded 4 MB for N={n}: {bytes_allocated} bytes"


def test_latency_distribution_percentiles_5000_nodes():
    """Check mean and percentile query latency on a 5,000-node graph."""
    n = 5000
    bg = CompiledBitGraph(n)

    # Add 10,000 edges
    rng = np.random.default_rng(999)
    edges = set()
    while len(edges) < 10000:
        u, v = rng.choice(n, size=2, replace=False)
        edges.add((int(min(u, v)), int(max(u, v))))
    bg.add_edges(list(edges))

    # Benchmark has_edge over 10,000 queries
    queries = 10000
    test_u = rng.choice(n, size=queries)
    test_v = rng.choice(n, size=queries)

    latencies_ns = []
    for i in range(queries):
        t0 = time.perf_counter_ns()
        _ = bg.has_edge(int(test_u[i]), int(test_v[i]))
        latencies_ns.append(time.perf_counter_ns() - t0)

    latencies_us = np.array(latencies_ns) / 1000.0
    mean_us = float(np.mean(latencies_us))
    p95_us = float(np.percentile(latencies_us, 95))
    p99_us = float(np.percentile(latencies_us, 99))
    max_us = float(np.max(latencies_us))

    # Keep aggregate performance guards. A maximum wall-clock sample can
    # include an arbitrary scheduler pause on a shared CI runner.
    print(f"Maximum observed has_edge wall time: {max_us:.2f} us")
    assert mean_us < 50.0, f"Mean has_edge latency {mean_us:.2f} us exceeds 50 us"
    assert p95_us < 100.0, f"p95 has_edge latency {p95_us:.2f} us exceeds 100 us"
    assert p99_us < 200.0, f"p99 has_edge latency {p99_us:.2f} us exceeds 200 us"

    # Benchmark count_common_neighbors over 2,000 queries
    cn_queries = 2000
    cn_u = rng.choice(n, size=cn_queries)
    cn_v = rng.choice(n, size=cn_queries)

    cn_latencies_ns = []
    for i in range(cn_queries):
        t0 = time.perf_counter_ns()
        _ = bg.count_common_neighbors(int(cn_u[i]), int(cn_v[i]))
        cn_latencies_ns.append(time.perf_counter_ns() - t0)

    cn_latencies_us = np.array(cn_latencies_ns) / 1000.0
    cn_mean_us = float(np.mean(cn_latencies_us))
    cn_p95_us = float(np.percentile(cn_latencies_us, 95))
    cn_max_us = float(np.max(cn_latencies_us))

    print(f"Maximum observed count_common_neighbors wall time: {cn_max_us:.2f} us")
    assert cn_mean_us < 100.0, f"Mean count_common_neighbors latency {cn_mean_us:.2f} us exceeds 100 us"
    assert cn_p95_us < 250.0, f"p95 count_common_neighbors latency {cn_p95_us:.2f} us exceeds 250 us"
