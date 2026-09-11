"""Automated tests for graph generator module."""

import itertools
from highs_turbo.graph_generator import (
    GraphInstance,
    generate_ab_pair,
    generate_exp87_instances,
    generate_exp91_family,
    generate_k5_cluster_graph,
    generate_k_multipartite_k5,
    generate_ks_minus_edges,
    generate_molecular_planar_benchmark,
)


def test_graph_instance_basics():
    edges = [(0, 1), (1, 2), (0, 2), (1, 0)]  # duplicate & unsorted
    g = GraphInstance(name="k3", num_nodes=3, edges=edges)
    assert g.num_edges == 3
    assert g.edges == [(0, 1), (0, 2), (1, 2)]
    tris = g.find_triangles()
    assert len(tris) == 1
    assert tris[0] == (0, 1, 2)


def test_ab_pair_isomorphism():
    g = generate_k5_cluster_graph(num_k5=3, num_bridges=2, seed=42)
    ga, gb, pi, pi_inv = generate_ab_pair(g, seed=123)

    assert ga.num_nodes == gb.num_nodes
    assert ga.num_edges == gb.num_edges
    assert len(ga.find_all_k5_cliques()) == len(gb.find_all_k5_cliques())
    assert len(ga.find_triangles()) == len(gb.find_triangles())

    # Verify bijection
    for u in range(ga.num_nodes):
        assert pi_inv[pi[u]] == u


def test_exp87_instances():
    instances = generate_exp87_instances()
    assert len(instances) == 4
    expected_targets = {
        "exp87_g1_target57": 57,
        "exp87_g2_target89": 89,
        "exp87_g3_target88": 88,
        "exp87_g4_target100": 100,
    }
    for name, target in expected_targets.items():
        assert name in instances
        g = instances[name]
        assert g.metadata.get("target_integer_maxcut") == target
        assert len(g.find_all_k5_cliques()) == g.metadata.get("num_k5")


def test_exp91_family():
    family = generate_exp91_family(num_graphs=6, seed=91)
    assert len(family) == 6
    for name, g in family.items():
        k5s = g.find_all_k5_cliques()
        assert len(k5s) >= 6


def test_exp73_multipartite():
    g = generate_k_multipartite_k5(r=2, seed=42)
    assert g.num_nodes == 10
    k5s = g.find_all_k5_cliques()
    assert len(k5s) == 32  # 2^5 = 32


def test_ks_minus_edges():
    g = generate_ks_minus_edges(s=4, removed_edges=2, seed=42)
    assert g.num_nodes == 7
    # K7 has 21 edges, minus 2 edges = 19 edges
    assert g.num_edges == 19


def test_molecular_planar():
    g = generate_molecular_planar_benchmark(kind="c60")
    assert g.num_nodes == 20
    assert g.num_edges == 30
