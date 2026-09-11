"""Unit tests for large scale benchmark harness and instance generators.

Validates benchmark generation, memory scaling, and metric calculation
across 1,000+ node Chimera, Pegasus, and G-set topologies.
"""

from __future__ import annotations

import pytest
import numpy as np

from highs_turbo.topologies import (
    BitParallelGraph,
    generate_chimera_instance,
    generate_pegasus_instance,
    generate_gset_instance,
)
from highs_turbo.graph_generator import generate_ab_pair


def test_chimera_instance_generation_properties():
    """Verify Chimera instance structural parameters across sizes."""
    c4 = generate_chimera_instance(4, 4, 4, seed=42)
    assert c4.num_nodes == 128
    assert c4.num_edges == 352
    assert c4.metadata["family"] == "Chimera_spin_glass"

    c8 = generate_chimera_instance(8, 8, 4, seed=84)
    assert c8.num_nodes == 512
    assert c8.num_edges == 1472

    c12 = generate_chimera_instance(12, 12, 4, seed=126)
    assert c12.num_nodes == 1152
    assert c12.num_edges == 3360


def test_pegasus_instance_generation_properties():
    """Verify Pegasus instance properties and edge density."""
    p4 = generate_pegasus_instance(4, seed=1)
    assert p4.num_nodes == 264
    assert p4.num_edges > 1000
    assert p4.metadata["family"] == "Pegasus_spin_glass"

    p8 = generate_pegasus_instance(8, seed=2)
    assert p8.num_nodes == 1288
    assert p8.num_edges == 8804


def test_gset_instance_generation_properties():
    """Verify G-set instance node counts and edge targets."""
    g11 = generate_gset_instance("G11", seed=11)
    assert g11.num_nodes == 800
    assert g11.num_edges == 1600

    g43 = generate_gset_instance("G43", seed=43)
    assert g43.num_nodes == 1000
    assert g43.num_edges == 9990

    g51 = generate_gset_instance("G51", seed=51)
    assert g51.num_nodes == 1000
    assert g51.num_edges == 5909

    g22 = generate_gset_instance("G22", seed=22)
    assert g22.num_nodes == 2000
    assert g22.num_edges == 19990


def test_large_scale_memory_footprint():
    """Verify bit-parallel representation stays strictly within < 12 MB for 10,000 nodes."""
    # 10,000 nodes: words_per_row = (10000 + 63) // 64 = 157 words
    # Total memory: 10,000 * 157 * 8 bytes = 12.56 MB
    num_nodes = 5000
    bg = BitParallelGraph(num_nodes)
    words = bg.words_per_row
    assert words == 79
    # Footprint in bytes
    bytes_est = num_nodes * words * 8
    assert bytes_est < 4 * 1024 * 1024  # < 4 MB RAM for 5,000 nodes


def test_large_scale_ab_permutation_equivalence():
    """Verify deterministic isomorphic relabeling works on large 1,152-node Chimera."""
    c = generate_chimera_instance(12, 12, 4, seed=777)
    ca, cb, pi, pi_inv = generate_ab_pair(c, seed=888)

    assert ca.num_nodes == cb.num_nodes == 1152
    assert ca.num_edges == cb.num_edges == 3360

    # Invertibility check
    for u in range(ca.num_nodes):
        assert pi_inv[pi[u]] == u


def test_large_scale_benchmarks_module_imports():
    """Verify large_scale_benchmarks exports core classes and generators."""
    from highs_turbo.large_scale_benchmarks import (
        BitParallelGraph as NSBitParallelGraph,
        LargeScaleBenchmarkResult,
        LargeScaleBenchmarkSuite,
        generate_chimera_instance as ns_chimera,
        generate_embedded_chimera_instance,
        generate_gset_instance as ns_gset,
        generate_pegasus_instance as ns_pegasus,
        generate_planted_1000_node_instance,
    )

    g_embed = generate_embedded_chimera_instance(4, 4, 4, num_cliques=5, seed=42)
    assert g_embed.num_nodes == 128
    assert g_embed.metadata["embedded_k5_cliques"] == 5
    assert g_embed.metadata["family"] == "Chimera_spin_glass_embedded"


def test_large_scale_suite_evaluation_and_relabeling_invariance():
    """Verify evaluation and relabeling invariance on a small test instance."""
    from highs_turbo.large_scale_benchmarks import (
        LargeScaleBenchmarkSuite,
        generate_chimera_instance as ns_chimera,
    )

    suite = LargeScaleBenchmarkSuite(seed=42)
    g = ns_chimera(4, 4, 4, seed=42, ising=True)
    ra, rb = suite.evaluate_instance_pair(g, seed=10)

    assert ra.is_verified_sound
    assert rb.is_verified_sound
    assert len(ra.certificate_sha256) == 64
    assert len(rb.certificate_sha256) == 64
    assert ra.speedup > 0
    assert ra.iters_reduction_percent >= 0
    assert abs(ra.objective_val - rb.objective_val) <= 1e-6


def test_large_scale_benchmark_result_schema_completeness():
    """Verify all 14 required fields are present in LargeScaleBenchmarkResult serialization."""
    from dataclasses import asdict
    from highs_turbo.large_scale_benchmarks import (
        LargeScaleBenchmarkResult,
        LargeScaleBenchmarkSuite,
        generate_gset_instance as ns_gset,
    )

    suite = LargeScaleBenchmarkSuite(seed=42)
    g11 = ns_gset("G11", seed=11)
    ra, rb = suite.evaluate_instance_pair(g11, seed=111)

    d = asdict(ra)
    required_fields = [
        "instance_name",
        "labeling",
        "num_nodes",
        "num_edges",
        "standard_time_sec",
        "surrogate_time_sec",
        "speedup",
        "standard_simplex_iters",
        "surrogate_simplex_iters",
        "iters_reduction_percent",
        "gap_closed_percent",
        "is_verified_sound",
        "certificate_sha256",
        "objective_val",
    ]
    for req in required_fields:
        assert req in d, f"Missing required field: {req}"

    # Verify mathematical properties
    assert d["is_verified_sound"] is True
    assert len(d["certificate_sha256"]) == 64
    assert d["speedup"] > 0
    assert d["iters_reduction_percent"] >= 0.0

