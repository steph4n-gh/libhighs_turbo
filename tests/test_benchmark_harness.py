"""Automated tests for benchmark harness and end-to-end integration."""

import os
import pytest
from highs_turbo.benchmark_harness import BenchmarkSuite
from highs_turbo.graph_generator import generate_k5_cluster_graph


def test_benchmark_suite_single_cell():
    suite = BenchmarkSuite(seed=42)
    g = generate_k5_cluster_graph(num_k5=2, num_bridges=1, seed=42)
    res = suite.run_single_cell(g, label_cell="A")

    assert res.num_nodes == 10
    assert res.k5_count == 2
    assert res.is_verified_sound
    assert len(res.certificate_sha256) == 64
    assert res.gap_closed_percent >= 85.0
    assert res.surrogate_obj <= res.cycle_obj


def test_benchmark_exp87_suite():
    suite = BenchmarkSuite(seed=42)
    results = suite.run_exp87_benchmark()
    # 4 instances x 2 A/B labelings = 8 cells
    assert len(results) == 8

    # All must be verified sound
    for r in results:
        assert r.is_verified_sound
        assert r.gap_closed_percent >= 85.0

    # A/B invariance check: variance across labelings = 0
    by_instance = {}
    for r in results:
        base_name = r.instance_name.replace("_label_A", "").replace("_label_B", "")
        by_instance.setdefault(base_name, {})[r.label_cell] = r

    for base, cells in by_instance.items():
        assert "A" in cells and "B" in cells
        obj_a = cells["A"].surrogate_obj
        obj_b = cells["B"].surrogate_obj
        assert abs(obj_a - obj_b) < 1e-5


def test_format_markdown_report():
    suite = BenchmarkSuite(seed=42)
    g = generate_k5_cluster_graph(num_k5=2, num_bridges=1, seed=42)
    res = suite.run_single_cell(g, label_cell="A")
    md = suite.format_markdown_report([res])
    assert "Neural-Surrogate Cutting Plane Engine Benchmark Report" in md
    assert "CERTIFIED" in md


def test_benchmark_exp91_suite():
    suite = BenchmarkSuite(seed=42)
    results = suite.run_exp91_benchmark(num_graphs=2)
    assert len(results) == 4

    for r in results:
        assert r.is_verified_sound
        assert r.gap_closed_percent >= 85.0
        assert len(r.certificate_sha256) == 64

    # A/B invariance check
    by_instance = {}
    for r in results:
        base_name = r.instance_name.replace("_label_A", "").replace("_label_B", "")
        by_instance.setdefault(base_name, {})[r.label_cell] = r

    for base, cells in by_instance.items():
        assert "A" in cells and "B" in cells
        obj_a = cells["A"].surrogate_obj
        obj_b = cells["B"].surrogate_obj
        assert abs(obj_a - obj_b) < 1e-5
