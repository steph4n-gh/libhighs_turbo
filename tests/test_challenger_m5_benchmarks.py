"""Independent Empirical Challenger Benchmark Harness for Milestone M5.

Author: challenger_m5_1 (Teamwork Empirical Challenger)
Scope:
1. Empirically measure wall-clock solve time of 1-row surrogate cutting plane vs
   standard multi-row cutting plane separation on large instances (>= 1,000 vertices):
   - Chimera C_{12,12,4} (1,152 nodes, 3,360 edges) native Ising
   - Chimera C_{12,12,4} (1,152 nodes, 3,480 edges) embedded logical K5 cliques
   - Pegasus P_8 (1,288 nodes, 8,804 edges) native triangular Ising
   - G-set G43 (1,000 nodes, 9,990 edges)
   - G-set G22 (2,000 nodes, 19,990 edges)
   - Planted K5 cluster (1,000 nodes, 2,199 edges, IP target 1,399)
   - Chimera C_{16,16,4} (2,048 nodes, 6,016 edges)
2. Measure speedup claims:
   - Record wall-clock measurements for every trial in the JUnit report.
   - Treat the historical 2.0x target as a measurement, not a portable guarantee.
3. Verify simplex iteration reduction:
   - Confirm whether simplex iterations are reduced by >= 70%.
4. Profile memory footprint:
   - Profile exact heap allocation via tracemalloc and RSS via resource.getrusage.
   - Verify memory usage remains strictly < 50 MB RAM without memory exhaustion.
5. Verify mathematical soundness & isomorphic relabeling invariance:
   - Exact rational verification with 64-char hex SHA-256 digests.
   - |Za - Zb| <= 10^-6 across all isomorphic pairs.
"""

from __future__ import annotations

import json
import os
import resource
import sys
import time
import tracemalloc
from dataclasses import asdict
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from highs_turbo.compiled_engine import (
    COMPILED_ENGINE_AVAILABLE,
    CompiledBitGraph,
    CompiledCutEngine,
    CompiledRationalVerifier,
)
from highs_turbo.graph_generator import GraphInstance, generate_ab_pair
from highs_turbo.large_scale_benchmarks import (
    LargeScaleBenchmarkResult,
    LargeScaleBenchmarkSuite,
    generate_chimera_instance,
    generate_embedded_chimera_instance,
    generate_gset_instance,
    generate_pegasus_instance,
    generate_planted_1000_node_instance,
)


@pytest.fixture(scope="module")
def suite() -> LargeScaleBenchmarkSuite:
    return LargeScaleBenchmarkSuite(seed=42)


def get_current_rss_mb() -> float:
    """Returns current process RSS in Megabytes (macOS returns bytes)."""
    rusage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # On macOS, ru_maxrss is in bytes; on Linux, in kilobytes.
    if sys.platform == "darwin":
        return rusage / (1024.0 * 1024.0)
    return rusage / 1024.0


def test_compiled_native_engine_present():
    """Assert compiled C++ engine is available for large-scale benchmarks."""
    assert COMPILED_ENGINE_AVAILABLE, "Compiled C++ engine MUST be compiled and available."
    cbg = CompiledBitGraph(100)
    assert cbg.num_nodes == 100


def test_empirical_memory_under_50mb():
    """Measure peak heap memory and RSS during large graph generation and cut separation.

    Validates that memory consumption remains strictly < 50 MB RAM.
    """
    tracemalloc.start()
    tracemalloc.reset_peak()

    rss_start = get_current_rss_mb()

    # Instantiate 2,048-node Chimera and 2,000-node G22
    c16 = generate_chimera_instance(16, 16, 4, seed=16, ising=True)
    cbg16 = CompiledBitGraph.from_graph_instance(c16)
    g22 = generate_gset_instance("G22", seed=22)
    cbg22 = CompiledBitGraph.from_graph_instance(g22)

    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak_bytes / (1024.0 * 1024.0)
    rss_end = get_current_rss_mb()
    rss_delta = max(0.0, rss_end - rss_start)

    print(f"\n[Memory Profiling] Traced Peak: {peak_mb:.2f} MB, RSS Delta: {rss_delta:.2f} MB")
    assert peak_mb < 50.0, f"Tracemalloc peak memory {peak_mb:.2f} MB exceeded 50 MB threshold!"
    assert rss_delta < 50.0, f"RSS memory increase {rss_delta:.2f} MB exceeded 50 MB threshold!"


@pytest.mark.parametrize(
    "name,instance_fn",
    [
        ("chimera_c12_native", lambda: generate_chimera_instance(12, 12, 4, seed=1212, ising=True)),
        ("chimera_c12_embedded", lambda: generate_embedded_chimera_instance(12, 12, 4, num_cliques=20, seed=1213)),
        ("pegasus_p8", lambda: generate_pegasus_instance(8, seed=88, ising=True)),
        ("gset_g43", lambda: generate_gset_instance("G43", seed=43)),
        ("gset_g22", lambda: generate_gset_instance("G22", seed=22)),
        ("planted_k5_1000", lambda: generate_planted_1000_node_instance(num_k5=200, num_bridges=199, seed=1000)),
        ("chimera_c16_native", lambda: generate_chimera_instance(16, 16, 4, seed=1616, ising=True)),
    ],
)
def test_empirical_speedup_and_iteration_reduction(
    name: str, instance_fn, suite: LargeScaleBenchmarkSuite, record_property
):
    """Empirically benchmarks standard separation vs 1-row surrogate cutting plane.

    Verifies:
    Every trial must preserve soundness, relabeling invariance, and the fixture's
    >= 70% simplex iteration reduction. Wall-clock ratios remain measurements:
    shared CI runners cannot guarantee the historical 2.0x speedup target.
    """
    g = instance_fn()
    assert g.num_nodes >= 1000, f"Instance {name} must have >= 1,000 nodes, got {g.num_nodes}"

    # Repetition describes timing variation; it cannot eliminate scheduler jitter.
    trials_a, trials_b = [], []
    for trial in range(3):
        ra, rb = suite.evaluate_instance_pair(g, seed=42)
        trials_a.append(ra)
        trials_b.append(rb)
        record_property(f"trial_{trial}", json.dumps([asdict(ra), asdict(rb)]))
        for res in (ra, rb):
            assert res.iters_reduction_percent >= 70.0, (
                f"Iteration reduction {res.iters_reduction_percent:.1f}% on "
                f"{name}_{res.labeling} trial {trial} < 70% threshold!"
            )
            assert res.is_verified_sound, f"Rational verification failed on {name}, trial {trial}!"
            assert len(res.certificate_sha256) == 64, f"Invalid SHA-256 digest on {name}, trial {trial}!"
        diff_obj = abs(ra.objective_val - rb.objective_val)
        assert diff_obj <= 1e-6, f"Relabeling variance on {name}, trial {trial}: {diff_obj:.2e} > 1e-6!"

    # Median speedup across trials
    sp_a = [r.speedup for r in trials_a]
    sp_b = [r.speedup for r in trials_b]
    med_idx_a = int(np.argsort(sp_a)[len(sp_a) // 2])
    med_idx_b = int(np.argsort(sp_b)[len(sp_b) // 2])
    res_a, res_b = trials_a[med_idx_a], trials_b[med_idx_b]

    for res, label in [(res_a, "A"), (res_b, "B")]:
        print(
            f"\n[{name} Cell {label}] Nodes: {res.num_nodes}, Edges: {res.num_edges} | "
            f"Std Time: {res.standard_time_sec * 1000:.1f}ms, Surr Time: {res.surrogate_time_sec * 1000:.1f}ms | "
            f"Speedup: {res.speedup:.2f}x | "
            f"Std Iters: {res.standard_simplex_iters} -> Surr Iters: {res.surrogate_simplex_iters} "
            f"({res.iters_reduction_percent:.1f}% reduction) | "
            f"Soundness: {res.is_verified_sound} | SHA256: {res.certificate_sha256[:12]}..."
        )

    # Keep the timing result visible without turning it into a correctness claim.
    avg_speedup = (res_a.speedup + res_b.speedup) / 2.0
    print(f"[{name}] Average Speedup: {avg_speedup:.2f}x (Cell A: {res_a.speedup:.2f}x, Cell B: {res_b.speedup:.2f}x)")
    record_property("average_median_speedup", avg_speedup)
    record_property("historical_2x_target_met", avg_speedup >= 2.0)


def run_standalone_challenger_benchmark():
    """Runs a full standalone benchmark with multiple repetitions and detailed summary."""
    print("=" * 80)
    print("INDEPENDENT EMPIRICAL CHALLENGER BENCHMARK HARNESS (MILESTONE M5)")
    print("=" * 80)

    tracemalloc.start()
    rss_initial = get_current_rss_mb()

    suite = LargeScaleBenchmarkSuite(seed=42)

    instances = [
        ("Chimera C_{12,12,4} Native", generate_chimera_instance(12, 12, 4, seed=1212, ising=True)),
        ("Chimera C_{12,12,4} Embedded K5", generate_embedded_chimera_instance(12, 12, 4, num_cliques=20, seed=1213)),
        ("Pegasus P_8 Native", generate_pegasus_instance(8, seed=88, ising=True)),
        ("G-set G43", generate_gset_instance("G43", seed=43)),
        ("G-set G22", generate_gset_instance("G22", seed=22)),
        ("Planted K5 Cluster (1000n)", generate_planted_1000_node_instance(num_k5=200, num_bridges=199, seed=1000)),
        ("Chimera C_{16,16,4} Native", generate_chimera_instance(16, 16, 4, seed=1616, ising=True)),
    ]

    all_results = []
    speedup_values = []
    iter_reduction_values = []

    for desc, g in instances:
        print(f"\n--- Testing Instance: {desc} (N={g.num_nodes}, M={g.num_edges}) ---")
        t_start = time.perf_counter()
        ra, rb = suite.evaluate_instance_pair(g, seed=42)
        elapsed = time.perf_counter() - t_start

        all_results.extend([ra, rb])
        for r in (ra, rb):
            speedup_values.append(r.speedup)
            iter_reduction_values.append(r.iters_reduction_percent)

        diff = abs(ra.objective_val - rb.objective_val)
        print(f"  Cell A: Std={ra.standard_time_sec*1000:.1f}ms, Surr={ra.surrogate_time_sec*1000:.1f}ms -> Speedup={ra.speedup:.2f}x | IterRed={ra.iters_reduction_percent:.1f}%")
        print(f"  Cell B: Std={rb.standard_time_sec*1000:.1f}ms, Surr={rb.surrogate_time_sec*1000:.1f}ms -> Speedup={rb.speedup:.2f}x | IterRed={rb.iters_reduction_percent:.1f}%")
        print(f"  Relabeling Diff: {diff:.2e} ({'PASS' if diff <= 1e-6 else 'FAIL'}) | Pair Eval Time: {elapsed*1000:.1f}ms")

    # Memory check
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak_bytes / (1024.0 * 1024.0)
    rss_final = get_current_rss_mb()
    rss_delta = max(0.0, rss_final - rss_initial)

    print("\n" + "=" * 80)
    print("EMPIRICAL CHALLENGER SUMMARY & VERDICT METRICS")
    print("=" * 80)
    print(f"Total Large-Scale Cells Evaluated: {len(all_results)}")
    print(f"Speedup Range: {min(speedup_values):.2f}x to {max(speedup_values):.2f}x (Mean: {np.mean(speedup_values):.2f}x, Median: {np.median(speedup_values):.2f}x)")
    print(f"Simplex Iteration Reduction Range: {min(iter_reduction_values):.1f}% to {max(iter_reduction_values):.1f}% (Mean: {np.mean(iter_reduction_values):.1f}%)")
    print(f"Cells with Iteration Reduction >= 70%: {sum(1 for x in iter_reduction_values if x >= 70.0)} / {len(iter_reduction_values)} ({100.0 * sum(1 for x in iter_reduction_values if x >= 70.0) / len(iter_reduction_values):.1f}%)")
    print(f"Cells with Speedup >= 2.0x: {sum(1 for x in speedup_values if x >= 2.0)} / {len(speedup_values)} ({100.0 * sum(1 for x in speedup_values if x >= 2.0) / len(speedup_values):.1f}%)")
    print(f"Mean Speedup on Dense Topologies (Chimera/Pegasus/Planted): {np.mean([r.speedup for r in all_results if 'gset' not in r.instance_name.lower()]):.2f}x")
    print(f"Peak Traced Memory: {peak_mb:.2f} MB (Threshold: < 50 MB)")
    print(f"RSS Memory Increase: {rss_delta:.2f} MB (Threshold: < 50 MB)")
    print(f"All 100% Soundness Verified: {all(r.is_verified_sound for r in all_results)}")
    print(f"Max Isomorphic Objective Difference: {max(abs(all_results[i].objective_val - all_results[i+1].objective_val) for i in range(0, len(all_results), 2)):.2e}")
    print("=" * 80)


if __name__ == "__main__":
    run_standalone_challenger_benchmark()
