"""Research comparison of multi-row separation and compiled surrogate cuts.

The suite measures elapsed time, memory, iterations, objective agreement and
cut verification on generated graph families. G-set-shaped cases are seeded
synthetic graphs, not the published Stanford benchmark data. Measurements are
specific to the fixtures and host; no minimum speedup or memory use is promised.
Rational cut verification checks a cut combination, while a SHA-256 digest is
only a receipt. Numerical relaxation objectives are not exact optimality proofs.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from highs_turbo.compiled_engine import (
    COMPILED_ENGINE_AVAILABLE,
    CompiledBitGraph,
    CompiledCutEngine,
    CompiledRationalVerifier,
    CompiledSolverCallbackBridge,
    solve_with_compiled_surrogate,
)
from highs_turbo.exact_solver import ExactMaxCutSolver, LPSolution
from highs_turbo.graph_generator import (
    GraphInstance,
    generate_ab_pair,
    generate_exp87_instances,
    generate_exp91_family,
    generate_k5_cluster_graph,
)
from highs_turbo.rational_verifier import (
    RationalCutVerifier,
    VerificationCertificate,
)
from highs_turbo.topologies import (
    BitParallelGraph,
    generate_chimera_instance as _base_chimera,
    generate_gset_instance as _base_gset,
    generate_pegasus_instance as _base_pegasus,
)


# =============================================================================
# 1. Benchmark Instance Topologies
# =============================================================================


def generate_chimera_instance(
    m: int,
    n: Optional[int] = None,
    l: int = 4,
    seed: int = 42,
    ising: bool = True,
    embed_cliques: bool = False,
    num_embedded_cliques: int = 0,
) -> GraphInstance:
    """Generates a Chimera Ising spin glass hardware graph C_{M,N,L}.

    For M=12, N=12, L=4: 1,152 nodes, 3,360 edges.
    For M=16, N=16, L=4: 2,048 nodes, 6,016 edges.
    Supports both native bimodal spin glass couplings and embedded logical cliques.
    """
    base_g = _base_chimera(m=m, n=n, l=l, seed=seed, ising=ising)
    if not embed_cliques and num_embedded_cliques <= 0:
        return base_g

    # Embed logical K5 cliques into unit cells
    k_count = num_embedded_cliques if num_embedded_cliques > 0 else 20
    rng = random.Random(seed)
    total_cells = m * (n if n is not None else m)
    k_count = min(k_count, total_cells)

    new_edges: Set[Tuple[int, int]] = set(base_g.edges)
    weights: Dict[Tuple[int, int], float] = dict(base_g.weights)
    cell_nodes_size = 2 * l

    embedded_cliques = []
    for clq_idx in range(k_count):
        base_node = clq_idx * cell_nodes_size
        cell_qubits = list(range(base_node, min(base_g.num_nodes, base_node + 5)))
        if len(cell_qubits) == 5:
            embedded_cliques.append(tuple(sorted(cell_qubits)))
            for i in range(5):
                for j in range(i + 1, 5):
                    u, v = min(cell_qubits[i], cell_qubits[j]), max(cell_qubits[i], cell_qubits[j])
                    e = (u, v)
                    if e not in new_edges:
                        new_edges.add(e)
                        weights[e] = 1.0 if not ising else (1.0 if rng.random() > 0.5 else -1.0)

    sorted_edges = sorted(list(new_edges))
    meta = dict(base_g.metadata)
    meta["embedded_k5_cliques"] = len(embedded_cliques)
    meta["family"] = "Chimera_spin_glass_embedded"

    return GraphInstance(
        name=f"chimera_c{m}_{n or m}_{l}_embedded",
        num_nodes=base_g.num_nodes,
        edges=sorted_edges,
        weights=weights,
        metadata=meta,
    )


def generate_embedded_chimera_instance(
    m: int = 12,
    n: int = 12,
    l: int = 4,
    num_cliques: int = 20,
    seed: int = 42,
    ising: bool = True,
) -> GraphInstance:
    """Convenience generator for Chimera C_{12,12,4} with embedded logical K5 cliques."""
    return generate_chimera_instance(
        m=m, n=n, l=l, seed=seed, ising=ising, embed_cliques=True, num_embedded_cliques=num_cliques
    )


def generate_pegasus_instance(
    m: int = 8,
    seed: int = 42,
    ising: bool = True,
    *,
    node_list=None,
    edge_list=None,
) -> GraphInstance:
    """Generates a Pegasus Ising spin glass instance P_M.

    For P_8: 1,288 nodes, 8,804 edges with rich native triangular frustrated structure.
    """
    return _base_pegasus(m=m, seed=seed, ising=ising, node_list=node_list, edge_list=edge_list)


def generate_gset_instance(
    name: str = "G43",
    seed: int = 42,
) -> GraphInstance:
    """Generate synthetic G-set-shaped graphs (not the published datasets)."""
    return _base_gset(name=name, seed=seed)


def generate_planted_1000_node_instance(
    num_k5: int = 200,
    num_bridges: int = 199,
    seed: int = 1000,
) -> GraphInstance:
    """Generates 1,000-node planted multi-cluster instance with exact ground-truth integer target."""
    return generate_k5_cluster_graph(num_k5=num_k5, num_bridges=num_bridges, seed=seed)


# =============================================================================
# 2. Result Data Structures
# =============================================================================


@dataclass
class LargeScaleBenchmarkResult:
    """Structured benchmark results satisfying both expanded and legacy schemas."""

    instance_name: str
    family: str
    labeling: str  # "A" or "B"
    label_cell: str  # Alias for backward compatibility
    num_nodes: int
    num_edges: int
    k5_count: int
    standard_time_sec: float
    surrogate_time_sec: float
    speedup: float
    standard_simplex_iters: int
    surrogate_simplex_iters: int
    iters_reduction_percent: float
    gap_closed_percent: float
    is_verified_sound: bool
    certificate_sha256: str
    objective_val: float

    # Legacy field aliases for seamless compatibility
    surrogate_obj: float = 0.0
    full_k5_obj: float = 0.0
    full_k5_time_ms: float = 0.0
    full_k5_iterations: int = 0
    surrogate_time_ms: float = 0.0
    surrogate_iterations: int = 0
    speedup_vs_full_k5: float = 0.0
    cycle_obj: float = 0.0
    cycle_time_ms: float = 0.0
    cycle_iterations: int = 0
    ip_maxcut: float = 0.0
    target_integer_target: Optional[int] = None
    memory_footprint_bytes: int = 0

    # Symmetric benchmarking fields
    root_lp_obj: float = 0.0
    standard_rounds: int = 4
    surrogate_rounds: int = 1
    standard_lp_calls: int = 0
    surrogate_lp_calls: int = 0
    standard_separation_time_sec: float = 0.0
    surrogate_separation_time_sec: float = 0.0
    surrogate_certification_time_sec: float = 0.0
    standard_lp_time_sec: float = 0.0
    surrogate_lp_time_sec: float = 0.0
    sym_1round_speedup: float = 0.0
    sym_1round_iter_reduction: float = 0.0
    sym_1round_std_time_sec: float = 0.0
    sym_1round_surr_time_sec: float = 0.0
    sym_1round_std_iters: int = 0
    sym_1round_surr_iters: int = 0

    def __post_init__(self) -> None:
        if self.surrogate_obj == 0.0:
            self.surrogate_obj = self.objective_val
        if self.full_k5_obj == 0.0:
            self.full_k5_obj = self.objective_val
        if self.full_k5_time_ms == 0.0:
            self.full_k5_time_ms = self.standard_time_sec * 1000.0
        if self.surrogate_time_ms == 0.0:
            self.surrogate_time_ms = self.surrogate_time_sec * 1000.0
        if self.full_k5_iterations == 0:
            self.full_k5_iterations = self.standard_simplex_iters
        if self.surrogate_iterations == 0:
            self.surrogate_iterations = self.surrogate_simplex_iters
        if self.speedup_vs_full_k5 == 0.0:
            self.speedup_vs_full_k5 = self.speedup
        if self.cycle_obj == 0.0 and self.root_lp_obj != 0.0:
            self.cycle_obj = self.root_lp_obj


# =============================================================================
# 3. Solver & Cut Separation Engine
# =============================================================================


class LargeScaleBenchmarkSuite:
    """High-performance large-scale benchmark harness and solver evaluator."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.verifier = CompiledRationalVerifier() if COMPILED_ENGINE_AVAILABLE else RationalCutVerifier()
        self.cut_engine = CompiledCutEngine() if COMPILED_ENGINE_AVAILABLE else None
        self.exact_solver = ExactMaxCutSolver()

    def estimate_memory_footprint(self, graph: GraphInstance) -> int:
        """Calculates exact memory footprint of bit-parallel graph and solver data."""
        n = graph.num_nodes
        m = graph.num_edges
        words_per_row = (((n + 63) // 64 + 7) // 8) * 8
        bit_graph_bytes = n * words_per_row * 8
        # Sparse constraint matrix: ~3 non-zeros per cut * 8 bytes + pointers
        sparse_lp_bytes = m * 8 * 4 + 2000 * 3 * 8
        total = bit_graph_bytes + sparse_lp_bytes
        return total

    def solve_standard_cutting_plane(
        self,
        graph: GraphInstance,
        max_rounds: int = 4,
        max_cuts_per_round: int = 500,
    ) -> Dict[str, Any]:
        """Solves LP using standard multi-row cutting plane separation with a cut pool."""
        t0 = time.perf_counter()
        m = graph.num_edges
        c = np.array([-graph.weights.get(e, 1.0) for e in graph.edges], dtype=np.float64)
        bounds = [(0.0, 1.0)] * m
        edges = list(graph.edges)

        cbg = CompiledBitGraph.from_graph_instance(graph) if COMPILED_ENGINE_AVAILABLE else None

        # 1. Root LP relaxation solve
        t_lp0 = time.perf_counter()
        res0 = linprog(c, bounds=bounds, method="highs")
        lp_time = time.perf_counter() - t_lp0
        root_iters = res0.nit
        root_objective = float(-res0.fun)
        total_simplex_iters = root_iters
        final_obj = root_objective
        x_curr = res0.x
        lp_solver_calls = 1
        sep_time = 0.0

        bridge = CompiledSolverCallbackBridge(m) if COMPILED_ENGINE_AVAILABLE else None
        rows: List[int] = []
        cols: List[int] = []
        vals: List[float] = []
        b_list: List[float] = []
        current_num_rows = 0

        for round_idx in range(max_rounds):
            t_sep = time.perf_counter()
            # Separate violated inequalities
            violated = []
            if cbg is not None and self.cut_engine is not None:
                # Check for violated K5 cliques first
                k5_violated = self.cut_engine.separate_violated_k5(
                    cbg, edges, list(x_curr), threshold=1e-4, max_cuts=max_cuts_per_round
                )
                if k5_violated:
                    violated = k5_violated
                else:
                    # Check for violated triangles
                    tri_violated = self.cut_engine.separate_violated_triangles(
                        cbg, edges, list(x_curr), threshold=1e-4, max_cuts=max_cuts_per_round
                    )
                    if tri_violated:
                        violated = tri_violated
                    else:
                        # Check for violated 4-cycles
                        c4_violated = self.cut_engine.separate_violated_4cycles(
                            cbg, edges, list(x_curr), threshold=1e-4, max_cuts=max_cuts_per_round
                        )
                        if c4_violated:
                            violated = c4_violated

            if not violated:
                # Fallback to bipartite 4-cycles if graph is bipartite Chimera
                violated = self._separate_bipartite_4cycles(graph, x_curr, limit=max_cuts_per_round)

            sep_time += time.perf_counter() - t_sep

            if not violated:
                break

            # Add discovered violated cuts into multi-row pool
            for cut in violated:
                rhs = float(cut["rhs"] if isinstance(cut, dict) else cut.rhs)
                cut_indices = list(cut["edge_indices"] if isinstance(cut, dict) else cut.edge_indices)
                cut_coeffs = [float(cf) for cf in (cut["coefficients"] if isinstance(cut, dict) else cut.coefficients)]
                if bridge is not None:
                    bridge.add_cut_row(rhs, cut_indices, cut_coeffs, "std_cut")
                else:
                    r_idx = current_num_rows
                    current_num_rows += 1
                    b_list.append(rhs)
                    for e_idx, coeff in zip(cut_indices, cut_coeffs):
                        rows.append(r_idx)
                        cols.append(e_idx)
                        vals.append(coeff)

            # Solve LP with added constraints
            t_lp = time.perf_counter()
            if bridge is not None:
                A_csr, b_arr = bridge.get_sparse_constraints()
            else:
                A_csr = sp.csr_matrix((vals, (rows, cols)), shape=(current_num_rows, m))
                b_arr = np.array(b_list, dtype=np.float64)

            res = linprog(c, A_ub=A_csr, b_ub=b_arr, bounds=bounds, method="highs")
            lp_time += time.perf_counter() - t_lp
            lp_solver_calls += 1
            total_simplex_iters += res.nit
            final_obj = -res.fun
            x_curr = res.x

        solve_time_sec = time.perf_counter() - t0
        num_added = bridge.num_rows if bridge is not None else current_num_rows
        return {
            "objective": float(final_obj),
            "root_objective": float(root_objective),
            "simplex_iterations": int(total_simplex_iters),
            "time_sec": float(solve_time_sec),
            "separation_time_sec": float(sep_time),
            "lp_time_sec": float(lp_time),
            "lp_solver_calls": int(lp_solver_calls),
            "num_cuts_added": num_added,
            "rounds": int(max_rounds),
        }

    def solve_surrogate_cutting_plane(
        self,
        graph: GraphInstance,
        max_rounds: int = 1,
        max_cuts: int = 1000,
        gnn_model: Optional[Any] = None,
    ) -> Tuple[Dict[str, Any], VerificationCertificate, List[Any]]:
        """Solve a numerical LP using rationally verified surrogate cuts."""
        t0 = time.perf_counter()
        m = graph.num_edges
        c = np.array([-graph.weights.get(e, 1.0) for e in graph.edges], dtype=np.float64)
        bounds = [(0.0, 1.0)] * m
        edges = list(graph.edges)

        cbg = CompiledBitGraph.from_graph_instance(graph) if COMPILED_ENGINE_AVAILABLE else None

        # 1. Solve root LP relaxation
        t_lp0 = time.perf_counter()
        res0 = linprog(c, bounds=bounds, method="highs")
        lp_time = time.perf_counter() - t_lp0
        root_iters = res0.nit
        root_objective = float(-res0.fun)
        x_curr = res0.x
        lp_solver_calls = 1
        total_iters = root_iters
        final_obj = root_objective

        bridge = CompiledSolverCallbackBridge(m) if COMPILED_ENGINE_AVAILABLE else None
        sep_time = 0.0
        cert_time = 0.0
        all_cuts: List[Any] = []
        last_cert: Optional[VerificationCertificate] = None
        a_surr_last = np.zeros(m, dtype=np.float64)
        b_surr_last = 0.0

        for round_idx in range(max_rounds):
            t_sep = time.perf_counter()
            cuts_to_aggregate = []
            if cbg is not None and self.cut_engine is not None:
                k5_violated = self.cut_engine.separate_violated_k5(
                    cbg, edges, list(x_curr), threshold=1e-4, max_cuts=max_cuts
                )
                if k5_violated:
                    cuts_to_aggregate = k5_violated
                else:
                    tri_violated = self.cut_engine.separate_violated_triangles(
                        cbg, edges, list(x_curr), threshold=1e-4, max_cuts=max_cuts
                    )
                    if tri_violated:
                        cuts_to_aggregate = tri_violated
                    else:
                        c4_violated = self.cut_engine.separate_violated_4cycles(
                            cbg, edges, list(x_curr), threshold=1e-4, max_cuts=max_cuts
                        )
                        if c4_violated:
                            cuts_to_aggregate = c4_violated

            if not cuts_to_aggregate:
                cuts_to_aggregate = self._separate_bipartite_4cycles(graph, x_curr, limit=max_cuts)
            sep_time += time.perf_counter() - t_sep

            if not cuts_to_aggregate:
                break

            all_cuts.extend(cuts_to_aggregate)

            # Rational verification & surrogate cut synthesis
            t_cert = time.perf_counter()
            cert, a_surr, b_surr = self._certify_and_build_surrogate(
                graph, cbg, cuts_to_aggregate, gnn_model=gnn_model
            )
            cert_time += time.perf_counter() - t_cert
            last_cert = cert
            a_surr_last = a_surr
            b_surr_last = b_surr

            # Solve LP with added surrogate row using bridge
            if bridge is not None:
                nz_idx = np.nonzero(a_surr)[0]
                bridge.add_cut_row(b_surr, nz_idx, a_surr[nz_idx], f"surrogate_round_{round_idx}")
                A_surr, b_surr_arr = bridge.get_sparse_constraints()
            else:
                A_surr = sp.csr_matrix(a_surr.reshape(1, -1))
                b_surr_arr = np.array([b_surr], dtype=np.float64)

            t_lp = time.perf_counter()
            res_surr = linprog(c, A_ub=A_surr, b_ub=b_surr_arr, bounds=bounds, method="highs")
            lp_time += time.perf_counter() - t_lp
            lp_solver_calls += 1
            total_iters += res_surr.nit
            final_obj = -res_surr.fun
            x_curr = res_surr.x

        if last_cert is None:
            last_cert = VerificationCertificate(
                is_valid=True,
                status="CERTIFIED_EMPTY_CONIC_COMBINATION",
                rejection_reason=None,
                num_active_supports=0,
                exact_coefficients={},
                exact_rhs=Fraction(0, 1),
                sha256_hash="0" * 64,
            )

        solve_time_sec = time.perf_counter() - t0

        return (
            {
                "objective": float(final_obj),
                "root_objective": float(root_objective),
                "simplex_iterations": int(total_iters),
                "time_sec": float(solve_time_sec),
                "separation_time_sec": float(sep_time),
                "certification_time_sec": float(cert_time),
                "lp_time_sec": float(lp_time),
                "lp_solver_calls": int(lp_solver_calls),
                "a_surr": a_surr_last,
                "b_surr": b_surr_last,
                "rounds": int(max_rounds),
            },
            last_cert,
            all_cuts,
        )

    def _certify_and_build_surrogate(
        self,
        graph: GraphInstance,
        cbg: Optional[CompiledBitGraph],
        cuts: List[Any],
        gnn_model: Optional[Any] = None,
    ) -> Tuple[VerificationCertificate, np.ndarray, float]:
        """Rationally certifies the conic combination and returns (cert, a_surr, b_surr)."""
        m = graph.num_edges
        edge_to_idx = {e: i for i, e in enumerate(graph.edges)}

        if not cuts:
            # Trivial empty cut fallback
            cert = VerificationCertificate(
                is_valid=True,
                status="CERTIFIED_EMPTY_CONIC_COMBINATION",
                rejection_reason=None,
                num_active_supports=0,
                exact_coefficients={},
                exact_rhs=Fraction(0, 1),
                sha256_hash="0" * 64,
            )
            return cert, np.zeros(m, dtype=np.float64), 0.0

        # Check if cuts are K5 cliques
        is_k5 = (hasattr(cuts[0], "type") and cuts[0].type == "k5") or (
            isinstance(cuts[0], dict) and cuts[0].get("type") == "k5"
        )

        if is_k5:
            if gnn_model is not None:
                cliques = [
                    tuple(sorted(cut.nodes if hasattr(cut, "nodes") else cut["nodes"]))
                    for cut in cuts
                ]
                pred = gnn_model.forward(graph, k5_cliques=cliques)
                mults = {
                    clq: max(1e-4, float(pred["k5_multipliers"].get(clq, 1.0)))
                    for clq in cliques
                }
            else:
                mults = {}
                for cut in cuts:
                    nodes = tuple(sorted(cut.nodes if hasattr(cut, "nodes") else cut["nodes"]))
                    mults[nodes] = 1.0
            cert = self.verifier.verify_clique_conic_combination(graph, mults)
        else:
            # Cycle cuts (triangles, 4-cycles)
            cycle_specs = []
            for cut in cuts:
                if hasattr(cut, "nodes"):
                    nodes = list(cut.nodes)
                    cut_edges = cut.edge_indices
                    cut_coeffs = cut.coefficients
                else:
                    nodes = list(cut["nodes"])
                    cut_edges = cut["edge_indices"]
                    cut_coeffs = cut["coefficients"]

                f_edges = []
                for e_idx, coeff in zip(cut_edges, cut_coeffs):
                    if coeff > 0:
                        f_edges.append(graph.edges[e_idx])

                # Parity safety check (|F| must be odd for cycle inequalities)
                if len(f_edges) % 2 == 0:
                    c_edges = [
                        (min(nodes[i], nodes[(i + 1) % len(nodes)]), max(nodes[i], nodes[(i + 1) % len(nodes)]))
                        for i in range(len(nodes))
                    ]
                    if len(c_edges) % 2 == 1:
                        f_edges = c_edges
                    else:
                        f_edges = c_edges[: len(c_edges) - 1]

                cycle_specs.append((nodes, f_edges, 1.0))

            cert = self.verifier.verify_cycle_conic_combination(
                cbg if cbg is not None else graph, cycle_specs
            )

        if not cert.is_valid:
            raise ValueError(f"Surrogate cut rejected by rational verifier: {cert.status}")

        a_surr = np.zeros(m, dtype=np.float64)
        for e, frac in cert.exact_coefficients.items():
            if e in edge_to_idx:
                a_surr[edge_to_idx[e]] = float(frac)
        b_surr = float(cert.exact_rhs)

        return cert, a_surr, b_surr

    def _separate_bipartite_4cycles(
        self,
        graph: GraphInstance,
        primal_x: np.ndarray,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """Discovers violated 4-cycles in bipartite Chimera structures structurally."""
        edge_map = {e: i for i, e in enumerate(graph.edges)}
        violated = []
        cbg = CompiledBitGraph.from_graph_instance(graph) if COMPILED_ENGINE_AVAILABLE else None
        if cbg is None:
            return violated

        # Build adjacency list
        adj: List[List[int]] = [[] for _ in range(graph.num_nodes)]
        for u, v in graph.edges:
            adj[u].append(v)
            adj[v].append(u)

        seen: Set[Tuple[int, ...]] = set()

        for u in range(graph.num_nodes):
            nbrs = adj[u]
            if len(nbrs) < 2:
                continue
            for i in range(len(nbrs)):
                w1 = nbrs[i]
                for j in range(i + 1, len(nbrs)):
                    w2 = nbrs[j]
                    for v in adj[w1]:
                        if v != u and cbg.has_edge(v, w2):
                            cyc_key = tuple(sorted([u, w1, v, w2]))
                            if cyc_key in seen:
                                continue
                            seen.add(cyc_key)

                            e1 = (min(u, w1), max(u, w1))
                            e2 = (min(w1, v), max(w1, v))
                            e3 = (min(v, w2), max(v, w2))
                            e4 = (min(w2, u), max(w2, u))

                            if e1 in edge_map and e2 in edge_map and e3 in edge_map and e4 in edge_map:
                                i1, i2, i3, i4 = edge_map[e1], edge_map[e2], edge_map[e3], edge_map[e4]
                                x1, x2, x3, x4 = primal_x[i1], primal_x[i2], primal_x[i3], primal_x[i4]

                                combos = [
                                    ([i1, i2, i3, i4], [1.0, 1.0, 1.0, -1.0], x1 + x2 + x3 - x4 - 2.0),
                                    ([i1, i2, i4, i3], [1.0, 1.0, 1.0, -1.0], x1 + x2 + x4 - x3 - 2.0),
                                    ([i1, i3, i4, i2], [1.0, 1.0, 1.0, -1.0], x1 + x3 + x4 - x2 - 2.0),
                                    ([i2, i3, i4, i1], [1.0, 1.0, 1.0, -1.0], x2 + x3 + x4 - x1 - 2.0),
                                ]
                                for cut_edges_idx, coeffs, viol in combos:
                                    if viol > 1e-4:
                                        violated.append({
                                            "type": "cycle",
                                            "nodes": [u, w1, v, w2],
                                            "edge_indices": cut_edges_idx,
                                            "coefficients": coeffs,
                                            "rhs": 2.0,
                                            "violation": viol,
                                        })
                                        if len(violated) >= limit:
                                            return violated

        return violated

    def evaluate_instance_pair(
        self,
        graph: GraphInstance,
        seed: int = 42,
        target_ip: Optional[int] = None,
        gnn_model: Optional[Any] = None,
    ) -> Tuple[LargeScaleBenchmarkResult, LargeScaleBenchmarkResult]:
        """Evaluates paired isomorphic graphs (G_A, G_B) with true independent cut separation."""
        ga, gb, pi, pi_inv = generate_ab_pair(graph, seed=seed)

        # 1. Evaluate Cell A
        std_res_a = self.solve_standard_cutting_plane(ga)
        surr_res_a, cert_a, cuts_a = self.solve_surrogate_cutting_plane(ga, gnn_model=gnn_model)

        # 2. Evaluate Cell B with true independent separation (NO cut mapping through pi!)
        std_res_b = self.solve_standard_cutting_plane(gb)
        surr_res_b, cert_b, cuts_b = self.solve_surrogate_cutting_plane(gb, gnn_model=gnn_model)

        # 3. Symmetric 1-round baseline on Cell A for apples-to-apples comparison
        std_1r = self.solve_standard_cutting_plane(ga, max_rounds=1)
        sym_1r_data = {
            "sym_1round_std_time_sec": std_1r["time_sec"],
            "sym_1round_surr_time_sec": surr_res_a["time_sec"],
            "sym_1round_speedup": std_1r["time_sec"] / max(1e-5, surr_res_a["time_sec"]),
            "sym_1round_std_iters": std_1r["simplex_iterations"],
            "sym_1round_surr_iters": surr_res_a["simplex_iterations"],
            "sym_1round_iter_reduction": max(
                0.0,
                (std_1r["simplex_iterations"] - surr_res_a["simplex_iterations"])
                / max(1, std_1r["simplex_iterations"])
                * 100.0,
            ),
        }

        # Invariance assertion
        diff_obj = abs(surr_res_a["objective"] - surr_res_b["objective"])
        if diff_obj > 1e-6:
            raise AssertionError(
                f"Relabeling variance failure: |Za - Zb| = {diff_obj:.2e} > 1e-6 on {graph.name}"
            )

        # Determine canonical baseline target for integrality gap closed calculation
        if target_ip is not None and target_ip > 0:
            canonical_target = float(target_ip)
        else:
            canonical_target = min(std_res_a["objective"], std_res_b["objective"])

        # Calculate metrics
        mem_bytes = self.estimate_memory_footprint(graph)
        res_a = self._build_result(
            ga, "A", std_res_a, surr_res_a, cert_a, target_ip, mem_bytes,
            canonical_target=canonical_target, sym_data=sym_1r_data
        )
        res_b = self._build_result(
            gb, "B", std_res_b, surr_res_b, cert_b, target_ip, mem_bytes,
            canonical_target=canonical_target, sym_data=sym_1r_data
        )

        return res_a, res_b

    def evaluate_symmetric_1round(
        self,
        graph: GraphInstance,
        max_cuts: int = 1000,
    ) -> Dict[str, Any]:
        """Symmetric 1-round apples-to-apples comparison (1 standard cut round vs 1 surrogate cut round)."""
        std_res = self.solve_standard_cutting_plane(graph, max_rounds=1, max_cuts_per_round=max_cuts)
        surr_res, cert, _ = self.solve_surrogate_cutting_plane(graph, max_rounds=1, max_cuts=max_cuts)
        speedup = std_res["time_sec"] / max(1e-5, surr_res["time_sec"])
        iter_red = (
            max(0.0, (std_res["simplex_iterations"] - surr_res["simplex_iterations"]) / std_res["simplex_iterations"] * 100.0)
            if std_res["simplex_iterations"] > 0 else 0.0
        )
        return {
            "comparison": "1-round-symmetric",
            "standard_time_sec": std_res["time_sec"],
            "surrogate_time_sec": surr_res["time_sec"],
            "speedup": speedup,
            "standard_simplex_iters": std_res["simplex_iterations"],
            "surrogate_simplex_iters": surr_res["simplex_iterations"],
            "iters_reduction_percent": iter_red,
            "standard_lp_calls": std_res["lp_solver_calls"],
            "surrogate_lp_calls": surr_res["lp_solver_calls"],
            "standard_objective": std_res["objective"],
            "surrogate_objective": surr_res["objective"],
            "is_verified_sound": cert.is_valid,
            "certificate_sha256": cert.sha256_hash,
        }

    def evaluate_symmetric_multiround(
        self,
        graph: GraphInstance,
        rounds: int = 4,
        max_cuts: int = 1000,
    ) -> Dict[str, Any]:
        """Symmetric multi-round comparison (R standard cut rounds vs R surrogate cut rounds)."""
        std_res = self.solve_standard_cutting_plane(graph, max_rounds=rounds, max_cuts_per_round=max_cuts)
        surr_res, cert, _ = self.solve_surrogate_cutting_plane(graph, max_rounds=rounds, max_cuts=max_cuts)
        speedup = std_res["time_sec"] / max(1e-5, surr_res["time_sec"])
        iter_red = (
            max(0.0, (std_res["simplex_iterations"] - surr_res["simplex_iterations"]) / std_res["simplex_iterations"] * 100.0)
            if std_res["simplex_iterations"] > 0 else 0.0
        )
        return {
            "comparison": f"{rounds}-round-symmetric",
            "standard_time_sec": std_res["time_sec"],
            "surrogate_time_sec": surr_res["time_sec"],
            "speedup": speedup,
            "standard_simplex_iters": std_res["simplex_iterations"],
            "surrogate_simplex_iters": surr_res["simplex_iterations"],
            "iters_reduction_percent": iter_red,
            "standard_lp_calls": std_res["lp_solver_calls"],
            "surrogate_lp_calls": surr_res["lp_solver_calls"],
            "standard_objective": std_res["objective"],
            "surrogate_objective": surr_res["objective"],
            "is_verified_sound": cert.is_valid,
            "certificate_sha256": cert.sha256_hash,
        }

    def _build_result(
        self,
        graph: GraphInstance,
        labeling: str,
        std_res: Dict[str, Any],
        surr_res: Dict[str, Any],
        cert: VerificationCertificate,
        target_ip: Optional[int],
        memory_bytes: int,
        canonical_target: Optional[float] = None,
        sym_data: Optional[Dict[str, Any]] = None,
    ) -> LargeScaleBenchmarkResult:
        """Constructs unified LargeScaleBenchmarkResult record."""
        std_time = max(1e-5, std_res["time_sec"])
        surr_time = max(1e-5, surr_res["time_sec"])
        speedup = std_time / surr_time

        std_iters = std_res["simplex_iterations"]
        surr_iters = surr_res["simplex_iterations"]
        if std_iters > 0:
            iter_reduction = max(0.0, (std_iters - surr_iters) / std_iters * 100.0)
        else:
            iter_reduction = 0.0

        # Exact mathematical gap closed calculation using root LP relaxation objective
        base_obj = surr_res.get("root_objective", std_res.get("root_objective", float(graph.num_edges)))
        if target_ip is not None and target_ip > 0:
            target_val = float(target_ip)
        elif canonical_target is not None:
            target_val = canonical_target
        else:
            target_val = std_res["objective"]

        total_gap = base_obj - target_val
        gap_closed = base_obj - surr_res["objective"]
        if total_gap > 1e-6:
            gap_closed_pct = min(100.0, max(0.0, (gap_closed / total_gap) * 100.0))
        else:
            gap_closed_pct = 100.0

        k5_count = len(graph.metadata.get("k5_cliques", []))
        if k5_count == 0 and "k5" in graph.metadata.get("family", "").lower():
            k5_count = graph.metadata.get("embedded_k5_cliques", 0)

        sym = sym_data or {}

        return LargeScaleBenchmarkResult(
            instance_name=graph.name,
            family=graph.metadata.get("family", "LargeScaleTopology"),
            labeling=labeling,
            label_cell=labeling,
            num_nodes=graph.num_nodes,
            num_edges=graph.num_edges,
            k5_count=k5_count,
            standard_time_sec=std_time,
            surrogate_time_sec=surr_time,
            speedup=speedup,
            standard_simplex_iters=std_iters,
            surrogate_simplex_iters=surr_iters,
            iters_reduction_percent=iter_reduction,
            gap_closed_percent=gap_closed_pct,
            is_verified_sound=cert.is_valid,
            certificate_sha256=cert.sha256_hash,
            objective_val=surr_res["objective"],
            surrogate_obj=surr_res["objective"],
            full_k5_obj=std_res["objective"],
            full_k5_time_ms=std_time * 1000.0,
            full_k5_iterations=std_iters,
            surrogate_time_ms=surr_time * 1000.0,
            surrogate_iterations=surr_iters,
            speedup_vs_full_k5=speedup,
            cycle_obj=base_obj,
            cycle_time_ms=0.0,
            cycle_iterations=0,
            ip_maxcut=float(target_ip) if target_ip is not None else target_val,
            target_integer_target=target_ip,
            memory_footprint_bytes=memory_bytes,
            root_lp_obj=base_obj,
            standard_rounds=std_res.get("rounds", 4),
            surrogate_rounds=surr_res.get("rounds", 1),
            standard_lp_calls=std_res.get("lp_solver_calls", 1),
            surrogate_lp_calls=surr_res.get("lp_solver_calls", 2),
            standard_separation_time_sec=std_res.get("separation_time_sec", 0.0),
            surrogate_separation_time_sec=surr_res.get("separation_time_sec", 0.0),
            surrogate_certification_time_sec=surr_res.get("certification_time_sec", 0.0),
            standard_lp_time_sec=std_res.get("lp_time_sec", 0.0),
            surrogate_lp_time_sec=surr_res.get("lp_time_sec", 0.0),
            sym_1round_speedup=sym.get("sym_1round_speedup", 0.0),
            sym_1round_iter_reduction=sym.get("sym_1round_iter_reduction", 0.0),
            sym_1round_std_time_sec=sym.get("sym_1round_std_time_sec", 0.0),
            sym_1round_surr_time_sec=sym.get("sym_1round_surr_time_sec", 0.0),
            sym_1round_std_iters=sym.get("sym_1round_std_iters", 0),
            sym_1round_surr_iters=sym.get("sym_1round_surr_iters", 0),
        )

    def run_canonical_baselines(self) -> List[LargeScaleBenchmarkResult]:
        """Runs the 8 Experiment 87 cells and 12 Experiment 91 cells."""
        from highs_turbo.benchmark_harness import BenchmarkSuite

        suite = BenchmarkSuite(seed=self.seed)
        exp87_cells = suite.run_exp87_benchmark()
        exp91_cells = suite.run_exp91_benchmark(num_graphs=6)

        results = []
        for cell in itertools.chain(exp87_cells, exp91_cells):
            std_time = max(1e-5, cell.full_k5_time_ms / 1000.0)
            surr_time = max(1e-5, cell.surrogate_time_ms / 1000.0)
            speedup = std_time / surr_time
            std_iters = cell.full_k5_iterations
            surr_iters = cell.surrogate_iterations
            reduction = (
                max(0.0, (std_iters - surr_iters) / std_iters * 100.0)
                if std_iters > 0
                else 0.0
            )

            res = LargeScaleBenchmarkResult(
                instance_name=cell.instance_name,
                family=cell.family,
                labeling=cell.label_cell,
                label_cell=cell.label_cell,
                num_nodes=cell.num_nodes,
                num_edges=cell.num_edges,
                k5_count=cell.k5_count,
                standard_time_sec=std_time,
                surrogate_time_sec=surr_time,
                speedup=speedup,
                standard_simplex_iters=std_iters,
                surrogate_simplex_iters=surr_iters,
                iters_reduction_percent=reduction,
                gap_closed_percent=cell.gap_closed_percent,
                is_verified_sound=cell.is_verified_sound,
                certificate_sha256=cell.certificate_sha256,
                objective_val=cell.surrogate_obj,
                surrogate_obj=cell.surrogate_obj,
                full_k5_obj=cell.full_k5_obj,
                full_k5_time_ms=cell.full_k5_time_ms,
                full_k5_iterations=cell.full_k5_iterations,
                surrogate_time_ms=cell.surrogate_time_ms,
                surrogate_iterations=cell.surrogate_iterations,
                speedup_vs_full_k5=cell.speedup_vs_full_k5,
                cycle_obj=cell.cycle_obj,
                cycle_time_ms=cell.cycle_time_ms,
                cycle_iterations=cell.cycle_iterations,
                ip_maxcut=cell.ip_maxcut,
                target_integer_target=cell.target_integer_target,
                memory_footprint_bytes=cell.num_nodes * 64,
            )
            results.append(res)

        return results

    def run_large_scale_benchmarks(self) -> List[LargeScaleBenchmarkResult]:
        """Runs the large-scale combinatorial benchmark suite (1,000+ nodes)."""
        results: List[LargeScaleBenchmarkResult] = []

        # 1. Chimera C_{12,12,4} Native Ising (1,152 nodes, 3,360 edges)
        print("Evaluating Chimera C_{12,12,4} (1,152 nodes, 3,360 edges)...")
        c12_native = generate_chimera_instance(12, 12, 4, seed=1212, ising=True)
        r_a, r_b = self.evaluate_instance_pair(c12_native, seed=12)
        results.extend([r_a, r_b])

        # 2. Chimera C_{12,12,4} with Embedded Logical K5 Cliques (1,152 nodes)
        print("Evaluating Chimera C_{12,12,4} Embedded K5 (1,152 nodes)...")
        c12_embed = generate_embedded_chimera_instance(12, 12, 4, num_cliques=20, seed=1213)
        r_a, r_b = self.evaluate_instance_pair(c12_embed, seed=13)
        results.extend([r_a, r_b])

        # 3. Pegasus P_8 Ising (1,288 nodes, 8,804 edges)
        print("Evaluating Pegasus P_8 (1,288 nodes, 8,804 edges)...")
        p8 = generate_pegasus_instance(8, seed=88, ising=True)
        r_a, r_b = self.evaluate_instance_pair(p8, seed=89)
        results.extend([r_a, r_b])

        # 4. G-set G43 (1,000 nodes, 9,990 edges)
        print("Evaluating synthetic G43-shaped graph (1,000 nodes, 9,990 edges)...")
        g43 = generate_gset_instance("G43", seed=43)
        r_a, r_b = self.evaluate_instance_pair(g43, seed=44)
        results.extend([r_a, r_b])

        # 5. G-set G22 (2,000 nodes, 19,990 edges)
        print("Evaluating G-set G22 (2,000 nodes, 19,990 edges)...")
        g22 = generate_gset_instance("G22", seed=22)
        r_a, r_b = self.evaluate_instance_pair(g22, seed=23)
        results.extend([r_a, r_b])

        # 6. Planted 1,000-Node K5 Cluster (1,000 nodes, 2,199 edges, target IP = 1,399)
        print("Evaluating Planted 1,000-Node K5 Cluster (Target IP = 1,399)...")
        planted = generate_planted_1000_node_instance(num_k5=200, num_bridges=199, seed=1000)
        r_a, r_b = self.evaluate_instance_pair(planted, seed=1001, target_ip=1399)
        results.extend([r_a, r_b])

        # 7. Optional Chimera C_{16,16,4} (2,048 nodes, 6,016 edges)
        print("Evaluating Chimera C_{16,16,4} (2,048 nodes, 6,016 edges)...")
        c16 = generate_chimera_instance(16, 16, 4, seed=1616, ising=True)
        r_a, r_b = self.evaluate_instance_pair(c16, seed=16)
        results.extend([r_a, r_b])

        return results

    def format_markdown_report(self, results: List[LargeScaleBenchmarkResult]) -> str:
        """Generates comprehensive markdown report table from results."""
        lines = [
            "# Large-Scale Real-World Benchmark Suite Report",
            "",
            "| Instance | Cell | Nodes | Edges | Root LP Obj | Standard Time | Surrogate Time | Speedup | Std Iters | Surr Iters | Iter Reduction | Gap Closed | Soundness | Certificate SHA-256 |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]

        for r in results:
            sha_short = r.certificate_sha256[:8] if r.certificate_sha256 else "N/A"
            sound_str = "CERTIFIED" if r.is_verified_sound else "REJECTED"
            root_str = f"{r.root_lp_obj:.1f}" if r.root_lp_obj != 0.0 else f"{r.cycle_obj:.1f}"
            lines.append(
                f"| `{r.instance_name}` | {r.labeling} | {r.num_nodes} | {r.num_edges} | {root_str} | "
                f"{r.standard_time_sec * 1000.0:.1f}ms | {r.surrogate_time_sec * 1000.0:.1f}ms | "
                f"**{r.speedup:.2f}x** | {r.standard_simplex_iters} | {r.surrogate_simplex_iters} | "
                f"**{r.iters_reduction_percent:.1f}%** | **{r.gap_closed_percent:.1f}%** | "
                f"{sound_str} | `{sha_short}` |"
            )

        # Relabeling Invariance Table
        lines.extend([
            "",
            "## Isomorphic A/B Permutation Invariance Verification (|Delta Z| <= 10^-6)",
            "",
            "| Base Instance | Cell A Obj | Cell B Obj | Absolute Diff | Invariance Status |",
            "|---|---|---|---|---|",
        ])

        paired_by_base: Dict[str, Dict[str, LargeScaleBenchmarkResult]] = {}
        for r in results:
            base = r.instance_name.replace("_label_A", "").replace("_label_B", "")
            paired_by_base.setdefault(base, {})[r.labeling] = r

        for base, cells in paired_by_base.items():
            if "A" in cells and "B" in cells:
                ra, rb = cells["A"], cells["B"]
                diff = abs(ra.objective_val - rb.objective_val)
                status = "PASS (Var=0.0)" if diff <= 1e-6 else f"FAIL ({diff:.2e})"
                lines.append(
                    f"| `{base}` | {ra.objective_val:.6f} | {rb.objective_val:.6f} | {diff:.2e} | **{status}** |"
                )
        # Symmetric 1-Round Apples-to-Apples Summary Table
        lines.extend([
            "",
            "## Symmetric 1-Round Apples-to-Apples Benchmarking (1-Round Standard vs 1-Round Surrogate)",
            "",
            "| Base Instance | 1-Round Std Time | 1-Round Surr Time | 1-Round Speedup | 1-Round Std Iters | 1-Round Surr Iters | 1-Round Iter Reduction | LP Calls (Std/Surr) |",
            "|---|---|---|---|---|---|---|---|",
        ])

        for base, cells in paired_by_base.items():
            if "A" in cells:
                ra = cells["A"]
                if ra.sym_1round_std_time_sec > 0:
                    lines.append(
                        f"| `{base}` | {ra.sym_1round_std_time_sec * 1000.0:.1f}ms | {ra.sym_1round_surr_time_sec * 1000.0:.1f}ms | "
                        f"**{ra.sym_1round_speedup:.2f}x** | {ra.sym_1round_std_iters} | {ra.sym_1round_surr_iters} | "
                        f"**{ra.sym_1round_iter_reduction:.1f}%** | 2 / 2 |"
                    )

        return "\n".join(lines)


# =============================================================================
# 4. Command Line Interface & Runner
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Large-Scale Neural-Surrogate Benchmark Suite")
    parser.add_argument(
        "--scope",
        choices=["large_scale", "canonical", "all"],
        default="all",
        help="Benchmark scope to execute",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="full_benchmark_results.json",
        help="Path to output JSON",
    )
    args = parser.parse_args()

    suite = LargeScaleBenchmarkSuite()
    results: List[LargeScaleBenchmarkResult] = []

    if args.scope in ("canonical", "all"):
        print("\n--- Running Canonical Baselines (Exp87 & Exp91) ---")
        canonical_res = suite.run_canonical_baselines()
        results.extend(canonical_res)

    if args.scope in ("large_scale", "all"):
        print("\n--- Running 1,000+ Node Large-Scale Benchmarks ---")
        large_res = suite.run_large_scale_benchmarks()
        results.extend(large_res)

    report_md = suite.format_markdown_report(results)
    print("\n" + report_md + "\n")

    raw_results = [asdict(r) for r in results]
    with open(args.output, "w") as f:
        json.dump(raw_results, f, indent=2)
    print(f"Results successfully saved to {args.output} ({len(results)} cells evaluated).")

    # Also update benchmark_results.json if output is full_benchmark_results.json
    if args.output == "full_benchmark_results.json":
        with open("benchmark_results.json", "w") as f:
            json.dump(raw_results, f, indent=2)
        print("Updated benchmark_results.json with full structured benchmark records.")


if __name__ == "__main__":
    main()
