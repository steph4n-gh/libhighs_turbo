"""High-performance Python wrapper and bridge for the compiled C++ engine.

Provides zero-copy, cache-aligned bit-parallel graph operations, exact rational
verification, cut discovery, and in-memory HiGHS solver integration.
"""

from __future__ import annotations

import time
from fractions import Fraction
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.sparse as sp

try:
    from highs_turbo import _compiled_engine as _ce

    COMPILED_ENGINE_AVAILABLE = True
except ImportError:
    _ce = None
    COMPILED_ENGINE_AVAILABLE = False

from highs_turbo.exact_solver import ExactMaxCutSolver, LPSolution
from highs_turbo.graph_generator import GraphInstance
from highs_turbo.rational_verifier import VerificationCertificate


class CompiledBitGraph:
    """High-performance bit-parallel graph core backed by C++ BitGraph."""

    def __init__(self, num_nodes: int):
        if not COMPILED_ENGINE_AVAILABLE:
            raise RuntimeError("Compiled native engine (_compiled_engine) is not available.")
        self._native = _ce.BitGraph(num_nodes)
        self._num_nodes = num_nodes

    @classmethod
    def from_graph_instance(cls, graph: GraphInstance) -> CompiledBitGraph:
        """Constructs a CompiledBitGraph from a Python GraphInstance."""
        cbg = cls(graph.num_nodes)
        edges = [(int(u), int(v)) for u, v in graph.edges]
        cbg._native.add_edges(edges)
        return cbg

    @property
    def num_nodes(self) -> int:
        return self._native.num_nodes

    @property
    def num_edges(self) -> int:
        return self._native.num_edges

    def add_edge(self, u: int, v: int) -> None:
        self._native.add_edge(int(u), int(v))

    def add_edges(self, edges: Sequence[Tuple[int, int]]) -> None:
        self._native.add_edges([(int(u), int(v)) for u, v in edges])

    def has_edge(self, u: int, v: int) -> bool:
        return self._native.has_edge(int(u), int(v))

    def degree(self, u: int) -> int:
        return self._native.degree(int(u))

    def count_common_neighbors(self, u: int, v: int) -> int:
        return self._native.count_common_neighbors(int(u), int(v))

    def count_triangles_on_edge(self, u: int, v: int) -> int:
        return self._native.count_triangles_on_edge(int(u), int(v))

    def count_triangles(self) -> int:
        return self._native.count_triangles()

    def find_triangles(self) -> List[Tuple[int, int, int]]:
        raw = self._native.find_triangles()
        return [(t[0], t[1], t[2]) for t in raw]

    def find_k5_cliques(self) -> List[Tuple[int, int, int, int, int]]:
        raw = self._native.find_k5_cliques()
        return [(c[0], c[1], c[2], c[3], c[4]) for c in raw]

    def get_edges(self) -> List[Tuple[int, int]]:
        return list(self._native.get_edges())

    def compute_1wl_colors(self, max_iters: int = 3) -> List[int]:
        return list(self._native.compute_1wl_colors(max_iters))

    def extract_topological_features(
        self,
        edge_list: Sequence[Tuple[int, int]],
        edge_weights: Optional[Sequence[float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        edges = [(int(u), int(v)) for u, v in edge_list]
        weights = [float(w) for w in edge_weights] if edge_weights is not None else []
        feats = self._native.extract_topological_features(edges, weights)
        return np.array(feats.node_features, copy=True), np.array(feats.edge_features, copy=True)

    @property
    def native(self) -> Any:
        return self._native


class CompiledRationalVerifier:
    """Compiled rational verifier with Stein binary GCD and GMP fallback."""

    def __init__(self, rational_denominator_limit: int = 100000):
        if not COMPILED_ENGINE_AVAILABLE:
            raise RuntimeError("Compiled native engine (_compiled_engine) is not available.")
        self.denominator_limit = rational_denominator_limit
        self._native = _ce.RationalVerifier(rational_denominator_limit)

    def verify_clique_conic_combination(
        self,
        graph: Union[GraphInstance, CompiledBitGraph],
        candidate_multipliers: Dict[Tuple[int, ...], Union[float, int, Fraction]],
        candidate_rhs: Optional[Union[float, Fraction]] = None,
        tol: float = 1e-6,
    ) -> VerificationCertificate:
        """Verifies candidate clique multipliers and returns a VerificationCertificate."""
        if isinstance(graph, GraphInstance):
            native_graph = CompiledBitGraph.from_graph_instance(graph).native
        elif isinstance(graph, CompiledBitGraph):
            native_graph = graph.native
        else:
            native_graph = graph

        # Format multipliers for C++
        formatted_mults = {}
        for clq, val in candidate_multipliers.items():
            if isinstance(val, (int, float)) and val < 0:
                cert = VerificationCertificate(
                    is_valid=False,
                    status="REJECTED_NEGATIVE_MULTIPLIER",
                    rejection_reason=f"Support {clq} has negative raw multiplier: {val}",
                )
                cert.compute_sha256()
                return cert
            if isinstance(val, Fraction):
                formatted_mults[tuple(int(x) for x in clq)] = _ce.ExactRational(str(val.numerator), str(val.denominator))
            elif isinstance(val, int):
                formatted_mults[tuple(int(x) for x in clq)] = _ce.ExactRational(val, 1)
            else:
                formatted_mults[tuple(int(x) for x in clq)] = _ce.ExactRational.from_double(float(val), self.denominator_limit)

        cand_rhs_native = None
        if candidate_rhs is not None:
            if isinstance(candidate_rhs, Fraction):
                cand_rhs_native = _ce.ExactRational(str(candidate_rhs.numerator), str(candidate_rhs.denominator))
            elif isinstance(candidate_rhs, int):
                cand_rhs_native = _ce.ExactRational(candidate_rhs, 1)
            else:
                cand_rhs_native = _ce.ExactRational.from_double(float(candidate_rhs), self.denominator_limit)

        native_cert = self._native.verify_clique_conic_combination(
            native_graph, formatted_mults, cand_rhs_native, tol
        )

        # Convert native certificate to standard Python VerificationCertificate
        exact_coeffs: Dict[Tuple[int, int], Fraction] = {}
        raw_coeffs = native_cert.get_coefficients()
        for edge, frac_tuple in raw_coeffs.items():
            exact_coeffs[edge] = Fraction(int(frac_tuple[0]), int(frac_tuple[1]))

        rhs_str = native_cert.exact_rhs.num_str()
        rhs_den = native_cert.exact_rhs.den_str()
        exact_rhs = Fraction(int(rhs_str), int(rhs_den))

        cert = VerificationCertificate(
            is_valid=native_cert.is_valid,
            status=native_cert.status,
            rejection_reason=native_cert.rejection_reason if native_cert.rejection_reason else None,
            num_active_supports=native_cert.num_active_supports,
            exact_coefficients=exact_coeffs,
            exact_rhs=exact_rhs,
            sha256_hash=native_cert.sha256_hash,
        )
        return cert

    def verify_cycle_conic_combination(
        self,
        graph: Union[GraphInstance, CompiledBitGraph],
        candidate_cycles: Sequence[Any],
        candidate_rhs: Optional[Union[float, int, Fraction]] = None,
        tol: float = 1e-6,
    ) -> VerificationCertificate:
        """Verifies candidate cycle multipliers and returns a VerificationCertificate."""
        if isinstance(graph, GraphInstance):
            native_graph = CompiledBitGraph.from_graph_instance(graph).native
        elif isinstance(graph, CompiledBitGraph):
            native_graph = graph.native
        else:
            native_graph = graph

        formatted_mults = []
        for item in candidate_cycles:
            if len(item) == 3:
                nodes, odd_edges, val = item
            elif len(item) == 2:
                support, val = item
                nodes, odd_edges = support
            else:
                raise ValueError(f"Invalid cycle format: {item}")

            if isinstance(val, (int, float)) and val < 0:
                cert = VerificationCertificate(
                    is_valid=False,
                    status="REJECTED_NEGATIVE_MULTIPLIER",
                    rejection_reason=f"Support ({nodes}, {odd_edges}) has negative multiplier: {val}",
                )
                cert.compute_sha256()
                return cert

            c_nodes = [int(x) for x in nodes]
            o_edges = [(int(u), int(v)) for u, v in odd_edges]

            if isinstance(val, Fraction):
                r = _ce.ExactRational(str(val.numerator), str(val.denominator))
            elif isinstance(val, int):
                r = _ce.ExactRational(val, 1)
            else:
                r = _ce.ExactRational.from_double(float(val), self.denominator_limit)

            formatted_mults.append(((c_nodes, o_edges), r))

        cand_rhs_native = None
        if candidate_rhs is not None:
            if isinstance(candidate_rhs, Fraction):
                cand_rhs_native = _ce.ExactRational(str(candidate_rhs.numerator), str(candidate_rhs.denominator))
            elif isinstance(candidate_rhs, int):
                cand_rhs_native = _ce.ExactRational(candidate_rhs, 1)
            else:
                cand_rhs_native = _ce.ExactRational.from_double(float(candidate_rhs), self.denominator_limit)

        native_cert = self._native.verify_cycle_conic_combination(
            native_graph, formatted_mults, cand_rhs_native, tol
        )

        exact_coeffs: Dict[Tuple[int, int], Fraction] = {}
        raw_coeffs = native_cert.get_coefficients()
        for edge, frac_tuple in raw_coeffs.items():
            exact_coeffs[edge] = Fraction(int(frac_tuple[0]), int(frac_tuple[1]))

        rhs_str = native_cert.exact_rhs.num_str()
        rhs_den = native_cert.exact_rhs.den_str()
        exact_rhs = Fraction(int(rhs_str), int(rhs_den))

        cert = VerificationCertificate(
            is_valid=native_cert.is_valid,
            status=native_cert.status,
            rejection_reason=native_cert.rejection_reason if native_cert.rejection_reason else None,
            num_active_supports=native_cert.num_active_supports,
            exact_coefficients=exact_coeffs,
            exact_rhs=exact_rhs,
            sha256_hash=native_cert.sha256_hash,
        )
        return cert


class CompiledCutEngine:
    """Compiled cut discovery and separation engine."""

    def __init__(self, rational_denominator_limit: int = 100000):
        if not COMPILED_ENGINE_AVAILABLE:
            raise RuntimeError("Compiled native engine (_compiled_engine) is not available.")
        self._native = _ce.CutEngine(rational_denominator_limit)
        self.denominator_limit = rational_denominator_limit

    def separate_violated_k5(
        self,
        graph: Union[GraphInstance, CompiledBitGraph],
        edge_list: Sequence[Tuple[int, int]],
        primal_sol: Sequence[float],
        threshold: float = 1e-4,
        max_cuts: int = 0,
    ) -> List[Any]:
        native_graph = graph.native if isinstance(graph, CompiledBitGraph) else CompiledBitGraph.from_graph_instance(graph).native
        edges = [(int(u), int(v)) for u, v in edge_list]
        return self._native.separate_violated_k5(native_graph, edges, list(primal_sol), threshold, max_cuts)

    def separate_violated_triangles(
        self,
        graph: Union[GraphInstance, CompiledBitGraph],
        edge_list: Sequence[Tuple[int, int]],
        primal_sol: Sequence[float],
        threshold: float = 1e-4,
        max_cuts: int = 0,
    ) -> List[Any]:
        native_graph = graph.native if isinstance(graph, CompiledBitGraph) else CompiledBitGraph.from_graph_instance(graph).native
        edges = [(int(u), int(v)) for u, v in edge_list]
        return self._native.separate_violated_triangles(native_graph, edges, list(primal_sol), threshold, max_cuts)

    def separate_violated_4cycles(
        self,
        graph: Union[GraphInstance, CompiledBitGraph],
        edge_list: Sequence[Tuple[int, int]],
        primal_sol: Sequence[float],
        threshold: float = 1e-4,
        max_cuts: int = 0,
    ) -> List[Any]:
        native_graph = graph.native if isinstance(graph, CompiledBitGraph) else CompiledBitGraph.from_graph_instance(graph).native
        edges = [(int(u), int(v)) for u, v in edge_list]
        return self._native.separate_violated_4cycles(native_graph, edges, list(primal_sol), threshold, max_cuts)

    def select_canonical_cuts(self, cuts: List[Any], max_cuts: int = 0) -> List[Any]:
        return self._native.select_canonical_cuts(cuts, max_cuts)

    def compile_surrogate_cut(
        self,
        graph: Union[GraphInstance, CompiledBitGraph],
        edge_list: Sequence[Tuple[int, int]],
        k5_multipliers: Sequence[Tuple[Sequence[int], float]],
        tol: float = 1e-6,
    ) -> Any:
        native_graph = graph.native if isinstance(graph, CompiledBitGraph) else CompiledBitGraph.from_graph_instance(graph).native
        edges = [(int(u), int(v)) for u, v in edge_list]
        mults = [([int(x) for x in clq], float(val)) for clq, val in k5_multipliers]
        return self._native.compile_surrogate_cut(native_graph, edges, mults, tol)

    def compile_violation_surrogate(
        self,
        graph: Union[GraphInstance, CompiledBitGraph],
        edge_list: Sequence[Tuple[int, int]],
        violated_cuts: List[Any],
    ) -> Any:
        native_graph = graph.native if isinstance(graph, CompiledBitGraph) else CompiledBitGraph.from_graph_instance(graph).native
        edges = [(int(u), int(v)) for u, v in edge_list]
        return self._native.compile_violation_surrogate(native_graph, edges, violated_cuts)


class CompiledSolverCallbackBridge:
    """Manages cut row pool, certified surrogate cuts, and solver constraint matrices."""

    def __init__(self, num_edges: int):
        if not COMPILED_ENGINE_AVAILABLE:
            raise RuntimeError("Compiled native engine (_compiled_engine) is not available.")
        self._native = _ce.SolverCallbackBridge(num_edges)
        self.num_edges = num_edges

    def add_cut_row(
        self,
        upper: float,
        indices: Sequence[int],
        values: Sequence[float],
        label: str = "",
    ) -> None:
        self._native.add_cut_row(float(upper), [int(i) for i in indices], [float(v) for v in values], label)

    def add_surrogate_cut(self, cut: Any, label: str = "") -> None:
        if hasattr(cut, "_native"):
            self._native.add_surrogate_cut(cut._native, label)
        else:
            self._native.add_surrogate_cut(cut, label)

    @property
    def num_rows(self) -> int:
        return self._native.get_num_rows()

    @property
    def num_certified_cuts(self) -> int:
        return self._native.get_num_certified_cuts()

    def get_flat_row_data(self) -> Tuple[List[float], List[float], List[int], List[int], List[float]]:
        return self._native.get_flat_row_data()

    def get_sparse_constraints(self) -> Tuple[sp.csr_matrix, np.ndarray]:
        lower, upper, starts, indices, values = self._native.get_flat_row_data()
        num_rows = len(upper)
        if num_rows == 0:
            return sp.csr_matrix((0, self.num_edges), dtype=np.float64), np.empty(0, dtype=np.float64)
        indptr = np.array(starts, dtype=np.int32)
        indices_arr = np.array(indices, dtype=np.int32)
        data_arr = np.array(values, dtype=np.float64)
        A_csr = sp.csr_matrix((data_arr, indices_arr, indptr), shape=(num_rows, self.num_edges), dtype=np.float64)
        b_arr = np.array(upper, dtype=np.float64)
        return A_csr, b_arr

    def clear(self) -> None:
        self._native.clear()

    @property
    def native(self) -> Any:
        return self._native


def solve_with_compiled_surrogate(
    graph: GraphInstance,
    surrogate_coefficients: Optional[np.ndarray] = None,
    surrogate_rhs: Optional[float] = None,
    k5_multipliers: Optional[Dict[Tuple[int, ...], float]] = None,
    method: str = "highs_in_memory",
) -> LPSolution:
    """Solves Max-Cut relaxation accelerated by the compiled cutting plane engine.

    If k5_multipliers is provided, compiles and rationally certifies the 1-row
    surrogate cutting plane before adding it to the HiGHS solver state via
    CompiledSolverCallbackBridge.
    """
    if not COMPILED_ENGINE_AVAILABLE:
        raise RuntimeError("Compiled engine (_compiled_engine) required for solve_with_compiled_surrogate.")

    t0 = time.perf_counter()
    cbg = CompiledBitGraph.from_graph_instance(graph)
    edge_list = list(graph.edges)
    bridge = CompiledSolverCallbackBridge(len(edge_list))

    if k5_multipliers is not None:
        cut_engine = CompiledCutEngine()
        mult_list = [(list(k), float(v)) for k, v in k5_multipliers.items()]
        surr_cut = cut_engine.compile_surrogate_cut(cbg, edge_list, mult_list)
        if not surr_cut.is_valid:
            raise ValueError(f"Surrogate cut rejected by rational verifier: {surr_cut.status}")
        bridge.add_surrogate_cut(surr_cut, "surrogate_k5_cut")
    elif surrogate_coefficients is not None and surrogate_rhs is not None:
        coeffs = np.array(surrogate_coefficients, dtype=np.float64)
        rhs = float(surrogate_rhs)
        nz_idx = np.where(np.abs(coeffs) > 1e-12)[0]
        nz_val = coeffs[nz_idx]
        bridge.add_cut_row(rhs, nz_idx, nz_val, "surrogate_external_cut")
    else:
        raise ValueError("Must provide either k5_multipliers or (surrogate_coefficients, surrogate_rhs).")

    # In-memory HiGHS solve via bridge constraint matrix
    A_csr, b_arr = bridge.get_sparse_constraints()
    solver = ExactMaxCutSolver()
    if A_csr.shape[0] > 0:
        row_coeffs = np.asarray(A_csr.toarray()[0], dtype=np.float64)
        solution = solver.solve_surrogate_relaxation(graph, row_coeffs, float(b_arr[0]))
    else:
        solution = solver.solve_surrogate_relaxation(graph, np.zeros(len(edge_list)), 0.0)
    return solution

def solve_milp_with_highs(
    c: Sequence[float],
    col_lower: Sequence[float],
    col_upper: Sequence[float],
    row_lower: Sequence[float],
    row_upper: Sequence[float],
    row_starts: Sequence[int],
    row_indices: Sequence[int],
    row_values: Sequence[float],
    integrality: Sequence[int],
    bridge: Optional[CompiledSolverCallbackBridge] = None
) -> Dict[str, Any]:
    """Legacy array interface using the same public runtime as the other APIs.

    Queued bridge rows are added before solving. The graph extension only
    manages and verifies cuts; it no longer loads a second HiGHS library.
    The legacy integrality convention is preserved: positive means integer.
    """
    import highspy
    from highs_turbo.lp_accelerator import _check_status, _add_rows

    costs = np.asarray(c, dtype=float)
    lower, upper = np.asarray(col_lower, dtype=float), np.asarray(col_upper, dtype=float)
    row_lower, row_upper = np.asarray(row_lower, dtype=float), np.asarray(row_upper, dtype=float)
    if (costs.ndim != 1 or lower.shape != costs.shape or upper.shape != costs.shape
            or row_lower.ndim != 1 or row_lower.shape != row_upper.shape):
        raise ValueError("Column and row arrays must have matching one-dimensional shapes")
    starts = np.asarray(row_starts, dtype=np.int64)
    if len(starts) == len(row_upper):
        starts = np.r_[starts, len(row_values)]
    matrix = sp.csr_matrix((row_values, row_indices, starts), shape=(len(row_upper), len(costs)), dtype=float)
    matrix.check_format(full_check=True)
    types = np.asarray(integrality)
    if types.size and types.shape != costs.shape:
        raise ValueError("integrality must be empty or have one entry per column")
    session = highspy.Highs()
    _check_status(session.setOptionValue("output_flag", False))
    _check_status(session.addCols(len(costs), costs, lower, upper, 0, [], [], []))
    _add_rows(session, matrix, row_lower, row_upper)
    if types.size:
        _check_status(session.changeColsIntegrality(len(costs), np.arange(len(costs)), (types > 0).astype(np.uint8)))
    if bridge is not None:
        lo, hi, ptr, idx, values = bridge.get_flat_row_data()
        cuts = sp.csr_matrix((values, idx, ptr), shape=(len(hi), len(costs)))
        cuts.check_format(full_check=True)
        _add_rows(session, cuts, lo, hi)
    _check_status(session.run())
    info, solution = session.getInfo(), session.getSolution()
    return {"status": int(session.getModelStatus()), "fun": info.objective_function_value,
            "x": np.array(solution.col_value, copy=True) if solution.value_valid else None,
            "simplex_iterations": info.simplex_iteration_count, "mip_node_count": info.mip_node_count}
