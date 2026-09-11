"""Public API and high-level drop-in plugin for HiGHS and SciPy (highs_turbo).

Provides:
1. `highs_turbo.linprog(...)`: Exact signature and return format of scipy.optimize.linprog,
   accelerating large inequality systems without changing the supplied problem.
2. `highs_turbo.solve_maxcut(G)`: High-level 1-line Max-Cut solver returning cut value,
   partition vector, and SHA-256 cryptographic receipt.
3. `highs_turbo.solve_qubo(Q)`: High-level QUBO / Ising spin glass solver.
4. `highs_turbo.TurboSolver`: Configurable solver class.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import scipy.sparse as sp
except ImportError:
    sp = None

from scipy.optimize import Bounds, LinearConstraint, OptimizeResult, linprog as scipy_linprog, milp as scipy_milp

# Import compiled engine components with graceful fallback
from highs_turbo.compiled_engine import (
    COMPILED_ENGINE_AVAILABLE,
    CompiledBitGraph,
    CompiledCutEngine,
    CompiledRationalVerifier,
    CompiledSolverCallbackBridge,
)

from highs_turbo.detector import TopologyDetector, TopologyScanResult, detect_topology
from highs_turbo.exact_solver import ExactMaxCutSolver
from highs_turbo.graph_generator import GraphInstance
from highs_turbo.rational_verifier import VerificationCertificate


@dataclass
class MaxCutResult:
    """Result of solve_maxcut, unpackable as (cut_value, partition, certificate)."""

    cut_value: float
    partition: np.ndarray
    certificate: str
    is_rationally_certified: bool = True
    upper_bound: float = 0.0
    simplex_iterations: int = 0
    solve_time_ms: float = 0.0
    status: str = "OPTIMAL"
    exact_rational_bound: Optional[Fraction] = None
    num_active_supports: int = 0

    def __iter__(self):
        """Enables 3-element tuple unpacking: cut_val, partition, cert = solve_maxcut(G)."""
        return iter([self.cut_value, self.partition, self.certificate])

    def __getitem__(self, idx: int) -> Any:
        return [self.cut_value, self.partition, self.certificate][idx]

    def __len__(self) -> int:
        return 3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cut_value": self.cut_value,
            "partition": self.partition.tolist() if isinstance(self.partition, np.ndarray) else self.partition,
            "certificate": self.certificate,
            "is_rationally_certified": self.is_rationally_certified,
            "upper_bound": self.upper_bound,
            "simplex_iterations": self.simplex_iterations,
            "solve_time_ms": self.solve_time_ms,
            "status": self.status,
        }

    def __repr__(self) -> str:
        return (
            f"MaxCutResult(cut_value={self.cut_value:.4f}, "
            f"upper_bound={self.upper_bound:.4f}, "
            f"certified={self.is_rationally_certified}, "
            f"sha256='{self.certificate[:12]}...', "
            f"iterations={self.simplex_iterations})"
        )


@dataclass
class QuboResult:
    """Result of solve_qubo, unpackable as (energy, solution, certificate)."""

    energy: float
    solution: np.ndarray
    certificate: str
    is_rationally_certified: bool = True
    lower_bound: float = 0.0
    simplex_iterations: int = 0
    solve_time_ms: float = 0.0
    status: str = "OPTIMAL"
    exact_rational_bound: Optional[Fraction] = None
    num_active_supports: int = 0

    def __iter__(self):
        """Enables 3-element tuple unpacking: energy, solution, cert = solve_qubo(Q)."""
        return iter([self.energy, self.solution, self.certificate])

    def __getitem__(self, idx: int) -> Any:
        return [self.energy, self.solution, self.certificate][idx]

    def __len__(self) -> int:
        return 3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "energy": self.energy,
            "solution": self.solution.tolist() if isinstance(self.solution, np.ndarray) else self.solution,
            "certificate": self.certificate,
            "is_rationally_certified": self.is_rationally_certified,
            "lower_bound": self.lower_bound,
            "simplex_iterations": self.simplex_iterations,
            "solve_time_ms": self.solve_time_ms,
            "status": self.status,
        }

    def __repr__(self) -> str:
        return (
            f"QuboResult(energy={self.energy:.4f}, "
            f"lower_bound={self.lower_bound:.4f}, "
            f"certified={self.is_rationally_certified}, "
            f"sha256='{self.certificate[:12]}...', "
            f"iterations={self.simplex_iterations})"
        )


class TurboSolver:
    """High-performance Neural-Surrogate Cutting Plane solver engine and HiGHS orchestrator."""

    def __init__(
        self,
        rational_denominator_limit: int = 100000,
        max_cuts: int = 1000,
        verbose: bool = False,
        fallback_on_error: bool = True,
    ):
        self.rational_denominator_limit = rational_denominator_limit
        self.max_cuts = max_cuts
        self.verbose = verbose
        self.fallback_on_error = fallback_on_error

        self.detector = TopologyDetector()
        self.exact_solver = ExactMaxCutSolver(rational_denominator_limit=rational_denominator_limit)
        self._bench_suite = None
        self._gnn_model = None

        if COMPILED_ENGINE_AVAILABLE:
            self.cut_engine = CompiledCutEngine(rational_denominator_limit=rational_denominator_limit)
            self.verifier = CompiledRationalVerifier(rational_denominator_limit=rational_denominator_limit)
        else:
            self.cut_engine = None
            self.verifier = None

    @property
    def bench_suite(self):
        if self._bench_suite is None:
            from highs_turbo.large_scale_benchmarks import LargeScaleBenchmarkSuite

            self._bench_suite = LargeScaleBenchmarkSuite(seed=42)
        return self._bench_suite

    @property
    def gnn_model(self):
        if self._gnn_model is not None:
            return self._gnn_model
        try:
            import os
            import torch
            from highs_turbo.surrogate_model import EdgeEquivariantSurrogateGNN
            weights_path = os.path.join(os.path.dirname(__file__), "default_weights.pt")
            if os.path.exists(weights_path):
                self._gnn_model = EdgeEquivariantSurrogateGNN(hidden_dim=32, num_layers=2)
                self._gnn_model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
                self._gnn_model.eval()
        except Exception as e:
            if self.verbose:
                print(f"[highs_turbo] Could not load GNN weights: {e}")
                
        if self._gnn_model is None:
            class GeometricFallback:
                def forward(self, graph, k5_cliques=None, **kwargs):
                    if k5_cliques is None:
                        return {"k5_multipliers": {}}
                    return {"k5_multipliers": {clq: 0.5 ** i for i, clq in enumerate(k5_cliques)}}
            self._gnn_model = GeometricFallback()

        return self._gnn_model

    @gnn_model.setter
    def gnn_model(self, model):
        self._gnn_model = model

    def linprog(
        self,
        c: Sequence[float],
        A_ub: Optional[Any] = None,
        b_ub: Optional[Sequence[float]] = None,
        A_eq: Optional[Any] = None,
        b_eq: Optional[Sequence[float]] = None,
        bounds: Optional[Any] = (0, None),
        method: str = "highs",
        callback: Optional[Callable] = None,
        options: Optional[Dict[str, Any]] = None,
        x0: Optional[Sequence[float]] = None,
        integrality: Optional[Sequence[int]] = None,
        **kwargs: Any,
    ) -> OptimizeResult:
        """SciPy-compatible solve with exact row aggregation and row recovery.

        Graph hints never authorize changing the supplied feasible set. Large
        inequality systems can use a smaller working model. Sparse and equality
        models use native HiGHS, with competing methods for large LPs. Returned
        solutions satisfy the original model; methods, limits, and unsupported
        options retain ordinary SciPy execution.
        """
        if isinstance(bounds, Bounds):
            bounds = np.column_stack((
                np.broadcast_to(bounds.lb, (len(c),)),
                np.broadcast_to(bounds.ub, (len(c),)),
            ))
        method_clean = method.lower().replace("-", "_")
        use_turbo = method_clean in ("highs", "turbo", "highs_turbo")
        scipy_method = "highs" if use_turbo else method
        fallback_reason = None
        if use_turbo and callback is None and x0 is None:
            try:
                from highs_turbo.lp_accelerator import solve_reduced_lp

                result = solve_reduced_lp(c, A_ub, b_ub, A_eq, b_eq, bounds,
                                          integrality, dict(options or {}))
                if result is not None:
                    return result
            except Exception as exc:
                if not self.fallback_on_error:
                    raise
                fallback_reason = str(exc)
                if self.verbose:
                    print(f"[highs_turbo] Fallback to SciPy: {exc}")
        result = scipy_linprog(
            c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
            method=scipy_method, callback=callback, options=options, x0=x0,
            integrality=integrality,
        )
        result.turbo_accelerated = False
        if fallback_reason is not None:
            result.fallback_reason = fallback_reason
        return result

    def solve_maxcut(self, graph_or_adj: Any, **kwargs: Any) -> MaxCutResult:
        """High-level 1-line Max-Cut solver.

        Accepts GraphInstance, NetworkX Graph, 2D adjacency array, or scipy.sparse matrix.
        Returns MaxCutResult with cut value, partition vector, and SHA-256 cryptographic receipt.
        Supports 3-tuple unpacking: cut_val, partition, cert = solve_maxcut(G).
        """
        t0 = time.perf_counter()
        graph = self.detector._coerce_to_graph_instance(graph_or_adj, np.empty(0))
        n = graph.num_nodes
        m = graph.num_edges

        if n == 0 or m == 0:
            return MaxCutResult(
                cut_value=0.0,
                partition=np.zeros(n, dtype=int),
                certificate="0" * 64,
                is_rationally_certified=True,
                upper_bound=0.0,
                simplex_iterations=0,
                solve_time_ms=(time.perf_counter() - t0) * 1000.0,
            )

        # 1. Compute high-quality integer partition and cut value
        if n <= 200:
            # Solve exact integer Max-Cut via HiGHS MILP
            cut_val, partition = self.exact_solver.solve_integer_maxcut(graph)
        else:
            # Randomized hyperplane rounding + greedy 1-flip descent on relaxation
            partition = self._compute_approximate_partition(graph)
            cut_val = self._evaluate_cut_value(graph, partition)

        # 2. Compute certified dual upper bound and cryptographic SHA-256 certificate
        cbg = CompiledBitGraph.from_graph_instance(graph) if COMPILED_ENGINE_AVAILABLE else None
        c_weights = np.array([-graph.weights.get(e, 1.0) for e in graph.edges], dtype=np.float64)

        # Unconstrained root solve
        x_base = np.where(c_weights < 0.0, 1.0, 0.0)

        # Separate violated cuts
        if cbg is not None and self.cut_engine is not None:
            k5_violated = self.cut_engine.separate_violated_k5(cbg, graph.edges, list(x_base), threshold=1e-4, max_cuts=self.max_cuts)
            tri_violated = self.cut_engine.separate_violated_triangles(cbg, graph.edges, list(x_base), threshold=1e-4, max_cuts=self.max_cuts)
            c4_violated = self.cut_engine.separate_violated_4cycles(cbg, graph.edges, list(x_base), threshold=1e-4, max_cuts=self.max_cuts)

            if k5_violated:
                cuts = k5_violated
            elif tri_violated:
                cuts = tri_violated
            elif c4_violated:
                cuts = c4_violated
            else:
                cuts = self.bench_suite._separate_bipartite_4cycles(graph, x_base, limit=self.max_cuts)
        else:
            cuts = self.bench_suite._separate_bipartite_4cycles(graph, x_base, limit=self.max_cuts)

        cert, a_surr, b_surr = self.bench_suite._certify_and_build_surrogate(graph, cbg, cuts, gnn_model=self.gnn_model)

        # Solve surrogate LP in 0-1 pivots
        if cbg is not None and COMPILED_ENGINE_AVAILABLE:
            bridge = CompiledSolverCallbackBridge(m)
            nz = np.nonzero(a_surr)[0]
            bridge.add_cut_row(b_surr, nz, a_surr[nz], "certified_surrogate")
            A_surr, b_arr = bridge.get_sparse_constraints()
        else:
            A_surr = a_surr.reshape(1, -1)
            b_arr = np.array([b_surr], dtype=np.float64)

        bounds = [(0.0, 1.0)] * m
        res_lp = scipy_linprog(c_weights, A_ub=A_surr, b_ub=b_arr, bounds=bounds, method="highs")
        upper_bound = -float(res_lp.fun) if res_lp.success else float("inf")
        iters = res_lp.nit if hasattr(res_lp, "nit") else 0

        solve_time_ms = (time.perf_counter() - t0) * 1000.0

        return MaxCutResult(
            cut_value=float(cut_val),
            partition=partition,
            certificate=cert.sha256_hash,
            is_rationally_certified=cert.is_valid,
            upper_bound=upper_bound,
            simplex_iterations=iters,
            solve_time_ms=solve_time_ms,
            exact_rational_bound=cert.exact_rhs,
            num_active_supports=cert.num_active_supports,
            status="OPTIMAL" if n <= 200 else "HEURISTIC",
        )

    def solve_qubo(self, Q: Any, **kwargs: Any) -> QuboResult:
        """High-level QUBO / Ising spin glass solver: min x^T Q x for x in {0, 1}^n.

        Accepts 2D numpy array, scipy.sparse matrix, or dictionary {(i, j): val}.
        Returns QuboResult with minimum energy, binary solution vector, and SHA-256 certificate.
        Supports 3-tuple unpacking: energy, sol, cert = solve_qubo(Q).
        """
        t0 = time.perf_counter()

        # Parse and symmetrize Q
        if isinstance(Q, dict):
            max_idx = max((max(i, j) for i, j in Q.keys()), default=-1)
            n = max_idx + 1
            Q_mat = np.zeros((n, n), dtype=np.float64)
            for (i, j), v in Q.items():
                Q_mat[i, j] += float(v)
        elif sp is not None and sp.issparse(Q):
            Q_mat = Q.toarray()
            n = Q_mat.shape[0]
        else:
            Q_mat = np.asarray(Q, dtype=np.float64)
            n = Q_mat.shape[0] if Q_mat.ndim > 0 else 0

        if Q_mat.ndim != 2 or Q_mat.shape[0] != Q_mat.shape[1]:
            raise ValueError("Q must be a square matrix")
        if not np.isfinite(Q_mat).all():
            raise ValueError("Q must contain only finite values")

        Q_sym = 0.5 * (Q_mat + Q_mat.T)
        np.fill_diagonal(Q_sym, np.diag(Q_mat))

        # Solve integer problem
        if n <= 100:
            energy, solution = self._solve_qubo_exact_milp(Q_sym, n)
        else:
            solution = self._solve_qubo_heuristic(Q_sym, n)
            energy = float(solution @ Q_sym @ solution)

        # Derive certified lower bound & SHA-256 proof receipt
        cert = self._certify_qubo_lower_bound(Q_mat, n)
        lower_bound = float(cert.exact_rhs) if cert.is_valid else float("-inf")

        solve_time_ms = (time.perf_counter() - t0) * 1000.0

        return QuboResult(
            energy=energy,
            solution=solution,
            certificate=cert.sha256_hash,
            is_rationally_certified=cert.is_valid,
            lower_bound=lower_bound,
            simplex_iterations=0,
            solve_time_ms=solve_time_ms,
            exact_rational_bound=cert.exact_rhs,
            num_active_supports=cert.num_active_supports,
            status="OPTIMAL" if n <= 18 else "HEURISTIC",
        )

    def verify_candidate_cut(
        self,
        graph: GraphInstance,
        multipliers: Dict[Tuple[int, ...], Union[float, int, Fraction]],
        rhs: Optional[Union[float, Fraction]] = None,
    ) -> VerificationCertificate:
        """Exposes direct rational verification for adversarial stress tests."""
        if self.verifier is not None:
            return self.verifier.verify_clique_conic_combination(graph, multipliers, rhs)
        return self.bench_suite.verifier.verify_clique_conic_combination(graph, multipliers, rhs)

    def _compute_approximate_partition(self, graph: GraphInstance) -> np.ndarray:
        """Fast greedy 1-flip local search for finding cut partitions on large graphs."""
        n = graph.num_nodes
        rng = np.random.default_rng(42)
        s = rng.integers(0, 2, size=n, dtype=int)

        # Build adjacency
        adj: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
        for (u, v), w in graph.weights.items():
            adj[u].append((v, w))
            adj[v].append((u, w))

        # 2 passes of greedy 1-flip descent
        for _ in range(2):
            improved = False
            for u in range(n):
                # Net gain if flipping s[u]
                gain = 0.0
                curr_spin = s[u]
                for v, w in adj[u]:
                    if s[v] == curr_spin:
                        gain += w
                    else:
                        gain -= w
                if gain > 1e-6:
                    s[u] = 1 - curr_spin
                    improved = True
            if not improved:
                break
        return s

    def _evaluate_cut_value(self, graph: GraphInstance, partition: np.ndarray) -> float:
        cut = 0.0
        for (u, v), w in graph.weights.items():
            if partition[u] != partition[v]:
                cut += w
        return cut

    def _solve_qubo_exact_milp(self, Q: np.ndarray, n: int) -> Tuple[float, np.ndarray]:
        """Solves exact QUBO via exact bit enumeration (n <= 18) or multi-start local search."""
        if n <= 18:
            num_configs = 1 << n
            best_energy = float("inf")
            best_x = np.zeros(n, dtype=int)
            for mask in range(num_configs):
                x_vec = np.array([(mask >> k) & 1 for k in range(n)], dtype=np.float64)
                e = float(x_vec @ Q @ x_vec)
                if e < best_energy:
                    best_energy = e
                    best_x = x_vec.astype(int)
            return best_energy, best_x

        # For larger n, use multi-start local search
        best_energy = float("inf")
        best_x = np.zeros(n, dtype=int)
        for seed in range(20):
            rng = np.random.default_rng(seed)
            x_cand = rng.integers(0, 2, size=n, dtype=int)
            for _ in range(10):
                improved = False
                for i in range(n):
                    curr = x_cand[i]
                    new = 1 - curr
                    delta = (new - curr) * (Q[i, i] + 2.0 * np.dot(Q[i], x_cand) - 2.0 * Q[i, i] * curr)
                    if delta < -1e-6:
                        x_cand[i] = new
                        improved = True
                if not improved:
                    break
            e = float(x_cand @ Q @ x_cand)
            if e < best_energy:
                best_energy = e
                best_x = x_cand.copy()

        return best_energy, best_x

    def _solve_qubo_heuristic(self, Q: np.ndarray, n: int) -> np.ndarray:
        """Fast 1-flip local search for finding low-energy QUBO states on large matrices."""
        rng = np.random.default_rng(42)
        x = rng.integers(0, 2, size=n, dtype=int)
        for _ in range(10):
            improved = False
            for i in range(n):
                curr = x[i]
                new = 1 - curr
                delta = (new - curr) * (Q[i, i] + 2.0 * np.dot(Q[i], x) - 2.0 * Q[i, i] * curr)
                if delta < -1e-6:
                    x[i] = new
                    improved = True
            if not improved:
                break
        return x

    def _certify_qubo_lower_bound(self, Q: np.ndarray, n: int) -> VerificationCertificate:
        """Bound each binary monomial using its exact input coefficient.

        x^T Q x = sum Q_ii x_i + sum_{i<j} (Q_ij + Q_ji) x_i x_j.
        Each monomial is in {0, 1}, so its contribution is at least
        min(0, coefficient). Do not round coefficients toward zero.
        """
        coefficients = {}
        for i in range(n):
            coefficients[(i, i)] = Fraction.from_float(float(Q[i, i]))
            for j in range(i + 1, n):
                coefficients[(i, j)] = (
                    Fraction.from_float(float(Q[i, j]))
                    + Fraction.from_float(float(Q[j, i]))
                )
        coefficients = {key: value for key, value in coefficients.items() if value}
        trivial_lb = sum((min(Fraction(0), value) for value in coefficients.values()), Fraction(0))
        cert = VerificationCertificate(
            is_valid=True,
            status="CERTIFIED_DIAGONAL_BOUND" if all(i == j for i, j in coefficients) else "CERTIFIED_TRIVIAL_LOWER_BOUND",
            rejection_reason=None,
            num_active_supports=0,
            exact_coefficients=coefficients,
            exact_rhs=trivial_lb,
            sha256_hash="0" * 64,
        )
        cert.compute_sha256()
        return cert


# Global default solver instance
_DEFAULT_SOLVER = TurboSolver()


def linprog(
    c: Sequence[float],
    A_ub: Optional[Any] = None,
    b_ub: Optional[Sequence[float]] = None,
    A_eq: Optional[Any] = None,
    b_eq: Optional[Sequence[float]] = None,
    bounds: Optional[Any] = (0, None),
    method: str = "highs",
    callback: Optional[Callable] = None,
    options: Optional[Dict[str, Any]] = None,
    x0: Optional[Sequence[float]] = None,
    integrality: Optional[Sequence[int]] = None,
    **kwargs: Any,
) -> OptimizeResult:
    """Exact 3-line drop-in replacement for scipy.optimize.linprog with Neural-Surrogate acceleration."""
    return _DEFAULT_SOLVER.linprog(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method=method,
        callback=callback,
        options=options,
        x0=x0,
        integrality=integrality,
        **kwargs,
    )


def solve_maxcut(graph_or_adj: Any, **kwargs: Any) -> MaxCutResult:
    """High-level 1-line Max-Cut solver returning cut value, partition vector, and SHA-256 certificate."""
    return _DEFAULT_SOLVER.solve_maxcut(graph_or_adj, **kwargs)


def solve_qubo(Q: Any, **kwargs: Any) -> QuboResult:
    """High-level 1-line QUBO / Ising spin glass solver returning minimum energy, state, and certificate."""
    return _DEFAULT_SOLVER.solve_qubo(Q, **kwargs)
