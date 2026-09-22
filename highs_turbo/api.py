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

import time
from dataclasses import dataclass
from fractions import Fraction
from numbers import Real
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import scipy.sparse as sp
except ImportError:
    sp = None

from scipy.optimize import Bounds, OptimizeResult, linprog as scipy_linprog

# Import compiled engine components with graceful fallback
from highs_turbo.compiled_engine import (
    COMPILED_ENGINE_AVAILABLE,
    CompiledBitGraph,
    CompiledCutEngine,
    CompiledRationalVerifier,
)

from highs_turbo.detector import TopologyDetector, TopologyScanResult, detect_topology
from highs_turbo.exact_solver import ExactMaxCutSolver
from highs_turbo.graph_generator import GraphInstance
from highs_turbo.rational_verifier import VerificationCertificate


@dataclass
class MaxCutResult:
    """Result unpackable as (cut_value, partition, certificate).

    The legacy certification flag describes the bound, not exact optimality.
    ``certificate`` is a receipt digest; ``bound_certificate`` is the witness.
    ``to_dict()`` preserves the legacy summary format, without the full witness.
    """

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
    bound_certificate: Any = None

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
    """Result unpackable as (energy, solution, certificate).

    The legacy certification flag describes the bound, not exact optimality.
    ``certificate`` is a receipt digest; ``bound_certificate`` is the witness.
    ``to_dict()`` preserves the legacy summary format, without the full witness.
    """

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
    bound_certificate: Any = None

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
        attempt_started = time.perf_counter()
        if use_turbo and callback is None and x0 is None:
            try:
                if integrality is not None and np.any(integrality):
                    from highs_turbo.ising_linearized import solve_linearized_binary
                    result = solve_linearized_binary(c, A_ub, b_ub, A_eq, b_eq, bounds,
                                                     integrality, dict(options or {}))
                    if result is not None:
                        return result
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
            # Both a declined model and a failed attempt consume preparation
            # time. Ordinary SciPy receives only the remaining caller budget.
            limit = (options or {}).get("time_limit")
            if isinstance(limit, Real) and np.isfinite(limit) and limit >= 0:
                options = {**options, "time_limit": max(0., limit-(time.perf_counter()-attempt_started))}
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
                exact_rational_bound=Fraction(),
                simplex_iterations=0,
                solve_time_ms=(time.perf_counter() - t0) * 1000.0,
            )

        from hashlib import sha256
        import json
        from highs_turbo.ising import solve_ising, _downward

        options = dict(kwargs)
        options.setdefault("time_limit", None if n <= 200 else 5.)
        weights = {edge: Fraction(float(graph.weights.get(edge, 1.))) for edge in graph.edges}
        if options["time_limit"] is not None and options["time_limit"] >= 0:
            options["time_limit"] = max(0., options["time_limit"]-(time.perf_counter()-t0))
        result = solve_ising(dict.fromkeys(range(n), 0), weights, **options)
        partition = np.asarray([(1-result.spins[i])//2 for i in range(n)])
        total = sum(weights.values(), Fraction())
        cut_value = (total-result.exact_energy)/2
        upper = (total-result.exact_cut_lower_bound)/2
        receipt = sha256(json.dumps(result.certificate.to_dict(), sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()
        return MaxCutResult(
            cut_value=float(cut_value), partition=partition, certificate=receipt,
            is_rationally_certified=True, upper_bound=-_downward(-upper),
            solve_time_ms=(time.perf_counter()-t0)*1000,
            exact_rational_bound=upper, num_active_supports=len(result.certificate.cuts),
            status="OPTIMAL" if result.status == "OPTIMAL" else "HEURISTIC",
            bound_certificate=result.certificate,
        )

    def solve_qubo(self, Q: Any, **kwargs: Any) -> QuboResult:
        """High-level QUBO / Ising spin glass solver: min x^T Q x for x in {0, 1}^n.

        Accepts 2D numpy array, scipy.sparse matrix, or dictionary {(i, j): val}.
        Returns QuboResult with minimum energy, binary solution vector, and SHA-256 certificate.
        Supports 3-tuple unpacking: energy, sol, cert = solve_qubo(Q).
        """
        t0 = time.perf_counter()

        # Iterate stored coefficients without densifying sparse inputs. Sparse
        # duplicates and opposite orientations are added as exact fractions;
        # floating-point sparse canonicalization can change the objective.
        if isinstance(Q, dict):
            for key in Q:
                if (not isinstance(key, tuple) or len(key) != 2
                        or any(isinstance(i, (bool, np.bool_))
                               or not isinstance(i, (int, np.integer)) or i < 0
                               for i in key)):
                    raise ValueError("Q dictionary keys must be pairs of nonnegative integer indices")
            n = 1 + max((int(i) for key in Q for i in key), default=-1)
            coefficients = ((int(i), int(j), value) for (i, j), value in Q.items())
        elif sp is not None and sp.issparse(Q):
            if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
                raise ValueError("Q must be a square matrix")
            n = Q.shape[0]
            stored = Q.tocoo(copy=False)
            if np.iscomplexobj(stored.data):
                raise ValueError("Q must contain only finite real values")
            coefficients = zip(stored.row, stored.col, stored.data)
        else:
            if np.iscomplexobj(Q):
                raise ValueError("Q must contain only finite real values")
            Q_mat = np.asarray(Q, dtype=np.float64)
            if Q_mat.ndim != 2 or Q_mat.shape[0] != Q_mat.shape[1]:
                raise ValueError("Q must be a square matrix")
            n = Q_mat.shape[0]
            rows, columns = np.nonzero(Q_mat)
            coefficients = zip(rows, columns, Q_mat[rows, columns])

        from hashlib import sha256
        import json
        from highs_turbo.ising import solve_ising, _downward

        fields = dict.fromkeys(range(n), Fraction())
        constant = Fraction()
        couplings = {}
        for i, j, coefficient in coefficients:
            if np.iscomplexobj(coefficient):
                raise ValueError("Q must contain only finite real values")
            try:
                value = float(coefficient)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("Q must contain only finite real values") from exc
            if not np.isfinite(value):
                raise ValueError("Q must contain only finite real values")
            value = Fraction(value)
            i, j = int(i), int(j)
            if i == j:
                fields[i] += value/2
                constant += value/2
            elif value:
                pair = (min(i, j), max(i, j))
                couplings[pair] = couplings.get(pair, Fraction()) + value/4
        couplings = {pair: value for pair, value in sorted(couplings.items()) if value}
        for (i, j), value in couplings.items():
            fields[i] += value
            fields[j] += value
            constant += value
        options = dict(kwargs)
        options.setdefault("time_limit", None if n <= 18 else 5.)
        if options["time_limit"] is not None and options["time_limit"] >= 0:
            options["time_limit"] = max(0., options["time_limit"]-(time.perf_counter()-t0))
        result = solve_ising(fields, couplings, offset=constant, **options)
        solution = np.asarray([(1+result.spins[i])//2 for i in range(n)], dtype=int)
        receipt = sha256(json.dumps(result.certificate.to_dict(), sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()
        return QuboResult(
            energy=float(result.exact_energy), solution=solution, certificate=receipt,
            is_rationally_certified=True, lower_bound=_downward(result.exact_cut_lower_bound),
            solve_time_ms=(time.perf_counter()-t0)*1000,
            exact_rational_bound=result.exact_cut_lower_bound,
            num_active_supports=len(result.certificate.cuts),
            status="OPTIMAL" if result.status == "OPTIMAL" else "HEURISTIC",
            bound_certificate=result.certificate,
        )

    def solve_ising(self, h, J, **kwargs):
        """Solve an Ising model, returning spins, bounds, and termination status."""
        from highs_turbo.ising import solve_ising

        return solve_ising(h, J, **kwargs)

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
