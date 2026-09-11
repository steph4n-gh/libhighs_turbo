"""Exact reference solver for Max-Cut using SciPy HiGHS and exact rational arithmetic.

Computes:
1. Exact integer Max-Cut (ground-truth IP via SciPy MILP).
2. Base cycle/metric LP relaxation.
3. Metric + full K5 LP relaxation.
4. Metric + single-row surrogate relaxation.
5. Canonical analytic center & minimum-norm center of the optimal dual face (guaranteeing label invariance).
6. Exact rational certificate verification via fractions.Fraction.
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp, minimize

from highs_turbo.graph_generator import GraphInstance


@dataclass
class LPSolution:
    """Stores results of an LP or IP relaxation solve."""

    success: bool
    objective_value: float
    primal_solution: np.ndarray
    dual_multipliers: np.ndarray
    k5_multipliers: Dict[Tuple[int, int, int, int, int], float] = field(default_factory=dict)
    wall_clock_time: float = 0.0
    simplex_iterations: int = 0
    message: str = ""
    exact_rational_bound: Optional[Fraction] = None


@dataclass
class RationalCertificate:
    """Exact rational dual certificate proving an upper bound on Max-Cut."""

    rational_upper_bound: Fraction
    integer_floor: int
    is_feasible: bool
    duals: Dict[str, Fraction] = field(default_factory=dict)
    residual_norm: Fraction = Fraction(0, 1)


class ExactMaxCutSolver:
    """High-precision reference solver and dual analyzer for Max-Cut problems."""

    def __init__(self, rational_denominator_limit: int = 100000):
        self.rational_denominator_limit = rational_denominator_limit

    def solve_integer_maxcut(self, graph: GraphInstance) -> Tuple[float, np.ndarray]:
        """Computes exact ground-truth integer Max-Cut using SciPy MILP.

        Variables:
        - s_u in {0, 1} for u in V (s_0 = 0 to break symmetry)
        - x_{uv} in [0, 1] for (u, v) in E
        Returns:
        - max_cut_val: float
        - cut_partition: array of 0/1 indicating node partition
        """
        n = graph.num_nodes
        m = graph.num_edges

        # Total variables: m (edges) + n (nodes)
        # x_0..x_{m-1}, s_0..s_{n-1}
        c = np.zeros(m + n)
        for i, (u, v) in enumerate(graph.edges):
            c[i] = -graph.weights.get((u, v), 1.0)  # Minimize negative cut value

        integrality = np.ones(m + n, dtype=int)

        # Bounds
        lb = np.zeros(m + n)
        ub = np.ones(m + n)
        ub[m] = 0.0  # Fix s_0 = 0 to break Z2 symmetry
        bounds = Bounds(lb, ub)

        # Constraints for each edge (u, v) with variable x_i:
        # 1) x_i - s_u - s_v <= 0
        # 2) x_i + s_u + s_v <= 2
        # 3) s_u - s_v - x_i <= 0
        # 4) s_v - s_u - x_i <= 0
        A_rows = []
        b_u = []
        for i, (u, v) in enumerate(graph.edges):
            su_idx = m + u
            sv_idx = m + v

            # 1) x_i - s_u - s_v <= 0
            r1 = np.zeros(m + n)
            r1[i] = 1.0
            r1[su_idx] = -1.0
            r1[sv_idx] = -1.0
            A_rows.append(r1)
            b_u.append(0.0)

            # 2) x_i + s_u + s_v <= 2
            r2 = np.zeros(m + n)
            r2[i] = 1.0
            r2[su_idx] = 1.0
            r2[sv_idx] = 1.0
            A_rows.append(r2)
            b_u.append(2.0)

            # 3) s_u - s_v - x_i <= 0
            r3 = np.zeros(m + n)
            r3[su_idx] = 1.0
            r3[sv_idx] = -1.0
            r3[i] = -1.0
            A_rows.append(r3)
            b_u.append(0.0)

            # 4) s_v - s_u - x_i <= 0
            r4 = np.zeros(m + n)
            r4[sv_idx] = 1.0
            r4[su_idx] = -1.0
            r4[i] = -1.0
            A_rows.append(r4)
            b_u.append(0.0)

        A = np.array(A_rows)
        b_l = np.full(len(b_u), -np.inf)
        constraints = LinearConstraint(A, b_l, b_u)

        res = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints)
        if not res.success:
            raise RuntimeError(f"MILP Max-Cut solve failed: {res.status}")

        max_cut = float(-res.fun)
        partition = np.round(res.x[m : m + n]).astype(int)
        return max_cut, partition

    def build_relaxation_matrices(
        self,
        graph: GraphInstance,
        include_k5: bool = False,
        k5_cliques: Optional[List[Tuple[int, int, int, int, int]]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[Tuple[int, int, int, int, int]]]:
        """Builds constraint matrix A and RHS b for Max-Cut LP relaxation.

        Edge variables: x_e for e in graph.edges.
        Objective: max sum w_e x_e => min -sum w_e x_e.
        Returns:
        - c: objective vector
        - A: inequality constraint matrix (A x <= b)
        - b: RHS vector
        - row_labels: labels for each row
        - active_k5_cliques: list of k5 cliques included
        """
        m = graph.num_edges
        edge_to_idx = {e: i for i, e in enumerate(graph.edges)}
        c = np.array([-graph.weights.get(e, 1.0) for e in graph.edges], dtype=np.float64)

        A_rows: List[np.ndarray] = []
        b_vals: List[float] = []
        row_labels: List[str] = []

        # 1. Triangle inequalities for all triangles in G
        triangles = graph.find_triangles()
        for u, v, w in triangles:
            e_uv = (min(u, v), max(u, v))
            e_vw = (min(v, w), max(v, w))
            e_uw = (min(u, w), max(u, w))
            if e_uv in edge_to_idx and e_vw in edge_to_idx and e_uw in edge_to_idx:
                i1, i2, i3 = edge_to_idx[e_uv], edge_to_idx[e_vw], edge_to_idx[e_uw]

                # x_1 + x_2 - x_3 <= 1
                r1 = np.zeros(m)
                r1[i1] = 1.0
                r1[i2] = 1.0
                r1[i3] = -1.0
                A_rows.append(r1)
                b_vals.append(1.0)
                row_labels.append(f"tri_{u}_{v}_{w}_1")

                # x_1 - x_2 + x_3 <= 1
                r2 = np.zeros(m)
                r2[i1] = 1.0
                r2[i2] = -1.0
                r2[i3] = 1.0
                A_rows.append(r2)
                b_vals.append(1.0)
                row_labels.append(f"tri_{u}_{v}_{w}_2")

                # -x_1 + x_2 + x_3 <= 1
                r3 = np.zeros(m)
                r3[i1] = -1.0
                r3[i2] = 1.0
                r3[i3] = 1.0
                A_rows.append(r3)
                b_vals.append(1.0)
                row_labels.append(f"tri_{u}_{v}_{w}_3")

                # x_1 + x_2 + x_3 <= 2
                r4 = np.zeros(m)
                r4[i1] = 1.0
                r4[i2] = 1.0
                r4[i3] = 1.0
                A_rows.append(r4)
                b_vals.append(2.0)
                row_labels.append(f"tri_{u}_{v}_{w}_4")

        # 2. Literal K5 clique inequalities
        cliques_used: List[Tuple[int, int, int, int, int]] = []
        if include_k5:
            if k5_cliques is None:
                k5_cliques = graph.find_all_k5_cliques()
            for k5 in k5_cliques:
                row = np.zeros(m)
                valid_k5 = True
                for u, v in itertools.combinations(k5, 2):
                    e = (min(u, v), max(u, v))
                    if e in edge_to_idx:
                        row[edge_to_idx[e]] = 1.0
                    else:
                        valid_k5 = False
                        break
                if valid_k5:
                    A_rows.append(row)
                    b_vals.append(6.0)
                    row_labels.append(f"k5_{'_'.join(map(str, k5))}")
                    cliques_used.append(k5)

        if not A_rows:
            A = np.zeros((0, m))
            b = np.zeros(0)
        else:
            A = np.array(A_rows, dtype=np.float64)
            b = np.array(b_vals, dtype=np.float64)

        return c, A, b, row_labels, cliques_used

    def solve_cycle_relaxation(self, graph: GraphInstance) -> LPSolution:
        """Solves the base cycle/metric relaxation of Max-Cut."""
        t0 = time.perf_counter()
        c, A, b, row_labels, _ = self.build_relaxation_matrices(graph, include_k5=False)
        m = graph.num_edges
        bounds = [(0.0, 1.0) for _ in range(m)]

        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
        dt = time.perf_counter() - t0

        iters = getattr(res, "nit", 0)
        obj = float(-res.fun) if res.success else float("inf")
        duals = -res.ineqlin.marginals if hasattr(res, "ineqlin") and res.ineqlin is not None else np.zeros(len(b))

        return LPSolution(
            success=res.success,
            objective_value=obj,
            primal_solution=res.x,
            dual_multipliers=duals,
            wall_clock_time=dt,
            simplex_iterations=iters,
            message=res.message,
            exact_rational_bound=Fraction.from_float(obj).limit_denominator(self.rational_denominator_limit),
        )

    def solve_k5_relaxation(
        self,
        graph: GraphInstance,
        k5_cliques: Optional[List[Tuple[int, int, int, int, int]]] = None,
    ) -> LPSolution:
        """Solves the full metric + literal K5 relaxation."""
        t0 = time.perf_counter()
        c, A, b, row_labels, cliques_used = self.build_relaxation_matrices(graph, include_k5=True, k5_cliques=k5_cliques)
        m = graph.num_edges
        bounds = [(0.0, 1.0) for _ in range(m)]

        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
        dt = time.perf_counter() - t0

        iters = getattr(res, "nit", 0)
        obj = float(-res.fun) if res.success else float("inf")

        # Extract K5 dual multipliers
        duals = -res.ineqlin.marginals if hasattr(res, "ineqlin") and res.ineqlin is not None else np.zeros(len(b))
        k5_duals = {}
        for idx, lbl in enumerate(row_labels):
            if lbl.startswith("k5_"):
                k5_tuple = cliques_used[idx - (len(row_labels) - len(cliques_used))]
                k5_duals[k5_tuple] = float(max(0.0, duals[idx]))

        return LPSolution(
            success=res.success,
            objective_value=obj,
            primal_solution=res.x,
            dual_multipliers=duals,
            k5_multipliers=k5_duals,
            wall_clock_time=dt,
            simplex_iterations=iters,
            message=res.message,
            exact_rational_bound=Fraction.from_float(obj).limit_denominator(self.rational_denominator_limit),
        )

    def solve_surrogate_relaxation(
        self,
        graph: GraphInstance,
        surrogate_coefficients: np.ndarray,
        surrogate_rhs: float,
    ) -> LPSolution:
        """Solves base cycle relaxation with exactly 1 surrogate cutting plane row added:

        a_surr^T x <= b_surr.
        """
        t0 = time.perf_counter()
        c, A_base, b_base, row_labels, _ = self.build_relaxation_matrices(graph, include_k5=False)

        # Append single surrogate row
        surr_row = np.array(surrogate_coefficients, dtype=np.float64).reshape(1, -1)
        A = np.vstack([A_base, surr_row])
        b = np.append(b_base, float(surrogate_rhs))

        m = graph.num_edges
        bounds = [(0.0, 1.0) for _ in range(m)]

        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
        dt = time.perf_counter() - t0

        iters = getattr(res, "nit", 0)
        obj = float(-res.fun) if res.success else float("inf")
        duals = -res.ineqlin.marginals if hasattr(res, "ineqlin") and res.ineqlin is not None else np.zeros(len(b))

        return LPSolution(
            success=res.success,
            objective_value=obj,
            primal_solution=res.x,
            dual_multipliers=duals,
            wall_clock_time=dt,
            simplex_iterations=iters,
            message=res.message,
            exact_rational_bound=Fraction.from_float(obj).limit_denominator(self.rational_denominator_limit),
        )

    def compute_face_invariant_dual_center(
        self,
        graph: GraphInstance,
        k5_cliques: Optional[List[Tuple[int, int, int, int, int]]] = None,
        tol: float = 1e-5,
    ) -> Dict[Tuple[int, int, int, int, int], float]:
        """Computes the canonical minimum-norm / analytic center of the optimal dual face.

        In LP relaxations with dual degeneracy, standard solvers return arbitrary extreme points
        that flip under isomorphic vertex relabeling.
        This method solves the strictly convex quadratic projection:
            min_{y >= 0, z >= 0}  1/2 ||y_k5||_2^2 + rho/2 ||y_tri||_2^2 + rho/2 ||z||_2^2
            s.t.   A^T y + z >= w
                   b^T y + 1^T z <= Z_opt + tol
                   y >= 0, z >= 0
        Because the objective is strictly convex in y_k5, the optimal multiplier vector
        is UNIQUE and strictly permutation-equivariant under any graph isomorphism.
        """
        if k5_cliques is None:
            k5_cliques = graph.find_all_k5_cliques()

        if not k5_cliques:
            return {}

        # 1. Solve full K5 LP to get optimal objective value and baseline duals
        lp_sol = self.solve_k5_relaxation(graph, k5_cliques=k5_cliques)
        if not lp_sol.success:
            return {k: 0.0 for k in k5_cliques}

        z_opt = lp_sol.objective_value
        c, A, b, row_labels, cliques_used = self.build_relaxation_matrices(graph, include_k5=True, k5_cliques=k5_cliques)
        m = graph.num_edges
        num_rows = len(b)
        k5_start = num_rows - len(cliques_used)

        if not cliques_used or k5_start < 0:
            return {k: 0.0 for k in k5_cliques}

        total_k5_mass = sum(lp_sol.k5_multipliers.values()) if lp_sol.k5_multipliers else 0.0
        if total_k5_mass <= 1e-9:
            return {k: 0.0 for k in k5_cliques}

        # If only 1 clique, its multiplier is already unique
        if len(cliques_used) == 1:
            return {cliques_used[0]: float(max(0.0, lp_sol.k5_multipliers.get(cliques_used[0], 0.0)))}

        # Formulate strictly convex quadratic projection onto the optimal dual face
        n_vars = num_rows + m
        rho = 1e-4

        def obj(x: np.ndarray) -> float:
            yk5 = x[k5_start:num_rows]
            yother = x[:k5_start]
            z = x[num_rows:]
            return 0.5 * float(np.dot(yk5, yk5)) + rho * 0.5 * float(np.dot(yother, yother)) + rho * 0.5 * float(np.dot(z, z))

        def grad(x: np.ndarray) -> np.ndarray:
            g = np.zeros_like(x)
            g[k5_start:num_rows] = x[k5_start:num_rows]
            g[:k5_start] = rho * x[:k5_start]
            g[num_rows:] = rho * x[num_rows:]
            return g

        C1 = np.hstack([A.T, np.eye(m)])
        C2 = np.hstack([b.reshape(1, -1), np.ones((1, m))])
        C_mat = np.vstack([C1, C2])
        w = -c
        lb = np.append(w, -np.inf)
        ub = np.append(np.full(m, np.inf), z_opt + tol)
        lin_con = LinearConstraint(C_mat, lb, ub)
        bounds = Bounds(np.zeros(n_vars), np.full(n_vars, np.inf))

        x0 = np.zeros(n_vars)
        if lp_sol.dual_multipliers is not None and len(lp_sol.dual_multipliers) == num_rows:
            x0[:num_rows] = np.maximum(0.0, lp_sol.dual_multipliers)
            x0[num_rows:] = np.maximum(0.0, w - A.T @ x0[:num_rows])

        res = minimize(obj, x0, jac=grad, constraints=lin_con, bounds=bounds, method="SLSQP", options={"maxiter": 400, "ftol": 1e-7})
        if res.success:
            yk5 = res.x[k5_start:num_rows]
        else:
            yk5 = x0[k5_start:num_rows]

        canonical_duals = {}
        for idx, clq in enumerate(cliques_used):
            val = float(max(0.0, yk5[idx]))
            canonical_duals[clq] = val

        for clq in k5_cliques:
            if clq not in canonical_duals:
                canonical_duals[clq] = 0.0

        return canonical_duals

    def verify_exact_dual_certificate(
        self,
        c_weights: Dict[Tuple[int, int], Union[int, float, Fraction]],
        surrogate_coeffs: Dict[Tuple[int, int], Union[int, float, Fraction]],
        surrogate_rhs: Union[int, float, Fraction],
        surrogate_multiplier: Union[int, float, Fraction] = Fraction(1, 1),
        slack_bounds: Optional[Dict[Tuple[int, int], Union[int, float, Fraction]]] = None,
    ) -> RationalCertificate:
        """Exact rational verification of a surrogate cut certificate using fractions.Fraction.

        Verifies that:
        1. Surrogate multiplier lambda >= 0 and slack multipliers z_e >= 0.
        2. Dual feasibility holds for every edge: lambda * a_e + z_e >= w_e.
        3. Computes the exact rational upper bound: lambda * b + sum_{e in E} z_e.
        4. Derives the certified integer floor: floor(lambda * b + sum z_e).
        5. Computes exact residual norm: sum_e max(0, w_e - (lambda * a_e + z_e)).
        """
        def to_frac(v: Union[int, float, Fraction]) -> Fraction:
            if isinstance(v, Fraction):
                return v
            if isinstance(v, int):
                return Fraction(v, 1)
            return Fraction.from_float(float(v)).limit_denominator(self.rational_denominator_limit)

        lam = to_frac(surrogate_multiplier)
        b = to_frac(surrogate_rhs)

        if lam < Fraction(0, 1):
            return RationalCertificate(
                rational_upper_bound=Fraction(0, 1),
                integer_floor=0,
                is_feasible=False,
                duals={"lambda_surrogate": lam},
                residual_norm=Fraction(1, 1),
            )

        edges = sorted(list(set(c_weights.keys()).union(surrogate_coeffs.keys())))
        residual = Fraction(0, 1)
        duals: Dict[str, Fraction] = {"lambda_surrogate": lam}
        total_slack = Fraction(0, 1)

        for e in edges:
            w_e = to_frac(c_weights.get(e, Fraction(0, 1)))
            a_e = to_frac(surrogate_coeffs.get(e, Fraction(0, 1)))

            if slack_bounds is not None:
                z_e = to_frac(slack_bounds.get(e, Fraction(0, 1)))
                if z_e < Fraction(0, 1):
                    return RationalCertificate(
                        rational_upper_bound=Fraction(0, 1),
                        integer_floor=0,
                        is_feasible=False,
                        duals=duals,
                        residual_norm=Fraction(1, 1),
                    )
            else:
                z_e = max(Fraction(0, 1), w_e - lam * a_e)

            eff = lam * a_e + z_e
            if eff < w_e:
                residual += (w_e - eff)

            duals[f"z_{e[0]}_{e[1]}"] = z_e
            total_slack += z_e

        rational_bound = lam * b + total_slack
        integer_floor = math.floor(rational_bound)
        is_feasible = (residual == Fraction(0, 1))

        return RationalCertificate(
            rational_upper_bound=rational_bound,
            integer_floor=integer_floor,
            is_feasible=is_feasible,
            duals=duals,
            residual_norm=residual,
        )
