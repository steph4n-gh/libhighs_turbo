"""Use verified Ising bounds inside exact binary-product linearizations.

All three McCormick rows must be present with their exact coefficients.
Additional user constraints remain in the original HiGHS model. The Ising
problem relaxes those constraints, so its lower bound remains valid for them.
"""

from fractions import Fraction
import math
import time

import highspy
import numpy as np
import scipy.sparse as sp
from scipy.optimize import Bounds, OptimizeResult

from highs_turbo.ising import solve_ising, verify_ising_certificate, _downward
from highs_turbo.lp_accelerator import _add_rows, _check_status


def binary_products(matrix, rhs, integrality):
    """Recognize y=x_u*x_v only from complete, exact envelope rows."""
    upper, lower = {}, []
    for row in range(matrix.shape[0]):
        begin, end = matrix.indptr[row : row + 2]
        indices, values = matrix.indices[begin:end], matrix.data[begin:end]
        if end - begin == 2 and rhs[row] == 0 and set(values) == {-1.0, 1.0}:
            y, x = int(indices[values == 1][0]), int(indices[values == -1][0])
            upper.setdefault(y, {}).setdefault(x, set()).add(row)
        elif end - begin == 3 and rhs[row] == 1 and sorted(values) == [-1.0, 1.0, 1.0]:
            y = int(indices[values == -1][0])
            nodes = tuple(map(int, indices[values == 1]))
            lower.append((row, y, nodes))
    products, covered = {}, set()
    for row, y, nodes in lower:
        if set(nodes) <= upper.get(y, {}).keys():
            if y in products and products[y] != nodes:
                return None
            products[y] = nodes
            covered.add(row)
            for node in nodes:
                covered.update(upper[y][node])
    nodes = sorted(set(range(matrix.shape[1])) - products.keys())
    node_set = set(nodes)
    if (
        not products
        or not 32 <= len(nodes) <= 2047
        or any(integrality[i] != 1 for i in nodes)
        or any(u not in node_set or v not in node_set for u, v in products.values())
    ):
        return None
    return nodes, products, len(covered) == matrix.shape[0]


def ising_objective(c, nodes, products):
    """Expand x=(1+s)/2 using exact input coefficients, without rounding sums."""
    fields = {i: Fraction(float(c[i])) / 2 for i in nodes}
    constant = sum(fields.values(), Fraction())
    couplings = {}
    for y, (u, v) in products.items():
        value = Fraction(float(c[y])) / 4
        couplings[u, v] = couplings.get((u, v), Fraction()) + value
        fields[u] += value
        fields[v] += value
        constant += value
    return fields, couplings, constant


def verify_linprog_certificate(c, certificate, *, A_ub, b_ub, bounds, integrality):
    """Verify a binary-product objective bound without running an optimizer.

    Additional inequalities or equalities only restrict the feasible set and
    do not invalidate this bound. Complete product envelopes remain required.
    """
    try:
        c = np.asarray(c, dtype=float)
        if c.ndim != 1 or not c.size:
            return False
        if sp.issparse(A_ub) and not getattr(A_ub, "has_canonical_format", False):
            return False
        matrix = sp.csr_matrix(A_ub, dtype=float, copy=True)
        matrix.eliminate_zeros()
        matrix.sort_indices()
        rhs = np.asarray(b_ub, dtype=float)
        types = np.broadcast_to(np.asarray(integrality), c.shape)
        limits = (
            np.column_stack(
                (
                    np.broadcast_to(bounds.lb, c.shape),
                    np.broadcast_to(bounds.ub, c.shape),
                )
            )
            if isinstance(bounds, Bounds)
            else np.broadcast_to(np.asarray(bounds, dtype=float), (len(c), 2))
        )
        if (
            matrix.shape[1] != len(c)
            or rhs.shape != (matrix.shape[0],)
            or not np.isin(types, [0, 1]).all()
            or np.any(limits[:, 0] != 0)
            or np.any(limits[:, 1] != 1)
            or not all(np.isfinite(a).all() for a in (c, matrix.data, rhs))
        ):
            return False
        recognized = binary_products(matrix, rhs, types)
        if recognized is None:
            return False
        nodes, products, _ = recognized
        fields, couplings, constant = ising_objective(c, nodes, products)
        return verify_ising_certificate(fields, couplings, certificate, offset=constant)
    except (TypeError, ValueError, IndexError, OverflowError):
        return False


def solve_linearized_binary(c, A_ub, b_ub, A_eq, b_eq, bounds, integrality, options):
    """Return a SciPy-style result, or decline before doing any solver work."""
    began = time.perf_counter()
    if integrality is None or not np.any(integrality):
        return None
    allowed = {"time_limit", "mip_rel_gap", "presolve", "disp"}
    if (
        set(options) - allowed
        or options.get("disp", False)
        or options.get("presolve", True) is not True
    ):
        return None
    c = np.asarray(c, dtype=float)
    if c.ndim != 1 or not c.size or A_ub is None or b_ub is None:
        return None
    count = len(c)
    types = np.broadcast_to(np.asarray(integrality), (count,))
    limits = np.asarray(bounds, dtype=float)
    if limits.shape == (2,):
        limits = np.broadcast_to(limits, (count, 2))
    if (
        limits.shape != (count, 2)
        or np.any(limits[:, 0] != 0)
        or np.any(limits[:, 1] != 1)
        or not np.isin(types, [0, 1]).all()
    ):
        return None
    # Combining duplicate sparse entries in floating point could turn an
    # inexact envelope into an apparently exact one. Let SciPy handle them.
    if sp.issparse(A_ub) and not getattr(A_ub, "has_canonical_format", False):
        return None
    matrix = sp.csr_matrix(A_ub, dtype=float, copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    rhs = np.asarray(b_ub, dtype=float)
    equality = (
        sp.csr_matrix((0, count)) if A_eq is None else sp.csr_matrix(A_eq, dtype=float)
    )
    eq_rhs = np.empty(0) if b_eq is None else np.asarray(b_eq, dtype=float)
    if (
        matrix.shape[1] != count
        or rhs.shape != (matrix.shape[0],)
        or equality.shape[1] != count
        or eq_rhs.shape != (equality.shape[0],)
        or not all(
            np.isfinite(a).all() for a in (c, matrix.data, rhs, equality.data, eq_rhs)
        )
    ):
        return None
    recognized = binary_products(matrix, rhs, types)
    if recognized is None:
        return None
    nodes, products, only_products = recognized
    only_products = only_products and equality.shape[0] == 0
    seconds = options.get("time_limit", math.inf)
    gap = options.get("mip_rel_gap", 0.0001)
    if (
        not isinstance(seconds, (int, float))
        or seconds <= 0
        or math.isnan(seconds)
        or not isinstance(gap, (int, float))
        or not math.isfinite(gap)
        or gap < 0
    ):
        return None
    fields, couplings, constant = ising_objective(c, nodes, products)
    preparation = (
        (
            None
            if math.isinf(seconds)
            else max(0.0, seconds - (time.perf_counter() - began))
        )
        if only_products
        else min(5.0, 0.4 * seconds)
    )
    result = solve_ising(
        fields,
        couplings,
        offset=constant,
        time_limit=preparation,
        relative_gap=float(gap) if only_products else 0.0,
    )
    candidate = np.zeros(count)
    for node in nodes:
        candidate[node] = (1 + result.spins[node]) / 2
    for y, (u, v) in products.items():
        candidate[y] = candidate[u] * candidate[v]
    # A start is installed only when it satisfies every additional constraint.
    feasible = np.all(matrix @ candidate <= rhs) and np.array_equal(
        equality @ candidate, eq_rhs
    )
    lower_bound = _downward(result.exact_cut_lower_bound)
    if only_products:
        # The complete row set defines a bijection with the spin problem.
        # Check its lifted solution against the original matrix before use.
        if not feasible:
            raise RuntimeError(
                "The lifted Ising solution violates the original product model"
            )
        x = candidate
        status = (
            0
            if result.status in ("OPTIMAL", "GAP_LIMIT")
            else 1
            if result.status in ("TIME_LIMIT", "GAP_LIMIT")
            else 4
        )
        model_status = (
            highspy.HighsModelStatus.kOptimal
            if status == 0
            else highspy.HighsModelStatus.kTimeLimit
            if status == 1
            else highspy.HighsModelStatus.kSolveError
        )
        dual_bound = max(lower_bound, result.lower_bound)
        node_count = result.node_count
        gap_value = result.relative_gap
        description = result.message
    else:
        session = highspy.Highs()
        for key, value in {
            "output_flag": False,
            "parallel": "off",
            "mip_rel_gap": float(gap),
        }.items():
            _check_status(session.setOptionValue(key, value))
        _check_status(
            session.addCols(count, c, np.zeros(count), np.ones(count), 0, [], [], [])
        )
        integer = np.flatnonzero(types)
        _check_status(
            session.changeColsIntegrality(
                len(integer), integer, np.ones(len(integer), dtype=np.uint8)
            )
        )
        _add_rows(session, matrix, np.full(len(rhs), -np.inf), rhs)
        _add_rows(session, equality, eq_rhs, eq_rhs)
        _add_rows(session, sp.csr_matrix(c.reshape(1, -1)), [lower_bound], [np.inf])
        if feasible:
            _check_status(session.setSolution(count, np.arange(count), candidate))
        _check_status(
            session.setOptionValue(
                "time_limit", max(0.0, seconds - (time.perf_counter() - began))
            )
        )
        _check_status(session.run())
        solution, info = session.getSolution(), session.getInfo()
        model_status = session.getModelStatus()
        status = {
            highspy.HighsModelStatus.kOptimal: 0,
            highspy.HighsModelStatus.kTimeLimit: 1,
            highspy.HighsModelStatus.kIterationLimit: 1,
            highspy.HighsModelStatus.kInfeasible: 2,
            highspy.HighsModelStatus.kUnbounded: 3,
        }.get(model_status, 4)
        x = np.asarray(solution.col_value) if solution.value_valid else None
        dual_bound = (
            max(lower_bound, info.mip_dual_bound) if info.valid else lower_bound
        )
        node_count = max(0, info.mip_node_count)
        gap_value = info.mip_gap
        description = session.modelStatusToString(model_status)
    slack = None if x is None else rhs - matrix @ x
    con = None if x is None else eq_rhs - equality @ x
    objective = None if x is None else float(c @ x)
    if objective is not None:
        dual_bound = min(dual_bound, objective)
        gap_value = (
            (objective - dual_bound) / abs(objective)
            if objective
            else 0.0
            if dual_bound == 0
            else math.inf
        )
    message = {
        0: "Optimization terminated successfully.",
        1: "Time limit reached.",
        2: "The problem is infeasible.",
        3: "The problem is unbounded.",
        4: "The HiGHS solver did not return an optimal solution.",
    }[status]
    message += f" (HiGHS Status {int(model_status)}: {description})"
    return OptimizeResult(
        x=x,
        fun=objective,
        success=status == 0,
        status=status,
        message=message,
        nit=-1,
        crossover_nit=-1,
        slack=slack,
        con=con,
        ineqlin=OptimizeResult(
            residual=slack, marginals=None if x is None else np.zeros(len(rhs))
        ),
        eqlin=OptimizeResult(
            residual=con, marginals=None if x is None else np.zeros(len(eq_rhs))
        ),
        lower=OptimizeResult(
            residual=x, marginals=None if x is None else np.zeros(count)
        ),
        upper=OptimizeResult(
            residual=None if x is None else 1 - x,
            marginals=None if x is None else np.zeros(count),
        ),
        mip_node_count=node_count,
        mip_dual_bound=dual_bound,
        mip_gap=gap_value,
        turbo_accelerated=True,
        turbo_strategy="ising_certificate",
        turbo_exact_lower_bound=result.exact_cut_lower_bound,
        turbo_bound_certificate=result.certificate,
        turbo_preparation_seconds=result.solve_time,
        turbo_equivalent_model=only_products,
    )
