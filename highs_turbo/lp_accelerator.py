"""Solve a smaller relaxation, restoring original rows with a retained HiGHS basis.

Every working model contains a subset of the original inequalities and, when
exact integer aggregation is possible, nonnegative sums of original rows.
An optimal working solution that satisfies the original model is therefore
optimal for that model too. No graph interpretation changes the feasible set.
"""

import highspy
import numpy as np
import scipy.sparse as sp
from scipy.optimize import OptimizeResult


def _check_status(status):
    if status == highspy.HighsStatus.kError:
        raise RuntimeError("HiGHS rejected the model or solver operation")


def _add_rows(session, matrix, lower, upper):
    if matrix.shape[0]:
        _check_status(session.addRows(matrix.shape[0], lower, upper, matrix.nnz,
                                      matrix.indptr, matrix.indices, matrix.data))


def _integer_surrogates(matrix, rhs):
    """Aggregate small integer rows without any floating-point summation error."""
    # Every intermediate integer sum is exactly representable in binary64.
    if (np.any(np.abs(matrix.data) > 2**30) or np.any(np.abs(rhs) > 2**30)
            or not np.equal(matrix.data, np.trunc(matrix.data)).all()
            or not np.equal(rhs, np.trunc(rhs)).all()
            or np.abs(matrix.data).sum() > 2**50 or np.abs(rhs).sum() > 2**50):
        return None
    rows, cols = matrix.shape
    nonempty = np.diff(matrix.indptr) != 0
    first_col = np.zeros(rows, dtype=np.int64)
    first_col[nonempty] = matrix.indices[matrix.indptr[:-1][nonempty]]
    blocks = min(16, max(1, cols // 16))
    # Keep rows with opposite orientations from cancelling one another.
    row_sum = np.asarray(matrix.sum(axis=1)).ravel()
    groups = 2 * (first_col * blocks // cols) + (row_sum < 0)
    _, groups = np.unique(groups, return_inverse=True)
    weights = sp.csr_matrix((np.ones(rows), (groups, np.arange(rows))))
    return weights, (weights @ matrix).tocsr(), np.asarray(weights @ rhs)


def _conic_solution(c, matrix, rhs, equality, eq_rhs, lower, upper, selected, tolerance):
    """Close the primal-dual gap directly when a conic row combination permits it.

    For example, uniform triangle relaxations have x_e=2/3 and a uniform
    nonnegative combination of triangle rows matching the objective. Validate
    both sides against the actual input, so a near match is never assumed.
    """
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        return None
    support = matrix[selected]
    if not selected.size or np.any(support.data < 0):
        return None
    coverage = np.asarray(support.sum(axis=0)).ravel()
    if not np.isfinite(coverage).all():
        return None
    touched = coverage > 0
    if np.any(c[touched] >= 0):
        return None
    corner = np.where(c < 0, upper, lower)
    origin = np.where(touched, lower, corner)
    direction = corner - origin
    change = support @ direction
    moving = change > 0
    if not moving.any():
        return None
    fraction = np.min((rhs[selected] - support @ origin)[moving] / change[moving])
    if not 0 <= fraction <= 1:
        return None
    x = origin + fraction * direction
    multiplier = np.min(-c[touched] / coverage[touched])
    marginal = np.zeros(len(rhs))
    marginal[selected] = -multiplier
    reduced = c + multiplier * coverage
    lower_dual = np.where(x == lower, np.maximum(reduced, 0), 0)
    upper_dual = np.where(x == upper, np.minimum(reduced, 0), 0)
    slack = rhs - matrix @ x
    con = eq_rhs - equality @ x
    primal = float(c @ x)
    dual = float(rhs @ marginal + lower @ lower_dual + upper @ upper_dual)
    if (not all(np.isfinite(a).all() for a in (x, slack, con, reduced))
            or not np.isfinite(primal) or not np.isfinite(dual)
            or np.any(slack < -tolerance) or np.any(np.abs(con) > tolerance)
            or np.any(np.abs(reduced - lower_dual - upper_dual) > tolerance)
            or np.any(np.abs(slack * marginal) > tolerance)
            or abs(primal - dual) > tolerance):
        return None
    return OptimizeResult(
        x=x, fun=primal, success=True, status=0, nit=0, crossover_nit=0,
        message="Optimization terminated successfully. (Primal-dual optimality verified)",
        slack=slack, con=con,
        ineqlin=OptimizeResult(residual=slack, marginals=marginal),
        eqlin=OptimizeResult(residual=con, marginals=np.zeros(len(eq_rhs))),
        lower=OptimizeResult(residual=x - lower, marginals=lower_dual),
        upper=OptimizeResult(residual=upper - x, marginals=upper_dual),
        turbo_accelerated=True, turbo_strategy="conic_certificate", turbo_rounds=0,
        turbo_initial_rows=len(selected), turbo_original_rows=len(rhs),
        turbo_rows_used=len(selected), turbo_surrogate_rows=1,
    )


def solve_reduced_lp(c, A_ub, b_ub, A_eq, b_eq, bounds, integrality, options):
    """Return a solution of the original model, or None to use ordinary SciPy.

    Calls with explicit solve budgets or unfamiliar options stay with SciPy:
    several related solves must not reset a caller's time/iteration/node limit.
    """
    if A_ub is None:
        return None
    if integrality is not None and np.any(np.asarray(integrality)):
        return None
    num_rows = A_ub.shape[0] if hasattr(A_ub, "shape") else len(A_ub)
    if num_rows < 512:
        return None
    supported = {"presolve", "primal_feasibility_tolerance", "dual_feasibility_tolerance",
                 "simplex_dual_edge_weight_strategy", "mip_rel_gap", "disp"}
    if set(options) - supported or options.get("disp", False):
        return None

    c = np.asarray(c, dtype=float)
    if c.ndim != 1 or not c.size:
        return None
    matrix = sp.csr_matrix(A_ub, dtype=float)
    rows, cols = matrix.shape
    if cols != c.size or rows < max(512, 2 * cols):
        return None
    if not matrix.has_canonical_format:
        matrix = matrix.copy()
        matrix.sum_duplicates()
    rhs = np.asarray(b_ub, dtype=float)
    if A_eq is not None and np.ndim(A_eq) != 2:
        return None
    equality = sp.csr_matrix((0, cols)) if A_eq is None else sp.csr_matrix(A_eq, dtype=float)
    eq_rhs = np.empty(0) if b_eq is None else np.asarray(b_eq, dtype=float)
    if (rhs.shape != (rows,) or equality.shape[1] != cols
            or eq_rhs.shape != (equality.shape[0],)):
        return None
    if not all(np.isfinite(a).all() for a in (c, matrix.data, rhs, equality.data, eq_rhs)):
        return None

    limits = np.asarray((0, np.inf) if bounds is None else bounds, dtype=float)
    if limits.shape == (2,):
        limits = np.broadcast_to(limits, (cols, 2)).copy()
    elif limits.shape == (cols, 2):
        limits = limits.copy()
    else:
        return None
    limits[np.isnan(limits[:, 0]), 0] = -np.inf
    limits[np.isnan(limits[:, 1]), 1] = np.inf
    lower, upper = limits.T
    if np.any(lower > upper) or np.any(lower == np.inf) or np.any(upper == -np.inf):
        return None
    types = np.broadcast_to(np.asarray(0 if integrality is None else integrality), (cols,))
    if not np.isin(types, [0, 1, 2, 3]).all():
        return None

    native_options = {"presolve": "on" if options.get("presolve", True) else "off"}
    for name in ("primal_feasibility_tolerance", "dual_feasibility_tolerance", "mip_rel_gap"):
        value = options.get(name)
        if value is not None:
            if not isinstance(value, (float, int)) or not np.isfinite(value) or value < 0:
                return None
            if name != "mip_rel_gap" and value < 1e-10:
                return None
            native_options[name] = float(value)
    strategy = options.get("simplex_dual_edge_weight_strategy")
    if strategy is not None:
        strategies = {"dantzig": 0, "devex": 1, "steepest": 2, "steepest-devex": -1}
        if strategy not in strategies:
            return None
        native_options["simplex_dual_edge_weight_strategy"] = strategies[strategy]

    # Seed the working set using violations at the objective's box minimizer.
    point = np.where(c < 0, upper, lower)
    point = np.where(np.isfinite(point), point, 0.0)
    scale = np.maximum(1.0, np.asarray(abs(matrix).sum(axis=1)).ravel())
    score = (matrix @ point - rhs) / scale
    count = min(max(16, cols // 4), rows // 8)
    selected = np.unique(np.concatenate((
        np.argpartition(score, -count)[-count:],
        np.linspace(0, rows - 1, count, dtype=int),
    )))
    restrictive = np.flatnonzero(score > 0)
    use_restrictive = 0 < len(restrictive) <= rows // 3
    if use_restrictive:
        selected = restrictive
    elif matrix.nnz < rows * cols * 0.1:
        # Sparse systems without an obvious small working set usually benefit
        # more from HiGHS' own presolve than from several row-recovery rounds.
        return None
    tolerance = min(options.get("primal_feasibility_tolerance") or 1e-7,
                    options.get("dual_feasibility_tolerance") or 1e-7)
    if use_restrictive:
        certified = _conic_solution(c, matrix, rhs, equality, eq_rhs, lower, upper,
                                   selected, tolerance)
        if certified is not None:
            return certified
    session = highspy.Highs()
    _check_status(session.setOptionValue("output_flag", False))
    _check_status(session.setOptionValue("solver", "simplex"))
    _check_status(session.addCols(cols, c, lower, upper, 0, [], [], []))
    for name, value in native_options.items():
        _check_status(session.setOptionValue(name, value))
    _add_rows(session, equality, eq_rhs, eq_rhs)
    aggregate = None if use_restrictive else _integer_surrogates(matrix, rhs)
    num_surrogates = 0
    if aggregate is not None:
        weights, surrogate, surrogate_rhs = aggregate
        num_surrogates = surrogate.shape[0]
        _add_rows(session, surrogate, np.full(num_surrogates, -np.inf), surrogate_rhs)
    initial_count = len(selected)
    active = np.zeros(rows, dtype=bool)
    row_order = []
    total_iterations = 0

    for round_index in range(5):
        active[selected] = True
        row_order.extend(selected.tolist())
        _add_rows(session, matrix[selected], np.full(len(selected), -np.inf), rhs[selected])
        _check_status(session.run())
        info = session.getInfo()
        solution = session.getSolution()
        total_iterations += info.simplex_iteration_count + info.ipm_iteration_count
        if session.getModelStatus() != highspy.HighsModelStatus.kOptimal or not solution.value_valid:
            # An unbounded relaxation does not establish an unbounded original.
            # Restore all rows before making any conclusion about the model.
            if active.all():
                return None
            selected = np.flatnonzero(~active)
            continue

        x = np.asarray(solution.col_value)
        residual = rhs - matrix @ x
        violated = residual < -tolerance
        if not violated.any():
            marginal = eq_marginal = lower_marginal = upper_marginal = None
            if solution.dual_valid:
                row_dual = np.asarray(solution.row_dual)
                col_dual = np.asarray(solution.col_dual)
                n_eq = len(eq_rhs)
                eq_marginal = row_dual[:n_eq]
                marginal = np.zeros(rows)
                if aggregate is not None:
                    marginal += weights.T @ row_dual[n_eq:n_eq + num_surrogates]
                marginal[row_order] += row_dual[n_eq + num_surrogates:]
                column_status = np.asarray(session.getBasis().col_status, dtype=int)
                lower_marginal = np.where(column_status == int(highspy.HighsBasisStatus.kLower), col_dual, 0.0)
                upper_marginal = np.where(column_status == int(highspy.HighsBasisStatus.kUpper), col_dual, 0.0)
            con = eq_rhs - equality @ x
            result = OptimizeResult(
                x=x, fun=float(c @ x), success=True, status=0,
                message="Optimization terminated successfully. (HiGHS Status 7: Optimal)",
                nit=total_iterations, crossover_nit=info.crossover_iteration_count,
                slack=residual, con=con,
                ineqlin=OptimizeResult(residual=residual, marginals=marginal),
                eqlin=OptimizeResult(residual=con, marginals=eq_marginal),
                lower=OptimizeResult(residual=x - lower, marginals=lower_marginal),
                upper=OptimizeResult(residual=upper - x, marginals=upper_marginal),
                turbo_accelerated=True, turbo_strategy="row_recovery", turbo_rounds=round_index + 1,
                turbo_initial_rows=initial_count, turbo_original_rows=rows,
                turbo_rows_used=int(active.sum()), turbo_surrogate_rows=num_surrogates,
            )
            return result

        if np.any(violated & active):
            return None  # Let SciPy handle a numerically troublesome model.
        missing = np.flatnonzero(violated & ~active)
        if round_index >= 3 or active.sum() + len(missing) >= rows * 0.6:
            selected = np.flatnonzero(~active)
        else:
            batch = min(len(missing), max(32, cols))
            worst = np.argpartition(residual[missing] / scale[missing], batch - 1)[:batch]
            selected = missing[worst]
    return None
