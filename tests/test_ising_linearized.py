"""The drop-in bridge must preserve every original row and coefficient."""

from fractions import Fraction
from itertools import product
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
import scipy.sparse as sp
from scipy.optimize import Bounds, linprog as scipy_linprog

from highs_turbo import linprog, verify_ising_certificate, verify_linprog_certificate
from highs_turbo.ising_linearized import binary_products, ising_objective, solve_linearized_binary


def model():
    pairs = [(0, 1), (1, 2), (2, 3), (0, 3)]
    c = np.r_[[-0.1, 0.2, -0.7, 0.3], np.ones(28), [0.3, -0.6, 0.2, -0.9]]
    matrix = np.zeros((3 * len(pairs), len(c)))
    rhs = np.tile([0.0, 0.0, 1.0], len(pairs))
    for k, (u, v) in enumerate(pairs):
        y = 32 + k
        matrix[3 * k, [y, u]] = [1, -1]
        matrix[3 * k + 1, [y, v]] = [1, -1]
        matrix[3 * k + 2, [u, v, y]] = [1, 1, -1]
    return c, sp.csr_matrix(matrix), rhs, np.r_[np.ones(32), np.zeros(4)], pairs


@pytest.mark.parametrize("additional", ["none", "equality", "inequality"])
def test_drop_in_certificate_preserves_binary_energy_and_extra_constraints(additional):
    c, matrix, rhs, types, pairs = model()
    equality, eq_rhs = None, None
    if additional == "equality":
        equality = sp.csr_matrix(([1.0], ([0], [0])), shape=(1, len(c)))
        eq_rhs = [0.0]
    elif additional == "inequality":
        extra = sp.csr_matrix(([1.0, 1.0], ([0, 0], [0, 3])), shape=(1, len(c)))
        matrix, rhs = sp.vstack([matrix, extra], format="csr"), np.r_[rhs, 0.0]
    result = linprog(
        c,
        A_ub=matrix,
        b_ub=rhs,
        A_eq=equality,
        b_eq=eq_rhs,
        bounds=(0, 1),
        integrality=types,
    )
    assert result.success and result.turbo_strategy == "ising_certificate"
    assert result.turbo_equivalent_model == (additional == "none")
    assert np.all(result.slack >= -1e-9)
    if additional != "none":
        assert result.x[0] == 0
    assert result.ineqlin.marginals.shape == rhs.shape
    assert result.eqlin.marginals.shape == ((1,) if additional == "equality" else (0,))
    assert result.lower.marginals.shape == result.upper.marginals.shape == c.shape
    energies = []
    for bits in product([0, 1], repeat=4):
        state = [*bits, *([0] * 28), *(bits[u] * bits[v] for u, v in pairs)]
        value = sum(Fraction(float(a)) * b for a, b in zip(c, state))
        assert result.turbo_exact_lower_bound <= value
        if (
            additional == "none"
            or additional == "equality"
            and bits[0] == 0
            or additional == "inequality"
            and bits[0] == bits[3] == 0
        ):
            energies.append(value)
    assert abs(result.fun - float(min(energies))) < 1e-12
    fields, couplings, constant = ising_objective(
        c, range(32), dict(enumerate(pairs, 32))
    )
    assert verify_ising_certificate(
        fields, couplings, result.turbo_bound_certificate, offset=constant
    )
    assert verify_linprog_certificate(
        c,
        result.turbo_bound_certificate.to_dict(),
        A_ub=matrix,
        b_ub=rhs,
        bounds=Bounds(0, 1),
        integrality=types,
    )
    assert not verify_linprog_certificate(
        c + 0.125,
        result.turbo_bound_certificate,
        A_ub=matrix,
        b_ub=rhs,
        bounds=(0, 1),
        integrality=types,
    )


def test_incomplete_or_inexact_envelope_keeps_scipy_model():
    c, matrix, rhs, types, _ = model()
    for changed, bound in [(matrix[:-1], rhs[:-1]), (matrix.copy(), rhs.copy())]:
        if changed.shape == matrix.shape:
            changed.data[0] += 1e-8
        actual = linprog(c, A_ub=changed, b_ub=bound, bounds=(0, 1), integrality=types)
        reference = scipy_linprog(
            c, A_ub=changed, b_ub=bound, bounds=(0, 1), integrality=types
        )
        assert actual.success == reference.success
        assert abs(actual.fun - reference.fun) < 1e-12
        assert actual.turbo_accelerated is False


def test_duplicate_sparse_entries_are_declined_before_certificate_recognition(monkeypatch):
    import highs_turbo.ising_linearized as bridge
    from highs_turbo.ising_cuts import make_certificate

    c, matrix, rhs, types, pairs = model()
    values = np.insert(matrix.data, 0, matrix.data[0] / 2)
    values[1] /= 2
    indices = np.insert(matrix.indices, 0, matrix.indices[0])
    starts = matrix.indptr.copy()
    starts[1:] += 1
    duplicate = sp.csr_matrix((values, indices, starts), shape=matrix.shape)
    assert not duplicate.has_canonical_format
    np.testing.assert_array_equal(duplicate.toarray(), matrix.toarray())
    monkeypatch.setattr(bridge, "solve_ising", lambda *a, **k: pytest.fail("Decline before solving"))
    assert solve_linearized_binary(c, duplicate, rhs, None, None, (0, 1), types, {}) is None
    fields, couplings, constant = ising_objective(c, range(32), dict(enumerate(pairs, 32)))
    # The certificate is valid for the canonical model, but cannot authorize
    # floating-point summation of duplicate coefficients in an envelope.
    weights = dict(couplings)
    weights.update({(i, 32): w for i, w in fields.items() if w})
    proof = make_certificate([], [], sorted(weights), weights, constant)
    assert verify_linprog_certificate(c, proof, A_ub=matrix, b_ub=rhs,
                                     bounds=(0, 1), integrality=types)
    assert not verify_linprog_certificate(c, proof, A_ub=duplicate, b_ub=rhs,
                                         bounds=(0, 1), integrality=types)


@pytest.mark.parametrize("nodes, recognized", [(31, False), (32, True), (8191, True), (8192, False)])
def test_binary_product_size_boundaries(nodes, recognized):
    matrix = sp.csr_matrix(([-1., 1., -1., 1., 1., 1., -1.],
                            ([0, 0, 1, 1, 2, 2, 2], [0, nodes, 1, nodes, 0, 1, nodes])),
                           shape=(3, nodes + 1))
    result = binary_products(matrix, np.array([0., 0., 1.]), np.r_[np.ones(nodes), 0])
    assert (result is not None) == recognized


@pytest.mark.parametrize("options", [
    {"time_limit": 0}, {"time_limit": -1}, {"presolve": False},
    {"disp": True}, {"maxiter": 1}, {"mip_rel_gap": -1},
])
def test_unsupported_bridge_options_decline_without_solver_work(options, monkeypatch):
    import highs_turbo.ising_linearized as bridge
    c, matrix, rhs, types, _ = model()
    monkeypatch.setattr(bridge, "solve_ising", lambda *a, **k: pytest.fail("Unsupported option"))
    assert solve_linearized_binary(c, matrix, rhs, None, None, (0, 1), types, options) is None


@pytest.mark.parametrize("status, expected", [
    ("OPTIMAL", 0), ("GAP_LIMIT", 0), ("TIME_LIMIT", 1), ("SOLVER_ERROR", 4),
])
def test_equivalent_bridge_preserves_termination_and_checked_bound(status, expected, monkeypatch):
    import highs_turbo.ising_linearized as bridge
    from highs_turbo import solve_ising

    c, matrix, rhs, types, pairs = model()
    fields, couplings, constant = ising_objective(c, range(32), dict(enumerate(pairs, 32)))
    original = solve_ising(fields, couplings, offset=constant, time_limit=0)
    simulated = replace(original, status=status, message=status)
    monkeypatch.setattr(bridge, "solve_ising", lambda *a, **k: simulated)
    result = solve_linearized_binary(c, matrix, rhs, None, None, (0, 1), types, {})
    assert result.status == expected and result.success == (expected == 0)
    assert result.turbo_exact_lower_bound == original.exact_cut_lower_bound
    assert np.all(result.slack >= 0)
    assert verify_linprog_certificate(c, result.turbo_bound_certificate.to_dict(),
                                     A_ub=matrix, b_ub=rhs, bounds=(0, 1), integrality=types)


@pytest.mark.parametrize("additional", [False, True])
def test_bridge_preparation_does_not_restart_expired_budget(additional, monkeypatch):
    import highs_turbo.ising_linearized as bridge
    c, matrix, rhs, types, _ = model()
    equality = sp.csr_matrix(([1.], ([0], [0])), shape=(1, len(c))) if additional else None
    ticks = iter([10., 12.])
    monkeypatch.setattr(bridge, "time", SimpleNamespace(perf_counter=lambda: next(ticks)))

    class ReachedSolver(Exception):
        pass

    def solve(*args, **kwargs):
        assert kwargs["time_limit"] == 0
        raise ReachedSolver

    monkeypatch.setattr(bridge, "solve_ising", solve)
    with pytest.raises(ReachedSolver):
        solve_linearized_binary(c, matrix, rhs, equality, [0.] if additional else None,
                                (0, 1), types, {"time_limit": 1.})


@pytest.mark.parametrize("additional", ["equality", "inequality"])
def test_infeasible_additional_constraints_keep_scipy_status(additional):
    c, matrix, rhs, types, _ = model()
    extra = sp.csr_matrix(([1.], ([0], [0])), shape=(1, len(c)))
    equality, eq_rhs = (extra, [-1.]) if additional == "equality" else (None, None)
    if additional == "inequality":
        matrix, rhs = sp.vstack([matrix, extra], format="csr"), np.r_[rhs, -1.]
    result = solve_linearized_binary(c, matrix, rhs, equality, eq_rhs, (0, 1), types, {})
    assert result.status == 2 and not result.success
    assert result.x is None and result.fun is None
    assert verify_linprog_certificate(c, result.turbo_bound_certificate, A_ub=matrix,
                                     b_ub=rhs, bounds=(0, 1), integrality=types)


@pytest.mark.parametrize("failure", ["product", "equality", "inequality", "bounds",
                                     "integrality", "nan", "shape", "missing"])
def test_native_bridge_rejects_invalid_final_primal(failure, monkeypatch):
    import highs_turbo.ising_linearized as bridge
    from highs_turbo import solve_ising

    c, matrix, rhs, types, pairs = model()
    fields, couplings, constant = ising_objective(c, range(32), dict(enumerate(pairs, 32)))
    initial = solve_ising(fields, couplings, offset=constant, time_limit=0)
    monkeypatch.setattr(bridge, "solve_ising", lambda *a, **k: initial)
    equality = sp.csr_matrix(([1.], ([0], [0])), shape=(1, len(c)))
    extra = sp.csr_matrix(([1.], ([0], [3])), shape=(1, len(c)))
    matrix, rhs = sp.vstack([matrix, extra], format="csr"), np.r_[rhs, 0.]
    x = np.zeros(len(c))
    if failure == "product":
        x[32] = 1.
    elif failure == "equality":
        x[0] = 1.
    elif failure == "inequality":
        x[3] = 1.
    elif failure == "bounds":
        x[10] = -1.
    elif failure == "integrality":
        x[10] = .5
    elif failure == "nan":
        x[10] = np.nan
    elif failure == "shape":
        x = x[:-1]
    monkeypatch.setattr(bridge.highspy.Highs, "getSolution", lambda self:
                        SimpleNamespace(value_valid=failure != "missing", col_value=x))
    with pytest.raises(RuntimeError, match="Binary-product solve"):
        solve_linearized_binary(c, matrix, rhs, equality, [0.], (0, 1), types, {})
    if failure == "product":
        # The public API's existing error fallback must also keep the original
        # problem intact when the accelerated native result fails validation.
        actual = linprog(c, A_ub=matrix, b_ub=rhs, A_eq=equality, b_eq=[0.],
                         bounds=(0, 1), integrality=types)
        reference = scipy_linprog(c, A_ub=matrix, b_ub=rhs, A_eq=equality, b_eq=[0.],
                                 bounds=(0, 1), integrality=types)
        assert actual.success and actual.fun == pytest.approx(reference.fun)
        assert not actual.turbo_accelerated
        assert "original rows" in actual.fallback_reason
