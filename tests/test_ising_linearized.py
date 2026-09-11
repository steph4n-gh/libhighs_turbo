"""The drop-in bridge must preserve every original row and coefficient."""

from fractions import Fraction
from itertools import product
import numpy as np
import pytest
import scipy.sparse as sp
from scipy.optimize import Bounds, linprog as scipy_linprog

from highs_turbo import linprog, verify_ising_certificate, verify_linprog_certificate
from highs_turbo.ising_linearized import ising_objective


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
