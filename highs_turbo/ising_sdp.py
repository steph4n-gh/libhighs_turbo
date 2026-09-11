"""A numerical global relaxation with an exact sum-of-squares bound witness.

Any rational factor B/D proves s' B B' s / D**2 >= 0. We subtract it
from the residual Ising matrix and bound every remaining off-diagonal term
by its absolute value. Neither eigenvalues nor Cholesky success are trusted.
"""

from dataclasses import replace
from fractions import Fraction
import time

import numpy as np
import scipy.sparse as sp
from scipy.linalg import cholesky, eigh
from scipy.optimize import minimize

from highs_turbo.ising_cuts import _bound, check_certificate, cut_matrix, make_certificate


MAX_GRAM_VERTICES = 2048


def gram_lower_bound(edges, residual, scale, factor, denominator):
    """Exactly bound sum(r_uv*s_u*s_v), including all Gram fill-in terms.

    Integer dot products use binary64 BLAS only after proving that every
    product and every possible partial sum has magnitude below 2**53.
    Those integer operations are exact under IEEE-754 binary64 arithmetic.
    Final sums and rational corrections use unbounded Python integers.
    """
    n = max((max(e) for e in edges), default=-1)+1
    if (not 0 < n <= MAX_GRAM_VERTICES or len(factor) != n
            or type(denominator) is not int or denominator <= 0):
        raise ValueError("Invalid Gram witness dimensions or denominator")
    dense = np.zeros((n, n), dtype=np.float64)
    magnitude = 0
    for i, row in enumerate(factor):
        values = np.asarray(row)
        if values.shape != (i+1,) or values.dtype.kind not in 'iu':
            raise ValueError("Gram factor must contain triangular integer rows")
        row_max = max((abs(int(x)) for x in values), default=0)
        magnitude = max(magnitude, row_max)
        if n*magnitude*magnitude >= 2**53:
            raise ValueError("Gram dot products exceed the exact integer range")
        dense[i, :i+1] = values
    gram = (dense @ dense.T).astype(np.int64)
    # Row sums also fit exactly in int64; accumulate the rows in Python.
    maximum = int(np.max(np.abs(gram)))
    if n*maximum >= 2**63:
        raise ValueError("Gram row sum exceeds the integer range")
    total = sum(map(int, np.abs(gram).sum(axis=1)))
    square = denominator*denominator
    # Start with Q=0, then replace the two symmetric entries for each edge.
    # Nonedges are deliberately included: factorization creates dense fill-in.
    common = scale*square
    numerator = -total*scale
    for (u, v), weight in zip(edges, residual):
        value = int(gram[u, v])
        numerator += 2*abs(value)*scale-abs(weight*square-2*value*scale)
    return Fraction(numerator, common)


def strengthen_certificate(proof, edges, weights, constant, *, deadline, seed=0, cuts=()):
    """Try a low-rank vector relaxation, retaining the stronger valid witness.

    The dense factor and eigenvalue work are limited to 2048 vertices. Larger
    inputs retain their existing certificate. The cooperative deadline can
    overrun by the final factorization and exact check.
    """
    n = max((max(e) for e in edges), default=-1)+1
    if not 1 < n <= MAX_GRAM_VERTICES or time.perf_counter() >= deadline:
        return proof
    residual = dict(weights)
    for cut, numerator in zip(proof.cuts, proof.multipliers):
        multiplier = Fraction(numerator, proof.denominator)
        for index, coefficient in zip(cut.indices, cut.coefficients):
            residual[edges[index]] -= multiplier*coefficient
    u, v = np.asarray(edges).T
    values = np.asarray([float(residual[e])/2 for e in edges])
    if not np.isfinite(values).all() or not np.any(values):
        return proof
    # Normalize to avoid underflow/overflow in the numerical search. The
    # witness is rescaled before exact checking against the original input.
    magnitude = float(np.max(np.abs(values)))
    rng = np.random.default_rng(seed)
    initial = rng.normal(size=(n, min(16, n)))
    initial /= np.linalg.norm(initial, axis=1)[:, None]
    latest = initial.ravel()
    original = 2*values/magnitude
    matrix = None

    def fit(edge_values, iterations):
        nonlocal latest, matrix
        matrix = sp.csr_matrix((np.r_[edge_values, edge_values]/2,
                               (np.r_[u, v], np.r_[v, u])), shape=(n, n))

        def objective(flat):
            vectors = flat.reshape(n, -1)
            norms = np.maximum(np.linalg.norm(vectors, axis=1)[:, None], 1e-100)
            vectors = vectors/norms
            product = matrix @ vectors
            diagonal = np.sum(vectors*product, axis=1)
            gradient = 2*(product-vectors*diagonal[:, None])/norms
            return diagonal.sum(), gradient.ravel()

        def checkpoint(flat):
            nonlocal latest
            latest = flat.copy()
            if time.perf_counter() >= deadline:
                raise StopIteration

        try:
            result = minimize(objective, latest, jac=True, method='L-BFGS-B', callback=checkpoint,
                              options={'maxiter': iterations, 'ftol': 1e-10,
                                       'gtol': 1e-7, 'maxcor': 5})
            latest = result.x
        except StopIteration:
            pass
        return objective(latest)[0]

    if cuts:
        # Optimize the actual SDP Lagrangian rather than freezing multipliers
        # chosen for a box relaxation. Its ascent direction is cut violation
        # at the current vector correlations. The numerical solve is only a
        # proposal; quantized multipliers and the final Gram factor are checked.
        rows = cut_matrix(cuts, len(edges))
        shift = np.asarray(rows.sum(axis=1)).ravel()-2*np.asarray([c.rhs for c in cuts])
        chosen = np.zeros(len(cuts))
        calls = 0

        def dual_objective(multipliers):
            nonlocal chosen, calls
            if calls and time.perf_counter() >= deadline:
                raise StopIteration
            chosen = multipliers.copy()
            value = fit(original-rows.T @ multipliers, 150 if calls == 0 else 25)
            vectors = latest.reshape(n, -1)
            vectors = vectors/np.maximum(np.linalg.norm(vectors, axis=1)[:, None], 1e-100)
            correlation = np.sum(vectors[u]*vectors[v], axis=1)
            calls += 1
            return -value-shift @ multipliers, rows @ correlation-shift

        try:
            result = minimize(dual_objective, chosen, jac=True, method='L-BFGS-B',
                              bounds=[(0, None)]*len(cuts),
                              options={'maxiter': 60, 'maxls': 10, 'maxcor': 5,
                                       'ftol': 1e-9, 'gtol': 1e-7})
            chosen = result.x
        except StopIteration:
            pass
        extra = make_certificate(cuts, -chosen*magnitude, edges, weights, constant)
        combined = make_certificate(
            (*proof.cuts, *extra.cuts),
            [-Fraction(x, proof.denominator) for x in proof.multipliers]
            + [-Fraction(x, extra.denominator) for x in extra.multipliers],
            edges, weights, constant)
        # Refit against the exact rounded coefficients that the checker sees.
        residual = dict(weights)
        for cut, numerator in zip(combined.cuts, combined.multipliers):
            for index, coefficient in zip(cut.indices, cut.coefficients):
                residual[edges[index]] -= Fraction(numerator*coefficient, combined.denominator)
        original = np.asarray([float(residual[e])/magnitude for e in edges])
        source = combined
    else:
        source = proof
    fit(original, 150)
    vectors = latest.reshape(n, -1)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1)[:, None], 1e-100)
    diagonal = np.sum(vectors*(matrix @ vectors), axis=1)
    slack = matrix.toarray()
    slack.flat[::n+1] -= diagonal
    try:
        smallest = eigh(slack, subset_by_index=[0, 0], eigvals_only=True, check_finite=False)[0]
        slack.flat[::n+1] += max(0., -smallest)+1e-7
        factor = cholesky(slack, lower=True, check_finite=False)*np.sqrt(magnitude)
        # Choose a dyadic scale that keeps integer dot products exact.
        largest = float(np.max(np.abs(factor)))
        if not np.isfinite(largest) or largest <= 0:
            return proof
        bits = min(20, int(np.floor(np.log2(np.sqrt((2**52)/n)/largest))))
        if bits < 0:
            return proof
        denominator = 2**bits
        integers = np.rint(factor*denominator).astype(np.int64)
        packed = tuple(tuple(map(int, integers[i, :i+1])) for i in range(n))
        lower = _bound(source.cuts, source.multipliers, source.denominator,
                       edges, weights, constant, packed, denominator)
        if lower <= proof.lower_bound:
            return proof
        candidate = replace(source, lower_bound=lower, gram_factor=packed, gram_denominator=denominator)
        if not check_certificate(candidate, edges, weights, constant):
            raise RuntimeError("Sum-of-squares certificate failed exact verification")
        return candidate
    except (ValueError, OverflowError, np.linalg.LinAlgError):
        return proof
