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
from highs_turbo.ising_sparse import MAX_SPARSE_VERTICES


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


def factor_witness(matrix, vectors, magnitude=1.):
    """Construct a dyadic Gram proposal; validity is decided by the checker."""
    n = len(vectors)
    slack = matrix.toarray()
    slack.flat[::n+1] -= np.sum(vectors*(matrix @ vectors), axis=1)
    smallest = eigh(slack, subset_by_index=[0, 0], eigvals_only=True, check_finite=False)[0]
    slack.flat[::n+1] += max(0., -smallest)+1e-7
    factor = np.tril(cholesky(slack, lower=True, check_finite=False))*np.sqrt(magnitude)
    largest = float(np.max(np.abs(factor)))
    if not np.isfinite(largest) or largest <= 0:
        raise ValueError("Invalid numerical Gram factor")
    bits = min(20, int(np.floor(np.log2(np.sqrt((2**52)/n)/largest))))
    if bits < 0:
        raise ValueError("Gram factor exceeds the exact integer range")
    denominator = 2**bits
    integers = np.rint(factor*denominator).astype(np.int64)
    return tuple(tuple(map(int, integers[i, :i+1])) for i in range(n)), denominator


def strengthen_certificate(proof, edges, weights, constant, *, deadline, seed=0, cuts=(),
                           geometry=False, certificate_fallbacks=None):
    """Try a low-rank vector relaxation, retaining the stronger valid witness.

    Dense factors are limited to 2048 vertices. Larger problems use sparse
    factors up to 8192 vertices, subject to work and storage limits. The
    cooperative deadline can overrun by final factorization and exact checks.
    """
    n = max((max(e) for e in edges), default=-1)+1
    if not 1 < n <= MAX_SPARSE_VERTICES or time.perf_counter() >= deadline:
        return proof
    sparse = n > MAX_GRAM_VERTICES
    if sparse:
        if len(edges) > 16*n:
            return proof
        # Reserve part of the budget for factorization and exact checking.
        remaining = deadline-time.perf_counter()
        deadline -= min(2., .2*remaining) if np.isfinite(remaining) else 0.
    geometry_deadline = deadline
    if geometry and n >= 32:
        deadline = min(deadline, time.perf_counter()+max(.3, .4*(deadline-time.perf_counter())))
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
    initial = rng.normal(size=(n, min(32 if sparse else 16, n)))
    initial /= np.linalg.norm(initial, axis=1)[:, None]
    latest = initial.ravel()
    original = 2*values/magnitude
    matrix = None
    if sparse:
        from highs_turbo.ising_sparse import independent_groups, fit_vectors
        groups = independent_groups(edges, n)

    def fit(edge_values, iterations):
        nonlocal latest, matrix
        matrix = sp.csr_matrix((np.r_[edge_values, edge_values]/2,
                               (np.r_[u, v], np.r_[v, u])), shape=(n, n))
        if sparse:
            vectors = fit_vectors(matrix, latest.reshape(n, -1), groups,
                                  deadline=deadline, iterations=iterations)
            latest = vectors.ravel()
            return float(np.sum(vectors*(matrix @ vectors)))

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
    # A useful cut-only witness survives a failed numerical factor proposal.
    if source.lower_bound > proof.lower_bound and check_certificate(source, edges, weights, constant):
        proof = source
    fit(original, 700 if sparse else 150)
    vectors = latest.reshape(n, -1)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1)[:, None], 1e-100)
    try:
        if sparse:
            from highs_turbo.ising_sparse import factor_witness as sparse_factor_witness
            packed, denominator = sparse_factor_witness(matrix, vectors, magnitude)
        else:
            packed, denominator = factor_witness(matrix, vectors, magnitude)
        lower = _bound(source.cuts, source.multipliers, source.denominator,
                       edges, weights, constant, () if sparse else packed, denominator,
                       packed if sparse else ())
        candidate = replace(source, lower_bound=lower, gram_factor=() if sparse else packed,
                            sparse_gram_factor=packed if sparse else (), gram_denominator=denominator)
        # _bound already checked the sparse arithmetic and factor structure;
        # solve_ising independently checks the assembled certificate at return.
        if not sparse and not check_certificate(candidate, edges, weights, constant):
            raise RuntimeError("Sum-of-squares certificate failed exact verification")
        if lower > proof.lower_bound:
            proof = candidate
    except (ValueError, OverflowError, np.linalg.LinAlgError, RuntimeError) as error:
        if certificate_fallbacks is not None:
            certificate_fallbacks.append({"stage": "sdp_factor", "reason": str(error)})
        return proof
    if geometry and n >= 32 and time.perf_counter()+.75 < geometry_deadline:
        from highs_turbo.ising_geometry import strengthen_geometry
        proof = strengthen_geometry(proof, edges, weights, constant, vectors,
                                    deadline=geometry_deadline, seed=seed,
                                    certificate_fallbacks=certificate_fallbacks)
    return proof
