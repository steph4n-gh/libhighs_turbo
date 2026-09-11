"""Find nonlocal cuts in vector space and certify their full objective bound.

The nearest-neighbor search proposes inequalities only. The ordinary cycle
checker validates every rule, including edges absent from the input graph.
"""

from dataclasses import replace
from fractions import Fraction
import time
import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from highs_turbo.ising_cuts import (
    IsingCut,
    make_certificate,
    cut_matrix,
    _bound,
    check_certificate,
)
from highs_turbo.ising_sdp import factor_witness


def geometric_triangles(vectors, edges, *, limit=2048, seed=0, epsilon=0.5):
    began = time.perf_counter()
    n = len(vectors)
    if n < 3:
        return [], dict(seconds=0.0, candidates=0)
    rng = np.random.default_rng(seed)
    pairs = np.asarray(edges, dtype=int).reshape(-1, 2)
    if len(pairs) > 16 * n:
        pairs = pairs[rng.choice(len(pairs), 16 * n, replace=False)]
    pairs = np.r_[pairs, rng.integers(0, n, size=(8 * n, 2))]
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    count = len(pairs)
    pairs = np.tile(pairs, (2, 1))
    sign = np.repeat([1, -1], count)
    targets = -(vectors[pairs[:, 0]] + sign[:, None] * vectors[pairs[:, 1]])
    tree = cKDTree(np.r_[vectors, -vectors])
    # The excluded endpoint vectors are at distance at least one from these
    # targets. Any neighbor inside the unit ball is therefore a third spin.
    distances, third = tree.query(targets, k=1, eps=epsilon, workers=1)
    triples = np.c_[pairs, third % n]
    coefficients = np.c_[
        np.ones(len(pairs), dtype=int), sign, np.where(third < n, 1, -1)
    ]
    order = np.argsort(triples, axis=1)
    triples = np.take_along_axis(triples, order, axis=1)
    coefficients = np.take_along_axis(coefficients, order, axis=1)
    coefficients *= coefficients[:, 0, None]
    chosen = []
    seen = set()
    usage = np.zeros(n, dtype=int)
    eligible = np.flatnonzero(
        (distances < 0.999) & (third % n != pairs[:, 0]) & (third % n != pairs[:, 1])
    )
    for index in eligible[np.argsort(distances[eligible], kind="stable")]:
        nodes = triples[index]
        if np.any(usage[nodes] >= 16):
            continue
        key = tuple(zip(map(int, nodes), map(int, coefficients[index])))
        if key in seen:
            continue
        seen.add(key)
        chosen.append(key)
        usage[nodes] += 1
        if len(chosen) >= limit:
            break
    return chosen, dict(seconds=time.perf_counter() - began, candidates=len(eligible))


def triangle_rows(triangles, edges):
    extra = sorted(
        {
            tuple(sorted((u, v)))
            for t in triangles
            for (u, a), (v, b) in ((t[0], t[1]), (t[0], t[2]), (t[1], t[2]))
        }
        - set(edges)
    )
    expanded = list(edges) + extra
    location = {e: i for i, e in enumerate(expanded)}
    cuts = []
    for triangle in triangles:
        terms = sorted(
            (location[tuple(sorted((u, v)))], a * b)
            for (u, a), (v, b) in (
                (triangle[0], triangle[1]),
                (triangle[0], triangle[2]),
                (triangle[1], triangle[2]),
            )
        )
        cuts.append(
            IsingCut(
                tuple(i for i, a in terms),
                tuple(a for i, a in terms),
                (sum(a for i, a in triangle) ** 2 - 1) // 4,
                "cycle",
            )
        )
    return expanded, cuts


def strengthen_geometry(proof, edges, weights, constant, vectors, *, deadline, seed=0):
    """Fit a batch of nonlocal cuts, retaining the stronger exact certificate."""
    triangles, _ = geometric_triangles(vectors, edges, seed=seed)
    expanded, new = triangle_rows(triangles, edges)
    known = set(proof.cuts)
    new = [cut for cut in new if cut not in known]
    if not new or time.perf_counter() + 0.25 >= deadline:
        return proof
    original_weights = weights
    weights = {**weights, **dict.fromkeys(expanded[len(edges) :], Fraction())}
    cuts = (*proof.cuts, *new)
    n = len(vectors)
    row = cut_matrix(cuts, len(expanded))
    shift = np.asarray(row.sum(axis=1)).ravel() - 2 * np.asarray([c.rhs for c in cuts])
    original = np.asarray([float(weights[e]) for e in expanded])
    magnitude = float(np.max(np.abs(original)))
    if not np.isfinite(magnitude) or magnitude <= 0:
        return proof
    original /= magnitude
    chosen = np.r_[
        [float(Fraction(a, proof.denominator)) / magnitude for a in proof.multipliers],
        np.zeros(len(new)),
    ]
    u, v = np.asarray(expanded).T
    latest = vectors.copy().ravel()
    matrix = None
    calls = 0
    phase_deadline = time.perf_counter() + 0.75 * (deadline - time.perf_counter())

    def fit(multipliers, iterations):
        nonlocal latest, matrix
        residual = original - row.T @ multipliers
        matrix = sp.csr_matrix(
            (np.r_[residual, residual] / 2, (np.r_[u, v], np.r_[v, u])), shape=(n, n)
        )

        def objective(flat):
            V = flat.reshape(n, -1)
            norms = np.maximum(np.linalg.norm(V, axis=1)[:, None], 1e-100)
            V = V / norms
            applied = matrix @ V
            diagonal = np.sum(V * applied, axis=1)
            return sum(diagonal), (
                2 * (applied - V * diagonal[:, None]) / norms
            ).ravel()

        def checkpoint(flat):
            nonlocal latest
            latest = flat.copy()
            if time.perf_counter() >= phase_deadline:
                raise StopIteration

        result = minimize(
            objective,
            latest,
            jac=True,
            method="L-BFGS-B",
            callback=checkpoint,
            options={"maxiter": iterations, "maxcor": 5, "ftol": 1e-10, "gtol": 1e-7},
        )
        latest = result.x
        return objective(latest)[0]

    def dual(multipliers):
        nonlocal chosen, calls
        if calls and time.perf_counter() >= phase_deadline:
            raise StopIteration
        chosen = multipliers.copy()
        value = fit(chosen, 150 if calls == 0 else 75)
        calls += 1
        V = latest.reshape(n, -1)
        V = V / np.maximum(np.linalg.norm(V, axis=1)[:, None], 1e-100)
        return -value - shift @ chosen, row @ np.sum(V[u] * V[v], axis=1) - shift

    try:
        result = minimize(
            dual,
            chosen,
            jac=True,
            method="L-BFGS-B",
            bounds=[(0, None)] * len(chosen),
            options={
                "maxiter": 60,
                "maxls": 10,
                "maxcor": 5,
                "ftol": 1e-9,
                "gtol": 1e-7,
            },
        )
        chosen = result.x
    except StopIteration:
        pass
    source = make_certificate(cuts, -chosen * magnitude, expanded, weights, constant)
    exact = {
        cut: float(Fraction(a, source.denominator)) / magnitude
        for cut, a in zip(source.cuts, source.multipliers)
    }
    phase_deadline = deadline
    fit(np.asarray([exact.get(c, 0.0) for c in cuts]), 150)
    V = latest.reshape(n, -1)
    V = V / np.maximum(np.linalg.norm(V, axis=1)[:, None], 1e-100)
    try:
        factor, denominator = factor_witness(matrix, V, magnitude)
        lower = _bound(
            source.cuts,
            source.multipliers,
            source.denominator,
            expanded,
            weights,
            constant,
            factor,
            denominator,
        )
        if lower <= proof.lower_bound:
            return proof
        candidate = replace(
            source,
            problem_digest=proof.problem_digest,
            lower_bound=lower,
            gram_factor=factor,
            gram_denominator=denominator,
            extra_edges=tuple(expanded[len(edges) :]),
        )
        if not check_certificate(candidate, edges, original_weights, constant):
            raise RuntimeError("Geometric certificate failed independent verification")
        return candidate
    except (ValueError, OverflowError, np.linalg.LinAlgError):
        return proof
