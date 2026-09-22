"""Research only: cubic SOS and ordinary-cut controls on a fixed Schur block.

This experiment is not called by the public solver. It requires optional
cvxpy/SCS and a prepared, previously checked source proof. See
CUBIC_SCHUR_RESULTS.md for the preparation contract, mathematics, and limits.
"""

import argparse
import gzip
import json
import time
from collections import defaultdict
from fractions import Fraction
from itertools import combinations, product
from math import lcm
from pathlib import Path

import cvxpy as cp
import numpy as np
import scipy.sparse as sp
import scs
from cvxpy.reductions.solvers.conic_solvers.scs_conif import dims_to_solver_dict
from scipy.linalg import cholesky, eigh
from scipy.optimize import minimize
from scipy.sparse.linalg import splu, spsolve_triangular

from highs_turbo.ising_cuts import IsingCertificate
from highs_turbo.ising_sparse import (
    MAX_FACTOR_ENTRIES,
    _unpack_factor,
    gram_lower_bound,
)


def cubic_identities(n, triples):
    triples = sorted({tuple(sorted(map(int, t))) for t in triples})
    if any(
        len(t) != 3 or len(set(t)) != 3 or min(t) < 0 or max(t) >= n for t in triples
    ):
        raise ValueError("Expected three distinct original vertices")
    index = {t: n + i for i, t in enumerate(triples)}
    products = defaultdict(set)
    quads = set()
    for t in triples:
        for vertex in t:
            pair = tuple(v for v in t if v != vertex)
            products[pair].add((vertex, index[t]))
    for i, left in enumerate(triples):
        for right in triples[i + 1 :]:
            union = set(left) | set(right)
            mask = tuple(sorted(set(left) ^ set(right)))
            products[mask].add((index[left], index[right]))
            if len(union) == 4:
                quads.add(tuple(sorted(union)))
            if len(mask) == 4:
                quads.add(mask)
    # A repeated quartic from cubic*singleton requires two triples within
    # the same four-set. Those two triples share exactly two vertices.
    for quad in quads:
        for vertex in quad:
            triple = tuple(v for v in quad if v != vertex)
            if triple in index:
                products[quad].add((vertex, index[triple]))
    for mask, occurrences in products.items():
        if len(mask) == 2:
            occurrences.add(mask)
    groups = []
    for mask in sorted(products):
        occurrences = sorted(products[mask])
        if len(occurrences) > 1:
            groups.append((mask, occurrences))
    rows = []
    cols = []
    labels = []
    for group, (_, occurrences) in enumerate(groups):
        for u, v in occurrences:
            rows.append(u)
            cols.append(v)
            labels.append(group)
    labels = np.asarray(labels, dtype=int)
    return (
        triples,
        groups,
        np.asarray(rows, dtype=int),
        np.asarray(cols, dtype=int),
        labels,
        np.bincount(labels),
    )


def identity_check(n, triples, edges, integer_weights, scale, expected):
    """Verify the entire lifted quadratic polynomial pulls back to the source."""
    basis = [(i,) for i in range(n)] + list(triples)
    actual = defaultdict(int)
    for (u, v), weight in zip(edges, integer_weights):
        if type(weight) is not int:
            raise ValueError("Expected integer coefficients")
        mask = tuple(sorted(set(basis[u]) ^ set(basis[v])))
        actual[mask] += weight
    actual = {key: Fraction(value, scale) for key, value in actual.items() if value}
    return actual == {tuple(sorted(k)): v for k, v in expected.items() if v}


def template(support):
    # g=(r*r-1)/2, r=sum(t_i), t_i=b_i*s_i, |support|=5.
    # p_i=t_i*(r*r-1). On the five-spin cube:
    # g = (11/640)*sum(p_i*p_i) - (1/384)*(sum(p_i))**2.
    # The 5x5 coefficient matrix has eigenvalues 11/640 and 1/240 > 0.
    nodes = sorted(i for i, b in support)
    signs = dict(support)
    assert (
        len(nodes) == 5
        and len(set(nodes)) == 5
        and all(abs(signs[i]) == 1 for i in nodes)
    )
    basis = [(i,) for i in nodes] + list(combinations(nodes, 3))
    P = []
    for i in nodes:
        P.append(
            [
                4 if mask == (i,) else 2 if len(mask) == 1 or i in mask else 0
                for mask in basis
            ]
        )
    totals = [sum(row[j] for row in P) for j in range(len(basis))]
    gram = {}
    for i, left in enumerate(basis):
        for j in range(i, len(basis)):
            a = (
                Fraction(11, 640) * sum(row[i] * row[j] for row in P)
                - Fraction(1, 384) * totals[i] * totals[j]
            )
            sign = 1
            for vertex in left + basis[j]:
                sign *= signs[vertex]
            gram[i, j] = a * sign
    return basis, gram


def correction(n, triples, supports, multipliers):
    rows = {(i,): i for i in range(n)}
    rows.update({t: n + j for j, t in enumerate(triples)})
    result = defaultdict(Fraction)
    for support, multiplier in zip(supports, multipliers):
        basis, gram = template(support)
        for (i, j), value in gram.items():
            if i != j:
                pair = tuple(sorted((rows[basis[i]], rows[basis[j]])))
                result[pair] += 2 * multiplier * value
        for (u, a), (v, b) in combinations(support, 2):
            result[tuple(sorted((u, v)))] -= multiplier * a * b
    return {edge: value for edge, value in result.items() if value}


def run(
    input_dir,
    output,
    *,
    mode="cubic",
    optimizer="mixing",
    all_cuts=False,
    solver_seconds=15,
    tolerance=1e-5,
):
    if optimizer == "mixing" and mode != "cubic":
        raise ValueError(
            "The low-rank control is implemented for cubic identities only"
        )
    if all_cuts and mode != "scalar":
        raise ValueError("All signed cuts are a scalar control")
    if solver_seconds <= 0 or tolerance <= 0:
        raise ValueError("Expected positive time and tolerance")
    folder = Path(input_dir)
    data = json.loads((folder / "input.json").read_text())
    n = data["n"]
    edges = list(map(tuple, data["edges"]))
    prior = {e: Fraction(w) for e, w in zip(edges, data["residual"])}
    constant = Fraction(data["constant"])
    supports = [[tuple(x) for x in support] for support in data["pent"]]
    parent_lam = list(map(Fraction, data["multipliers"]))
    parent = IsingCertificate.from_dict(
        json.loads(gzip.decompress((folder / "source-proof.json.gz").read_bytes()))
    )
    triples = (
        sorted(
            {
                tuple(t)
                for support in supports
                for t in combinations(sorted(i for i, b in support), 3)
            }
        )
        if mode == "cubic"
        else []
    )
    affected = sorted({i for support in supports for i, b in support})
    outside = sorted(set(range(n)) - set(affected))
    U = affected + list(range(n, n + len(triples)))
    r = len(U)
    size = n + len(triples)
    m = len(affected)
    local = {vertex: i for i, vertex in enumerate(U)}
    start = time.perf_counter()
    # Exact diagonal-dominance repair: C + diag(beta) = G + a PSD diagonally
    # dominant residual, with the same raw bound as the original Gram receipt.
    parent_residual = dict(prior)
    for support, lam in zip(supports, parent_lam):
        for (u, a), (v, b) in combinations(support, 2):
            parent_residual[u, v] -= lam * a * b
    scale_parent = lcm(*(w.denominator for w in parent_residual.values()))
    raw_parent = gram_lower_bound(
        edges,
        [int(parent_residual[e] * scale_parent) for e in edges],
        scale_parent,
        parent.sparse_gram_factor,
        parent.gram_denominator,
    )
    B = _unpack_factor(parent.sparse_gram_factor, n)
    F = B.astype(float)
    G = (F @ F.T).tocsr()
    G.data = G.data.astype(np.int64)
    diagonal = G.diagonal().copy()
    G.setdiag(0)
    G.eliminate_zeros()
    square = parent.gram_denominator**2
    assert all((w * square / 2).denominator == 1 for w in parent_residual.values())
    u, v = np.array(edges).T
    values = np.array(
        [int(parent_residual[e] * square / 2) for e in edges], dtype=np.int64
    )
    Cinteger = sp.csr_matrix(
        (np.r_[values, values], (np.r_[u, v], np.r_[v, u])), shape=(n, n)
    )
    error = Cinteger - G
    assert n * int(np.max(abs(error.data), initial=0)) < 2**63
    row_abs = np.asarray(abs(error).sum(axis=1)).ravel()
    beta_exact = [Fraction(int(a) + int(b), square) for a, b in zip(diagonal, row_abs)]
    assert -sum(beta_exact) == raw_parent
    beta = np.array(list(map(float, beta_exact)))
    del B, F, G, Cinteger, error
    prior_values = np.array([float(prior[e]) / 2 for e in edges])
    Q = sp.csr_matrix(
        (np.r_[prior_values, prior_values], (np.r_[u, v], np.r_[v, u])),
        shape=(size, size),
    )
    A = (Q[outside][:, outside] + sp.diags(beta[outside])).tocsc()
    lu = splu(
        A,
        permc_spec="MMD_AT_PLUS_A",
        diag_pivot_thresh=0,
        options={"SymmetricMode": True, "Equil": False},
    )
    pivots = lu.U.diagonal()
    assert np.min(pivots) > 0 and np.array_equal(lu.perm_r, lu.perm_c)
    L0 = (lu.L @ sp.diags(np.sqrt(pivots))).tocsr()[lu.perm_r, :]
    cross = Q[outside][:, affected].toarray()
    Z = (
        spsolve_triangular(
            lu.L.tocsr(), cross[np.argsort(lu.perm_r)], lower=True, unit_diagonal=True
        )
        / np.sqrt(pivots)[:, None]
    )
    W = np.zeros((r, len(outside)))
    W[:m] = Z.T
    H = W @ W.T
    C = Q[U][:, U].toarray()
    base = C - H
    if all_cuts:

        def canonical_support(support):
            support = sorted(support)
            flip = support[0][1]
            return tuple((i, b * flip) for i, b in support)

        parent_by_support = {
            canonical_support(support): lam
            for support, lam in zip(supports, parent_lam)
        }
        all_supports = set()
        for support in supports:
            vertices = sorted(i for i, b in support)
            for k in (3, 5):
                for subset in combinations(vertices, k):
                    for tail in product((-1, 1), repeat=k - 1):
                        all_supports.add(tuple(zip(subset, (1,) + tail)))
        supports = sorted(all_supports)
        parent_lam = [
            parent_by_support.get(support, Fraction()) for support in supports
        ]
        right = np.array([len(support) // 2 for support in supports])
    else:
        right = np.full(len(supports), 2)
    b = cp.Variable(r)
    variables = []
    oprows = []
    opcols = []
    opvalues = []
    identity_groups = []
    if mode == "cubic":
        triples, identity_groups, *_ = cubic_identities(n, triples)
        for mask, occurrences in identity_groups:
            first = occurrences[0]
            for pair in occurrences[1:]:
                column = len(variables)
                variables.append((first, pair))
                for (a, c), sign in [(first, -1), (pair, 1)]:
                    i, j = local[a], local[c]
                    oprows.extend([i + r * j, j + r * i])
                    opcols.extend([column, column])
                    opvalues.extend([sign / 2, sign / 2])
        theta = cp.Variable(len(variables))
        operator = sp.csc_matrix(
            (opvalues, (oprows, opcols)), shape=(r * r, len(variables))
        )
        objective = cp.sum(b)
    else:
        theta = cp.Variable(len(supports))
        for column, support in enumerate(supports):
            for (a, sign_a), (c, sign_c) in combinations(support, 2):
                i, j = local[a], local[c]
                oprows.extend([i + r * j, j + r * i])
                opcols.extend([column, column])
                opvalues.extend([-sign_a * sign_c / 2] * 2)
        operator = sp.csc_matrix(
            (opvalues, (oprows, opcols)), shape=(r * r, len(supports))
        )
        objective = cp.sum(b) + right @ theta
    slack = base + cp.diag(b) + cp.reshape(operator @ theta, (r, r), order="F")
    constraints = [slack >> 0]
    if mode == "scalar":
        constraints.append(theta >= 0)
    problem = cp.Problem(cp.Minimize(objective), constraints)
    warm_b = np.r_[beta[affected], np.zeros(len(triples))]
    if mode == "cubic":
        seed = correction(n, triples, supports, parent_lam)
        warm_theta = np.array(
            [float(seed.get(pair, Fraction())) for first, pair in variables]
        )
        basis_rows = {(i,): i for i in range(n)}
        basis_rows.update({t: n + j for j, t in enumerate(triples)})
        for support, lam in zip(supports, parent_lam):
            masks, gram = template(support)
            for i, mask in enumerate(masks):
                warm_b[local[basis_rows[mask]]] += float(lam * gram[i, i])
    else:
        warm_theta = np.array(list(map(float, parent_lam)))
    initial_slack = (
        base
        + np.diag(warm_b)
        + np.asarray(operator @ warm_theta).reshape((r, r), order="F")
    )
    initial_min = float(
        eigh(
            initial_slack, subset_by_index=[0, 0], eigvals_only=True, check_finite=False
        )[0]
    )
    initial_bound = (
        float(constant)
        - sum(beta[outside])
        - sum(warm_b)
        - (right @ warm_theta if mode == "scalar" else 0)
    )
    print(
        json.dumps(
            {"phase": "warm_start", "minimum": initial_min, "bound": initial_bound}
        ),
        flush=True,
    )
    setup = time.perf_counter() - start
    print(
        json.dumps(
            {
                "phase": "schur_setup",
                "mode": mode,
                "n": n,
                "lifted_size": size,
                "affected": m,
                "schur_size": r,
                "variables": theta.size,
                "seconds": setup,
                "raw_parent": float(constant - 2 * sum(parent_lam) + raw_parent),
            }
        ),
        flush=True,
    )
    solve_start = time.perf_counter()
    if optimizer == "mixing":
        _, _, global_rows, global_cols, labels, count = cubic_identities(n, triples)
        rr = np.array([local[int(i)] for i in global_rows])
        cc = np.array([local[int(i)] for i in global_cols])
        alpha = np.array(
            [
                float(seed.get((int(a), int(c)), Fraction()))
                for a, c in zip(global_rows, global_cols)
            ]
        )
        vectors = np.load(folder / "vectors.npy")
        V = np.empty((r, vectors.shape[1]))
        V[:m] = vectors[affected]
        rng = np.random.default_rng(0)
        for j, triple in enumerate(triples, m):
            A = vectors[list(triple)]
            target = np.array([A[1] @ A[2], A[0] @ A[2], A[0] @ A[1]])
            value = np.linalg.lstsq(A, target, rcond=1e-10)[0]
            norm = value @ value
            if norm > 1:
                value /= np.sqrt(norm)
            else:
                extra = rng.normal(size=V.shape[1])
                extra -= A.T @ np.linalg.lstsq(A @ A.T, A @ extra, rcond=1e-10)[0]
                if np.linalg.norm(extra) > 1e-10:
                    value += extra * np.sqrt(max(0, 1 - norm)) / np.linalg.norm(extra)
            V[j] = value

        # Balanced redundant multipliers avoid privileging the reference occurrence.
        def project(values):
            return (
                values
                - np.bincount(labels, weights=values, minlength=len(count))[labels]
                / count[labels]
            )

        penalty = 0.2
        run_iterations = 0
        deadline = solve_start + solver_seconds

        def augmented(flat):
            trial = flat.reshape(r, -1)
            norms = np.maximum(np.linalg.norm(trial, axis=1)[:, None], 1e-100)
            unit = trial / norms
            moments = np.sum(unit[rr] * unit[cc], axis=1)
            delta = project(moments)
            effective = alpha + penalty * delta
            matrix = sp.csr_matrix(
                (np.r_[effective, effective] / 2, (np.r_[rr, cc], np.r_[cc, rr])),
                shape=(r, r),
            )
            applied = base @ unit + matrix @ unit
            gradient = (
                2 * (applied - unit * np.sum(unit * applied, axis=1)[:, None]) / norms
            )
            value = (
                np.sum(unit * (base @ unit))
                + alpha @ moments
                + penalty / 2 * (delta @ delta)
            )
            return value, gradient.ravel()

        def checkpoint(flat):
            nonlocal V
            V = flat.reshape(r, -1).copy()
            V /= np.maximum(np.linalg.norm(V, axis=1)[:, None], 1e-100)
            if time.perf_counter() >= deadline:
                raise StopIteration

        delta = project(np.sum(V[rr] * V[cc], axis=1))
        while time.perf_counter() < deadline and run_iterations < 10000:
            result = minimize(
                augmented,
                V.ravel(),
                jac=True,
                method="L-BFGS-B",
                callback=checkpoint,
                options={
                    "maxiter": 150 if run_iterations == 0 else 70,
                    "maxls": 15,
                    "maxcor": 5,
                    "ftol": 1e-9,
                    "gtol": 1e-6,
                },
            )
            V = result.x.reshape(r, -1)
            V /= np.maximum(np.linalg.norm(V, axis=1)[:, None], 1e-100)
            delta = project(np.sum(V[rr] * V[cc], axis=1))
            alpha = project(alpha + penalty * delta)
            run_iterations += 1
            if run_iterations > 3 and np.max(abs(delta), initial=0) < 1e-6:
                break
        alpha_by_pair = {
            (int(a), int(c)): value
            for a, c, value in zip(global_rows, global_cols, alpha)
        }
        theta.value = np.array([alpha_by_pair[pair] for first, pair in variables])
        local_matrix = base + np.asarray(operator @ theta.value).reshape(
            (r, r), order="F"
        )
        # The diagonal and multipliers remain numerical proposals; the full exact
        # factor checker below decides the bound after repairing this slack.
        b.value = -np.sum(V * (local_matrix @ V), axis=1)
        run_status = "research_al"
        print(
            json.dumps(
                {
                    "phase": "local_al",
                    "max_moment_violation": float(np.max(abs(delta), initial=0))
                    if optimizer == "mixing"
                    else None,
                    "rank": V.shape[1] if optimizer == "mixing" else None,
                }
            ),
            flush=True,
        )
    else:
        canonical, chain, inverse = problem.get_problem_data(cp.SCS)
        offsets = next(
            item.var_offsets
            for item in reversed(inverse)
            if hasattr(item, "var_offsets")
            and b.id in item.var_offsets
            and theta.id in item.var_offsets
        )
        x0 = np.zeros(canonical["A"].shape[1])
        x0[offsets[b.id] : offsets[b.id] + r] = warm_b
        x0[offsets[theta.id] : offsets[theta.id] + theta.size] = warm_theta
        s0 = canonical["b"] - canonical["A"] @ x0
        solver = scs.SCS(
            {key: canonical[key] for key in ["A", "b", "c"]},
            dims_to_solver_dict(canonical["dims"]),
            time_limit_secs=solver_seconds,
            max_iters=30000,
            eps_abs=tolerance,
            eps_rel=tolerance,
            verbose=False,
        )
        solution = solver.solve(warm_start=True, x=x0, s=s0, y=np.zeros_like(s0))
        problem.unpack_results(solution, chain, inverse)
        run_status = problem.status
        run_iterations = problem.solver_stats.num_iters
    solve_done = time.perf_counter() - start
    assert b.value is not None and theta.value is not None
    print(
        json.dumps(
            {
                "phase": "schur_solve",
                "mode": mode,
                "status": run_status,
                "seconds": time.perf_counter() - solve_start,
                "numerical_bound": float(constant)
                - sum(beta[outside])
                - objective.value,
                "iterations": run_iterations,
            }
        ),
        flush=True,
    )
    # Rebuild the exact polynomial coefficients after rounding each free identity
    # coordinate, assigning its opposite to the group's reference occurrence.
    den = 2**20
    rounded = [round(float(x) * den) for x in theta.value]
    expected = dict(prior)
    alpha = {}
    if mode == "cubic":
        for value, (first, pair) in zip(rounded, variables):
            alpha[first] = alpha.get(first, Fraction()) - Fraction(value, den)
            alpha[pair] = alpha.get(pair, Fraction()) + Fraction(value, den)
    else:
        rounded = [max(0, value) for value in rounded]
        for value, support in zip(rounded, supports):
            for (a, sa), (c, sc) in combinations(support, 2):
                expected[a, c] -= Fraction(value, den) * sa * sc
        constant -= sum(
            Fraction(value * int(rhs), den) for value, rhs in zip(rounded, right)
        )
    expanded = sorted(set(edges) | set(alpha))
    scale = lcm(den, *(w.denominator for w in expected.values()))
    weights = [
        int((expected.get(e, Fraction()) + alpha.get(e, Fraction())) * scale)
        for e in expanded
    ]
    assert identity_check(n, triples, expanded, weights, scale, expected)
    local_delta = np.asarray(operator @ np.array(rounded, dtype=float) / den).reshape(
        (r, r), order="F"
    )
    T = base + np.diag(b.value) + local_delta
    minimum = float(
        eigh(T, subset_by_index=[0, 0], eigvals_only=True, check_finite=False)[0]
    )
    shift = max(0, -minimum) + 1e-9
    small = np.tril(cholesky(T + np.eye(r) * shift, lower=True, check_finite=False))
    ordered = sp.bmat(
        [[L0, None], [sp.csr_matrix(W), sp.csr_matrix(small)]], format="csr"
    )
    order = np.array(outside + U)
    factor = ordered[np.argsort(order), :]
    assert factor.nnz <= MAX_FACTOR_ENTRIES
    norm = float(np.max(np.asarray(factor.multiply(factor).sum(axis=1))))
    bits = min(24, int(np.floor(np.log2(np.sqrt(2**50 / norm)))))
    assert bits >= 0
    D = 2**bits
    factor.data = np.rint(factor.data * D).astype(np.int64)
    factor.eliminate_zeros()
    factor.sort_indices()
    packed = tuple(
        tuple(map(int, row)) for row in (factor.indptr, factor.indices, factor.data)
    )
    factor_done = time.perf_counter() - start
    lower = constant + gram_lower_bound(expanded, weights, scale, packed, D)
    elapsed = time.perf_counter() - start
    witness = {
        "n": n,
        "triples": triples,
        "edges": expanded,
        "weights": weights,
        "scale": scale,
        "factor": packed,
        "denominator": D,
        "constant": str(constant),
    }
    check_start = time.perf_counter()
    assert identity_check(n, triples, expanded, weights, scale, expected)
    assert constant + gram_lower_bound(expanded, weights, scale, packed, D) == lower
    check = time.perf_counter() - check_start
    record = {
        "mode": mode,
        "optimizer": optimizer,
        "all_cuts": all_cuts,
        "tolerance": tolerance,
        "warm_start": True,
        "initial_minimum": initial_min,
        "initial_bound": initial_bound,
        "solver_seconds": solver_seconds,
        "starting_bound": data["scalar_pentagonal_bound"],
        "bound": float(lower),
        "exact": str(lower),
        "setup": setup,
        "solve_done": solve_done,
        "factor_done": factor_done,
        "seconds": elapsed,
        "independent_seconds": check,
        "total_additional_seconds": elapsed + check,
        "n": n,
        "lifted_size": size,
        "schur_size": r,
        "variables": theta.size,
        "status": run_status,
        "iterations": run_iterations,
        "local_shift": shift,
        "factor_entries": len(packed[2]),
        "identity_valid": True,
        "independent_valid": True,
        "max_moment_violation": float(np.max(abs(delta), initial=0))
        if optimizer == "mixing"
        else None,
        "rank": V.shape[1] if optimizer == "mixing" else None,
    }
    print(json.dumps(record), flush=True)
    output = Path(output)
    output.with_suffix(".json").write_text(json.dumps(record, indent=2))
    output.with_name(output.name + "-proof.json.gz").write_bytes(
        gzip.compress(json.dumps(witness, separators=(",", ":")).encode())
    )
    return record


def prepare(case_path, proof_path, vectors_path, output):
    """Check a source proof and select the common 32-support input, untimed."""
    from dataclasses import replace

    import highs_turbo.ising_geometry as geometry
    from experiments.sparse_geometric_followup import pair_parity, parity_rows
    from highs_turbo.ising import _normalize
    from highs_turbo.ising_cuts import check_certificate, cut_matrix, problem_digest
    from highs_turbo.ising_sparse import fit_vectors, independent_groups

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    case = json.loads(Path(case_path).read_text())
    J = {(u, v): w for u, v, w in case["couplings"]}
    labels, h, weights, constant = _normalize(case["fields"], J, 0)
    weights.update({(i, len(labels)): value for i, value in enumerate(h) if value})
    base_edges = sorted(weights)
    source = IsingCertificate.from_dict(
        json.loads(gzip.decompress(Path(proof_path).read_bytes()))
    )
    assert check_certificate(source, base_edges, weights, constant)
    edges = [*base_edges, *source.extra_edges]
    weights = {**weights, **dict.fromkeys(source.extra_edges, Fraction())}
    source = replace(
        source, problem_digest=problem_digest(edges, weights, constant), extra_edges=()
    )
    V = np.load(vectors_path)
    row = cut_matrix(source.cuts, len(edges))
    lam = np.array([float(Fraction(a, source.denominator)) for a in source.multipliers])
    residual = np.array([float(weights[e]) for e in edges]) - row.T @ lam
    u, v = np.array(edges).T
    M = sp.csr_matrix(
        (np.r_[residual, residual] / 2, (np.r_[u, v], np.r_[v, u])),
        shape=(len(V), len(V)),
    )
    start = time.perf_counter()
    V = fit_vectors(M, V, independent_groups(edges, len(V)), deadline=start + 2)
    warmup = time.perf_counter() - start
    start = time.perf_counter()
    supports, stats = pair_parity(V, edges, limit=32, seed=0)
    selection = time.perf_counter() - start
    old_search, old_rows = geometry.geometric_triangles, geometry.triangle_rows
    try:
        geometry.geometric_triangles = lambda *args, **kwargs: (supports, stats)
        geometry.triangle_rows = parity_rows
        start = time.perf_counter()
        proposal = geometry.strengthen_geometry(
            source, edges, weights, constant, V, deadline=start + 8, seed=0
        )
        fit_time = time.perf_counter() - start
    finally:
        geometry.geometric_triangles, geometry.triangle_rows = old_search, old_rows
    start = time.perf_counter()
    assert check_certificate(proposal, edges, weights, constant)
    checked = time.perf_counter() - start
    all_edges = [*edges, *proposal.extra_edges]
    all_weights = {**weights, **dict.fromkeys(proposal.extra_edges, Fraction())}
    prior = []
    pent = []
    mult = []
    for cut, lam_num in zip(proposal.cuts, proposal.multipliers):
        if cut.kind == "subgraph" and len(cut.indices) == 10:
            nodes = sorted({v for i in cut.indices for v in all_edges[i]})
            coefs = {all_edges[i]: a for i, a in zip(cut.indices, cut.coefficients)}
            signs = {nodes[0]: 1, **{v: coefs[nodes[0], v] for v in nodes[1:]}}
            assert all(coefs[a, b] == signs[a] * signs[b] for a, b in coefs)
            pent.append([(v, signs[v]) for v in nodes])
            mult.append(Fraction(lam_num, proposal.denominator))
        else:
            prior.append((cut, Fraction(lam_num, proposal.denominator)))
    residual = dict(all_weights)
    fixed_constant = constant
    for cut, multiplier in prior:
        fixed_constant += multiplier * (sum(cut.coefficients) - 2 * cut.rhs)
        for index, coef in zip(cut.indices, cut.coefficients):
            residual[all_edges[index]] -= multiplier * coef
    assert pent
    np.save(output / "vectors.npy", V)
    record = {
        "case": Path(case_path).stem,
        "n": len(V),
        "edges": all_edges,
        "residual": [str(residual[e]) for e in all_edges],
        "constant": str(fixed_constant),
        "pent": pent,
        "multipliers": list(map(str, mult)),
        "before": float(source.lower_bound),
        "scalar_pentagonal_bound": float(proposal.lower_bound),
        "warmup_seconds": warmup,
        "selection_seconds": selection,
        "fit_seconds": fit_time,
        "check_seconds": checked,
        "selection": stats,
    }
    (output / "input.json").write_text(json.dumps(record, separators=(",", ":")))
    (output / "source-proof.json.gz").write_bytes(
        gzip.compress(json.dumps(proposal.to_dict(), separators=(",", ":")).encode())
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in record.items()
                if k not in ("edges", "residual", "pent", "multipliers")
            }
        ),
        flush=True,
    )

    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("scalar", "cubic"), default="cubic")
    parser.add_argument("--optimizer", choices=("scs", "mixing"), default="mixing")
    parser.add_argument("--all-cuts", action="store_true")
    parser.add_argument("--seconds", type=float, default=15)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    args = parser.parse_args()
    run(
        args.input,
        args.output,
        mode=args.mode,
        optimizer=args.optimizer,
        all_cuts=args.all_cuts,
        solver_seconds=args.seconds,
        tolerance=args.tolerance,
    )


if __name__ == "__main__":
    main()
