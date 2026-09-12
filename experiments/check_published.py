"""Apply the same exact residual bounds to published solvers' numerical output.

Repairs are proposals only: every returned bound is checked using rational
arithmetic, then rounded to the original energy lattice. Retain the strongest
of the trivial bound, direct PSD projection, and coefficient-matching repair.
"""

from dataclasses import replace
from fractions import Fraction
from functools import reduce
from math import gcd, lcm
from operator import xor
import json
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.linalg import cholesky, eigh

from highs_turbo.ising_cuts import IsingCut, make_certificate, _bound, check_certificate


def lattice_bound(bound, weights):
    weights = list(map(Fraction, weights))
    scale = lcm(*(w.denominator for w in weights))
    lattice = reduce(gcd, (w.numerator * (scale // w.denominator) for w in weights), 0)
    if lattice:
        base = sum(weights, Fraction())
        step = Fraction(2 * lattice, scale)
        distance = (bound - base) / step
        bound = base + (-(-distance.numerator // distance.denominator)) * step
    return bound


def pack_factor(matrix):
    n = len(matrix)
    factor = np.tril(cholesky(matrix, lower=True, check_finite=False))
    maximum = abs(factor).max()
    bits = min(20, int(np.floor(np.log2(np.sqrt(2**52 / n) / maximum))))
    if bits < 0:
        raise ValueError("Gram proposal exceeds the exact integer range")
    denominator = 2**bits
    integers = np.rint(factor * denominator).astype(np.int64)
    packed = tuple(tuple(map(int, integers[i, : i + 1])) for i in range(n))
    return packed, denominator


def mixing(data, output, path):
    began = time.perf_counter()
    n = data["n"]
    edges = [tuple(e) for e in data["edges"]]
    weights = {e: Fraction(w) for e, w in zip(edges, data["weights"])}
    cuts = [
        IsingCut(tuple(c["indices"]), tuple(c["coefficients"]), c["rhs"], c["kind"])
        for c in data.get("cuts", [])
    ]
    source = make_certificate(
        cuts, -np.maximum(output["dual"][n:], 0), edges, weights, Fraction()
    )
    candidates = {
        "trivial": -sum(map(abs, weights.values())),
        "cuts": source.lower_bound,
    }
    if n > 2048:
        from highs_turbo.ising_sparse import factor_witness

        residual = dict(weights)
        for cut, numerator in zip(source.cuts, source.multipliers):
            for index, coefficient in zip(cut.indices, cut.coefficients):
                residual[edges[index]] -= Fraction(
                    numerator * coefficient, source.denominator
                )
        u, v = np.asarray(edges).T
        values = np.asarray([float(residual[e]) / 2 for e in edges])
        matrix = sp.csr_matrix(
            (np.r_[values, values], (np.r_[u, v], np.r_[v, u])), shape=(n, n)
        )
        vectors = (
            np.fromfile(path + ".vectors", dtype=np.float64)
            .reshape(output["rank"], n, order="F")
            .T
        )
        vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1)[:, None], 1e-100)
        # Offer both returned dual coefficients and returned vector directions
        # to the exact same sparse proposal/checker used by the new engine.
        # No optimization or refitting is hidden in certificate repair.
        errors = {}
        for name, objective, point in [
            ("sparse_vectors", matrix, vectors),
            (
                "sparse_dual",
                matrix - sp.diags(np.asarray(output["dual"][:n])),
                np.zeros((n, 1)),
            ),
        ]:
            try:
                factor, denominator = factor_witness(objective, point)
                bound = _bound(
                    source.cuts,
                    source.multipliers,
                    source.denominator,
                    edges,
                    weights,
                    Fraction(),
                    (),
                    denominator,
                    factor,
                )
                proof = replace(
                    source,
                    lower_bound=bound,
                    sparse_gram_factor=factor,
                    gram_denominator=denominator,
                )
                assert check_certificate(proof, edges, weights, Fraction())
                candidates[name] = bound
            except (ValueError, RuntimeError, OverflowError) as error:
                errors[name] = str(error)
        bound = max(candidates.values())
        return dict(
            verified=float(bound),
            certificate_seconds=time.perf_counter() - began,
            exact=[bound.numerator, bound.denominator],
            repairs={k: float(v) for k, v in candidates.items()},
            repair_errors=errors,
        )
    slack = np.fromfile(path + ".gram", dtype=np.float64).reshape(n, n, order="F")
    eig, vec = eigh((slack + slack.T) / 2, check_finite=False)
    projected = (vec * np.maximum(eig, 0)) @ vec.T
    projected.flat[:: n + 1] += 1e-8

    # Match the rounded cut multipliers and original objective exactly in the
    # off-diagonals. A diagonal PSD repair charges the trace correction only.
    # This is the same construction used by our own low-rank SDP solver.
    residual = dict(weights)
    for cut, numerator in zip(source.cuts, source.multipliers):
        for index, coefficient in zip(cut.indices, cut.coefficients):
            residual[edges[index]] -= Fraction(
                numerator * coefficient, source.denominator
            )
    matched = np.diag(-np.array(output["dual"][:n]))
    for (u, v), value in residual.items():
        matched[u, v] = matched[v, u] = float(value) / 2
    smallest = eigh(
        matched, eigvals_only=True, subset_by_index=[0, 0], check_finite=False
    )[0]
    matched.flat[:: n + 1] += max(0.0, -smallest) + 1e-8
    for name, gram in [("projection", projected), ("coefficient_repair", matched)]:
        packed, denominator = pack_factor(gram)
        bound = _bound(
            source.cuts,
            source.multipliers,
            source.denominator,
            edges,
            weights,
            Fraction(),
            packed,
            denominator,
        )
        proof = replace(
            source, lower_bound=bound, gram_factor=packed, gram_denominator=denominator
        )
        assert check_certificate(proof, edges, weights, Fraction())
        candidates[name] = bound
    bound = max(candidates.values())
    return dict(
        verified=float(bound),
        certificate_seconds=time.perf_counter() - began,
        exact=[bound.numerator, bound.denominator],
        repairs={k: float(v) for k, v in candidates.items()},
    )


def tssos(data, output):
    from experiments.lifted_ising import polynomial, quantize_gram, check_sos

    began = time.perf_counter()
    coefficients = polynomial(data["n"], data["edges"], data["weights"])
    bases, grams = [], []
    for block in output["blocks"]:
        bases.append([reduce(xor, (1 << i for i in m), 0) for m in block["basis"]])
        grams.append(np.array(block["gram"]))
    candidates = {"trivial": -sum(abs(v) for m, v in coefficients.items() if m)}
    direct = [quantize_gram(basis, gram) for basis, gram in zip(bases, grams)]
    candidates["projection"] = lattice_bound(
        check_sos(coefficients, direct), data["weights"]
    )

    # Nonconstant coefficient constraints form disjoint groups of off-diagonal
    # entries. Their least-squares projection is one scalar adjustment per group.
    # Diagonal PSD shifts then affect the constant only, since m_S(s)^2 = 1.
    expansion, counts, pairs = {}, {}, []
    for basis, gram in zip(bases, grams):
        indices = [
            (i, j, basis[i] ^ basis[j])
            for i in range(len(basis))
            for j in range(i + 1, len(basis))
        ]
        pairs.append(indices)
        for i, j, mask in indices:
            if mask:
                counts[mask] = counts.get(mask, 0) + 1
                expansion[mask] = expansion.get(mask, 0.0) + gram[i, j] + gram[j, i]
    adjustment = {
        m: (float(coefficients.get(m, 0)) - expansion[m]) / (2 * count)
        for m, count in counts.items()
    }
    repaired = []
    for basis, gram, indices in zip(bases, grams, pairs):
        gram = (gram + gram.T) / 2
        for i, j, mask in indices:
            if mask:
                gram[i, j] += adjustment[mask]
                gram[j, i] = gram[i, j]
        smallest = eigh(
            gram, eigvals_only=True, subset_by_index=[0, 0], check_finite=False
        )[0]
        gram.flat[:: len(gram) + 1] += max(0.0, -smallest) + 1e-9
        repaired.append(quantize_gram(basis, gram))
    candidates["coefficient_repair"] = lattice_bound(
        check_sos(coefficients, repaired), data["weights"]
    )
    bound = max(candidates.values())
    return dict(
        verified=float(bound),
        certificate_seconds=time.perf_counter() - began,
        exact=[bound.numerator, bound.denominator],
        repairs={k: float(v) for k, v in candidates.items()},
    )


if __name__ == "__main__":
    data_path, result_path = sys.argv[1:]
    data = json.load(open(data_path))
    output = json.load(open(result_path))
    checked = (
        mixing(data, output, result_path)
        if output["method"] == "mixing"
        else tssos(data, output)
    )
    print(
        json.dumps(
            dict(
                method=output["method"],
                seconds=output["seconds"],
                numerical=output["numerical"],
                **checked,
            )
        )
    )
