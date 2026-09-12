"""Research experiments; these candidates are not used by the public solver.

The two-pass candidate failed its G55 comparison. The five-variable separator
uses known pentagonal inequalities and did not beat another triangle pass.
See SPARSE_FOLLOWUP_RESULTS.md for timings, limitations, and reproduction.
"""

import argparse
import gzip
import json
import time
from collections import Counter
from dataclasses import replace
from fractions import Fraction
from itertools import combinations
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.spatial import cKDTree

import highs_turbo.ising_geometry as geometry
from highs_turbo.ising import _normalize
from highs_turbo.ising_cuts import (
    IsingCertificate,
    IsingCut,
    _bound,
    check_certificate,
    cut_matrix,
    make_certificate,
    problem_digest,
)
from highs_turbo.ising_geometry import geometric_triangles, triangle_rows
from highs_turbo.ising_sparse import factor_witness, fit_vectors, independent_groups


def pair_parity(vectors, edges, *, limit=512, seed=0, epsilon=4, queries=4096):
    started = time.perf_counter()
    n = len(vectors)
    rng = np.random.default_rng(seed)
    pairs = np.asarray(edges, dtype=int).reshape(-1, 2)
    if len(pairs) > 16 * n:
        pairs = pairs[rng.choice(len(pairs), 16 * n, replace=False)]
    pairs = np.r_[pairs, rng.integers(0, n, size=(4 * n, 2))]
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    sums = np.r_[
        vectors[pairs[:, 0]] + vectors[pairs[:, 1]],
        vectors[pairs[:, 0]] - vectors[pairs[:, 1]],
    ]
    pair_nodes = np.tile(pairs, (2, 1))
    pair_signs = np.c_[
        np.ones(2 * len(pairs), dtype=int), np.repeat([1, -1], len(pairs))
    ]
    index = cKDTree(np.r_[sums, -sums])
    pair_nodes = np.tile(pair_nodes, (2, 1))
    pair_signs = np.r_[pair_signs, -pair_signs]
    neighbors = [[] for _ in range(n)]
    for u, v in edges:
        neighbors[u].append(v)
        neighbors[v].append(u)
    triples = []
    for _ in range(8 * n):
        middle = int(rng.integers(n))
        near = neighbors[middle]
        if len(near) < 2:
            continue
        u, v = rng.choice(near, size=2, replace=False)
        triples.append((int(u), middle, int(v)))
    if not triples:
        return [], {
            "seconds": time.perf_counter() - started,
            "candidates": 0,
            "queries": 0,
        }
    triples = np.asarray(triples)
    signs = np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1]])
    values = np.asarray(
        [sum(a * vectors[triples[:, j]] for j, a in enumerate(sign)) for sign in signs]
    )
    norms = np.sum(values * values, axis=2)
    # Seed with triples not already separated by a triangle inequality.
    valid = np.flatnonzero(np.min(norms, axis=0) > 1.0001)
    if len(valid) > queries:
        valid = rng.choice(valid, queries, replace=False)
    selected_sign = np.argmin(norms[:, valid], axis=0)
    targets = -values[selected_sign, valid]
    triples = triples[valid]
    triple_signs = signs[selected_sign]
    distances, indices = index.query(targets, k=4, eps=epsilon, workers=1)
    candidates = {}
    for q, j in zip(*np.where(distances < 0.999)):
        p = indices[q, j]
        nodes = np.r_[triples[q], pair_nodes[p]]
        if len(set(nodes)) != 5:
            continue
        coefficients = np.r_[triple_signs[q], pair_signs[p]]
        order = np.argsort(nodes)
        nodes = nodes[order]
        coefficients = coefficients[order]
        coefficients *= coefficients[0]
        key = tuple(zip(map(int, nodes), map(int, coefficients)))
        candidates[key] = float(distances[q, j])
    chosen = []
    usage = Counter()
    for key in sorted(candidates, key=candidates.get):
        if any(usage[i] >= 16 for i, a in key):
            continue
        chosen.append(key)
        usage.update(i for i, a in key)
        if len(chosen) >= limit:
            break
    return chosen, {
        "seconds": time.perf_counter() - started,
        "candidates": len(candidates),
        "queries": len(targets),
        "pairs": len(pair_nodes),
        "best": min(candidates.values(), default=1),
    }


def parity_rows(parity, edges):
    extra = sorted(
        {tuple(sorted((u, v))) for b in parity for (u, a), (v, c) in combinations(b, 2)}
        - set(edges)
    )
    expanded = [*edges, *extra]
    locations = {e: i for i, e in enumerate(expanded)}
    cuts = []
    for b in parity:
        terms = sorted(
            (locations[tuple(sorted((u, v)))], a * c)
            for (u, a), (v, c) in combinations(b, 2)
        )
        cuts.append(
            IsingCut(
                tuple(i for i, a in terms),
                tuple(a for i, a in terms),
                (sum(a for i, a in b) ** 2 - 1) // 4,
                "subgraph",
            )
        )
    return expanded, cuts


def strengthen_two_pass(proof, edges, weights, constant, vectors, *, deadline, seed=0):
    n = len(vectors)
    expanded = list(edges)
    cuts = list(proof.cuts)
    magnitude = max(abs(float(w)) for w in weights.values())
    chosen = np.array(
        [float(Fraction(a, proof.denominator)) / magnitude for a in proof.multipliers]
    )
    V = vectors.copy()
    matrix = None
    calls = 0

    def fit(multipliers, iterations):
        nonlocal V, matrix
        residual = original - row.T @ multipliers
        matrix = sp.csr_matrix(
            (np.r_[residual, residual] / 2, (np.r_[u, v], np.r_[v, u])),
            shape=(n, n),
        )
        V = fit_vectors(
            matrix,
            V,
            groups,
            deadline=phase_deadline,
            iterations=iterations,
            tolerance=1e-6,
        )
        return float(np.sum(V * (matrix @ V)))

    def dual(multipliers):
        nonlocal chosen, calls
        if calls and time.perf_counter() >= phase_deadline:
            raise StopIteration
        chosen = multipliers.copy()
        value = fit(chosen, 300 if calls == 0 else 150)
        calls += 1
        return -value - shift @ chosen, row @ np.sum(V[u] * V[v], axis=1) - shift

    for stage in range(2):
        phase_start = time.perf_counter()
        phase_deadline = phase_start + (0.45 if stage == 0 else 0.75) * (
            deadline - phase_start
        )
        batch, stats = geometric_triangles(V, expanded, seed=seed + stage, epsilon=4)
        expanded, new = triangle_rows(batch, expanded)
        known = set(cuts)
        new = [c for c in new if c not in known]
        cuts += new
        chosen = np.r_[chosen, np.zeros(len(new))]
        weights = {**weights, **dict.fromkeys(expanded[len(edges) :], Fraction())}
        row = cut_matrix(cuts, len(expanded))
        shift = np.asarray(row.sum(axis=1)).ravel() - 2 * np.array(
            [c.rhs for c in cuts]
        )
        original = np.array([float(weights[e]) / magnitude for e in expanded])
        u, v = np.array(expanded).T
        groups = independent_groups(expanded, n)
        calls = 0

        if new and time.perf_counter() < phase_deadline:
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
        print(
            json.dumps(
                {
                    "phase": "adaptive_geometry",
                    "stage": stage,
                    "seconds": time.perf_counter() - phase_start,
                    "cuts": len(cuts),
                    "calls": calls,
                    "search": stats,
                }
            ),
            flush=True,
        )
        if time.perf_counter() + 0.3 >= deadline:
            break
    source = make_certificate(cuts, -chosen * magnitude, expanded, weights, constant)
    exact = {
        c: float(Fraction(a, source.denominator)) / magnitude
        for c, a in zip(source.cuts, source.multipliers)
    }
    phase_deadline = deadline
    fit(np.array([exact.get(c, 0) for c in cuts]), 300)
    try:
        packed, denominator = factor_witness(matrix, V, magnitude)
        lower = _bound(
            source.cuts,
            source.multipliers,
            source.denominator,
            expanded,
            weights,
            constant,
            (),
            denominator,
            packed,
        )
        print(
            json.dumps(
                {
                    "phase": "candidate_bound",
                    "bound": float(lower),
                    "base": float(proof.lower_bound),
                    "factor_entries": len(packed[1]),
                }
            ),
            flush=True,
        )
        if lower <= proof.lower_bound:
            return proof
        return replace(
            source,
            problem_digest=proof.problem_digest,
            lower_bound=lower,
            sparse_gram_factor=packed,
            gram_denominator=denominator,
            extra_edges=tuple(expanded[len(edges) :]),
        )
    except (ValueError, OverflowError, np.linalg.LinAlgError, RuntimeError) as error:
        print(json.dumps({"phase": "candidate_error", "error": str(error)}), flush=True)
        return proof


def append_batch(args):
    case_name = args.case.stem
    mode = args.mode.removeprefix("append-")
    seconds = args.seconds
    case = json.loads(args.case.read_text())
    labels, fields, weights, constant = _normalize(
        case["fields"], {(u, v): w for u, v, w in case["couplings"]}, 0
    )
    weights.update({(i, len(labels)): v for i, v in enumerate(fields) if v})
    edges = sorted(weights)
    source = IsingCertificate.from_dict(read_proof(args.proof))
    expanded = [*edges, *source.extra_edges]
    expanded_weights = {**weights, **dict.fromkeys(source.extra_edges, Fraction())}
    rebased = replace(
        source,
        problem_digest=problem_digest(expanded, expanded_weights, constant),
        extra_edges=(),
    )
    V = np.load(args.vectors)
    start = time.perf_counter()
    row = cut_matrix(source.cuts, len(expanded))
    residual = np.array(
        [float(expanded_weights[e]) for e in expanded]
    ) - row.T @ np.array(
        [float(Fraction(a, source.denominator)) for a in source.multipliers]
    )
    u, v = np.array(expanded).T
    matrix = sp.csr_matrix(
        (np.r_[residual, residual] / 2, (np.r_[u, v], np.r_[v, u])),
        shape=(len(V), len(V)),
    )
    groups = independent_groups(expanded, len(V))
    fit_vectors(matrix, V, groups, deadline=time.perf_counter() + 2)
    print(
        json.dumps(
            {
                "case": case_name,
                "phase": "warmup",
                "seconds": time.perf_counter() - start,
                "vector_objective": float(np.sum(V * (matrix @ V))),
            }
        ),
        flush=True,
    )
    search = pair_parity if mode == "five" else geometry.geometric_triangles
    stats = {}

    def measured_search(*args, **kwargs):
        batch, data = search(*args, **kwargs)
        stats.update(data)
        stats["kept"] = len(batch)
        print(
            json.dumps(dict(case=case_name, phase="search", mode=mode, **stats)),
            flush=True,
        )
        return batch, data

    geometry.geometric_triangles = measured_search
    if mode == "five":
        geometry.triangle_rows = parity_rows
    start = time.perf_counter()
    proposal = geometry.strengthen_geometry(
        rebased,
        expanded,
        expanded_weights,
        constant,
        V,
        deadline=start + seconds,
        seed=case.get("seed", 55),
    )
    proposal = replace(
        proposal,
        problem_digest=source.problem_digest,
        extra_edges=(*source.extra_edges, *proposal.extra_edges),
    )
    finished = time.perf_counter()
    valid = check_certificate(proposal, edges, weights, constant)
    end = time.perf_counter()
    record = {
        "case": case_name,
        "mode": mode,
        "budget": seconds,
        "previous": float(source.lower_bound),
        "bound": float(proposal.lower_bound),
        "valid": valid,
        "solve_seconds": finished - start,
        "check_seconds": end - finished,
        "elapsed_seconds": end - start,
        "cuts": len(proposal.cuts),
        "newcuts": sum(c not in set(source.cuts) for c in proposal.cuts),
        "extra_edges": len(proposal.extra_edges),
        "factor_entries": len(proposal.sparse_gram_factor[1]),
        "search": stats,
    }
    print(json.dumps(record), flush=True)


def read_proof(path):
    raw = path.read_bytes()
    if path.suffix == ".gz":
        raw = gzip.decompress(raw)
    data = json.loads(raw)
    return data.get("source_certificate", data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("production", "two-pass", "append-three", "append-five"),
        required=True,
    )
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--proof", type=Path)
    parser.add_argument("--vectors", type=Path)
    args = parser.parse_args()
    if args.mode.startswith("append-"):
        if args.proof is None or args.vectors is None:
            parser.error(
                "append experiments require the common input proof and vectors"
            )
        append_batch(args)
    else:
        from examples.benchmark_geometric_ising import worker

        if args.mode == "two-pass":
            geometry.strengthen_geometry = strengthen_two_pass
        worker(
            argparse.Namespace(
                case=args.case,
                seconds=args.seconds,
                source=None,
                method="auto",
                proof=None,
            )
        )


if __name__ == "__main__":
    main()
