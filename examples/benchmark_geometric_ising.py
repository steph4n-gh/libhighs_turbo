"""Focused before/after and drop-in comparisons on saved Ising instances.

Each worker is a fresh process. Inputs and imports are prepared before the
timer; everything inside the optimizer call, including its proof, is timed.
"""

import argparse
from fractions import Fraction
import json
from pathlib import Path
import sys
import time

import highspy
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog as scipy_linprog


def linearize(fields, couplings):
    n = len(fields)
    edges = list(couplings)
    m = len(edges)
    c = np.r_[2 * np.asarray(fields), 4 * np.asarray(list(couplings.values()))]
    for (u, v), weight in couplings.items():
        c[u] -= 2 * weight
        c[v] -= 2 * weight
    u, v = np.asarray(edges).T
    y = n + np.arange(m)
    rows = np.repeat(np.arange(3 * m), [2, 2, 3] * m)
    columns = np.stack([y, u, y, v, u, v, y], axis=1).ravel()
    values = np.tile([1, -1, 1, -1, 1, 1, -1], m)
    matrix = sp.csr_matrix((values, (rows, columns)), shape=(3 * m, n + m))
    rhs = np.tile([0.0, 0.0, 1.0], m)
    constant = sum((Fraction(w) for w in couplings.values()), Fraction()) - sum(
        map(Fraction, fields)
    )
    return c, matrix, rhs, np.r_[np.ones(n), np.zeros(m)], constant


def worker(args):
    root = args.source or Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from highs_turbo import solve_ising, verify_ising_certificate, linprog
    from highs_turbo.lp_accelerator import _add_rows, _check_status

    data = json.loads(args.case.read_text())
    fields = data["fields"]
    couplings = {(u, v): w for u, v, w in data["couplings"]}
    n = len(fields)
    if args.method in ("auto", "previous_hybrid", "cuts", "sdp"):
        options = (
            {"relaxation": "hybrid"}
            if args.method == "previous_hybrid"
            else ({"relaxation": args.method} if args.method in ("cuts", "sdp") else {})
        )
        began = time.perf_counter()
        result = solve_ising(
            fields, couplings, time_limit=args.seconds, seed=0, threads=1, **options
        )
        elapsed = time.perf_counter() - began
        assert verify_ising_certificate(fields, couplings, result.certificate)
        if args.proof:
            from highs_turbo.ising import _normalize

            labels, h, weights, constant = _normalize(fields, couplings, 0)
            weights.update(
                {(i, len(labels)): value for i, value in enumerate(h) if value}
            )
            edges = sorted(weights) + list(result.certificate.extra_edges)
            args.proof.write_text(
                json.dumps(
                    dict(
                        n=len(labels) + bool(any(h)),
                        edges=edges,
                        weights=[float(weights.get(e, 0)) for e in edges],
                        cuts=[
                            dict(
                                indices=c.indices,
                                coefficients=c.coefficients,
                                rhs=c.rhs,
                                kind=c.kind,
                            )
                            for c in result.certificate.cuts
                        ],
                        seed=0,
                        source_certificate=result.certificate.to_dict(),
                    ),
                    separators=(",", ":"),
                )
            )
        spins = result.spins
        record = dict(
            energy=result.energy,
            lower_bound=result.lower_bound,
            verified_bound=float(result.exact_cut_lower_bound),
            extra_edges=len(getattr(result.certificate, "extra_edges", ())),
            sparse_factor_entries=(
                len(result.certificate.sparse_gram_factor[2])
                if getattr(result.certificate, "sparse_gram_factor", ())
                else 0
            ),
            status=result.status,
            progress=result.progress,
        )
    else:
        c, matrix, rhs, types, constant = linearize(fields, couplings)
        began = time.perf_counter()
        if args.method == "native_highs":
            session = highspy.Highs()
            for key, value in {"output_flag": False, "parallel": "off"}.items():
                _check_status(session.setOptionValue(key, value))
            _check_status(
                session.addCols(
                    len(c), c, np.zeros(len(c)), np.ones(len(c)), 0, [], [], []
                )
            )
            _check_status(
                session.changeColsIntegrality(
                    n, np.arange(n), np.ones(n, dtype=np.uint8)
                )
            )
            _add_rows(session, matrix, np.full(len(rhs), -np.inf), rhs)
            _check_status(
                session.setOptionValue(
                    "time_limit", max(0.0, args.seconds - (time.perf_counter() - began))
                )
            )
            _check_status(session.run())
            info, solution = session.getInfo(), session.getSolution()
            x = np.asarray(solution.col_value) if solution.value_valid else None
            lower = info.mip_dual_bound if info.valid else -float("inf")
            verified = None
            status = session.modelStatusToString(session.getModelStatus())
            strategy = "original_highspy_model"
        else:
            solve = linprog if args.method == "turbo_linprog" else scipy_linprog
            result = solve(
                c,
                A_ub=matrix,
                b_ub=rhs,
                bounds=(0, 1),
                integrality=types,
                options={"time_limit": args.seconds},
            )
            x = result.x
            lower = result.get("mip_dual_bound", -float("inf"))
            verified = result.get("turbo_exact_lower_bound")
            status = result.message
            strategy = result.get("turbo_strategy", "scipy")
        elapsed = time.perf_counter() - began
        spins = (
            None if x is None else dict(enumerate(2 * np.rint(x[:n]).astype(int) - 1))
        )
        if x is not None:
            assert np.max(matrix @ x - rhs) <= 1e-6
            assert np.max(abs(x[:n] - np.rint(x[:n]))) <= 1e-6
        record = dict(
            lower_bound=float(lower + constant),
            verified_bound=None if verified is None else float(verified + constant),
            status=status,
            strategy=strategy,
        )
    if spins is not None:
        exact = sum((Fraction(w) * spins[i] for i, w in enumerate(fields)), Fraction())
        exact += sum(
            Fraction(w) * spins[u] * spins[v] for (u, v), w in couplings.items()
        )
        record["energy"] = float(exact)
        if record["verified_bound"] is not None:
            assert record["verified_bound"] <= float(exact)
    else:
        record["energy"] = None
    record.update(
        case=data["name"],
        method=args.method,
        seconds=elapsed,
        budget=args.seconds,
        nodes=n,
        edges=len(couplings),
    )
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=5)
    parser.add_argument("--source", type=Path)
    parser.add_argument(
        "--proof",
        type=Path,
        help="Export the active relaxation for published baselines",
    )
    parser.add_argument(
        "--method",
        choices=[
            "auto",
            "previous_hybrid",
            "cuts",
            "sdp",
            "turbo_linprog",
            "scipy_linprog",
            "native_highs",
        ],
        required=True,
    )
    worker(parser.parse_args())
