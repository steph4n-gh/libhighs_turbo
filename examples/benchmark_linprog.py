"""One command comparing complete solves of identical LPs, including preparation.

python examples/benchmark_linprog.py --repeats 3
python examples/benchmark_linprog.py --mps /path/to/model.mps

The native baseline uses the same HiGHS library as turbo, separating the
algorithm's contribution from differences in SciPy's bundled HiGHS version.
Generated cases are labelled synthetic. Optional MPS files are read using
SciPy's bundled model reader (SciPy >= 1.15); no files are downloaded.
"""

import argparse
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import scipy
import scipy.sparse as sp
from scipy.optimize import linprog

import highs_turbo
from highs_turbo.compiled_engine import _ce
from highs_turbo.exact_solver import ExactMaxCutSolver
from highs_turbo.graph_generator import generate_k5_cluster_graph


def native_full(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=(0, 1), integrality=None):
    limits = np.broadcast_to(np.asarray(bounds, dtype=float), (len(c), 2))
    types = np.broadcast_to(0 if integrality is None else integrality, (len(c),))
    session = _ce.HighsSession(c, limits[:, 0], limits[:, 1], types)
    for matrix, rhs, is_equality in [(A_ub, b_ub, False), (A_eq, b_eq, True)]:
        if matrix is not None:
            matrix = sp.csr_matrix(matrix)
            session.add_rows(rhs if is_equality else np.full(len(rhs), -np.inf), rhs,
                             matrix.indptr, matrix.indices, matrix.data)
    result = session.solve()
    if result["model_status"] != 7:
        raise RuntimeError(result["message"])
    return result


def generated_cases():
    rng = np.random.default_rng(4)
    for name, rows, cols, integer in [
        ("synthetic dense LP", 5000, 80, False),
        ("synthetic integer-row LP", 2000, 40, True),
    ]:
        A = rng.integers(0, 5, (rows, cols)).astype(float) if integer else rng.random((rows, cols))
        b = A @ np.ones(cols) + rng.integers(0, 10, rows)
        yield name, -rng.random(cols), dict(A_ub=A, b_ub=b, bounds=(0, 2))
    A = sp.random(5000, 100, density=0.08, random_state=rng, format="csr")
    yield "synthetic sparse LP", -rng.random(100), dict(
        A_ub=A, b_ub=A @ np.ones(100) + rng.integers(0, 10, 5000), bounds=(0, 2))
    graph = generate_k5_cluster_graph(100, 99)
    c, A, b, _, _ = ExactMaxCutSolver().build_relaxation_matrices(graph)
    yield "synthetic graph LP", c, dict(A_ub=sp.csr_matrix(A), b_ub=b, bounds=(0, 1))
    A = rng.integers(0, 5, (600, 8)).astype(float)
    yield "synthetic binary MIP", -rng.random(8), dict(
        A_ub=A, b_ub=np.floor(A.sum(axis=1) * 0.6), bounds=(0, 1), integrality=1)


def read_mps(path):
    from scipy.optimize._highspy._core import _Highs

    reader = _Highs()
    reader.setOptionValue("output_flag", False)
    if int(reader.readModel(str(path))) < 0:
        raise ValueError(f"Cannot read {path}")
    lp = reader.getLp()
    if lp.offset_ != 0 or int(lp.sense_) != 1:
        raise ValueError("This comparison expects minimization with no objective offset")
    matrix = sp.csc_matrix((lp.a_matrix_.value_, lp.a_matrix_.index_, lp.a_matrix_.start_),
                           shape=(lp.num_row_, lp.num_col_)).tocsr()
    lower = np.asarray(lp.row_lower_)
    upper = np.asarray(lp.row_upper_)
    equality = lower == upper
    below = (~equality) & (upper < 1e20)
    above = (~equality) & (lower > -1e20)
    return Path(path).name, np.asarray(lp.col_cost_), dict(
        A_ub=sp.vstack([matrix[below], -matrix[above]], format="csr"),
        b_ub=np.concatenate([upper[below], -lower[above]]),
        A_eq=matrix[equality], b_eq=upper[equality],
        bounds=np.column_stack([lp.col_lower_, lp.col_upper_]),
        integrality=np.asarray([int(value) for value in lp.integrality_]) if lp.integrality_ else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--mps", nargs="*", default=[])
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if _ce is None or not hasattr(_ce, "HighsSession"):
        parser.error("Build the native extension before comparing native solves")
    linprog([-1.0], bounds=(0, 1))
    native_full(np.array([-1.0]))
    version = _ce.HighsSession(np.array([0.0]), np.array([0.0]), np.array([1.0]), np.array([0])).version()
    print(f"SciPy {scipy.__version__}; native HiGHS {version}; "
          f"median of {args.repeats} complete solves; milliseconds")
    print("case | rows x cols | scipy | native full | turbo | vs scipy | vs native | rows used")
    cases = list(generated_cases()) + [read_mps(path) for path in args.mps]
    for name, c, kwargs in cases:
        times = {key: [] for key in ("scipy", "native", "turbo")}
        results = {}
        solvers = [("scipy", linprog), ("native", native_full), ("turbo", highs_turbo.linprog)]
        for repeat in range(args.repeats):
            for key, solve in solvers[repeat % 3:] + solvers[:repeat % 3]:
                started = perf_counter()
                result = solve(c, **kwargs)
                times[key].append((perf_counter() - started) * 1000)
                if result["x"] is None:
                    raise RuntimeError(f"{name}: {key} returned no solution")
                results[key] = result
        reference = results["scipy"]["fun"]
        for key, result in results.items():
            np.testing.assert_allclose(result["fun"], reference, rtol=1e-7, atol=1e-7,
                                       err_msg=f"{name}: {key} objective mismatch")
            assert np.max(kwargs["A_ub"] @ result["x"] - kwargs["b_ub"], initial=0) <= 1e-6
        elapsed = {key: np.median(values) for key, values in times.items()}
        shape = kwargs["A_ub"].shape
        used = results["turbo"].get("turbo_rows_used", "SciPy")
        print(f"{name} | {shape[0]} x {shape[1]} | {elapsed['scipy']:.2f} | "
              f"{elapsed['native']:.2f} | {elapsed['turbo']:.2f} | "
              f"{elapsed['scipy']/elapsed['turbo']:.2f}x | "
              f"{elapsed['native']/elapsed['turbo']:.2f}x | {used}", flush=True)


if __name__ == "__main__":
    main()
