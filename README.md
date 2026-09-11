# highs_turbo: Transparent LP Acceleration for HiGHS & SciPy

[![Tests](https://github.com/steph4n-gh/libhighs_turbo/actions/workflows/tests.yml/badge.svg)](https://github.com/steph4n-gh/libhighs_turbo/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

`highs_turbo.linprog` solves the supplied linear program using conic optimality
checks, smaller working sets, and competing HiGHS methods where these help. It preserves
the original problem and returns SciPy-style solutions, slacks, and marginals.
The package also provides Max-Cut and QUBO solvers and a C++20 cutting-plane
engine with rational verification.

## Installation

Python 3.10 or newer is required. NumPy, SciPy (1.9 or newer, for `milp`),
NetworkX, and the official HiGHS Python package (`highspy`) are installed
automatically. LP acceleration runs on Linux, macOS, and Windows without
compiling this project's C++ extension.

```bash
git clone https://github.com/steph4n-gh/libhighs_turbo.git
cd libhighs_turbo
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

The optional C++ cut engine uses macOS CommonCrypto. For its additional graph
cut operations, install the Xcode command line tools and native dependencies:

```bash
brew install gmp highs
python -m pip install .
```

If the optional extension cannot compile, LP acceleration still works through
`highspy`; graph operations use their Python implementations. Native cut
separation and native-only tests require a successful extension build.
Check the optional extension with:

```bash
python -c 'from highs_turbo.compiled_engine import COMPILED_ENGINE_AVAILABLE; print(COMPILED_ENGINE_AVAILABLE)'
```

To use the bundled neural weights or train models, install the optional extra:

```bash
python -m pip install '.[ml]'
```

The weights are shipped inside the installed package. Max-Cut uses geometric
weights when PyTorch is unavailable. Ordinary `linprog` calls do not load PyTorch
or require a trained model.

## Quickstart

### Linear programming

```python
import highs_turbo as opt

res = opt.linprog(
    [-1.0, 4.0],
    A_ub=[[-3.0, 1.0], [1.0, 2.0]],
    b_ub=[6.0, 4.0],
)
print(res.fun, res.nit)
print(res.turbo_accelerated)
```

The default `method="highs"` and the legacy `"turbo"` alias use the same
transparent path:

1. For suitable continuous LPs, construct a candidate primal solution and conic
   dual bound. Return it only when primal feasibility, dual stationarity,
   complementarity, and the objective gap pass checks against the original input.
2. Otherwise, solve a smaller working model in one persistent HiGHS instance.
   Small integer-coefficient rows can be summed exactly into surrogate rows.
   Restore violated original constraints while retaining the simplex basis.
3. Return a working-model optimum only after it satisfies every original
   inequality. Lift its dual multipliers back to the original row order.
   If recovery needs too many rounds, restore all rows and finish the solve.
4. Sparse and equality-heavy LPs that do not suit row reduction use HiGHS
   directly. For large models, give simplex a 5 ms head start; if it is still
   running, start an independent interior-point solve. Return an optimal result,
   cancel the other solve, and join both workers before returning. This can use
   two CPU cores and two model copies; each call starts from scratch.

The mathematical reason this preserves the answer is simple: the working model
is a relaxation of the original. An optimal relaxation solution that is feasible
for the original is also optimal for the original, subject to the requested
numerical tolerances. Graph hints never authorize adding constraints that change
the supplied LP. The LP conic check uses floating-point tolerances; it is not an
exact rational certificate of the LP optimum.

Integer models, explicit time/iteration/node budgets, other solver methods, and
unsupported options retain ordinary SciPy execution. Small continuous LPs use
the direct native path. An explicit simplex edge-weight strategy also keeps a
single native method. Use `method="highs-ds"` to retain SciPy's simplex path.
Successful native results expose `turbo_strategy`, `turbo_rows_used`,
`turbo_original_rows`, and `turbo_rounds`; `nit` includes all working solves.
The direct and competing-method paths also expose `turbo_solver` and
`turbo_solver_iterations`, including work done by a canceled method.
`turbo_accelerated` means a reduction or competing-method path was used; it is
not a claim that the particular call ran faster. Inequality row counts exclude
the equalities reported in `eqlin`.
Call `scipy.optimize.linprog` directly for a baseline comparison.

Graph edge relaxations need `bounds=(0, 1)`. Omitting bounds retains SciPy's
nonnegative, unbounded-above default. A complete runnable graph example is in
[`examples/01_quickstart_3_lines.py`](examples/01_quickstart_3_lines.py).

### Max-Cut

```python
import numpy as np
import highs_turbo

adjacency = np.ones((3, 3)) - np.eye(3)
result = highs_turbo.solve_maxcut(adjacency)
cut_value, partition, receipt = result
print(cut_value, partition, result.upper_bound, result.status)
```

Inputs may be a `GraphInstance`, NetworkX graph, dense adjacency matrix, or SciPy
sparse matrix. Graphs with at most 200 nodes use a MILP for the integer solution;
larger graphs use local search and report `status="HEURISTIC"`.

### QUBO

```python
result = highs_turbo.solve_qubo([[0.0, -1.0], [-1.0, 0.0]])
energy, state, receipt = result
print(energy, state, result.lower_bound, result.status)
```

This minimizes `x.T @ Q @ x` for binary `x`. Inputs may be a square dense or
sparse matrix, or a dictionary mapping `(i, j)` to coefficients. Up to 18
variables are enumerated exactly. Larger problems use local search and report
`status="HEURISTIC"`. The lower bound sums the minimum contribution of each
binary monomial using exact rational representations of the input floats,
including both off-diagonal entries `Q[i, j] + Q[j, i]`.

### Ising and genuine Pegasus subgraphs

```python
from highs_turbo import solve_ising

# E(s) = sum(h[i] * s[i]) + sum(J[i, j] * s[i] * s[j]), s[i] in {-1, +1}
result = solve_ising(
    {"a": 0.25, "b": -0.5},
    {("a", "b"): 1.0},
    time_limit=10,
)
print(result.spins, result.energy, result.lower_bound, result.gap, result.status)
```

This is a complete sparse integer solve using the public HiGHS library, available
on all supported platforms. Labels can be arbitrary hashable objects; `h` can
also be a sequence. Both orientations of a coupling add together. Self-couplings
and the optional `offset` contribute to the constant energy.
Coefficients are interpreted as finite binary64 numbers; the returned rational
values preserve those input values exactly.

The accelerated path adds verified triangle, square, and fundamental-cycle
inequalities, computes an exact rational bound from a nonnegative combination
of frustrated-cycle inequalities, and supplies a local-search spin assignment
to HiGHS. The integer solver retains every original constraint. Set
`accelerate=False` to run the same formulation without those cuts or the start.
Both paths use the same native settings with parallel search disabled.

`OPTIMAL` reports HiGHS' numerical conclusion or an exact bound match.
`TIME_LIMIT`, `INTERRUPTED`, and `GAP_LIMIT` return the best available spins and
the remaining absolute `gap` and `relative_gap`. A positive requested
`relative_gap` may terminate before proof of optimality. `time_limit` covers
preparation and search; mandatory input processing can exceed very small limits.
`SOLVER_ERROR` retains a feasible candidate and conservative cut bound when the
solver cannot finish. `lower_bound` may use a numerical HiGHS bound;
`exact_cut_lower_bound` and `exact_energy` are rational values.
`is_rationally_certified` is true only when those exact values agree. A valid
`cut_certificate` alone establishes the cut combination, not optimality.

Pegasus generation now uses D-Wave's maintained `dwave-graphs` package. The
default fabric has 264 spins / 1,604 couplers for P_4 and 1,288 / 8,804 for P_8.
`generate_pegasus_instance(m, node_list=..., edge_list=...)` validates supplied
D-Wave linear node labels and couplers. Its `metadata["node_labels"]` maps
contiguous internal indices back to the original labels. The previous
handcrafted substitute topology is no longer generated.

Run [`examples/02_dwave_pegasus_qpu.py`](examples/02_dwave_pegasus_qpu.py) for a
complete subgraph solve. No QPU connection or credentials are needed. The older
`KnownProblemSolver` application compares relaxation bounds; use `solve_ising`
when you need a spin assignment and an optimality gap.

```bash
python examples/benchmark_ising.py --sizes 32 48 64 --seeds 1 2 3 --repeats 3
python examples/benchmark_ising.py --m 8 --sizes 128 256 512 --time-limit 5
```

The benchmark includes model construction, cut verification, initialization,
and search. It alternates execution order and compares the same HiGHS version
under equal time limits. It reports speedups only when every compared run
reaches optimality; otherwise it reports energies and remaining gaps.

## Certificates and performance

A valid cut-combination receipt certifies that combination's coefficients and
right-hand side. It does not, on its own, certify that a returned partition is
optimal or that a floating-point LP objective is an exact dual bound. QUBO
receipts include the objective coefficients and the conservative lower bound.

Speedups and simplex iteration counts depend on the input, solver version, and
hardware. There is no fixed speedup or pivot-count guarantee. One timing script
compares complete solves of identical problems with SciPy, full native HiGHS,
and turbo. It includes all preparation/recovery time, rotates execution order,
reports medians, and checks objective agreement and original feasibility:

```bash
python examples/benchmark_linprog.py --repeats 5
python examples/benchmark_linprog.py --mps /path/to/afiro.mps /path/to/25fv47.mps
```

The built-in timing cases are synthetic; optional local MPS files use HiGHS'
public model reader. The full-native control uses the same HiGHS
library as turbo, so library-version differences cannot explain that comparison.
MPS maximization objectives are negated, and objective constants are omitted
equally for all solvers. Infeasible and unbounded statuses are compared too.

[Recorded Netlib results](NETLIB_RESULTS.md) cover complete cold solves of real
models, including slower cases and the additional CPU use of competing methods.

The generated G-set-style inputs are synthetic graphs; they are not downloaded
Stanford G-set benchmark files. `mock_benchmark.py` is only a simulated demo and
must not be used as performance evidence.

## Development and testing

For the full suite, use macOS with the native dependencies installed:

```bash
python -m pip install '.[test]' build
python setup.py build_ext --inplace
python -m pytest tests -q
python examples/01_quickstart_3_lines.py
```

Build a source distribution and then a wheel from that source distribution:

```bash
python -m build
```

Artifacts are written to `dist/`. The source archive includes the C++ headers;
the wheel includes the model weights and runtime topology helpers. To test the
installed wheel without importing the checkout:

```bash
python -m pip install --force-reinstall --no-deps dist/*.whl
python -I tests/smoke_installed.py --require-native --require-ml
```

CI runs the native suite on macOS and checks installed-package acceleration
without the optional extension or PyTorch on Linux and Windows, including the
minimum supported Python version.

## License

[MIT](LICENSE).
