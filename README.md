# highs_turbo: Transparent LP Acceleration for HiGHS & SciPy

[![Tests](https://github.com/steph4n-gh/libhighs_turbo/actions/workflows/tests.yml/badge.svg)](https://github.com/steph4n-gh/libhighs_turbo/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

`highs_turbo.linprog` solves the supplied linear program using conic optimality
checks and a smaller working set of constraints where these help. It preserves
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

The mathematical reason this preserves the answer is simple: the working model
is a relaxation of the original. An optimal relaxation solution that is feasible
for the original is also optimal for the original, subject to the requested
numerical tolerances. Graph hints never authorize adding constraints that change
the supplied LP. The LP conic check uses floating-point tolerances; it is not an
exact rational certificate of the LP optimum.

Small problems, integer models, explicit time/iteration/node budgets, other
solver methods, and unsupported options retain ordinary SciPy execution.
Some sparse systems are also delegated when reduction is unlikely to help.
Successful accelerated results expose `turbo_strategy`, `turbo_rows_used`,
`turbo_original_rows`, and `turbo_rounds`; `nit` includes all working solves.
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
