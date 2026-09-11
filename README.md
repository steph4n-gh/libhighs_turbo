# highs_turbo: Cutting Plane Experiments for HiGHS & SciPy

[![Tests](https://github.com/steph4n-gh/libhighs_turbo/actions/workflows/tests.yml/badge.svg)](https://github.com/steph4n-gh/libhighs_turbo/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

`highs_turbo` provides a SciPy-style linear programming interface, Max-Cut and
QUBO solvers, and an optional C++20 engine for discovering and combining graph
cutting planes. A rational verifier checks cut combinations and produces SHA-256
receipts. Optional PyTorch models predict combination weights.

## Installation

Python 3.10 or newer is required. NumPy, SciPy (1.9 or newer, for `milp`), and
NetworkX are installed automatically.

```bash
git clone https://github.com/steph4n-gh/libhighs_turbo.git
cd libhighs_turbo
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

The native engine currently uses macOS CommonCrypto. On macOS, install the
Xcode command line tools and the native dependencies before installing:

```bash
brew install gmp highs
python -m pip install .
```

If the native extension cannot compile, installation continues with Python and
SciPy implementations. Native cut separation and native-only tests require a
successful extension build. Check availability with:

```bash
python -c 'from highs_turbo.compiled_engine import COMPILED_ENGINE_AVAILABLE; print(COMPILED_ENGINE_AVAILABLE)'
```

To use the bundled neural weights or train models, install the optional extra:

```bash
python -m pip install '.[ml]'
```

The weights are shipped inside the installed package. Without PyTorch, the solver
uses its geometric weighting fallback.

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
print(getattr(res, "certificate_sha256", None))
```

The default `method="turbo"` detects graph structure and may add cuts that tighten
the supplied relaxation. This can change the LP objective. Use `method="highs"`
to delegate the original problem directly to SciPy. General LPs also fall back
to SciPy; they do not receive cut-certificate metadata.

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

Speedups and simplex iteration counts depend on the input, solver version,
hardware, and model weights. There is no fixed speedup or pivot-count guarantee.
Run the benchmark suite to obtain measurements for your environment:

```bash
python -m highs_turbo.large_scale_benchmarks --scope canonical --output benchmark_results.json
```

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

CI runs the native suite on macOS and installation smoke checks without a native
compiler or PyTorch on Linux, including the minimum supported Python version.

## License

[MIT](LICENSE).
