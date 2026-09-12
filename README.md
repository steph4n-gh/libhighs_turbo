# highs_turbo: Transparent LP Acceleration for HiGHS & SciPy

[![Tests](https://github.com/steph4n-gh/libhighs_turbo/actions/workflows/tests.yml/badge.svg)](https://github.com/steph4n-gh/libhighs_turbo/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

`highs_turbo.linprog` solves the supplied linear program using conic optimality
checks, smaller working sets, and competing HiGHS methods where these help. It preserves
the original problem and returns SciPy-style solutions, slacks, and marginals.
The package also provides Max-Cut and QUBO solvers and a C++20 cutting-plane
engine with rational verification.

## Plain English: what this does and why it is useful

Imagine thousands of switches, each pointing up or down. Some pairs prefer to
agree; others prefer to disagree. An **Ising problem** asks which settings give
the lowest total penalty, called the **energy**. A **QUBO** expresses a similar
problem using zero/one switches. **Max-Cut** asks how to split a network into two
groups so that the connections crossing between them have the greatest total
weight. These are closely related ways to describe difficult discrete choices.

Finding a good answer and proving how good it is are separate jobs. The best
answer found so far is the **incumbent**. A **lower bound** is a floor below which
no possible answer can fall. The **gap** is the distance between that floor and
the incumbent's energy. For example, an answer with energy -100 and a proven
floor of -103 is at most 3 energy units from the best possible answer. When an
exactly checked gap reaches zero, the answer is proven optimal.

A **relaxation** makes the problem easier by temporarily allowing more choices.
Its answer can be too optimistic, but that makes it useful as a bound. A **cut**
is a rule that removes impossible relaxed choices while keeping every real
switch setting. A **semidefinite program (SDP)** lets switches become vectors
and checks their relationships together. The default solver combines
that global view with cuts to raise the proven floor. It also looks for groups
of three vectors whose relationships cannot come from real switch settings,
even when those switches have no direct connection in the input.

A **certificate** is a mathematical receipt that a separate checker can verify.
Our Ising certificates use exact fractions. A **sum of squares (SOS)** certificate
also uses the fact that squared real numbers cannot be negative. Numerical
optimization proposes a receipt; exact arithmetic checks what it actually
proves, including rounding errors. A **higher-order** SOS method would also
reason about products of switches, giving it a richer language for expressing
their joint restrictions; that extension is currently research work.

A **sparse** proof stores the nonzero pieces of its square instead of a full
table. This lets the global bound engine handle larger networks without
requiring a dense proof. The checker first proves that its integer arithmetic
fits exactly, then checks the complete receipt, including connections created
by the square that were absent from the input.

The useful part is confidence: a better bound can tell you sooner that your
answer is good enough, and can help the integer solver discard fruitless search.
On six weighted Pegasus examples, the current hybrid option reduced the
remaining certified gap by 61–65% versus our earlier cut-based engine at the
same five-second budget. **Pegasus** is a particular sparse connection layout,
not a claim of quantum computation. These results are specific comparisons,
not evidence that every problem is faster or that the method is scientifically
new. See [the measurements and their limits](SDP_ISING_RESULTS.md).

**Drop-in** means existing application code can keep its optimizer calls and
result handling after switching its `linprog` import to `highs_turbo`. A speedup
inside that shared solver can therefore benefit many applications without
rewriting each one. The documented LP acceleration already uses this path.
The improved bound engine now runs automatically in `solve_ising`, `solve_qubo`,
and `solve_maxcut`. It also recognizes compatible binary-product models in
`linprog`: common linear formulations of products such as `y = x1*x2`. Any
additional application constraints stay in the solve. Other integer models keep
SciPy execution. Benefits depend on the problem structure and how much of an
application's runtime is spent solving it. See [the integration and fresh
comparisons](GEOMETRIC_ISING_RESULTS.md).

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
brew install gmp
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

The weights are shipped inside the installed package for the optional neural
components. Ordinary `linprog`, `solve_maxcut`, `solve_qubo`, and `solve_ising`
calls do not load PyTorch or require a trained model.

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

Compatible binary-product integer models use the checked Ising engine. All
variables must have bounds `[0, 1]`, with 32–2,047 binary base variables and
complete, exact rows `y-xu <= 0`, `y-xv <= 0`, `xu+xv-y <= 1` for each product.
Products may be continuous or binary. If these are all the constraints, the
model is converted exactly, solved, and its solution checked against every
original row. Additional inequalities and equalities stay in the original
HiGHS model, strengthened by the proven objective bound. The supported options
are `time_limit`, `mip_rel_gap`, and the defaults `presolve=True`, `disp=False`.
These results expose `turbo_strategy="ising_certificate"`,
`turbo_bound_certificate`, and `turbo_exact_lower_bound`; MIP marginals use
SciPy's zero-array convention. `verify_linprog_certificate(c, certificate,
A_ub=..., b_ub=..., bounds=..., integrality=...)` checks the bound without solving.

Other integer models, explicit continuous-LP time/iteration/node budgets, other
solver methods, and unsupported options retain ordinary SciPy execution. Small continuous LPs use
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
sparse matrix. All sizes use the shared Ising engine, including its checked
global bounds. The default time limit is unlimited up to 200 nodes and five
seconds above that; pass `time_limit` explicitly to change it. Unfinished solves
retain `status="HEURISTIC"` and a valid upper bound. `bound_certificate` contains
the full mathematical witness; the unpacked `receipt` remains its SHA-256 digest.

### QUBO

```python
result = highs_turbo.solve_qubo([[0.0, -1.0], [-1.0, 0.0]])
energy, state, receipt = result
print(energy, state, result.lower_bound, result.status)
```

This minimizes `x.T @ Q @ x` for binary `x`. Inputs may be a square dense or
sparse matrix, or a dictionary mapping `(i, j)` to coefficients. Up to 18
variables have no default time limit; larger problems default to five seconds.
All sizes use an exact objective conversion to the shared Ising engine, retaining
both off-diagonal entries `Q[i, j] + Q[j, i]` without rounding their sum. Pass
`time_limit` explicitly to change the budget. Unfinished solves retain
`status="HEURISTIC"` and a checked lower bound. The tuple layout is unchanged;
`bound_certificate` also exposes the full witness.

### Ising and genuine Pegasus subgraphs

```python
from highs_turbo import solve_ising

# E(s) = sum(h[i] * s[i]) + sum(J[i, j] * s[i] * s[j]), s[i] in {-1, +1}
h = {"a": 0.25, "b": -0.5}
J = {("a", "b"): 1.0}
result = solve_ising(
    h, J,
    time_limit=10,
)
print(result.spins, result.energy, result.lower_bound, result.gap, result.status)
```

This is a complete sparse integer solve using the public HiGHS library, available
on all supported platforms. Labels can be arbitrary hashable objects; `h` can
also be a sequence. Both orientations of a coupling add together. Self-couplings
and the optional `offset` contribute to the constant energy.
Ordinary coefficients are interpreted as finite binary64 numbers; `Fraction`
inputs retain their exact rational values. Returned rational bounds preserve
those input values.

The accelerated path repeatedly separates violated cycle inequalities and
stronger inequalities on supports of up to 12 vertices. Each small-support
inequality is verified against every local spin assignment. A fast multiplier
search supplies a bound before simplex; subsequent LP rounds reuse the model
and basis. Nonnegative cut combinations, an exact residual correction, and
objective-lattice rounding produce independently checkable rational bounds.
The integer solver retains every original constraint. Set
`accelerate=False` to run the same formulation without those cuts or the start.
Both paths use the same native settings with parallel search disabled.

Install `pip install '.[ising]'` for compiled simulated annealing and tabu
incumbents. Without that extra, a bounded multistart descent supplies the start.
`cut_policy="deterministic"` is the default. `"learned"` uses a bundled small
neural ranking model trained on measured marginal LP-bound improvements per
separator second; inference needs only NumPy. The policy orders candidates and
has no role in proof acceptance. `"static"` retains the original one-pass
algorithm for comparison. The fast multiplier kernel uses the optional macOS
extension; a slower Python implementation is available on other platforms.

The default `relaxation="auto"` combines a low-rank global semidefinite bound
with local cuts and geometric separation of nonlocal triangle inequalities.
Dense objectives skip enumeration of local cycles and start with the global
bound. `"hybrid"` explicitly chooses the combined path; `"sdp"` uses the basic
global bound and `"cuts"` selects the earlier sparse cut relaxation. The static
policy also uses cuts. These use existing NumPy/SciPy dependencies. Problems
up to 2,048 vertices including the reference spin use the combined dense path.
From 2,049 through 8,192 vertices, automatic selection uses a sparse global
bound; the binary-product `linprog` bridge accepts up to 8,191 base variables.
Dense inputs and factors exceeding the sparse work/storage limits fall back
to cuts. The dense witness needs quadratic memory. Final factorization, original-model
validation, and exact checking can overrun a short time limit. See
[the geometric algorithm and measurements](GEOMETRIC_ISING_RESULTS.md) and
[the sparse certificate comparison](SPARSE_ISING_RESULTS.md), including its
published baseline, and [the global-bound derivation](SDP_ISING_RESULTS.md).

```python
result = solve_ising(h, J, time_limit=5, relaxation="hybrid", certified_gap=2.0)
```

`OPTIMAL` reports HiGHS' numerical conclusion or an exact bound match.
`TIME_LIMIT`, `INTERRUPTED`, and `GAP_LIMIT` return the best available spins and
the remaining absolute `gap` and `relative_gap`. A positive requested
`relative_gap` may terminate before proof of optimality. `time_limit` covers
preparation and search; bounded preprocessing and verification can exceed very
small limits, and native solver limits are cooperative.
`SOLVER_ERROR` retains a feasible candidate and conservative cut bound when the
solver cannot finish. `lower_bound` may use a numerical HiGHS bound;
`exact_cut_lower_bound` and `exact_energy` are rational values.
`is_rationally_certified` is true only when those exact values agree. A valid
certificate can also establish a useful positive `exact_gap`.

```python
from highs_turbo import verify_ising_certificate

result = solve_ising(h, J, time_limit=10, certified_gap=2.0, cut_policy="learned")
witness = result.certificate.to_dict()  # JSON-serializable integers and fractions
assert verify_ising_certificate(h, J, witness)
print(result.exact_gap)  # exact_energy - independently verified lower bound
```

`certified_gap` requests an absolute gap from the exact certificate, including
the offset. It never relies on the numerical MIP bound. The standalone checker
runs no optimizer or model and rejects altered problem coefficients, invalid
cuts, negative multipliers, and incorrect bounds. Global witnesses include a
quantized triangular Gram factor; their checker accounts for every residual
entry, including fill-in on nonedges. It does not trust numerical eigenvalues
or a Cholesky status. Version 3 witnesses can include additional edges with
zero objective weight to express nonlocal cuts; the checker binds them to the
original problem and validates each inequality. Version 4 stores a sparse
integer Gram factor as row pointers, column indices, and values. Its checker
validates the structure, exact arithmetic range, and expansion size before
numeric multiplication. Versions 1–3 remain readable.
`progress` records checked bound/energy checkpoints with
elapsed times; `root_rounds` counts root relaxation rounds.
The older `cut_certificate` field is retained only for the static policy.
`threads=0` leaves HiGHS' thread count automatic; use a consistent thread setting
within a process because HiGHS shares its native scheduler across instances.
BLAS uses its usual environment settings independently of that HiGHS option.
If search stops for another reason, inspect `exact_gap` to determine whether the
requested certificate target was met, even if HiGHS reports numerical optimality.

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
See [the complete Pegasus measurements](PEGASUS_RESULTS.md), including weighted
instances and unfinished full-fabric runs. Those measurements describe the
earlier static implementation. The adaptive comparison and training recipe are
in [ADAPTIVE_ISING_RESULTS.md](ADAPTIVE_ISING_RESULTS.md).

## Certificates and performance

A valid cut-combination receipt certifies that combination's coefficients and
right-hand side. It does not, on its own, certify that a returned partition is
optimal or that a floating-point LP objective is an exact dual bound. QUBO
receipts bind the converted objective and its checked lower bound. The legacy
Max-Cut and QUBO `is_rationally_certified` flags mean that their bound is valid;
check `status` or equality of bound and objective to assess optimality.

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
