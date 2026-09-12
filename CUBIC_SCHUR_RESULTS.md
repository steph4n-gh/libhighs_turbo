# Cubic certificates: a stronger control removes the apparent gain

**Decision: keep this candidate out of the public solver.** On the development
Pegasus case, the best cubic result and the ordinary-cut control both certify
−11,052.75 in about 18.5 additional seconds. The ordinary-cut proof has fewer
factor entries. This experiment does not meet the novelty or performance target.
The shipped results in [the sparse geometric report](SPARSE_GEOMETRIC_RESULTS.md)
are unaffected.

In plain English, the experiment gave the proof extra variables representing
products of three switches. It also reused the existing proof outside the small
region being changed. At first, these extra variables appeared to help. Once
the comparison included all signed triangle and pentagonal rules on the same
five-switch groups, that advantage disappeared at the objective's exact
quarter-unit resolution. Equality here is an observation on one instance,
not a theorem that the two relaxations always agree.

## Common input and timing

All runs use development case Pegasus 6111: 5,641 spins, including the field
reference spin. The starting geometric certificate proves −11,058.5. A common
preparation selects 32 five-vertex supports and fits their pentagonal
multipliers, producing a checked bound of −11,054.75. The selected supports
contain 150 original vertices and 316 distinct cubic products.

Common preparation took 1.05 seconds to refit vectors, 0.78 seconds to select
supports, 5.83 seconds to fit the pentagonal proof, and 0.75 seconds for its
independent check. These costs, and producing the supplied geometric proof,
are outside the following incremental comparisons. They are not complete
solver calls or measurements on new holdout instances.

Processes ran sequentially with one BLAS thread. Actual times below include
reconstructing the exact parent slack, numerical optimization, factor
construction, the complete Gram calculation, and a second identity/Gram
check. Imports, input loading and final artifact writing are excluded. No
Gram work or storage limit was raised.

| Method | Solver budget | Raw checked bound | Quarter-unit bound | Actual additional seconds |
|---|---:|---:|---:|---:|
| 32 original pentagonal rules, SCS with parent start | 15 s | −11,053.590257 | −11,053.5 | 15.60 |
| Cubic identities, SCS with parent start | 15 s | −11,064.255302 | −11,064.25 | 18.50 |
| Cubic identities, SCS with parent start | 60 s | −11,054.510945 | −11,054.5 | 63.88 |
| Cubic identities, low-rank augmented Lagrangian | 15 s | −11,052.986340 | −11,052.75 | 18.50 |
| All signed triangles and pentagons on the same supports, SCS | 15 s | −11,052.988666 | −11,052.75 | 18.35 |

The final scalar control contains 1,776 distinct inequalities and a matrix of
order 150. The cubic formulation has 2,900 free identity coordinates and a
matrix of order 466. Its final factor contains 1,054,728 entries, versus
968,666 for the scalar control. Their numerical objective proposals differ
by less than 0.001, but those proposals are not certified bounds.

The final SCS control used absolute and relative tolerance `1e-6` and stopped
at its time limit with `optimal_inaccurate` status. The exact bound above is
still valid because the numerical proposal was repaired and checked. The
other SCS rows used `1e-5`; the cubic SCS runs also stopped before convergence.
A scalar run at `1e-5` finished in 12.26 seconds and certified −11,053. The
extra accuracy resolved the small apparent difference after rounding.

Earlier cold starts, the first low-rank run capped at 1,000 outer iterations,
and the global cubic prototypes are retained in the
[raw observations](benchmarks/cubic_schur_results.json). The iteration cap
ended the first local low-rank run after 3.98 seconds of optimization. Removing
that cap, while retaining the 15-second deadline, reduced its numerical
feasibility error. No broad parameter search or new holdout benchmark was run.

## How the boundary reduction works

Let the parent's residual energy be `sᵀ C s`, with zero diagonal in `C`, and
let its checked Gram matrix be `G = B Bᵀ / D²`. Define exactly

```
beta[i] = G[i,i] + sum(j != i, abs(C[i,j] - G[i,j])).
```

Then `C + diag(beta)` is positive semidefinite: subtracting `G` leaves a
symmetric diagonally dominant matrix with nonnegative diagonal. Moreover,
`-sum(beta)` is exactly the parent's raw Gram lower bound. This converts its
rounding allowance into a sparse positive semidefinite slack without losing
that raw bound. It does not preserve a potentially stronger separate cut-only
bound, so a production caller would still retain its earlier certificate.

Partition the original vertices into the affected set `U` and untouched set
`O`. Freeze `beta[O]` and factor the positive definite outside block `A[O,O]`.
Its Schur term is

```
H = C[U,O] inverse(A[O,O]) C[O,U].
```

The remaining optimization acts on `C[U,U] - H`. Added cubic rows have zero
outside cross terms. This reduces a 5,957-row lifted problem to 466 rows.
The numerical inverse, factorization and eigenvalue checks guide proposal
quality only. The final square is assembled over every original and added
row, quantized, and passed through the unchanged full sparse Gram checker.
All fill-in and rounding residuals remain in that calculation.

## Cubic identities and the starting proof

A row represents the Boolean character `m[S] = product(s[i], i in S)`.
Products satisfy `m[S] m[T] = m[S XOR T]`. The runner finds repeated products
of singleton and selected cubic rows. Reparameterization coefficients sum
to zero within every repeated-product group. Free coefficients are rounded
to dyadic fractions, with the reference occurrence receiving the exact
opposite sum. A separate exact expansion checks the entire resulting
polynomial, including cancellation of quartic and sextic terms.

The parent pentagonal certificate supplies a nonzero feasible start. For
five signed spins `t[i]`, set `r = sum(t)` and `p[i] = t[i] (r² - 1)`.
Reducing by `t[i]² = 1`, each `p[i]` has degree at most three, and

```
(r² - 1)/2 = (11/640) sum(p[i]²) - (1/384) (sum(p[i]))².
```

The coefficient matrix `(11/640) I - (1/384) 11ᵀ` has eigenvalues `11/640`
and `1/240`, both positive. Thus the familiar pentagonal rule has a rational
cubic SOS representation. Exact coefficient expansion verifies constant 2,
all signed pair coefficients, and zero higher-degree residual. This is a
representation of a known inequality, not a new inequality family.

The research witness combines that exact polynomial identity with a Gram
bound over the lifted spins. Valid original spins map into the lifted cube,
so a bound valid on that larger cube is also valid on the original instance.
These witnesses are research artifacts, not a new public certificate version.
The source certificate's cuts are checked during preparation.

## Prior work and interpretation

Selecting only some higher-order moment constraints is already established.
[Campos, Misener and Parpas](https://link.springer.com/article/10.1007/s11081-022-09763-y)
study partial Lasserre relaxations for sparse Max-Cut, including heuristics
for selecting submatrices and comparisons with BiqCrunch and CS-TSSOS.
[The MADAM paper](https://link.springer.com/article/10.1007/s10589-021-00310-6)
already uses triangle, pentagonal and heptagonal inequalities. Schur reduction,
low-rank optimization, and augmented Lagrangian updates are also established
tools. Combining them here does not by itself establish scientific priority.

This is a same-support control against known inequality families, not a new
execution of the authors' complete solvers. Earlier published-implementation
comparisons remain in the other result reports. Since the cubic candidate
fails this cheaper control, promoting it or claiming a published-method win
would be premature. A further candidate needs evidence of useful joint
restrictions beyond these ordinary local cuts.

## Reproduction and checks

The optional [research runner](experiments/cubic_schur.py) consolidates the
temporary prototypes. It requires CVXPY and SCS in the research environment;
neither is a package runtime dependency. To prepare a common input from an
existing checked geometric proof and its vectors:

```python
from experiments.cubic_schur import prepare

prepare("case.json", "geometric-proof.json.gz", "vectors.npy", "prepared")
```

Preparation uses the existing five-variable selector with limit 32 and seed
zero. Its vector refit and cut fit have two- and eight-second cooperative
budgets. Recreating a time-limited source proof can change the selected
supports and numerical results. The raw file records hashes of the actual
case, prepared input, source proof, vectors, production code and prototypes.
Large input and proof files are local artifacts rather than package data.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python -m experiments.cubic_schur --input prepared --output cubic \
  --mode cubic --optimizer mixing --seconds 15
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python -m experiments.cubic_schur --input prepared --output ordinary \
  --mode scalar --optimizer scs --all-cuts --seconds 15 --tolerance 1e-6
```

The large timing runs were not repeated after consolidation. The consolidated
code passed syntax and lint checks. A seven-spin control compared every
repeated basis product against an exhaustive oracle, expanded the signed
pentagonal template and seed correction exactly, and exercised all four
solver branches against the enumerated original optimum. The control checks
validity; its frozen outside block deliberately leaves a weaker bound, so it
is not presented as an optimality or performance result. Production code and
its verification limits are unchanged.
