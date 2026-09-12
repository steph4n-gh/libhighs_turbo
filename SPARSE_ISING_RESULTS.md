# Sparse exact Gram certificates for larger binary problems

The global bound path now supports up to 8,192 vertices, including the
reference spin used for local fields. It uses a sparse integer factor and an
exact range argument to check its square. Automatic Ising selection and the
existing QUBO, Max-Cut, and recognized binary-product `linprog` entry points
use the same engine. The `linprog` bridge accepts up to 8,191 base variables.

This is a verified scalability improvement over this repository's previous
large-graph path. It is **not a demonstrated algorithmic advance over the
published Mixing method**: the authors' unmodified implementation, given
our sparse certificate repair, matches or improves these basic bounds faster.

## In plain English

A lower bound is a floor below which the best possible answer cannot fall.
Raising that floor shrinks the remaining uncertainty about a feasible answer.
The proof is a mathematical receipt that another program can check.

The earlier receipt stored a full triangular table. The new receipt stores
only its nonzero entries. Before multiplying them, the checker proves that
all the arithmetic will be exact integers. This permits larger networks
without giving up the independently checked guarantee. Sparse inputs can
still produce large intermediate tables, so the implementation checks limits
and retains a conservative bound if it cannot construct a suitable receipt.

## Frozen comparison

The implementation was frozen before generating seeds 6111 and 6112 and
running G55. These are single runs, not statistical estimates. Each solver
ran sequentially in a fresh process with one BLAS thread. The requested
budget was ten seconds; the table reports actual time including preparation,
factor construction, exact verification, and any remaining HiGHS solve.

| Case | Spins | Common feasible energy | Previous hybrid bound | New bound | Remaining gap reduced | Previous / new seconds |
|---|---:|---:|---:|---:|---:|---:|
| Pegasus 6111 | 5,640 | -9,410.5 | -12,898 | -11,280.75 | 46.4% | 10.52 / 10.74 |
| Damaged Pegasus 6112 | 5,360 | -8,722 | -11,758 | -10,382.25 | 45.3% | 10.46 / 10.50 |
| G55 | 5,000 | -7,932 | -12,206 | -9,688 | 58.9% | 10.03 / 11.69 |

The comparison fixes the same feasible energy when calculating each gap.
The previous baseline is commit `f74a41d89550af0ca7c6bd86a625189e3450d483`
with `relaxation="hybrid"`; on these larger graphs its SDP phase cannot run.
That explicit hybrid setting reserves more time for root cuts than the old
automatic setting. The new run uses the default API settings.

The new `linprog` route returned exactly the same certified bounds in
10.76, 10.53, and 11.70 seconds respectively. It recognized and solved the
complete binary-product model, then checked the returned values against all
original rows. The same-version native HiGHS comparison returned numerical
bounds -21,409.25, -19,335, and -12,416; those are not independently certified.
This demonstrates the particular drop-in path, not a universal MIP speedup.

The saved factors have 862,540, 642,801, and 1,055,754 entries. The previous
dense triangular representation would require 15,913,261, 14,372,841, and
12,502,500 entries respectively. These counts exclude metadata and temporary
workspace, and are not measurements of peak process memory.

Inputs, source hashes, dependency versions, and every result are recorded in
[`benchmarks/sparse_ising_results.json`](benchmarks/sparse_ising_results.json).
The G55 input comes from
[Stanford's Gset collection](https://web.stanford.edu/~yyye/yyye/Gset/G55).
The generator uses the official D-Wave Pegasus graph, eighth-integer fields
and couplings, and removes 280 nodes for the damaged case before drawing its
weights. Its exact random draw order is in the reproduction script.

## Published baseline and the limit of the result

The closest baseline for the new *basic* relaxation is
[Wang, Chang, and Kolter's Mixing method](https://arxiv.org/abs/1706.00476).
We compiled the authors' unmodified
[C implementation](https://github.com/locuslab/mixing) at commit
`e7d84e744d1826d596f7dace449248a36de4d050` using its Makefile. Input conversion,
the executable's parse/solve/solution export, and the same sparse proposal
and two exact checks are included. Imports and compilation are excluded.
The program has iteration and tolerance controls but no wall-time limit.

| Case | Published settings | Rank | Checked bound | Total seconds |
|---|---|---:|---:|---:|
| Pegasus 6111 | Defaults | 107 | -11,280.25 | 4.00 |
| Pegasus 6111 | Rank 16, tolerance 1e-7 | 16 | -11,288 | 2.21 |
| G55 | Defaults | 100 | -9,582 | 4.61 |
| G55 | Rank 16, tolerance 1e-7 | 16 | -9,688 | 4.77 |

This comparison invalidates a speed or bound-quality superiority claim for
our current basic SDP search. It also identifies a concrete next direction:
spend less on basic vector optimization and investigate stronger constraints
that can still be certified sparsely. A fast numerical answer alone does
not establish a good exact bound: the G55 rank-16 numerical objective is
close to the rank-100 result, while its repaired bound is appreciably weaker.

An exploratory follow-up, outside the frozen API comparison, added nonlocal
geometric triangles to the sparse certificate. Starting with the published
rank-16 vectors raised Pegasus 6111's checked bound from -11,280.75 to
-11,076.5 in an additional 8.84 seconds. G55 improved from -9,688 to -9,632
in 12.31 additional seconds, still weaker than the published rank-100 basic
bound. These are research results on reused cases, not the current automatic
path or evidence of general superiority.

For that exploratory Pegasus follow-up, the unmodified
[AugmentedMixing implementation](https://github.com/jschwiddessen/AugmentedMixing.jl)
at `21f29340faefb4080e61f453075ab3ff76c62e1f` received the same 1,813 active
triangles without paying for their discovery. Defaults and the earlier
development settings (`mu_start=0.2`, `scaling=false`, `tau=1.3`) were each
given ten solver seconds. Actual solve plus certificate time was 50.66 and
53.87 seconds. Each stopped after one outer iteration and returned zero dual
multipliers. The two shared sparse repair proposals retained only the trivial
bound -22,608.5; failed proposals and their reasons are in the raw records.
This is evidence about these particular short-budget runs, not a general
comparison against a fully converged constrained SDP or all SDP solvers.

## Why the sparse checker is exact

After aggregating independently checked cuts with nonnegative rational
multipliers, let the residual energy be
`sum_(u,v) r_uv s_u s_v`, with every spin in `{-1,1}`. Any integer matrix
`B` and positive integer `D` provide a valid nonnegative square
`s' B B' s / D^2`. Subtract that square and bound each remaining term by its
absolute value. All entries created by the multiplication are included,
even when the original graph has no edge at that position.

The numerical proposal uses a symmetric SuperLU factorization with a small
diagonal shift and rounds its factor to integers. Neither positive pivots,
permutations, numerical factorization identities, nor an eigenvalue estimate
are premises of the certificate. They only affect how useful it is.

For each integer row `b_i`, the checker first verifies

```
sum_k B_ik^2 < 2^53.
```

Cauchy–Schwarz then gives, for every row pair,

```
sum_k |B_ik B_jk| <= ||b_i||_2 ||b_j||_2 < 2^53.
```

Each product and every possible partial sum in `B @ B.T` is therefore an
exactly representable binary64 integer. Standard CPU sparse multiplication
performs exact arithmetic under this checked precondition. This argument
does not assume a particular addition order and does not authorize reduced
precision arithmetic. Row norms and absolute row sums use int64 only after
their own overflow bounds have been checked. The final totals, rational
corrections, and objective-lattice rounding use Python integers and fractions.

Version 4 certificates contain CSR row pointers, column indices, integer
values, and a denominator. The reader rejects malformed pointers, repeated
columns, out-of-range indices, noninteger values, arithmetic overflow, and
mixed dense/sparse representations. The original problem digest and all cut
checks remain mandatory. Versions 1–3 remain readable.

The automatic sparse path requires at most 16 input edges per vertex. The
checker caps factors at eight million entries, scalar product work at two
billion contributions, and the expanded Gram pattern at sixteen million
entries. The symbolic product is bounded by the 8,192-vertex limit but may
temporarily exceed the accepted expanded-pattern limit before rejection.
Numerical factorization and checking have cooperative time limits; this is
why actual time can exceed a short requested budget.

## Prior work and scope

Sparse SDP methods, Cholesky factors, exact rational certificates, and using
floating-point arithmetic for suitably bounded integers are established
ideas. In particular,
[VSDP](https://optimization-online.org/wp-content/uploads/2006/12/1547.pdf)
already verifies sparse conic bounds while accounting for floating-point
rounding. The row-norm argument above is elementary; no priority claim is
made for it. Scientific novelty of a combined discovery/certification method
remains unestablished.

The sparse implementation adds four focused tests: exact Python expansion
and exhaustive spin states; malformed serialized receipts with optimization
disabled; work/storage limits; and automatic routing above the old size cap.
All 475 existing tests passed in the regression run. The four new tests pass
after aligning the new routing test with the suite's shared HiGHS thread
setting. The checker is also included in Linux and Windows CI coverage.

## Reproduce

```bash
python experiments/geometric_suite.py --suite sparse --seconds 10 \
  --directory /tmp/sparse-final --previous /path/to/f74a41d-checkout
python -m experiments.original_mixing_suite \
  --binary /path/to/authors/mixing --directory /tmp/sparse-final
```

Set `VECLIB_MAXIMUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and
`OMP_NUM_THREADS=1` for the published checker command. The suite sets them
for its solver workers. The published code is an optional research baseline,
not a runtime dependency of the package.
