# Reusing exact sparse proof calculations

The public solver now reuses an identical exact Gram-bound calculation within
one solve. Certificate construction and the final return check previously
expanded the same sparse factor separately. Reusing that calculation reduces
verification cost while retaining all certificate-field checks. A subsequent
standalone verification recomputes the arithmetic.

This is an implementation improvement, not a new relaxation or inequality
family. The stronger two-pass search and factor-compression candidates remain
research experiments. No new comparison against a published optimizer or claim
of scientific priority follows from these measurements.

## Complete API comparison

The final implementation was frozen before the comparison. Each of the three
previously studied sparse geometric cases received three repetitions per
implementation. Runs used sequential fresh processes, one BLAS thread, and
rotating order. Each requested ten seconds. The previous implementation is
commit `a1b300ff68b59ded1d1de3f2474a77a504c9edc0`.

Time includes the complete `solve_ising` call, including its final certificate
check and cache cleanup. Imports and case loading are excluded. After each
timed call, the public verifier independently recomputed the returned receipt;
that additional check is excluded for both implementations. These are existing
validation cases, not new holdouts or a statistical sample of all Ising models.

| Case | Previous / reuse median seconds | Time reduction | Previous / reuse median bound |
|---|---:|---:|---:|
| Pegasus 7111 | 10.75 / 10.30 | 4.2% | -11,050.125 / -11,050.125 |
| Damaged Pegasus 7112 | 10.56 / 10.21 | 3.4% | -10,145.125 / -10,145.125 |
| G56 | 12.57 / 10.21 | 18.8% | -9,538.000 / -9,538.000 |

All 18 final receipts passed independent verification. Feasible energies were
unchanged. Median bounds matched on all three cases. The previous damaged-
Pegasus bound ranged from -10,145.125 to -10,144.875; reuse returned -10,145.125
in all three runs. The other final-case bounds were constant across all runs.

The exact Gram formula and certificate format are unchanged. Cooperative
timing can still change which numerical iterate finishes, so equality of
arithmetic for an identical witness does not promise identical results from
every timed optimization run. Raw records retain every time and bound.

## Validation and lifetime

Every lookup follows the original denominator, integer-type, CSR-structure,
row-norm, and expansion-work checks. The full immutable arithmetic inputs form
the key, including the expanded edge list, residual coefficients, scale,
factor, denominator, and storage limit. Problem binding, cut validity,
multipliers, auxiliary-edge topology, and the claimed bound are checked on
every certificate verification. No digest or producer-supplied validity flag
stands in for the arithmetic.

The context holds at most one result and its input tuples. It is isolated per
solve and cleared on both normal return and exceptions; it does not retain a
large factor after the call. A standalone verification uses the ordinary
uncached calculation. The implementation reuses the already validated CSR
factor on misses, avoiding a second validation pass.

Tests cover malformed values that compare equal to integers, changed
arithmetic inputs, tightened resource limits, nested scopes, exception cleanup,
and fresh serialized verification. The existing exact integer expansion and
exhaustive spin tests continue to check the underlying bound formula.

## Why a second search pass is still experimental

A captured G55 two-pass run produced a factor with 1,751,727 entries and an
expansion-work estimate of 2,011,505,451, above the unchanged two-billion limit.
That run fell back to the basic bound -9,580. Its numerical vector objective
plus cut contribution was -9,473.25; this numerical proposal was not a certified
lower bound. This identifies the failure for the captured run, not retroactively
for the earlier fallback reported in [the first follow-up](SPARSE_FOLLOWUP_RESULTS.md).

Removing the smallest integer factor entries can reduce the work estimate.
Every experiment still accounts for the entire resulting Gram residual,
including nonedge fill-in. The following measurements reuse the saved proposal
and are not complete solves:

| Factor proposal | Work estimate | Checked bound | Compression + bound + separate check |
|---|---:|---:|---:|
| Remove entries of magnitude at most 4/8,388,608 | 1,891,501,694 | -9,498 | 6.71 s |
| Uniform threshold targeting half the work limit | 999,733,925 | -9,564 | 3.85 s |
| Entry magnitude times column mean, same half limit | 999,997,198 | -9,572 | 3.95 s |

The common initial factor construction took 1.70 seconds and is excluded from
these incremental times. The mildly compressed proof was stronger than the
production one-pass bound -9,544 on the development case. The more aggressive
proposals lost that advantage. This is an ordinary factor-thresholding proposal,
not evidence of a new proof family.

For the saved mildly compressed proof, an initial reuse prototype reduced the
full certificate check from 3.42 seconds without reuse to 0.23 seconds with
reuse. A fresh serialized check confirmed the same -9,498 bound. Malformed
numerically equal inputs were rejected before any cache hit.

The subsequent complete ten-second API experiment did not reproduce that
saved-proposal advantage on G55:

| Development case | Previous bound / seconds | Reuse only bound / seconds | Two passes + compression + reuse bound / seconds |
|---|---:|---:|---:|
| Pegasus 6111 | -11,058.75 / 10.83 | -11,058.75 / 10.40 | -11,009.25 / 10.76 |
| G55 | -9,544 / 12.69 | -9,544 / 10.65 | -9,576 / 13.52 |

Feasible energies were unchanged within each case, and every returned proof
passed an uncached independent check. G55 still regressed with the second
search pass. **Only exact-arithmetic reuse enters the production solver.**
Compression and two passes remain disabled by default.

An initial production implementation also completed a three-repetition
comparison on the validation cases. It saved 3.0%, 3.4%, and 17.9% of median
time, respectively, but redundantly validated factors on cache misses. Its
damaged-Pegasus bound was 0.25 weaker than its paired control. All of those
observations are retained separately from the final implementation's results.

## Artifacts and reproduction

[Raw observations](benchmarks/sparse_proof_reuse_results.json) contain the final
comparison, the initial implementation, every development observation, capture
hashes, software versions, and frozen production-source hashes. The large
captured matrices and factors are local research artifacts and are not bundled.
Recreating a proposal under a time limit can change its numerical iterate and
factor; hashes identify the actual saved inputs used in the incremental tests.

The existing case generator is `experiments.geometric_suite.generate` with
`suite="sparse-geometric"` for validation and `suite="sparse"` for development.
To measure the final public implementation on a generated case:

```bash
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python examples/benchmark_geometric_ising.py --method auto --seconds 10 \
  --case /path/to/G56.json
```

Pass `--source /path/to/a1b300f-checkout` to the same driver for the previous
implementation. Match its native extension and dependencies to the current
environment. Repeat three times with rotating order and retain all runs.

The consolidated research runner supports `--mode two-pass-compact`; it uses
the same 95%-of-work-limit threshold rule and the unchanged exact checker.
Those timed development prototypes predate consolidation and the final cache
implementation. The consolidated compaction control and CLI were checked;
the reported development timings were not replaced by new runs after editing.
