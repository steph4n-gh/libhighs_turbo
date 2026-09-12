# Follow-up experiments: room to improve, with regressions

The shipped Pegasus improvement is technically meaningful: the current
implementation reduced the remaining certified gap by 11.0% and 12.1% on two
fresh cases at roughly the previous runtime. Those percentages describe a
stronger guarantee, not better feasible solutions or a corresponding speedup.
See [the frozen comparison](SPARSE_GEOMETRIC_RESULTS.md) for the original
measurements and published baselines.

This follow-up tested two possible next steps on development cases only.
Neither search candidate is enabled in the public solver. At the time of
these experiments, the production source was commit
`33e038f89206563c259575e0d636be2ec6c5f252`. A subsequent
[proof-cost investigation](SPARSE_PROOF_REUSE_RESULTS.md) retains both failed
search candidates and describes the separate exact-arithmetic reuse change.

## Five-variable rules versus another triangle pass

The candidate indexes signed sums of pairs of vectors and queries them with
negated signed sums of triples. A five-vector sum with norm below one yields
a violated pentagonal inequality. The index uses objective and auxiliary
edges plus sampled pairs; triples come from sampled graph wedges. Their four
signed sums must all satisfy the triangle test. Accepted candidates have five
distinct vertices and receive the existing exhaustive subgraph check.

For five signed spins, the square of their sum is at least one. This is a
known pentagonal inequality, not a new inequality family. A regular simplex
on five vertices satisfies every triangle inequality but violates the
all-positive pentagonal inequality. The small control found that inequality
and certified the exact optimum -17/8 for a weighted five-vertex example.

[The MADAM paper](https://link.springer.com/article/10.1007/s10589-021-00310-6)
already strengthens SDP bounds with triangle, pentagonal, and heptagonal
inequalities. It describes simulated annealing separation inherited from
BiqBin and staged introduction of stronger families. The pair-sum search
here is a heuristic proposal; its novelty and advantage over published
separators have not been established.

Both candidates below start from the same existing geometric proof and
refitted rank-32 vectors. Each receives an additional eight-second cooperative
budget. Final factorization and independent verification are included in
actual time. Common vector refitting took 1.06 seconds on Pegasus and
0.58 seconds on G55, separately from the table. Producing the supplied input
proof is excluded. These are incremental experiments, not complete solves.

| Case | Starting bound | Another triangle pass | Five-variable pass | Triangle / five-variable seconds |
|---|---:|---:|---:|---:|
| Pegasus 6111 | -11,058.5 | -10,895 | -11,027.25 | 8.40 / 8.15 |
| G55 | -9,544 | -9,498 | -9,540 | 13.14 / 10.13 |

The five-variable search retained 512 rules in 0.76 and 0.65 seconds. It
improved both starting bounds, but another triangle pass produced larger
gains. G55 also illustrates why requested time alone is insufficient: proof
construction and checking can substantially overrun the cooperative budget.
All four returned proofs passed independent verification.

## Two passes inside the original API budget

A separate candidate divides the existing geometric phase into two passes,
refreshing triangles after the first fit and constructing one final proof.
The first pass receives 45% of the available geometric time. The second uses
75% of the remainder before final refitting and proof construction.

Each measurement below is a complete `solve_ising` call in a fresh process,
including initialization, search, certificate construction, independent
checking, and any remaining native solve. Imports and input loading are
excluded. Runs were sequential with one BLAS thread and requested ten seconds.

| Case | Common feasible energy | Production bound | Two-pass bound | Production / two-pass seconds |
|---|---:|---:|---:|---:|
| Pegasus 6111 | -9,410.5 | -11,058.75 | -10,994.75 | 10.79 / 11.04 |
| G55 | -7,932 | -9,544 | -9,580 | 12.57 / 11.66 |

The Pegasus remaining gap fell by a further 3.9%, at a 2.3% increase in actual
time. G55 regressed to its basic sparse proof. One subsequent G55 diagnostic
run, adding candidate-bound logging, returned -9,574 in 15.88 seconds: also
weaker than production. Both observations are retained. This diagnostic does
not establish the exact cause of the original fallback; cooperative timing
changed how many optimization calls finished.

**Decision: retain the current production default.** The Pegasus result
shows additional potential, but the two-pass candidate does not provide a
consistent improvement. The next design must address optimization and proof
cost before increasing the number of rules by default. These observations
do not measure downstream branch-and-bound speed or establish statistical
significance or scientific priority.

## Artifacts and reproduction

[Raw observations](benchmarks/sparse_followup_results.json) include all runs,
stage timings, input hashes, and the diagnostic result. The
[research runner](experiments/sparse_geometric_followup.py) consolidates the
temporary scripts used for these measurements. Timed runs were not repeated
after consolidation; the small mathematical control and CLI were checked.

For a complete API comparison on an existing saved case:

```bash
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m experiments.sparse_geometric_followup --mode production \
  --case /path/to/pegasus-6111.json --seconds 10
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m experiments.sparse_geometric_followup --mode two-pass \
  --case /path/to/pegasus-6111.json --seconds 10
```

The original case generator is `experiments.geometric_suite.generate` with
`suite="sparse"`. Incremental modes `append-three` and `append-five` additionally
require `--proof` (certificate JSON or gzip) and `--vectors` (NumPy array).
The raw file records hashes of the development inputs used here. Those large
intermediate proof and vector files are local research artifacts, not bundled
with the package. Recreating them with a different timed run can change the
incremental measurements.
