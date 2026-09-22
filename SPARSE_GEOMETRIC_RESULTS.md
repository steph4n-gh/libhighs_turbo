# Geometric certificates on larger sparse binary problems

Automatic Ising, QUBO, Max-Cut, and recognized binary-product `linprog`
models now use nonlocal geometric cuts on eligible problems with 2,049–8,192
vertices. This extends the earlier sparse basic SDP path. The integer proof
format and independent acceptance rules remain version 4.

On three fresh cases, the new default reduced the remaining certified gap by
3.7–12.1% relative to the previous sparse engine. This is a further improvement
on top of the earlier scalability result, not the same comparison repeated.
The new bounds also exceed the repaired basic Mixing bounds. These results
do not establish a universal speedup or scientific priority.

## In plain English

The solver represents each switch by an arrow while looking for a lower
bound. Some groups of three arrows have relationships that real switches
cannot have. Finding those groups lets it add valid rules and raise the
proven floor beneath the best possible answer.

The earlier large-graph solver spent most of its budget fitting the arrows.
The new path updates groups of unconnected arrows together, then uses a
fast approximate neighbor search to find useful triples. The proof checker
still checks every rule and every term of the final square exactly.

A separate setup bug was also fixed: reading the sampler's best answer
rebuilt its entire mapping once per spin. On G55 that performed 25 million
mapping lookups. Reading it once preserves the returned answer and releases
several seconds for the proof search. The basic-SDP control below includes
this fix, so its comparison isolates the additional benefit of geometric cuts.

## Frozen API comparison

The production files were frozen before generating Pegasus seeds 7111 and
7112 and downloading G56. Earlier seeds 6111 and 6112 and G55 were development
cases for this follow-up and are excluded from this table. The Pegasus
generator uses the official size-16 layout, eighth-integer weights and
fields, and removes 280 vertices before drawing weights for the damaged case.
G56 comes from [Stanford's Gset collection](https://web.stanford.edu/~yyye/yyye/Gset/G56).

Each method ran once, sequentially, in a fresh process with one BLAS thread.
The requested budget was ten seconds. Actual time includes all work inside
the solver API: normalization, initialization, vector fitting, cut discovery,
factor construction, independent verification, and any remaining integer
solve. Imports and constructing the input before calling the API are excluded.

| Case | Spins | Common feasible energy | Previous bound | New bound | Remaining gap reduced | Previous / new seconds |
|---|---:|---:|---:|---:|---:|---:|
| Pegasus 7111 | 5,640 | -9,408.375 | -11,252.125 | -11,050.125 | 11.0% | 10.66 / 10.79 |
| Damaged Pegasus 7112 | 5,360 | -8,635.875 | -10,352.625 | -10,144.875 | 12.1% | 10.52 / 10.56 |
| G56 | 5,000 | -7,914 | -9,600 | -9,538 | 3.7% | 11.78 / 12.53 |

The previous source is commit `615ece9b0b4bd74a82c270d430e0755804538b7f`,
with explicit `relaxation="hybrid"`. At these sizes that version routes to
its basic sparse SDP, the same large-graph route as its default setting.
Gap reductions use the common feasible energy, not different incumbents.
None of these cases was solved to proven optimality.

The new `relaxation="sdp"` control includes the faster sampler extraction,
rank-32 independent-group vector updates, and sparse proof construction,
but omits geometric cuts. Its checked bounds were -11,258.875, -10,359.625,
and -9,574. The geometric improvements of 208.75, 214.75, and 36 energy units
therefore persist beyond those engineering changes.

The `linprog` route produced bounds -11,050.125, -10,145.125, and -9,538 in
10.93, 10.60, and 12.54 seconds. The quarter-unit difference on the damaged
case reflects separate runs with cooperative deadlines. Every returned
solution was checked against the complete original linearized model.
Same-version native HiGHS produced numerical bounds -21,379.875,
-19,261.875, and -12,456. These native bounds are not independently certified.

All inputs, production hashes, versions, and individual runs are recorded in
[`benchmarks/sparse_geometric_results.json`](benchmarks/sparse_geometric_results.json).
These are focused observations, not repeated trials or statistical estimates.

Later [development experiments](SPARSE_FOLLOWUP_RESULTS.md) found another
3.9% gap reduction on Pegasus with two geometric passes, but a regression on
G55. That candidate remains outside the production default.

## Published basic Mixing comparison

The authors' unmodified [2017 Mixing implementation](https://github.com/locuslab/mixing)
at `e7d84e744d1826d596f7dace449248a36de4d050` receives the same original
objective and the same sparse certificate repair. Input conversion,
parse/solve/export, repair, and both exact checks are timed; imports and
compilation are excluded. Its executable has no wall-time stopping option.

| Case | Published default rank | Published checked bound | Published seconds | New checked bound | New seconds |
|---|---:|---:|---:|---:|---:|
| Pegasus 7111 | 107 | -11,251.625 | 3.55 | -11,050.125 | 10.79 |
| Damaged Pegasus 7112 | 104 | -10,352.375 | 2.96 | -10,144.875 | 10.56 |
| G56 | 100 | -9,574 | 4.64 | -9,538 | 12.53 |

The published rank-16, tolerance-1e-7 runs returned checked bounds
-11,258.875, -10,359.625, and -9,684 in 1.96, 1.64, and 4.83 seconds.
All six runs are retained in the raw results. The new path obtains stronger
proofs while spending more time; this table does not show a basic-SDP speed
advantage. Adding known triangle constraints also does not itself establish
an advance over published constrained-SDP methods.

## Constrained ConicBundle comparison

The unmodified [ConicBundle 1.a.2 library](https://www-user.tu-chemnitz.de/~helmberg/ConicBundle/)
was compiled with optimization and single-threaded Accelerate BLAS. A small
driver supplies the exact selected triangle set: 1,823 cuts for Pegasus 7111
and 1,447 for G56. The library pays no discovery cost. In addition to its
zero start, it receives diagonal multipliers derived from the published
default Mixing vectors, with that vector solve supplied free as well.

| Case | Initial multipliers | Numerical bound | Checked bound | Native call / total seconds |
|---|---|---:|---:|---:|
| Pegasus 7111 | Zero | -12,020.586 | -12,021.125 | 9.88 / 12.38 |
| Pegasus 7111 | Published Mixing | -11,189.503 | -11,190.375 | 10.33 / 13.77 |
| G56 | Zero | -9,730.784 | -9,730 | 9.86 / 21.08 |
| G56 | Published Mixing | -9,574.663 | -9,574 | 10.08 / 18.36 |

All four runs reached ConicBundle's ten-second user-CPU limit. Totals include
input conversion, the native call and output, and both shared sparse repair
proposals with independent checks. No numerical SDP value is treated as a
certificate. The new default's checked bounds were -11,050.125 and -9,538 in
10.79 and 12.53 seconds, including its own cut discovery and feasible answer.
Thus it produced a stronger checked result on these two cases within the
reported times, including against the supplied starting point.

This tests a particular short-budget regime and a fixed-cut driver for the
library. It does not compare fully converged relaxations, the tutorial's
dynamic separator, every bundle parameter setting, or a complete tuned
branch-and-cut solver. The baseline tries and charges two proof repairs;
using just one would change its total cost. These limits matter when
interpreting timing differences. The exact formulation, build commands,
license, and reproduction procedure are in
[`experiments/CONICBUNDLE.md`](experiments/CONICBUNDLE.md).

## Mechanism and limits

Independent groups come from a largest-first greedy coloring of every
potential objective edge, including cut edges whose current multipliers are
zero. Each group's unit-vector updates are ordinary coordinate minimizers.
This is a known parallel form of Mixing, not a new relaxation.

For cut discovery, the existing signed-vector search queries the nearest
neighbor to `-(v_i +/- v_j)` among `+/-v_k`. A distance below one yields a
violated triangle. Excluded endpoint vectors are at distance at least one,
so one sufficiently close neighbor suffices. On large inputs the approximation
parameter is four, retaining full 32-dimensional vectors; the returned
distance must still pass the violation test. It samples pairs and retains at
most 2,048 diversified triangles. It is not exhaustive separation, and no
worst-case subquadratic guarantee is claimed for the tree search.

Triangle inequalities, sparse semidefinite methods, and nonedge cuts are
established. [ConicBundle's tutorial](https://www-user.tu-chemnitz.de/~helmberg/ConicBundle/Manual/mctutorial.html)
explicitly discusses strengthening a sparse objective with nonedge triangles.
[Galli, Kaparis, and Letchford](https://www.lancaster.ac.uk/people/letchfoa/other-publications/2012-ISCO-gap.pdf)
describe heuristic triangle and more general gap-inequality separation.
The potential contribution here is the particular discovery and certification
procedure and its measured cost; scientific novelty remains unestablished.

The checker still includes all nonedge fill-in of the sparse factor's square.
The [existing exact range proof and work/storage limits](SPARSE_ISING_RESULTS.md)
apply unchanged. The solver retains its stronger existing proof if a proposed
factor fails or yields a weaker checked bound. Final factorization and exact
verification are cooperative operations and may exceed the requested limit,
as the G56 measurement shows.

All 480 tests pass. The focused sparse tests cover integer arithmetic,
malformed receipts, resource limits, large default routing, and independent
updates including an initially zero edge that later becomes active.

## Reproduce

```bash
python experiments/geometric_suite.py --suite sparse-geometric --seconds 10 \
  --directory /tmp/sparse-geometry-final --previous /path/to/615ece9-checkout
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m experiments.original_mixing_suite \
  --binary /path/to/authors/mixing --directory /tmp/sparse-geometry-final \
  --cases pegasus-7111 pegasus-7112 G56
```

The external published solver is an optional research baseline, not a runtime
dependency of the package.
