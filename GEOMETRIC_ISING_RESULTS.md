# Geometric Ising bounds and the drop-in binary-product bridge

The new implementation strengthens exact bounds on several unseen Ising
instances and makes that engine available through existing `linprog`, QUBO,
and Max-Cut calls. This is a demonstrated implementation improvement. Scientific
novelty remains unestablished: semidefinite bounds, triangle inequalities,
Boolean SOS, and exact residual correction all have prior art.

## What changed, in plain English

The solver first lets each binary switch become a short vector. This gives a
useful optimistic bound, but some relationships between the vectors cannot
come from real switches. For example, three switches cannot all disagree with
one another. The new search finds such contradictions even between switches
that were not connected in the original input, then adds the corresponding
valid rules. A separate exact checker verifies the resulting bound.

This now happens automatically in `solve_ising`, `solve_qubo`, and
`solve_maxcut`. Existing `linprog` calls also benefit when the supplied rows
exactly encode binary products. Extra application constraints remain in the
original model. General LPs and other MIPs retain their existing execution paths.
The recognition rules and result fields are documented in [README.md](README.md).

A smaller **gap** means a tighter guarantee on the solution's quality. It does
not by itself establish a solve-time speedup. All large cases below remained
unfinished at their time limits.

## Fresh comparison, 11 September 2026

Implementation choices were frozen before generating these four inputs.
Apple M4 Pro; Python 3.13; NumPy 2.5.3; SciPy 1.18.1; `highspy` 1.15.1.
Each method ran once in a separate process, sequentially, with one BLAS thread
and a nominal five-second budget. Method order rotates between cases. Timing
includes all work inside the optimizer, including construction and proof
checking; input generation and imports are outside the timer. These are four
focused observations, not statistical estimates of general performance.

Pegasus uses the maintained D-Wave P8 topology, random eighth-integer fields and
couplings, and previously unused seeds 4101 and 4102. The second case removes
70 nodes before assigning coefficients. Public G3 and G13 come from the
[Stanford Gset collection](https://web.stanford.edu/~yyye/yyye/Gset/).
The generator, raw input hashes, dependency versions, individual energies,
numerical bounds, checked bounds, and actual runtimes are preserved in
[the raw results](benchmarks/geometric_20260911.json).

The following gaps use the **same best observed feasible energy** for every
method on each input. This isolates bound strength from incumbent-search
quality. Higher lower bounds and smaller gaps are better.

| Case | Spins / couplings | Common energy | Previous hybrid checked gap | New automatic checked gap | Reduction |
|---|---:|---:|---:|---:|---:|
| Pegasus 4101 | 1,288 / 8,309 | -2,112.125 | 238.25 | 211.25 | 11.3% |
| Pegasus 4102 | 1,218 / 7,445 | -1,937.75 | 199 | 167 | 16.1% |
| G3 | 800 / 19,176 | -4,058 | 926 | 886 | 4.3% |
| G13 | 800 / 1,600 | -1,118 | 16 | 16 | 0% |

The previous hybrid is commit `4d27b7c9663cfa6be88ddb12573bfa957756a463`.
Actual runtimes were 5.009–5.105 seconds for that baseline and 5.027–5.211
seconds for the new automatic path. G13's older *numerical* bound was slightly
stronger than its checked bound and the new numerical bound. It is a neutral
result for certificates, not a universal improvement.

The next table compares the unchanged binary-product `linprog` input against
native `highspy` using the same HiGHS version and original rows. Native bounds
are numerical; the new column is independently checked.

| Case | Native common-energy gap | New `linprog` checked gap | Gap ratio | New actual seconds |
|---|---:|---:|---:|---:|
| Pegasus 4101 | 2,450.75 | 211 | 11.62× | 5.260 |
| Pegasus 4102 | 2,141 | 168 | 12.74× | 5.124 |
| G3 | 15,070 | 886 | 17.01× | 5.220 |
| G13 | 208 | 16 | 13.00× | 5.012 |

These are **11.6–17.0× smaller remaining gaps**, not runtime speedups. Native
`highspy` took 5.003–5.158 seconds. SciPy's own `linprog` was also measured and
is included in the raw results. Its bounds were the same except on G3, where
its gap was 15,062. The new path also found better feasible solutions, but the
table deliberately gives every method the common incumbent.

## Mathematical mechanism and its limits

For unit relaxation vectors `v_i`, choose distinct `i,j,k` and signs
`a_i,a_j,a_k ∈ {-1,+1}`. Every real spin assignment satisfies

```text
(a_i s_i + a_j s_j + a_k s_k)^2 >= 1
```

because the sum is an odd integer. Thus the corresponding valid triangle
inequality is violated by the relaxation exactly when

```text
||a_i v_i + a_j v_j + a_k v_k|| < 1.
```

Fix `a_i=1`, select a pair `(i,j)` and `a_j=±1`, and query the signed point set
`{v_k,-v_k}` near `q=-(v_i+a_j v_j)`. The four excluded endpoint vectors
`±v_i,±v_j` are all at distance at least one from `q`: two are exactly one
away, and the other two obey the reverse triangle inequality. Therefore a
single exact nearest-neighbor query detects a violated triangle for this
signed pair whenever one exists. No search for a fifth neighbor is required.

With an approximate neighbor guarantee `d_returned <= (1+epsilon) d_nearest`,
a triangle with `d_nearest < tau/(1+epsilon)` is guaranteed to be proposed
before batch filtering, when the acceptance threshold is `tau<1`.
[SciPy documents this distance guarantee](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.query.html).
The implementation uses `epsilon=0.5`, `tau=0.999`, at most `16n` original
pairs, and `8n` random pairs, each with both signs. It keeps at most 2,048 cuts
and limits each node to 16 selected triangles. Sampling and diversity filtering
mean it is not an exhaustive separation oracle. No worst-case subquadratic
running-time claim is made for the KD-tree search in high dimension.

Missing edges become auxiliary correlations with exactly zero objective
coefficient. Version 3 certificates retain the digest of the original input,
validate the added endpoints and all inequalities, and include every induced
Gram residual. The same exact checker accepts earlier certificate versions.
A numerical eigenvalue calculation or a successful Cholesky factorization is
never treated as proof. The final rational witness determines acceptance.

The 2,048-vertex limit bounds dense factorization memory. Larger inputs use the
existing sparse cuts. Deadline checks are cooperative: final factorization,
verification, or native solver work can exceed a short requested limit.

## Relation to published work

- [ConicBundle's Max-Cut tutorial](https://www-user.tu-chemnitz.de/~helmberg/ConicBundle/Manual/mctutorial.html)
  already combines SDP optimization and triangle separation, with a dense
  correlation matrix and enumeration in its example. This implementation instead
  queries the signed low-rank vectors and creates only selected missing edges.
- [Galli, Kaparis, and Letchford (2012)](https://www.lancaster.ac.uk/people/letchfoa/other-publications/2012-ISCO-gap.pdf)
  already study heuristic triangle separation, odd-clique and gap inequalities,
  eigenvector proposals, and compact cut collections. The triangle inequalities
  and the idea of selecting a small useful batch are not new here.
- [MADAM](https://link.springer.com/article/10.1007/s10589-021-00310-6)
  already strengthens SDP Max-Cut bounds with triangle, pentagonal, and heptagonal
  inequalities. Our exploratory five- and seven-spin parity cuts do not establish
  a new inequality family.
- [AugmentedMixing.jl](https://github.com/jschwiddessen/AugmentedMixing.jl)
  is a published low-rank method for constrained SDPs. The comparison below uses
  the unmodified package, not a reimplementation attributed to it.
- [TSSOS](https://github.com/wangjie212/TSSOS) provides sparse moment/SOS
  relaxations. The exploratory Boolean-monomial prototype belongs to that
  established family; selecting products does not automatically make it novel.

The signed nearest-neighbor construction is a concrete algorithmic candidate
for further research. The current literature search and measured improvements
do not prove that this construction is new, or that it outperforms the strongest
published Max-Cut solvers. The scientific novelty goal remains open.

## Published-method checks

`AugmentedMixing.jl` revision `21f29340faefb4080e61f453075ab3ff76c62e1f`
ran without source edits under Julia 1.13.0. It received every active cut in
the new five-second certificate, including nonlocal edges, with their original
objective coefficients. The selection cost was not charged to the baseline.
This favors the baseline and compares optimization of a common selected
relaxation; it does not compare two independent end-to-end cut selectors.

The default settings and one preset chosen on development cases were measured.
The preset is `mu_start=0.2`, `scaling=false`, `tau=1.3`. All runs used one BLAS
thread and `tol=1e-7`. A separate tiny instance warmed Julia compilation before
timing. Timed work includes formulation construction and the solve; the table
also adds the measured exact-repair time. Numerical duals are not treated as
certificates. Both direct PSD projection and coefficient matching with a
minimum-eigenvalue diagonal repair were checked, then the stronger of those
and the trivial bound was retained. This is the same residual-bound arithmetic
and objective-lattice rounding used for our witnesses.

| Case / method | Nominal solver seconds | Actual solve + verification seconds | Checked lower bound | Common-energy checked gap |
|---|---:|---:|---:|---:|
| Pegasus 4101, new auto | 5 | 5.211 | -2,323.375 | 211.25 |
| Augmented Mixing, default | 5 | 6.490 | -4,970.875 | 2,858.75 |
| Augmented Mixing, preset | 5 | 6.749 | -2,510.125 | 398 |
| Augmented Mixing, preset | 30 | 32.195 | -2,322.375 | 210.25 |
| G3, new auto | 5 | 5.085 | -4,944 | 886 |
| Augmented Mixing, default | 5 | 5.582 | -4,976 | 918 |
| Augmented Mixing, preset | 5 | 5.673 | -4,944 | 886 |
| Augmented Mixing, preset | 30 | 30.713 | -4,912 | 854 |

The new implementation improves the short-budget Pegasus result. It ties the
preset on G3 at five seconds, and the published solver achieves stronger
certificates when given thirty seconds. We did not measure first time to each
bound, so these observations do not establish a 6× time-to-target speedup.
Individual numerical bounds and repair outcomes are in
[the published-method raw results](benchmarks/geometric_published_20260911.json).

`TSSOS` revision `6a01185dadaedf3d36a7b1d18e1014ac4a9c8524` also ran unmodified,
using its documented SCS backend, Boolean reduction `nb=n`, order 2, `TS="MD"`,
and Gram output. The correlative-sparse formulation on a 96-spin, 4-regular
signed development graph produced a checked bound of -180 in 0.677 seconds
plus 0.108 seconds of repair. Its numerical objective was -181.7114. Both
coefficient-matching repair and direct PSD projection gave the same checked
bound after lattice rounding. The original trivial bound was -192. The current
engine obtained a checked bound of -150 and feasible energy -138 under the
same nominal five-second budget. That exploratory run overlapped a package
build, so its runtime is not used for a speed claim. See the
[small-instance raw comparison](benchmarks/geometric_tssos_20260911.json).

On the 1,288-spin Pegasus development case, CS-TSSOS order 2 was stopped after
roughly six minutes and about 9 GB of memory while building its JuMP model,
before the backend solver's time limit began. This is a formulation-construction
limit observed for that configuration. It is not evidence against every TSSOS
configuration or the licensed Mosek backend, which was not run. The generic
higher-order prototype did not earn production integration on these experiments.

## Reproduce

The previous source checkout must contain commit `4d27b7c` and its dependencies.
The optional compiled module can be built there normally. Both checkouts need
the same Python environment for a controlled comparison.

```bash
python experiments/geometric_suite.py \
  --directory /tmp/geometric-final --previous /path/to/4d27b7c

python experiments/published_suite.py --directory /tmp/geometric-final \
  --julia /path/to/julia --project /path/to/julia-research-environment
```

The Julia research environment needs `AugmentedMixing`, `TSSOS`, `JSON`,
`DynamicPolynomials`, `JuMP`, and `SCS`, with the two source revisions pinned as
above. These are research dependencies only. Set `JULIA_DEPOT_PATH` if using
an isolated depot. Each runner saves native output as well as final measurements.
No solver responses or certificates are reused as cached answers.

The implementation was checked by the full 475-test suite, including exhaustive
small-model energy comparisons, tampered-certificate rejection, and preservation
of extra equality and inequality rows in the binary-product bridge.
