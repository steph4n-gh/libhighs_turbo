# Adaptive Ising solver: method and held-out evaluation

This records the cut-only implementation. The subsequent global relaxation
and its separate fresh-instance measurements are in [SDP_ISING_RESULTS.md](SDP_ISING_RESULTS.md).

The implementation adds iterative separation, bounded subgraph inequalities,
a fast multiplier search, annealing/tabu incumbents, a learned candidate ranker,
and a serializable exact bound witness to `solve_ising`. The research target is
10x less total time to the same independently verified gap on previously unseen
weighted and damaged Pegasus graphs with more than 1,000 spins. The experiment completed 108 runs across six methods. Deterministic adaptive
separation reduced the per-instance median verified gap by 51.7–59.0% versus
the static implementation (median reduction 54.9%). No full instance was
proved optimal, no learned advantage was established, and the 10x research
target remains unproven.

## What is implemented

The original energy is converted to weighted Max-Cut, using a fixed reference
spin for fields. Write `E = c + sum(w) - 2*w*x`, with binary cut indicators
`x`. For verified inequalities `A*x <= b` and any nonnegative multipliers
`lambda`,

```
U = lambda*b + sum(max(0, w - A.T*lambda))
E_lower = c + sum(w) - 2*U
```

This bound does not assume that an LP solved correctly or that its numerical
duals satisfy a tolerance. Candidate multipliers are quantized to a common
positive denominator; the residual, aggregation, and final objective-lattice
rounding use integer/Fraction arithmetic. A digest binds the normalized input,
but supplies no mathematical evidence by itself. The standalone checker
verifies the rows and recomputes the bound without running an optimizer or
loading the learned model. The returned feasible spin energy is also evaluated
in exact arithmetic.

The initial pool contains frustrated triangles and chordless squares. A bounded
coordinate search improves the nonnegative multipliers. Its averaged residual
signs supply approximate fractional points for two early separation rounds;
these points need not be feasible. Later rounds solve the edge LP in one HiGHS
instance, retain its basis, and add violated short cycles, longer cycles found
through shortest paths in a double cover, and small-subgraph inequalities.
Separation uses bounded candidate batches; failure to find a cut does not prove
that the relaxation is the cut polytope.

For subgraphs with at most 12 vertices, the oracle first tries the local energy
direction, then a small separation LP over all local cut assignments. Proposed
coefficients are rounded to small integers and the right-hand side is recomputed
by exhaustive enumeration. The checker verifies every local assignment again.
For example, the all-positive K6 cycle relaxation allows energy -5, whereas the
new subgraph cut proves the correct optimum -3.

A remaining-budget MIP keeps the original sparse XOR formulation and every
retained verified cut. Its reported numerical branch-and-bound bound is separate
from the exact certificate. Numerical optimality is not promoted to rational
optimality. The `certified_gap` early-stop test uses only the exact witness; inspect
`exact_gap` if search stops for another reason, including numerical MIP optimality.
The fast native multiplier loop is currently available with the macOS extension;
the portable Python implementation follows the same coordinate rule and checks
its time budget within each sweep.

## Learned policy and training split

`cut_policy="deterministic"` remains the default. The optional learned model is a
12-feature, 16-hidden-unit ReLU network. It orders candidate supports using graph
density, cycle rank, current fractionality, weights, satisfaction, and boundary
statistics. It cannot create unchecked rows or alter proof acceptance. Production
inference uses NumPy; training uses the existing optional PyTorch dependency.

Training uses seeds 101–124 on weighted Pegasus subgraphs of 32, 64, 128, and 256
vertices, with 10% random coupler removal on alternating instances. Each proposed
cut is temporarily inserted into the current LP, its actual marginal energy-bound
improvement is measured, and the original basis/model is restored. Runs without
an optimal marginal LP measurement are excluded. The label is
`log1p(gain / max(separator_seconds, 0.001))`. Training explores candidates in
random order, independently of the current learned policy.

The collection recorded 3,441 candidate attempts; 1,559 supplied usable labels,
including 316 positive gains. The other marginal LP solves lacked an optimal
measurement within their budget. The saved model used the 1,559 measured candidates. Seeds 106, 112, 118, and 124 were
held out from fitting. Validation mean squared error was 2.175, versus 3.508 for
a constant predictor learned on the training split. This prediction metric is
not evidence of faster solving or generalization to full fabrics. Many candidates
have zero marginal gain. Deployment also encounters approximate early points,
while training labels were collected at LP solutions.

Reproduce data collection and fitting with:

```bash
python -m pip install '.[ml,ising]'
python examples/train_ising_policy.py --instances 24 --seconds 2 \
  --data /tmp/ising-policy-training.json
```

The exact recorded observations are in `benchmarks/ising/training.json.gz`;
model parameters and training metadata are in `highs_turbo/ising_policy.json`.
Time-based labels and bounded sampling can vary across machines and runs.

## Evaluation protocol

Measured on an Apple M4 Pro with 24 GiB RAM, macOS 26.6 arm64, Python 3.13.5,
NumPy 2.5.3, SciPy 1.18.1, HiGHS 1.15.1, dwave-graphs 1.0.0,
dwave-samplers 1.8.0, and dimod 0.12.22. Training used PyTorch 2.14.0.

The evaluation uses official full P8 fabrics (1,288 spins), random couplings in
`{-8,...,8}/8`, and fields in `{-3,...,3}/8`. The damaged variants remove each
spin with probability 5%, then each surviving coupler with probability 10%.
Seeds 1001, 1002, and 1003 are held out from training and development. No model
or algorithm tuning is performed from their results. Every input exceeds 1,000
spins. The normalized graph includes one additional reference vertex for fields.

Methods are the original HiGHS formulation without cuts/start, the previous
static acceleration, deterministic adaptive separation, learned adaptive
separation, a dedicated simulated-annealing/tabu portfolio, and SMS, a specialist
SCIP-based Max-Cut solver. Each receives a five-second search budget, with three
repeats and alternating/rotated order. Every method runs in a fresh worker.
Measured wall time includes normalization, model construction, sampling, cut
search, and proof checking. Imports are excluded for Python methods; SMS process
startup and file interchange are included. No warm starts or solved instances
are shared between methods. HiGHS, SMS, and native numerical libraries are limited
to one thread. Time limits are cooperative, so overruns are reported, not clipped.

The dedicated heuristic gets half its budget for 1,000-sweep annealing reads and
half for tabu search. It returns an independently evaluated feasible energy,
with no optimality bound. The adaptive solver spends at most about 100 ms on
its initial sampler and reserves part of the main budget for MIP search. The
`ising` extra was installed for all measurements.

The benchmark independently recomputes each returned spin energy. Exact witness
checks are included for all `solve_ising` results, including the static baseline.
Every numerical bound is also checked against the best feasible energy found
by any method on the identical input; this sanity check is not a proof of that
bound. SMS's global numerical gap is reconstructed from its subproblem gaps,
scaling factor, and recovered solution. Its reductions and branch-and-bound
proof are not verified by our checker.

An additional predetermined checkpoint target is an absolute certificate gap of
20% of `sum(abs(h)) + sum(abs(J))`. It is a coarse quality target, not a claim of
near-optimality. `time_to_target` records the first checked bound/energy checkpoint
meeting it. Missing targets are censored; they are never converted into invented
speedup ratios. SMS and the heuristic have no independently checked witness in
this experiment, so this protocol cannot establish a 10x certified-gap advantage
over those methods.

```bash
python examples/benchmark_adaptive_ising.py --seconds 5 --sizes 0 \
  --seeds 1001 1002 1003 --repeats 3 --sms /path/to/sms \
  --json benchmarks/ising/heldout.json
```

## Specialist baseline provenance

[SMS](https://github.com/CharJon/SMS) was built at commit
`88aff1a0b5caafa70dc8ced9f9d18b953068d96f`, with its pinned Networkit, MQLib,
JSON, and cxxopts submodules. The solver uses SCIP 10.0.2 and SoPlex 8.0.2 from
the official macOS arm64 SCIP Optimization Suite release. No algorithmic changes
were made to the baseline. The recorded macOS build patches use Accelerate's
CBLAS, remove the obsolete classic linker option, add missing standard includes,
and move one MQLib method definition after its complete type declaration. The
patches are in `benchmarks/ising/sms-macos.patch` and `mqlib-macos.patch`.

The build used LLVM 17, C++20, release optimization, libomp, GMP, MPFR, and Boost
headers. Tests were disabled for the external benchmark build. The SDK's missing
unversioned GMP/MPFR development symlinks were restored to its supplied versioned
libraries. SMS default presolve and search settings were retained; its source
explicitly sets SCIP's LP and search thread limits to one. Only the time limit,
seed, and output paths were passed at runtime.

## Measured results

All entries are medians of three runs; energy and gap medians are calculated
independently and need not come from the same run. The raw 108 records, including
all numerical bounds, exact gaps, timings, cut counts, and checked checkpoints,
are in [`benchmarks/ising/heldout.json`](benchmarks/ising/heldout.json).

| Input | Spins | Static verified gap | Adaptive verified gap | Learned verified gap | Reduction vs static |
|---|---:|---:|---:|---:|---:|
| Intact 1001 | 1288 | 1615.75 | 662.50 | 686.50 | 59.0% |
| Intact 1002 | 1288 | 1656.00 | 691.75 | 691.75 | 58.2% |
| Intact 1003 | 1288 | 1642.00 | 712.25 | 712.25 | 56.6% |
| Damaged 1001 | 1227 | 1063.75 | 499.25 | 502.50 | 53.1% |
| Damaged 1002 | 1206 | 955.00 | 461.25 | 462.50 | 51.7% |
| Damaged 1003 | 1217 | 1047.25 | 491.00 | 496.00 | 53.1% |

Energy and numerical-gap pairs (lower energy and smaller gap are better):

| Input | Native E / gap | Static E / gap | Adaptive E / gap | Learned E / gap | SMS E / gap | Heuristic E |
|---|---:|---:|---:|---:|---:|---:|
| Intact 1001 | 19.12 / 4574.50 | -1881.62 / 1615.75 | -2100.62 / 662.50 | -2100.62 / 686.50 | -748.38 / 2588.75 | -2111.88 |
| Intact 1002 | -46.50 / 4537.75 | -1870.50 / 1656.00 | -2101.25 / 691.75 | -2101.25 / 691.75 | -768.50 / 2593.88 | -2120.50 |
| Intact 1003 | -9.12 / 4555.75 | -1862.12 / 1642.00 | -2078.38 / 712.25 | -2078.38 / 712.25 | -795.88 / 2585.62 | -2100.38 |
| Damaged 1001 | 27.38 / 3741.75 | -1628.12 / 1063.75 | -1846.88 / 499.25 | -1846.88 / 502.50 | -726.62 / 1927.46 | -1864.12 |
| Damaged 1002 | -35.62 / 3482.25 | -1597.12 / 955.00 | -1787.12 / 461.25 | -1787.12 / 462.50 | -719.38 / 1825.28 | -1802.62 |
| Damaged 1003 | -64.00 / 3550.00 | -1603.25 / 1047.25 | -1801.75 / 491.00 | -1801.75 / 496.00 | -694.50 / 1929.50 | -1808.50 |

The deterministic method reduced the verified gap in every case. The median
per-instance reduction was **54.9%**, with a range of **51.7–59.0%**, relative to
the static implementation. These are fixed-budget gap reductions, not solve-time
speedup ratios. All exact gaps remain large; none of the six instances was
proved optimal by any method within the five-second budget.

Both adaptive policies met the coarse 20%-of-L1 certificate target in all 18
runs. Deterministic checkpoint times ranged from **0.701 to 0.957 s**, median
**0.851 s**; learned times ranged from **0.701 to 0.993 s**, median **0.834 s**.
The static and unaccelerated paths did not meet it in five seconds. Censoring
and the absence of independently checked SMS proofs prevent a valid 10x
comparison with the strongest available baseline.

Learning did not deliver a consistent benefit: across case medians, its verified
gap was equal to or larger than the deterministic gap, with a median increase
of 0.46%. Its small timing difference is insufficient evidence of a speedup.
Deterministic selection therefore remains the default. The observed improvement
belongs to the deterministic pipeline; it is not evidence that learning generates
better proofs.

The dedicated annealing/tabu method produced better median feasible energies
than the adaptive solver on all six cases. It provides no lower bound. SMS's
five-second numerical energy/gap pairs were worse in these runs, but this short
budget, one machine, one instance family, and one default specialist configuration
do not establish general superiority over specialist exact solvers.

Measured wall-time medians were 5.093 s (native), 5.077 s (static), 5.141 s
(deterministic), 5.166 s (learned), 5.078 s (SMS), and 5.050 s (heuristic).
The largest overrun was a native HiGHS run at 6.341 s; adaptive runs stayed
within 5.212 s. Actual times are retained in the raw records.

The measured Ising implementation is commit `02185f5de4254fa06ecbffe3b81e20371d15a3d1`.
A subsequent repair to the separate LP portfolio's early-error handling does not
change the timed Ising code. No other CPU-heavy tasks ran during measurement.

## Validation and remaining work

The test suite includes exhaustive small Ising comparisons, all-state validity
checks, a K6 subgraph strengthening example, long-cycle separation, perturbed
numerical multipliers, tampered serialized witnesses, and learned-policy proof
checks. Minimum-supported HiGHS 1.11 and the pure Python fallback pass 62
solver/Ising tests. Packaging includes the trained NumPy policy and optional
native multiplier kernel. CI also found and now covers an older LP portfolio
error path: an early simplex error must still launch its independent IPM method.

The next research obstacle is substantially tighter full-fabric proofs. The
12-vertex oracle and current learned ranker do not yet establish that capability.
A further attempt should measure larger recurring structures and train on
full-scale marginal **verified bound gain per total second**, including the
approximate early separation points. A common proof-checking path for specialist
baselines is also needed before claiming the strict certified-gap target.


## Scope of the research claim

Cycle separation and specialized branch-and-cut are established methods; see
[Rehfeldt, Koch, and Shinano (2023)](https://link.springer.com/article/10.1007/s12532-023-00236-6).
Learned cut selection also predates this work, for example
[Tang, Agrawal, and Faenza (2020)](https://proceedings.mlr.press/v119/tang20a.html).
Independent integer-programming proof checking is likewise established in
[VIPR](https://github.com/scipopt/vipr). This change implements and evaluates a
particular combination; it does not establish a new algorithm or a general
advantage over state-of-the-art exact solvers. SMS is one public specialist
baseline, not an exhaustive comparison with commercial and research solvers.
