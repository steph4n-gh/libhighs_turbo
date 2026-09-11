# Adaptive Ising solver: method and held-out evaluation

The implementation adds iterative separation, bounded subgraph inequalities,
a fast multiplier search, annealing/tabu incumbents, a learned candidate ranker,
and a serializable exact bound witness to `solve_ising`. The research target is
10x less total time to the same independently verified gap on previously unseen
weighted and damaged Pegasus graphs with more than 1,000 spins. Results below
will distinguish verified bounds from ordinary numerical solver bounds.

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
optimality. Specifying `certified_gap` stops only against the exact witness.
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

The saved model used 1,559 measured candidates. Seeds 106, 112, 118, and 124 were
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

The held-out run is in progress. No 10x performance claim has been established.
