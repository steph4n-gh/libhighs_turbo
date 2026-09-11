# Global Ising bounds from the September research leads

`solve_ising(..., relaxation="hybrid")` now optimizes cycle/subgraph cut
multipliers together with a global semidefinite relaxation. It returns a
serializable exact certificate and installs the resulting objective bound in
HiGHS. `relaxation="sdp"` provides the global relaxation without adaptive cuts;
the existing `"cuts"` default remains available unchanged.

## The useful research lead

September 5's `work/polynomial-residual-bound-certificate/DERIVATION.md` supplies
a rational PSD-factorization certificate for a signed residual matrix. That
is the lead implemented here, generalized to the full signed Ising objective.
Its rank-one block compression results require structural assumptions that
arbitrary signed Pegasus couplings do not satisfy. Its bounded-clique
counterexamples motivate looking beyond local rows, but do not establish a
ceiling for every arbitrary-coefficient subgraph cut in this solver.

`researchSept10` contains the earlier surrogate engine from which this package
developed. Its dual-face projection and nonnegative aggregation are useful
components, but aggregating a fixed cut pool does not strengthen its relaxation.
Here the important change is a global relaxation and multipliers optimized for
that relaxation, rather than multipliers frozen from the box/LP bound.

These are established mathematical ingredients. Low-rank SDP optimization is
studied in [Wang, Chang and Kolter's Mixing method](https://arxiv.org/abs/1706.00476),
and rigorous factorization/error bounds in [Rump's verification work](https://www.tuhh.de/ti3/paper/rump/Ru06c.pdf).
This implementation uses L-BFGS with normalized vectors, not that paper's
coordinate-descent implementation. No algorithmic novelty or priority is claimed.

## Exact bound, approximate search

For cut variables `x_uv=(1-s_u*s_v)/2`, verified rows `A*x <= b`, and
nonnegative rational multipliers `lambda`, define `r=w-A.T*lambda`. Then

```
E(s) >= c + sum(A.T*lambda) - 2*lambda*b + sum(r_uv*s_u*s_v).
```

Let `Q_uv=r_uv/2` symmetrically, with zero diagonal. Any rational factor
`B/D` gives a positive semidefinite Gram matrix `G=B*B.T/D^2`, so

```
s.T*Q*s >= trace(Q-G) - sum(i != j, abs((Q-G)_ij)).
```

The checker evaluates that expression exactly, including Gram fill-in on
nonedges. It keeps the stronger of this bound and the original box-residual
bound, then rounds up on the original objective's exact energy lattice.
An incorrect numerical eigensolver, stalled vector optimization, or inaccurate
Cholesky factor can only produce a weaker candidate; none supplies proof.

The candidate generator fits unit vectors of rank at most 16, estimates the
dual diagonal, shifts its slack, and quantizes a Cholesky factor. The hybrid
also optimizes nonnegative cut multipliers. The ascent direction is twice each
cut's violation at the current vector correlations. Inner solves are approximate,
so this search does not promise an optimal SDP solution. Every final multiplier
and factor is independently checked against the original rational problem.

Integer Gram products use binary64 BLAS after checking `n*max(abs(B))^2 < 2^53`.
Every integer product and partial sum is then exactly representable under
IEEE-754 binary64 arithmetic. Checked int64 row sums are accumulated in Python
integers; all objective/residual corrections use unbounded integer arithmetic.
The checker requires neither an eigensolver nor an SDP solver. Version-1 cut
certificates remain readable; version 2 also carries the triangular factor.

## Measurement protocol

The fresh full-fabric cases use seeds 3001–3003, both intact and damaged, from
the existing weighted Pegasus generator. Damage removes 5% of nodes and 10%
of the remaining couplers. These seeds were not used to develop the method.
The development probes used seeds 2001–2003. Fields/couplings are signed eighths.

All three methods receive five seconds on the same six problems, in fresh
processes with one BLAS/OpenMP/HiGHS thread and alternating order. Times include
normalization, preparation, search, witness construction and an additional
independent certificate check. Imports, graph generation and optional witness
serialization to disk are outside the clock. Spin energies are independently
recomputed from the original input. There is one run per case/method, so these
are direct observations, not distributions or claims of stable tail latency.

A separate, predetermined comparison uses intact seeds 3001 and 3003 and
damaged seed 3002. The target is an exact gap at most `0.075*sum(abs(h,J))`.
The hybrid receives five seconds; the previous cut engine receives 45 seconds.
A timeout with an open target gives a censored lower bound on time-to-target,
not a measured completion time. No optimum or speedup over specialist SDP
solvers is implied by this comparison.

Measurements run on an Apple M4 Pro / 24 GiB, macOS 26.6, Python 3.13.5,
NumPy 2.5.3, SciPy 1.18.1, HiGHS 1.15.1 and dwave-samplers 1.8.0. Dense
factorization/checking may exceed a requested time limit. The implementation
caps global factors at 2,048 vertices, including the field reference spin.

## Results

At five seconds, the hybrid reduced the verified gap by **60.7–65.2%**
relative to the previous adaptive cut engine (median **63.7%** across six cases).
Every displayed gap is backed by an independently checked witness. All
full-size instances still have open optimality gaps.

| Seed / fabric | Spins | Cut gap | SDP gap | Hybrid gap | Hybrid energy | Hybrid bound |
|---|---:|---:|---:|---:|---:|---:|
| 3001 / intact | 1288 | 702 | 384.25 | 246 | -2068.75 | -2314.75 |
| 3002 / intact | 1288 | 697.75 | 369.75 | 243 | -2082.5 | -2325.5 |
| 3003 / intact | 1288 | 653.25 | 371.25 | 231.75 | -2085 | -2316.75 |
| 3001 / damaged | 1225 | 484.75 | 304.5 | 180 | -1851.88 | -2031.88 |
| 3002 / damaged | 1220 | 487.25 | 309.25 | 185.25 | -1838.38 | -2023.62 |
| 3003 / damaged | 1216 | 473.75 | 318.25 | 186.25 | -1807.5 | -1993.75 |

The hybrid reached the preselected target in **3.25–3.71 seconds**; the prior
cut engine missed it after 45 seconds in all three cases. This establishes
a **greater-than-12x observed time-to-target advantage over that engine** in
these comparisons. Ratios below conservatively use the nominal 45-second
budget, not its slightly longer observed elapsed time.

| Seed / fabric | Target gap | Hybrid gap / seconds | Cut gap after 45 s | Time advantage |
|---|---:|---:|---:|---:|
| 3001 / intact | 369.769 | 246 / 3.582 s | 538.25 | >12.56x |
| 3003 / intact | 366.919 | 232.75 / 3.706 s | 514.75 | >12.14x |
| 3002 / damaged | 305.391 | 184.75 / 3.245 s | 353 | >13.87x |

These are 24 complete runs: 18 equal-budget observations plus six target
observations. The fixed-budget elapsed times span 5.08–5.35 seconds, including the extra
check. The result is a substantial improvement over this repository's previous
engine; it does **not** establish a speedup over specialist SDP/Max-Cut solvers,
a general 10x advantage, or a newly discovered mathematical algorithm.

[Raw measurements](benchmarks/ising/sdp.json) include every run and checkpoint.
The [saved witness](benchmarks/ising/sdp-witness.json.gz) is about 598 KiB compressed.
It corresponds to intact seed 3001 and certifies the original energy lower bound
of -2314.75. The measured source digest (concatenated files listed in the raw JSON) is
`bec5fcacf6948ee2a8167136004b785bd901778b321a595f20f868bf82012aac`.

```bash
python examples/benchmark_sdp_ising.py \
  --json benchmarks/ising/sdp.json \
  --witness benchmarks/ising/sdp-witness.json.gz
```

The checked-in witness corresponds to intact seed 3001 using the hybrid.
It can be checked without running a solver:

```python
import gzip, json, sys
sys.path.insert(0, "examples")
from benchmark_adaptive_ising import instance
from highs_turbo import verify_ising_certificate
h, J = instance(0, 3001, False)
with gzip.open("benchmarks/ising/sdp-witness.json.gz", "rt") as stream:
    assert verify_ising_certificate(h, J, json.load(stream))
```
