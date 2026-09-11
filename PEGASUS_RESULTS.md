# Pegasus Ising results

Measured on an Apple M4 Pro (12 cores), macOS 26.6, Python 3.13.5, HiGHS 1.15.1, and dwave-graphs 1.0.0. All runs use the local CPU and reproducible random couplings on the official Pegasus topology.

## Method

Each input is an induced breadth-first subgraph of the official D-Wave Pegasus fabric. The traversal root and coupling seed are specified by `examples/benchmark_ising.py`. Small and weighted cases use P_4; larger subgraphs use P_8. Both methods receive exactly the same coefficients and retain the original sparse XOR formulation. Both use HiGHS 1.15.1 with parallel search disabled, the same automatic native thread pool, the same random seed, and the same wall-clock budget. Every call constructs a fresh model; no solve cache or cross-call warm start is used.

Turbo adds verified triangle, square, and fundamental-cycle inequalities, computes a rational bound from a nonnegative combination of frustrated-cycle cuts, and supplies a bounded local-search incumbent. Timings include normalization, model construction, cut generation and verification, initialization, and integer search. Input graph generation and Python imports are shared and excluded. Execution order alternates; each reported value is the median of three cold solves.

For cut variables x, a verified aggregate a*x <= b implies w*x <= b + sum(max(0, w-a)). The corresponding Ising energy is at least offset + sum(w) - 2*(b + sum(max(0, w-a))). This arithmetic uses exact rational coefficients. An exact optimality certificate requires this bound to equal the independently evaluated spin energy; ordinary HiGHS optimality and dual bounds remain numerical. The individual cycle cuts are retained in the MIP.

## Both methods reached numerical optimality

All 54 runs below reached the same optimal energies (nine instances, two methods, three repeats). Every returned energy was recomputed from its spins and original coefficients. Separate exhaustive checks cover small Pegasus subgraphs with weighted couplings and local fields.

| Spins | Seed | Energy | Native ms | Turbo ms | Speedup |
|---:|---:|---:|---:|---:|---:|
| 32 | 1 | -57 | 80.83 | 14.15 | 5.71x |
| 32 | 2 | -61 | 292.73 | 18.17 | 16.11x |
| 32 | 3 | -53 | 243.36 | 17.78 | 13.68x |
| 48 | 1 | -82 | 854.37 | 362.97 | 2.35x |
| 48 | 2 | -84 | 1984.52 | 27.00 | 73.51x |
| 48 | 3 | -89 | 2736.96 | 528.77 | 5.18x |
| 64 | 1 | -106 | 3273.66 | 1628.79 | 2.01x |
| 64 | 2 | -118 | 6888.24 | 2561.41 | 2.69x |
| 64 | 3 | -139 | 3678.31 | 481.02 | 7.65x |

The sum of per-instance median times is 20.03 s for native HiGHS and 5.64 s for turbo (3.55x). The geometric mean of per-instance speedups is 7.11x. Individual large speedups reflect changes in the search tree and do not establish a general speed guarantee.

```bash
python examples/benchmark_ising.py --m 4 --sizes 32 48 64 --seeds 1 2 3 --repeats 3 --time-limit 10 --json small.json
```

## Weighted couplings and local fields

Couplings are random multiples of 1/8 from -1 through 1, and local fields are multiples of 1/8 from -3/8 through 3/8. The following cases use a five-second total budget per solve.

| Spins | Seed | Native energy | Turbo energy | Native gap | Turbo gap | Native ms | Turbo ms | Speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 1 | -37.5 | -37.5 | 0.00 | 0.00 | 48.62 | 12.44 | 3.91x |
| 32 | 2 | -35.5 | -35.5 | 0.00 | 0.00 | 175.16 | 38.25 | 4.58x |
| 32 | 3 | -32.375 | -32.375 | 0.00 | 0.00 | 154.19 | 19.91 | 7.74x |
| 64 | 1 | -66.75 | -66.75 | 0.00 | 0.00 | 3296.35 | 1205.42 | 2.73x |
| 64 | 2 | -77.5 | -77.5 | 0.00 | 0.00 | 4276.31 | 843.07 | 5.07x |
| 64 | 3 | -79.625 | -79.625 | 0.00 | 0.00 | 3920.36 | 1912.12 | 2.05x |

```bash
python examples/benchmark_ising.py --m 4 --sizes 32 64 --seeds 1 2 3 --repeats 3 --time-limit 5 --weighted --json weighted.json
```

## Larger subgraphs and full fabrics: five-second budget

These instances remain unfinished. No time-to-optimum speedup is assigned to them. Lower energy is better; a higher lower bound and a smaller remaining gap give more information about optimality. Every column is a median of three runs. The benchmark JSON retains the energy/bound pair for every run.

| Fabric / spins | Seed | Native energy | Turbo energy | Native bound | Turbo bound | Native gap | Turbo gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| P_8 / 128 | 1 | -201 | -273 | -519.00 | -309.41 | 318.00 | 36.41 |
| P_8 / 128 | 2 | -203 | -253 | -529.00 | -311.50 | 326.00 | 58.50 |
| P_8 / 128 | 3 | -201 | -251 | -519.00 | -311.00 | 318.00 | 60.00 |
| P_8 / 256 | 1 | -395 | -503 | -1215.00 | -706.50 | 820.00 | 203.50 |
| P_8 / 256 | 2 | -365 | -545 | -1267.00 | -735.50 | 902.00 | 190.50 |
| P_8 / 256 | 3 | -405 | -531 | -1267.00 | -734.00 | 862.00 | 203.00 |
| P_8 / 512 | 1 | -715 | -1101 | -2815.00 | -1939.00 | 2100.00 | 838.00 |
| P_8 / 512 | 2 | -765 | -1127 | -2911.00 | -1977.00 | 2146.00 | 850.00 |
| P_8 / 512 | 3 | -425 | -1085 | -2887.00 | -1969.00 | 2462.00 | 884.00 |
| P_4 / 264 | 1 | -434 | -568 | -1462.00 | -830.00 | 1028.00 | 262.00 |
| P_8 / 1288 | 1 | -96 | -3044 | -8550.00 | -5562.00 | 8454.00 | 2518.00 |

```bash
python examples/benchmark_ising.py --m 8 --sizes 128 256 512 --seeds 1 2 3 --repeats 3 --time-limit 5 --json large.json
python examples/benchmark_ising.py --m 4 --sizes 0 --seeds 1 --repeats 3 --time-limit 5 --json full-p4.json
python examples/benchmark_ising.py --m 8 --sizes 0 --seeds 1 --repeats 3 --time-limit 5 --json full-p8.json
```

The API passes the remaining budget to HiGHS after preparation. Native stopping checks and result evaluation can add a few milliseconds; a time limit is not a hard real-time deadline. The full P_8 result does not establish a ground state.

## Reproduction and limits

The measurements use the Ising implementation in commit `e1fabdc`. A subsequent compatibility fix routes the older array-based solver API through the same public HiGHS runtime and removes the system HiGHS link from the optional graph extension; it does not change the Ising formulation or acceleration algorithm. The benchmark records every selected instance, including unfinished runs. Performance depends on the coefficients, ordering, solver version, machine, and budget. The comparison isolates our cuts and starting solution against the same HiGHS formulation; it does not compare against specialized Ising optimizers or a QPU.

The topology comes from [D-Wave’s maintained graph package](https://pypi.org/project/dwave-graphs/1.0.0/) and its [official Pegasus generator](https://docs.dwavequantum.com/en/latest/ocean/api_ref_graphs/generated/dwave.graphs.pegasus_graph.html). Default P_4 has 264 spins and 1,604 couplers; default P_8 has 1,288 and 8,804. Hardware subgraphs can be supplied through explicit node and coupler lists.

Validation: 452 repository tests pass; the Ising and LP regressions pass with highspy 1.11.0; Linux, Windows, and macOS CI validate the new API and packaging. Cut feasibility and lower bounds are checked against exhaustive spin assignments on small problems.
