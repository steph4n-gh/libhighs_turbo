# Netlib cold-solve results

Measured September 11, 2026 on an Apple M4 Pro (12 CPU cores), macOS arm64,
Python 3.13.5, SciPy 1.18.1, and highspy 1.15.1. Values are the median of five
complete solves per method, in milliseconds. Each call creates fresh solver
instances. Model parsing and Python imports are outside the timed region;
normalization, model construction, row recovery, solver work, cancellation,
worker joins, and result construction are inside it. Execution order rotates.

The corpus contains all 91 models in [COIN-OR Data-Netlib at
f1cc423](https://github.com/coin-or-tools/Data-Netlib/tree/f1cc423067407d55d579c9c35fb01edf860dbc24),
plus the infeasible [woodinfe model from HiGHS](https://github.com/ERGO-Code/HiGHS/blob/73cac48c5340d775a477087198611862559be250/check/instances/woodinfe.mps).
No models were removed after timing. Matrix dimensions include equality rows.
All objective, inequality/equality feasibility, and status checks passed.
The MPS reader negates maximization objectives and omits constant offsets
equally for every solver; this preserves the optimizer.

| Across 92 models | SciPy | Full native HiGHS | Turbo |
| --- | ---: | ---: | ---: |
| Sum of per-model medians (seconds) | 7.883 | 7.626 | 4.137 |
| Turbo speedup by total time | 1.91× | 1.84× | — |
| Geometric mean of per-model speedups | 1.32× | 0.93× | — |
| Models where turbo measured faster | 82 | 10 | — |

The total-time improvement is concentrated in expensive models. The geometric
mean against full native HiGHS is below one: turbo adds overhead to many small
native solves. Its wider gains over SciPy include avoiding SciPy's frontend
work and differences in SciPy's bundled HiGHS. The full-native control uses the
same HiGHS 1.15.1 library as turbo and makes the algorithm comparison visible.

For large models that do not use row reduction, simplex starts first. If it is
still running after 5 ms, an independent interior-point method may start.
The first optimal result is checked against the supplied model, the other
method is canceled, and both workers are joined before the timing stops.
This uses up to two CPU cores and two model copies. Extra CPU work and occasional
slowdowns are the tradeoff; there is no promise that every model gets faster.
No previous solution or basis is cached between these cold solves.

`highs_direct` retains one native simplex solve. `highs_portfolio` used both
methods. `SciPy` indicates delegation. The larger speedups below are real
Netlib solves; they are not the separately reported synthetic graph examples.

## Reproduce

After installing this branch and its dependencies:

```bash
git clone https://github.com/coin-or-tools/Data-Netlib.git /tmp/netlib
git -C /tmp/netlib checkout f1cc423067407d55d579c9c35fb01edf860dbc24
curl -L https://raw.githubusercontent.com/ERGO-Code/HiGHS/73cac48c5340d775a477087198611862559be250/check/instances/woodinfe.mps -o /tmp/woodinfe.mps
python examples/benchmark_linprog.py --repeats 5 --mps /tmp/netlib/*.mps.gz /tmp/woodinfe.mps
```

The script also prints its existing synthetic comparisons. These are excluded
from every aggregate above. Runtime variation can change which method wins and
small timing differences should not be interpreted as stable advantages.

## Every model

| Model | Rows × columns | SciPy ms | Native ms | Turbo ms | vs SciPy | vs native | Strategy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 25fv47.mps | 821 × 1571 | 75.01 | 72.42 | 69.18 | 1.08× | 1.05× | highs_portfolio |
| 80bau3b.mps | 2262 × 9799 | 66.01 | 57.72 | 68.43 | 0.96× | 0.84× | highs_portfolio |
| adlittle.mps | 56 × 97 | 1.52 | 0.75 | 0.75 | 2.03× | 1.00× | highs_direct |
| afiro.mps | 27 × 32 | 0.79 | 0.20 | 0.28 | 2.82× | 0.71× | highs_direct |
| agg.mps | 488 × 163 | 2.20 | 1.48 | 1.91 | 1.15× | 0.77× | highs_direct |
| agg2.mps | 516 × 302 | 3.21 | 2.07 | 2.21 | 1.45× | 0.94× | highs_direct |
| agg3.mps | 516 × 302 | 3.46 | 2.17 | 2.52 | 1.37× | 0.86× | highs_direct |
| bandm.mps | 305 × 472 | 5.58 | 4.70 | 4.77 | 1.17× | 0.99× | highs_direct |
| beaconfd.mps | 173 × 262 | 1.82 | 0.98 | 1.13 | 1.61× | 0.87× | highs_direct |
| blend.mps | 74 × 83 | 1.46 | 0.79 | 0.81 | 1.80× | 0.98× | highs_direct |
| bnl1.mps | 643 × 1175 | 15.37 | 13.49 | 13.81 | 1.11× | 0.98× | highs_direct |
| bnl2.mps | 2324 × 3489 | 27.00 | 23.73 | 28.98 | 0.93× | 0.82× | highs_portfolio |
| boeing1.mps | 440 × 384 | 7.94 | 5.25 | 9.85 | 0.81× | 0.53× | highs_direct |
| boeing2.mps | 185 × 143 | 2.42 | 1.36 | 1.37 | 1.77× | 0.99× | highs_direct |
| bore3d.mps | 233 × 315 | 1.77 | 0.89 | 0.98 | 1.81× | 0.91× | highs_direct |
| brandy.mps | 220 × 249 | 3.41 | 2.73 | 2.75 | 1.24× | 0.99× | highs_direct |
| capri.mps | 271 × 353 | 2.61 | 1.77 | 1.83 | 1.43× | 0.97× | highs_direct |
| cycle.mps | 1903 × 2857 | 65.67 | 62.58 | 73.85 | 0.89× | 0.85× | highs_portfolio |
| czprob.mps | 929 × 3523 | 16.87 | 13.82 | 18.52 | 0.91× | 0.75× | highs_portfolio |
| d2q06c.mps | 2171 × 5167 | 357.48 | 347.18 | 242.64 | 1.47× | 1.43× | highs_portfolio |
| d6cube.mps | 415 × 6184 | 62.88 | 56.13 | 58.74 | 1.07× | 0.96× | highs_direct |
| degen2.mps | 444 × 534 | 8.90 | 7.51 | 8.04 | 1.11× | 0.93× | highs_direct |
| degen3.mps | 1503 × 1818 | 81.73 | 85.80 | 96.22 | 0.85× | 0.89× | highs_portfolio |
| dfl001.mps | 6071 × 12230 | 2791.43 | 2706.03 | 793.64 | 3.52× | 3.41× | highs_portfolio |
| e226.mps | 223 × 282 | 4.62 | 3.66 | 3.95 | 1.17× | 0.93× | highs_direct |
| etamacro.mps | 400 × 688 | 7.75 | 5.92 | 6.46 | 1.20× | 0.92× | highs_direct |
| fffff800.mps | 524 × 854 | 6.59 | 5.16 | 5.45 | 1.21× | 0.95× | highs_direct |
| finnis.mps | 497 × 614 | 3.51 | 2.45 | 2.75 | 1.28× | 0.89× | highs_direct |
| fit1d.mps | 24 × 1026 | 7.27 | 5.27 | 5.61 | 1.30× | 0.94× | highs_direct |
| fit1p.mps | 627 × 1677 | 21.85 | 20.21 | 20.75 | 1.05× | 0.97× | highs_direct |
| fit2d.mps | 25 × 10500 | 80.63 | 72.01 | 74.22 | 1.09× | 0.97× | highs_direct |
| fit2p.mps | 3000 × 13525 | 497.00 | 463.54 | 204.93 | 2.43× | 2.26× | highs_portfolio |
| forplan.mps | 162 × 421 | 4.10 | 2.99 | 3.31 | 1.24× | 0.90× | highs_direct |
| ganges.mps | 1309 × 1681 | 8.28 | 6.13 | 6.71 | 1.23× | 0.91× | highs_direct |
| gfrd-pnc.mps | 616 × 1092 | 4.42 | 3.03 | 3.63 | 1.22× | 0.83× | highs_direct |
| greenbea.mps | 2392 × 5405 | 124.29 | 115.12 | 130.47 | 0.95× | 0.88× | highs_portfolio |
| greenbeb.mps | 2392 × 5405 | 228.30 | 223.42 | 145.87 | 1.57× | 1.53× | highs_portfolio |
| grow15.mps | 300 × 645 | 17.26 | 15.97 | 16.44 | 1.05× | 0.97× | highs_direct |
| grow22.mps | 440 × 946 | 46.72 | 44.84 | 44.87 | 1.04× | 1.00× | highs_direct |
| grow7.mps | 140 × 301 | 5.19 | 4.18 | 4.47 | 1.16× | 0.94× | highs_direct |
| israel.mps | 174 × 142 | 2.24 | 1.57 | 1.71 | 1.31× | 0.92× | highs_direct |
| kb2.mps | 43 × 41 | 0.93 | 0.34 | 0.39 | 2.38× | 0.87× | highs_direct |
| lotfi.mps | 153 × 308 | 1.75 | 1.08 | 1.19 | 1.47× | 0.91× | highs_direct |
| maros-r7.mps | 3136 × 9408 | 299.80 | 288.06 | 258.32 | 1.16× | 1.12× | highs_portfolio |
| maros.mps | 846 × 1443 | 23.71 | 20.62 | 21.85 | 1.09× | 0.94× | highs_direct |
| modszk1.mps | 687 × 1620 | 9.39 | 7.34 | 8.40 | 1.12× | 0.87× | highs_direct |
| nesm.mps | 750 × 2923 | 99.45 | 97.05 | 64.37 | 1.54× | 1.51× | highs_portfolio |
| perold.mps | 625 × 1376 | 31.69 | 29.86 | 30.62 | 1.03× | 0.98× | highs_direct |
| pilot.mps | 1441 × 3652 | 550.63 | 600.17 | 349.29 | 1.58× | 1.72× | highs_portfolio |
| pilot4.mps | 410 × 1000 | 19.27 | 17.56 | 17.63 | 1.09× | 1.00× | highs_direct |
| pilot87.mps | 2030 × 4883 | 1791.57 | 1790.55 | 854.41 | 2.10× | 2.10× | highs_portfolio |
| pilotnov.mps | 975 × 2172 | 74.24 | 71.52 | 81.36 | 0.91× | 0.88× | highs_portfolio |
| recipe.mps | 91 × 180 | 1.13 | 0.47 | 0.58 | 1.95× | 0.81× | highs_direct |
| sc105.mps | 105 × 103 | 0.96 | 0.42 | 0.50 | 1.92× | 0.84× | highs_direct |
| sc205.mps | 205 × 203 | 1.52 | 0.96 | 1.03 | 1.48× | 0.93× | highs_direct |
| sc50a.mps | 50 × 48 | 0.81 | 0.40 | 0.34 | 2.38× | 1.18× | highs_direct |
| sc50b.mps | 50 × 48 | 0.72 | 0.23 | 0.30 | 2.40× | 0.77× | highs_direct |
| scagr25.mps | 471 × 500 | 4.62 | 3.56 | 3.85 | 1.20× | 0.92× | highs_direct |
| scagr7.mps | 129 × 140 | 1.35 | 0.77 | 0.82 | 1.65× | 0.94× | highs_direct |
| scfxm1.mps | 330 × 457 | 4.59 | 3.57 | 3.71 | 1.24× | 0.96× | highs_direct |
| scfxm2.mps | 660 × 914 | 10.18 | 8.77 | 9.30 | 1.09× | 0.94× | highs_direct |
| scfxm3.mps | 990 × 1371 | 17.59 | 15.13 | 15.91 | 1.11× | 0.95× | highs_direct |
| scorpion.mps | 388 × 358 | 2.13 | 1.29 | 1.58 | 1.35× | 0.82× | highs_direct |
| scrs8.mps | 490 × 1169 | 5.92 | 4.44 | 5.09 | 1.16× | 0.87× | highs_direct |
| scsd1.mps | 77 × 760 | 2.53 | 1.30 | 1.68 | 1.51× | 0.77× | highs_direct |
| scsd6.mps | 147 × 1350 | 5.23 | 4.00 | 4.41 | 1.19× | 0.91× | highs_direct |
| scsd8.mps | 397 × 2750 | 24.58 | 21.41 | 22.36 | 1.10× | 0.96× | highs_direct |
| sctap1.mps | 300 × 480 | 3.89 | 2.72 | 2.92 | 1.33× | 0.93× | highs_direct |
| sctap2.mps | 1090 × 1880 | 7.77 | 5.79 | 6.59 | 1.18× | 0.88× | highs_direct |
| sctap3.mps | 1480 × 2480 | 9.58 | 7.23 | 7.98 | 1.20× | 0.91× | highs_direct |
| seba.mps | 522 × 1028 | 3.11 | 1.73 | 2.18 | 1.43× | 0.79× | highs_direct |
| share1b.mps | 117 × 225 | 2.33 | 1.47 | 1.58 | 1.47× | 0.93× | highs_direct |
| share2b.mps | 96 × 79 | 1.64 | 0.95 | 1.00 | 1.64× | 0.95× | highs_direct |
| shell.mps | 536 × 1775 | 7.94 | 6.24 | 6.94 | 1.14× | 0.90× | highs_direct |
| ship04l.mps | 402 × 2118 | 5.65 | 3.60 | 4.19 | 1.35× | 0.86× | highs_direct |
| ship04s.mps | 402 × 1458 | 4.24 | 2.54 | 3.07 | 1.38× | 0.83× | highs_direct |
| ship08l.mps | 778 × 4283 | 11.22 | 8.11 | 11.35 | 0.99× | 0.71× | highs_portfolio |
| ship08s.mps | 778 × 2387 | 6.17 | 4.08 | 5.05 | 1.22× | 0.81× | highs_direct |
| ship12l.mps | 1151 × 5427 | 15.51 | 11.35 | 15.26 | 1.02× | 0.74× | highs_portfolio |
| ship12s.mps | 1151 × 2763 | 7.42 | 4.89 | 5.60 | 1.33× | 0.87× | highs_direct |
| sierra.mps | 1227 × 2036 | 8.10 | 5.99 | 6.57 | 1.23× | 0.91× | highs_direct |
| stair.mps | 356 × 467 | 7.95 | 6.71 | 7.00 | 1.14× | 0.96× | highs_direct |
| standata.mps | 359 × 1075 | 3.13 | 1.78 | 2.23 | 1.40× | 0.80× | highs_direct |
| standgub.mps | 361 × 1184 | 3.13 | 1.81 | 2.20 | 1.42× | 0.82× | highs_direct |
| standmps.mps | 467 × 1075 | 3.57 | 2.13 | 2.52 | 1.42× | 0.85× | highs_direct |
| stocfor1.mps | 117 × 111 | 1.22 | 0.58 | 0.71 | 1.72× | 0.82× | highs_direct |
| stocfor2.mps | 2157 × 2031 | 16.19 | 13.47 | 14.66 | 1.10× | 0.92× | highs_direct |
| tuff.mps | 333 × 587 | 4.96 | 3.73 | 4.02 | 1.23× | 0.93× | highs_direct |
| vtpbase.mps | 198 × 203 | 1.37 | 0.66 | 0.74 | 1.85× | 0.89× | highs_direct |
| wood1p.mps | 244 × 2594 | 42.93 | 36.58 | 37.72 | 1.14× | 0.97× | highs_direct |
| woodw.mps | 1098 × 8405 | 40.50 | 33.51 | 39.81 | 1.02× | 0.84× | highs_portfolio |
| woodinfe.mps | 35 × 89 | 0.43 | 0.08 | 0.54 | 0.80× | 0.15× | SciPy |
