# Stabilization regression observations — 2026-09-22

All 96 frozen observations completed: 36 LP solves and 60 binary solves. Every saved binary certificate passed the current verifier in a fresh process. All LP solutions satisfied their original constraints, and objectives agreed across methods and repetitions within 3.15e-11 (the gate was 1e-7). Both small exhaustive oracles agreed with returned feasible optima. No median API runtime regressed by more than 20% against the starting source revision.

Two short-budget Pegasus comparisons returned slightly weaker median checked bounds; their ranges, observed search variation, and disposition are recorded below. These measurements support a maintenance release with disclosed cooperative timing behavior. They do not establish universal speedups, deterministic quality at a wall-clock deadline, or an application/customer advantage.

## Frozen inputs and execution

- Baseline: `35b54e6863dfdd8e17c6a76e9b0569b10a14181f`; candidate production Python hashes and matching native-extension hashes are recorded in [raw observations](stabilization_results.json). Both revisions used the same interpreter, dependencies, and native extension.
- Run: `2026-09-22T14:44:54.378057+00:00` through `2026-09-22T14:49:36.295295+00:00`; platform `macOS-27.0-arm64-arm-64bit-Mach-O`, Python `3.13.5`.
- Versions: highspy 1.15.1, numpy 2.5.3, scipy 1.18.1, networkx 3.6.1, dwave-graphs 1.0.0, dwave-samplers 1.8.0.
- [Protocol](stabilization_protocol.json) and [compressed corpus](stabilization_corpus.json.gz) were frozen before measured runs. The ten cases are fresh seeds/models in familiar regression families. G57 is the canonical Stanford download, absent from the repository's previous benchmark reports; its source-byte SHA-256 is `1206f13e1b2a1034685abe9a25fcecc85b9d21c21bfc4de876246928b7012d66`.
- Three repetitions use fresh sequential processes with rotating order. No local tests/builds ran concurrently. BLAS thread environment variables were set to one; Ising calls use `threads=1`; native LP controls use `parallel=off`. The LP portfolio can run two competing native methods, so it can consume more than one CPU at once.
- API time includes complete solver work, its internal proof check, and cleanup. Imports and caller input construction are excluded. Independent verification times measure the public verifier in a second process, after imports and saved JSON decoding. Process peak RSS includes imports and input materialization. CPU time covers work during the API interval. Solver-worker wall time, compressed proof sizes, actual overruns, numerical reported bounds/gaps, exact bounds/gaps, and available fallback diagnostics remain in the raw data.
- LPs run without solver deadlines because explicit LP time limits route the wrapper to SciPy. A 180-second process watchdog protects the run. Native simplex and IPM controls use the same highspy-linked HiGHS as the portfolio; no bundled-SciPy version difference is being labeled an algorithm speedup.
- Binary requested budgets are 2–5 seconds. Basic SDP is `relaxation="sdp"` through the current public API. The previously documented published Mixing executable was absent, so no specialist published-solver positioning claim is supported.
- Baseline and QUBO wrapper fallback telemetry is unavailable (`null`), not evidence that no fallback occurred. Current Ising observations reported no certificate fallback events.

## LP comparison

Times are medians of three complete calls, in seconds. LP results are numerical; they are not presented as rational certificates.

| Case | Baseline | Candidate | Native simplex | Native IPM | Candidate strategy | Objective spread |
|---|---:|---:|---:|---:|---|---:|
| lp-dense-92101 | 0.0079 | 0.0079 | 0.0075 | 0.0158 | highs_direct | 9.9e-14 |
| lp-sparse-92102 | 0.4348 | 0.4425 | 1.1744 | 0.4126 | highs_portfolio | 3.1e-11 |
| lp-redundant-92103 | 0.0257 | 0.0262 | 0.0258 | 0.0443 | row_recovery | 3.6e-14 |

The sparse 1,200-variable model exercises the actual portfolio. Its candidate median is 2.65× faster than native simplex, while native IPM alone is faster than the portfolio. The portfolio consumed 0.921 CPU-seconds and 152.8 MiB peak RSS versus IPM's 0.445 CPU-seconds and 134.9 MiB. This is a workload-dependent tradeoff, not a free speedup.

## Binary comparison

Lower bounds are for minimization; a larger (less negative) checked lower bound is stronger. Exact gaps are independently recomputed feasible energy minus checked bound. Each table entry is a median over the three corresponding observations, so the median gap need not equal the difference of separately computed median energy and median bound.

| Case | Baseline / candidate seconds | Baseline / candidate checked lower bound | Baseline / candidate exact gap | Basic SDP lower bound |
|---|---:|---:|---:|---:|
| ising-dense-oracle-92104 | 0.094 / 0.092 | -16 / -16 | 0 / 0 | -18 |
| ising-sparse-oracle-92105 | 0.125 / 0.119 | -13 / -13 | 0 / 0 | -27/2 |
| ising-dense-92106 | 3.009 / 3.009 | -1215/4 / -1215/4 | 97/4 / 97/4 | -321 |
| qubo-csr-92107 | 2.282 / 2.273 | -1244 / -1244 | 8 / 8 | not run for QUBO |
| pegasus-92108 | 5.253 / 5.269 | -89869/8 / -89871/8 | 3673/2 / 7347/4 | -90363/8 |
| pegasus-damaged-92109 | 5.104 / 5.123 | -8475/4 / -8477/4 | 179 / 719/4 | -4565/2 |
| G57 | 5.058 / 5.059 | -7572 / -7572 | 666 / 666 | -7818 |

All baseline/candidate median feasible energies matched. The dense 12-spin and sparse 14-spin oracles have optima -16 and -13; default baseline/candidate proofs closed both exact gaps. Basic SDP reached the same feasible optima but retained exact gaps 2 and 1/2, illustrating why a numerical OPTIMAL result is distinct from an exact zero-gap certificate.

| Case | Candidate fresh verification seconds | Baseline / candidate median RSS MiB | Candidate maximum overrun seconds |
|---|---:|---:|---:|
| ising-dense-oracle-92104 | 0.00056 | 119.8 / 118.1 | 0.000 |
| ising-sparse-oracle-92105 | 0.00105 | 120.1 / 119.2 | 0.000 |
| ising-dense-92106 | 0.00690 | 191.5 / 204.6 | 0.009 |
| qubo-csr-92107 | 0.05966 | 248.6 / 244.3 | 0.000 |
| pegasus-92108 | 0.80178 | 527.4 / 502.8 | 0.273 |
| pegasus-damaged-92109 | 0.11149 | 382.7 / 383.7 | 0.125 |
| G57 | 0.09789 | 241.7 / 254.3 | 0.059 |

The largest overrun in the complete comparison was 1.988 seconds for basic SDP on G57 (requested 5 seconds, measured 6.988 seconds). Its median was 6.944 seconds. The candidate default G57 median was 5.059 seconds. These are cooperative limits and include work needed to return a checked result; applications requiring a hard deadline must supervise a separate process.

## Regression disposition

No case crossed the frozen 20% median runtime regression gate. The largest median increase was 1.9% on the small redundant LP; on the nontrivial sparse LP it was 1.8%. No measured runtime result supports a new general acceleration claim.

| Timed quality comparison | Baseline bound range | Candidate bound range | Median bound change | Median exact-gap change |
|---|---:|---:|---:|---:|
| pegasus-92108 | -89873/8 to -89615/8 | -89871/8 to -89869/8 | -1/4 (weaker) | +1/4, +0.0136% |
| pegasus-damaged-92109 | -8477/4 to -8473/4 | -4239/2 to -4237/2 | -1/2 (weaker) | +3/4, +0.4190% |

These are retained quality regressions at the requested five-second budget, not rounded away or replaced by favorable reruns. They are small relative to the existing positive gaps and occur with overlapping baseline/candidate ranges. The unchanged baseline itself varied by 32.25 on P16 and 1 on damaged P8. P16 witnesses contained 931,065–936,319 sparse entries for baseline and 931,177–931,389 for candidate; both retained 2,048 cuts. Damaged P8 retained 5,571–5,717 certificate cuts for baseline and 5,599–5,684 for candidate. These observations establish that different deadline-limited iterates were returned. Added input-certificate checks can also consume part of that budget; attributing the exact shift to one cause would require a separate fixed-work experiment.

The damaged P8 candidate also returned a weaker feasible answer (-1,917 instead of -1,939.75) in one of three runs; basic SDP did the same in one run. Its median feasible energy was unchanged. The reported median gap reflects that individual-run variation. Every retained witness remains valid, and current independent verification accepted all baseline proofs. Disposition: retain the correctness fixes, disclose timing-sensitive quality, and do not promise per-instance fixed-budget bound monotonicity or deterministic incumbents. No algorithms or budgets were tuned after seeing these results.

RSS is the complete fresh-process high-water mark, not a retained-allocation or leak measurement. The two upward median shifts were 191.5→204.6 MiB for dense72 and 241.7→254.3 MiB for G57. Their run ranges overlap:

| Case | Baseline RSS range MiB | Candidate RSS range MiB |
|---|---:|---:|
| ising-dense-92106 | 190.5–192.5 | 190.6–208.4 |
| G57 | 235.9–286.7 | 244.3–289.8 |
| pegasus-92108 | 491.5–557.1 | 493.8–557.1 |

Dense72 retained the same proof structure in all runs (182 cuts and a 73-row dense factor), and its third candidate run used less memory than the baseline median. G57 varied across both versions; one candidate witness also retained 11 additional cuts and 1,131–1,136 additional factor entries. P16 median memory decreased. These process-level observations are consistent with allocator/search-path variation and do not show a sustained material memory-growth pattern; they do not constitute a proof of leak freedom. The sparse QUBO median was 248.6→244.3 MiB, with the same incumbent, bound, and exact gap. Its preservation of sparse input is additionally covered by routing/allocation regression tests rather than inferred from this modest end-to-end memory difference.

## Artifacts and reproduction

- [Raw observations](stabilization_results.json): every repetition, certificate hash, fallback field, exact result, resource measurement, source hash, and LP objective spread.
- [Frozen protocol](stabilization_protocol.json), [frozen corpus](stabilization_corpus.json.gz), and [runner](stabilization.py). Existing independent-verification and rational-repair APIs are reused; no new runtime dependency was added.
- Sixty verified, compressed proof bundles are archived in the [release proof asset](https://github.com/steph4n-gh/libhighs_turbo/releases/download/v0.2.0/stabilization-proofs.tar.gz). It contains relative filenames only, is 33,000,399 bytes, and has SHA-256 `491b0158fc299019484b740c7da557839f18ede31c66b484c85ca9940344852a`. Every archive entry's hash was compared with the verified observation. The local run retained originals in `/tmp/highs-stabilization-proofs/`; the proof payloads are not added to the source tree.
- [External-answer example](../examples/certify_external_answer.py) exports a complete model-bound certificate, then launches a separate process to check the supplied heuristic assignment and its exact gap. Its two tests cover portable verification, rejection of a changed model/invalid assignment, and exact rational string coefficients.

Run from the checkout using the interpreter whose dependencies you intend to compare. Archive the recorded baseline revision into a separate directory, and copy the identical optional native extension into that archive if the candidate uses it (or run both without it). The runner refuses differing native-extension hashes. The committed corpus can be reused without network access; `prepare` is only needed to regenerate inputs.

```bash
mkdir -p /tmp/libhighs-stabilization-baseline-35b54e6
git archive 35b54e6863dfdd8e17c6a76e9b0569b10a14181f | tar -x -C /tmp/libhighs-stabilization-baseline-35b54e6
python - <<'COPY_NATIVE'
from pathlib import Path
import shutil
for binary in Path("highs_turbo").glob("*.so"):
    shutil.copy2(binary, Path("/tmp/libhighs-stabilization-baseline-35b54e6/highs_turbo") / binary.name)
COPY_NATIVE
python benchmarks/stabilization.py run \
  --baseline /tmp/libhighs-stabilization-baseline-35b54e6 \
  --artifacts /tmp/highs-stabilization-proofs
python examples/certify_external_answer.py --output /tmp/answer-proof.json
python examples/certify_external_answer.py --verify /tmp/answer-proof.json
```

To check an extracted benchmark proof without solving, run `python benchmarks/stabilization.py verify --proof /path/to/CASE-METHOD-REPETITION.json.gz`. The committed corpus supplies the original model. Whole-run reproduction can change finite-time iterates; the published hashes identify the actual checked witnesses.
