# Stabilization evidence for 0.2.0

This report records the stabilization implementation and measured release
evidence. The source baseline is `35b54e6863dfdd8e17c6a76e9b0569b10a14181f`.
The release additionally requires a passing distribution workflow on its tagged
source revision; the GitHub release notes link that run and its exact artifacts.

## Correctness

- Before changes: 482 tests passed in 35.35 seconds on macOS arm64,
  Python 3.13.5, NumPy 2.5.3, SciPy 1.18.1 and highspy 1.15.1.
- Focused wrapper validation: 66 tests passed, including exact sparse duplicate
  sums, dense/sparse/dictionary equivalence, memory behavior and caller budgets.
- Focused Ising/bridge/sparse validation: 74 tests passed, including certificate
  fallback, resource rejection, original-model validation and native failure.
- Two external-answer example tests passed, including separate-process proof
  checking, changed-model rejection and exact rational input.
- Complete candidate suite: 542 tests passed in 38.22 seconds on the same
  macOS/Python 3.13 environment. The first native CI run also passed all 542 tests
  in 90.88 seconds. Its installed smoke check caught the test-order issue below.

## Compatibility and distributions

A clean CPython 3.10.17 environment successfully installed NumPy 2.0.0,
SciPy 1.13.0, highspy 1.11.0, NetworkX 3.2 and dwave-graphs 1.0.0. The initial
82-test compatibility selection passed in 3.90 seconds. This macOS/NumPy 2.0
run emitted numerical RuntimeWarnings; the assertions passed, but the warning
cause has not been established. Linux CI subsequently passed the expanded
140-test solver selection and installed-wheel smoke at those exact minimums.
Windows and macOS forced-compilation-failure CI also passed that 140-test
selection and installed smoke.

The former SciPy 1.9 minimum cannot coexist with mandatory dwave-graphs 1.0's
NumPy 2 requirement. Package metadata now declares the tested compatible floors.

The optional ML minimum is PyTorch 2.4.1. Earlier PyTorch/NumPy combinations do
not reliably support the required `from_numpy`/`numpy` interoperability across
platforms. A clean Python 3.10 / NumPy 2.0 / PyTorch 2.4.1 environment passed that
roundtrip, all four public solvers' lazy-import checks, and the three existing
surrogate-model tests. Linux CI enforces this optional minimum with CPU wheels.

The first native installed-smoke run detected a defect in the new smoke check:
the script explicitly imported a research helper that loads optional PyTorch
before asserting ordinary solves do not import it. Research-helper checks now
run after the ordinary-solver assertion. Both checks remain enabled. This
correction changes the test, not the production import path.

The distribution workflow builds fresh sdists and wheels from those sdists,
installs outside the checkout, verifies resources and package versions, and
checks public solvers plus independent certificate verification. It uploads
artifacts only after these checks pass. Existing local `dist/` files are not
release inputs. Native graph operations require the documented macOS/GMP
prerequisites; the portable wheel uses the Python/HiGHS fallback.

## Frozen regression experiment

[The protocol](benchmarks/stabilization_protocol.json) and compressed original
inputs were prepared before measurements. It covers LP, exhaustive small Ising,
dense/sparse binary models, fresh Pegasus seeds and Stanford G57. The benchmark
uses three repetitions in fresh sequential processes and checks each binary
certificate in a separate process. Native LP simplex and IPM use the same
highspy library as the public solver.

All 96 observations completed: 36 LP solves and 60 binary solves. Every saved
binary proof passed a separate-process verification. LP objectives agreed
within 3.15e-11, and both small exhaustive-oracle cases agreed with returned
optima. No median API runtime regressed by more than 20%; the largest observed
median increase was about 1.9%.

The two five-second Pegasus comparisons returned slightly weaker median checked
bounds: 0.25 and 0.5 energy units. Their baseline/candidate ranges overlap. The
median exact gaps increased by approximately 0.014% and 0.419%, respectively;
one damaged-Pegasus run also had a weaker incumbent. These observations are
retained, and no algorithm or budget was tuned after measurement. They support
retaining the correctness fixes with explicit timing-sensitive quality limits.

Two upward peak-RSS median shifts also had overlapping run ranges; the large
Pegasus median decreased. This process-level experiment did not demonstrate a
sustained material growth pattern, and it is not a proof of leak freedom.

The largest budget overrun was 1.988 seconds for the basic SDP control under a
five-second request. The default candidate's largest observed overrun was
0.273 seconds. Applications requiring a hard deadline need process supervision.

See [the full report](benchmarks/stabilization_report.md) for every median,
range, resource tradeoff and regression disposition, and
[raw observations](benchmarks/stabilization_results.json) for all repetitions
and source/proof hashes. Sixty verified proof bundles are provided as the
release asset `stabilization-proofs.tar.gz`, SHA-256
`491b0158fc299019484b740c7da557839f18ede31c66b484c85ca9940344852a`.

The previously documented published Mixing executable is unavailable locally,
so this round does not renew an external specialist superiority claim. Public
application/customer evidence is outside this maintenance experiment.

## Release gate

The final [distribution workflow](.github/workflows/tests.yml) must pass native
macOS, fallback macOS/Windows, and Linux minimum-runtime/minimum-ML checks at the
release revision. Only its tested source distribution and portable wheel are
selected for release; the proof archive is checked against the measured hashes.
The release notes and checksums provide the final artifact identities.
