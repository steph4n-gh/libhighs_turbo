# Stabilization evidence for 0.2.0

This report is being completed against the release candidate. Pending checks
below are not passing results. The source baseline is
`35b54e6863dfdd8e17c6a76e9b0569b10a14181f`.

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
  macOS/Python 3.13 environment. Final cross-platform CI results are pending.

## Compatibility and distributions

A clean CPython 3.10.17 environment successfully installed NumPy 2.0.0,
SciPy 1.13.0, highspy 1.11.0, NetworkX 3.2 and dwave-graphs 1.0.0. The initial
82-test compatibility selection passed in 3.90 seconds. This macOS/NumPy 2.0
run emitted numerical RuntimeWarnings; the assertions passed, but the warning
cause has not been established. The same exact dependency set is now covered
by Linux CI.

The former SciPy 1.9 minimum cannot coexist with mandatory dwave-graphs 1.0's
NumPy 2 requirement. Package metadata now declares the tested compatible floors.

Fresh sdist-to-wheel builds, installed-artifact checks, and native/fallback
cross-platform CI are pending. Existing files in `dist/` are not release inputs.

## Frozen regression experiment

[The protocol](benchmarks/stabilization_protocol.json) and compressed original
inputs were prepared before measurements. It covers LP, exhaustive small Ising,
dense/sparse binary models, fresh Pegasus seeds and Stanford G57. The benchmark
uses three repetitions in fresh sequential processes and checks each binary
certificate in a separate process. Native LP simplex and IPM use the same
highspy library as the public solver.

Measurements and their disposition are pending. The previously documented
published Mixing executable is unavailable locally, so this round does not
renew an external specialist superiority claim. Public application/customer
evidence is outside this maintenance experiment.

## Release status

Release preparation is in progress. No release is authorized by a passing unit
test alone: final source, artifact and platform gates must all complete.
