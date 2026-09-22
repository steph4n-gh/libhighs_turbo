# Changelog

## 0.2.0 — 2026-09-22

This release integrates the checked geometric and sparse Ising bounds into the
public solvers and stabilizes their input, fallback and distribution contracts.

- Automatic Ising, QUBO, Max-Cut and recognized binary-product models use checked
  global/geometric bounds where supported, with sparse proofs through 8,192
  vertices including the reference spin. Exact sparse Gram calculations are
  reused within one solve; standalone verification remains independent.
- Sparse and dictionary QUBO inputs no longer allocate a dense matrix during
  conversion. Duplicate stored coefficients are added exactly. Invalid
  dictionary indices and nonfinite/nonreal coefficients raise `ValueError`.
- Proof-construction failures and deadline interruptions retain the strongest
  available checked certificate. Ising results expose `certificate_fallbacks`
  with the stage and reason for caught proof failures.
- Binary-product preparation and ordinary SciPy fallback retain only the
  remaining caller time budget. Deadline enforcement remains cooperative.
- Binary-product results are checked against the complete original model,
  including native solver candidates. Final native Ising failures preserve the
  prior feasible answer and checked bound instead of losing the partial result.
- Public documentation distinguishes numerical optimality, exact optimality,
  valid positive-gap bounds and receipt digests. Legacy tuple unpacking and
  certification-flag semantics are preserved.
- Source distributions contain documented examples, certificate fixtures and
  installed-package smoke checks. CI checks native macOS, Linux minimum
  dependencies, Windows fallback and macOS optional-compilation failure.
- CI retains every native benchmark trial as a test-report artifact. The older
  host-dependent 2x timing target is recorded rather than asserted; soundness,
  objective agreement and fixture iteration reduction are checked on every trial.
- Dependency minimums are NumPy 2.0, SciPy 1.13, NetworkX 3.2, highspy 1.11 and
  dwave-graphs 1.0, with Python 3.10 or newer. The earlier SciPy 1.9 claim was
  incompatible with the NumPy requirement of the mandatory graph dependency.
  Optional ML/test extras require PyTorch 2.4.1 or newer for NumPy 2
  interoperability; the minimum CPU configuration is exercised in CI.

See [the API contract](API_CONTRACT.md) for result and resource semantics and
[stabilization evidence](STABILIZATION_RESULTS.md) for exact environments,
test outcomes and regression measurements.

### Scope and known limits

Native graph operations require macOS and GMP. Portable Python/HiGHS operation
does not require the optional extension or PyTorch. Global factor limits can
select a weaker cut relaxation. Factorization and independent verification can
overrun a requested short time budget. Neither a numeric solver success nor a
receipt digest alone establishes exact optimality.

Performance depends on problem structure, solver configuration and hardware.
The release does not establish universal solver superiority, scientific novelty
or suitability for hard real-time operation. Higher-order cubic experiments,
additional sparse search passes and learned policy improvements remain research.

## 0.1.0

Initial development version: SciPy-compatible LP acceleration, graph/QUBO/Ising
solvers, optional native cut operations, exact certificate verification and
benchmark/research harnesses.
