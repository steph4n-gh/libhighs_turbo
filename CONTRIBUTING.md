# Contributing to highs_turbo

Small, focused fixes, reproducible bug reports, documentation improvements, and
benchmark comparisons are welcome. Start with the [README](README.md) and
[API contract](API_CONTRACT.md). Prefer reusing existing solver, verification,
and benchmark code over adding another framework or dependency.

## Reporting a problem

Use [GitHub Issues](https://github.com/steph4n-gh/libhighs_turbo/issues) for bugs
and proposed improvements. Include a minimal input or runnable example, expected
and actual behavior, package/revision and dependency versions, operating system,
and whether the optional native engine is available. For solver reports, include
the options, seed, time budget, and any relevant serialized certificate.

Use [private vulnerability reporting](https://github.com/steph4n-gh/libhighs_turbo/security/advisories/new)
for security findings that should not be posted in a public issue.

## Developing and checking a change

Work on a branch and keep changes limited to the demonstrated problem. A portable
environment can run the core solver regression tests without the optional ML
or native components:

```bash
python -m pip install -e . pytest
python -m pytest tests/test_solver_regressions.py tests/test_ising.py tests/test_ising_linearized.py tests/test_ising_sparse.py -q
```

See [development and testing](README.md#development-and-testing) for the full
macOS native suite, optional test dependencies, source distributions, and
installed-wheel checks. Run checks relevant to the change; documentation-only
changes do not need new solver tests.

Correctness fixes should include a small regression case. Preserve the caller's
original optimization model and the distinction between numerical success,
exact optimality, a valid bound, and a receipt digest. Proof generation must not
bypass independent verification. Preserve existing tuple and result contracts.

Performance comparisons should use identical inputs and resource budgets,
record solver/library versions, repeat measurements, and include preparation
and certificate-checking time. Report slower cases as well as improvements.
Existing benchmark scripts and [recorded results](STABILIZATION_RESULTS.md)
show the expected evidence.

## Pull requests

Open a pull request against `main` with the concrete problem, final behavior,
and relevant validation. Main-branch protection applies to administrators too:

- A pull request is required; direct pushes, force pushes, and branch deletion
  are blocked.
- The branch must be up to date, and review conversations must be resolved.
- All four CI configurations must pass: macOS native, Linux minimum dependencies,
  Windows fallback, and macOS forced-compilation-failure fallback.
- A separate approving review is optional; the other merge requirements
  still apply.

Update the API contract or changelog when a change affects documented behavior.
