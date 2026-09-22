# highs_turbo stabilization plan

Prepared 2026-09-22. Estimated effort: 5–7 working days, depending on defects
found during validation. This is a release plan, not a promise of a release date.

## Outcome and scope

Deliver a reproducible release candidate for the existing public API:
`linprog`, `solve_ising`, `solve_qubo`, `solve_maxcut`, and their certificate
verification functions. Installation, result semantics, resource limits and
fallback behavior must match the documentation.

Keep the current default algorithms. Cubic certificates, additional sparse
search passes and learned solver selection remain experiments. System1
integration is outside this stabilization phase. Prefer small fixes and reuse
the existing test, benchmark and packaging tools.

Correctness and reproducibility are release gates. A new speedup is not required
to stabilize the package; comparative performance determines which claims and
use cases the release can support.

## Starting evidence

- Source baseline: `35b54e6863dfdd8e17c6a76e9b0569b10a14181f`, on
  `codex/higher-order-ising`; working tree was clean before this plan was added.
- Draft PR #5 is open and mergeable. Its latest checks passed for macOS native,
  Linux/Python 3.10 fallback and Windows/Python 3.13 fallback.
- Local collection found 482 tests. Collection is not a new passing test run.
- Declared package version is 0.1.0; no GitHub releases were returned by the
  release listing at the time of inspection.
- Existing `dist/` files are older than current packaging metadata. They are
  not release candidates; rebuild from the final source revision.

## Phase 1 — Establish the baseline and public contract (0.5–1 day)

- Run the complete existing suite and record environment versions, passes,
  failures and skips. Reproduce failures before changing implementation.
- Write one API contract table covering numerical optimality, exact optimality,
  valid positive-gap bounds, timeout/interruption, solver failure and empty
  inputs. Explain the legacy certification flags without silently changing them.
- Specify which fields serialize the full certificate and how to verify it
  independently. Preserve documented tuple/result compatibility.
- Correct the README's conflicting binary-product limits: one section says
  2,047 base variables, another says 8,191. Check the intended boundary against
  the implementation and tests.

Exit: a reproducible baseline and an agreed, internally consistent API contract.

## Phase 2 — Fix demonstrated reliability gaps (1.5–2 days)

- Exercise timeout and fallback paths with deterministic failure injection:
  acceleration that consumes budget before failing, rejected factor proposals,
  resource-limit fallback and return of the strongest already checked bound.
  Keep cooperative deadline behavior explicit; do not claim hard real-time limits.
- Cover the binary-product bridge's status mapping and original-model checks:
  infeasibility, positive relative gap, exhausted budget, unsupported options,
  duplicate sparse entries and routing size boundaries. Use small fixtures or
  routing-only checks where a full boundary-sized solve is unnecessary.
- Address the eager dense conversion of sparse/dictionary QUBO inputs. Prefer
  preserving sparse coefficients through the existing path. If a conversion
  remains necessary, establish and document its memory limit before allocating.
- Add focused tests only for uncovered contracts and reproduced defects. Reuse
  existing exact enumeration, malformed-certificate and worker-cleanup tests.
- Expose enough benchmark diagnostics to distinguish an accepted improvement
  from a rejected proposal and a valid fallback; avoid a new telemetry subsystem.

Exit: no unresolved incorrect results, invalid accepted certificates, broken
documented status contracts or uncontrolled allocation in the supported paths.

## Phase 3 — Verify installation and compatibility (1 day)

- Build the source distribution and wheel from the frozen candidate into a new,
  empty output directory. Build the wheel from that source distribution.
- Check that documented distribution instructions are executable: include the
  examples/smoke files they require, or clearly identify checkout-only commands.
- Install into clean environments outside the checkout. Exercise all public
  solver entry points and standalone certificate verification from the artifact.
- Retain macOS native and Linux/Windows fallback checks. Verify the supported
  minimum dependency combination, or raise declared minimum versions to the
  versions actually supported by evidence. Record exact tested combinations.
- Force an optional-extension build failure on macOS and verify fallback works.
  Verify native-wheel loading with the documented GMP prerequisites; same-machine
  installation alone does not establish portability to another Mac.
- Check representative previously supported certificate versions, policy data,
  optional neural weights and operation without optional ML/native components.

Exit: the built artifacts, dependency declarations and installation documentation
agree, with passing checks on the supported platform matrix.

## Phase 4 — Check performance and resource regressions (1–2 days)

- Freeze a compact corpus before changing performance code: representative LPs,
  small exhaustively checkable binary models, dense/sparse cases, and previously
  unused Pegasus/Gset cases. Label reused regression cases and fresh cases.
- Compare the candidate with the initial source baseline. Use three repetitions,
  fresh sequential processes, rotating order and controlled thread settings.
- For LP positioning, compare native simplex and interior-point methods as well
  as the existing portfolio, recording its extra CPU and memory costs.
- For certification positioning, compare basic SDP and published Mixing with
  the same exact repair where available. Keep numerical and checked bounds
  separate. Record methods that do not support equal wall-time stopping.
- Record full API elapsed time, independent verification time, actual overruns,
  peak memory, feasible energy, exact bound, exact gap and fallback reasons.
  A fixed-budget run that misses a target does not establish time-to-target.
- Investigate any repeated runtime regression above 20% on nontrivial cases,
  any unexplained memory growth, or any weaker checked bound. A threshold
  crossing requires explanation and disposition, not automatic benchmark tuning.
- Add one runnable example that checks an external heuristic's feasible answer
  against an exported certificate in a separate process, using existing APIs.

Exit: no unexplained material regression, every saved proof verifies, and the
release's performance claims are supported by the measured workloads.

## Phase 5 — Prepare and verify the release (0.5–1 day)

- Freeze changes; select the next version after checking package release history.
- Prepare release notes with fixes, supported environments, known limitations,
  certificate/result semantics and benchmark reproduction commands.
- Run required tests and artifact checks against the final candidate revision.
  Recheck affected gates after any subsequent source or packaging change.
- Bring PR #5's title and description into line with its final scope. Release
  completion requires merging the reviewed change, tagging the tested revision
  and publishing only artifacts built from that revision through the chosen
  release process. This plan does not itself perform those external actions.
- Retain the previous revision and artifacts as the rollback reference.

Exit: one traceable release whose source, tests, documentation and distribution
artifacts match. Describe its demonstrated scope rather than declaring universal
solver superiority or unrestricted production readiness.

## Completion criteria

- Required tests pass; skips and platform exclusions are accounted for.
- Small exact-oracle comparisons agree; numerical LP results satisfy their
  documented tolerances; serialized certificates pass independent verification.
- Timeouts, errors and resource limits produce documented results or exceptions.
- Clean installations work across the declared support matrix.
- No unresolved release-blocking defect or unexplained material regression.
- Release notes disclose cooperative time limits, memory limits and heuristic
  versus exact result semantics.

Missing application data or an external specialist baseline limits market and
performance conclusions. It does not justify inventing evidence, nor does it
prevent a correctly scoped maintenance release. Numerical wrong answers,
invalid accepted certificates and broken installations block release.
