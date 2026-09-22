# Public solver and certificate contract

This contract applies to highs_turbo 0.2.0. Existing tuple unpacking, result
fields and legacy certification flags retain their meanings.

## Results and proof

| Entry point | Numerical result | Independently checked evidence |
| --- | --- | --- |
| Continuous `linprog` | SciPy-style `OptimizeResult`; success means the model passed the applicable numerical tolerances. | LP feasibility and dual checks use floating point, not a rational optimality certificate. |
| Recognized binary-product `linprog` | SciPy-style MIP termination, including the requested relative-gap tolerance. Every returned candidate is checked against the original rows and bounds. | `turbo_bound_certificate` and `turbo_exact_lower_bound` certify a lower bound. A successful MIP termination alone does not prove exact optimality. |
| `solve_ising` | `status`, `lower_bound`, `gap` and `relative_gap` may incorporate HiGHS' numerical conclusions. | `certificate`, `exact_energy`, `exact_cut_lower_bound` and `exact_gap` describe the checked rational result. `is_rationally_certified` means the exact gap is zero. |
| `solve_qubo` | `energy`, binary `solution`, and `OPTIMAL` or `HEURISTIC` status. `OPTIMAL` includes numerical solver termination. | `exact_rational_bound` is a checked lower bound; `bound_certificate` is its Ising witness. The legacy `is_rationally_certified` flag means a valid bound, including when its gap is positive. |
| `solve_maxcut` | `cut_value`, binary `partition`, and `OPTIMAL` or `HEURISTIC` status. `OPTIMAL` includes numerical solver termination. | `exact_rational_bound` is a checked upper bound; `bound_certificate` is the converted Ising witness. The legacy certification flag means a valid bound, not necessarily an exact optimum. |

For Ising, an exact zero gap establishes optimality. For QUBO and Max-Cut,
compare the bound with the candidate objective recomputed from original
coefficients using exact rational arithmetic. Floating-point equality alone
does not establish an exact gap of zero.

Finite floating-point objective coefficients are interpreted as their exact
binary64 values, not as intended decimal fractions. Ising also accepts
`fractions.Fraction` coefficients directly. Bounds certify the supplied
mathematical model; they do not validate whether that model describes the
application correctly.

## Serialization and verification

Use `result.certificate.to_dict()` for an Ising witness. Save the dictionary as
JSON, then call `verify_ising_certificate(h, J, witness, offset=offset)` against
the original objective in a new process. This verifier does not run a solver.
Its result certifies the bound; separately check the candidate's spin values
and energy. The external-answer example demonstrates the complete workflow.

For the binary-product bridge, use
`result.turbo_bound_certificate.to_dict()` and
`verify_linprog_certificate(c, witness, A_ub=..., b_ub=..., bounds=...,
integrality=...)`. Additional constraints restrict feasible answers and do not
invalidate the objective lower bound, but the caller must still check its
candidate against those constraints.

QUBO and Max-Cut retain three-element unpacking. Their string `certificate`
field is a SHA-256 digest, not the full proof. Their `to_dict()` method is a
legacy summary and intentionally omits the full witness; serialize
`bound_certificate.to_dict()` separately and verify it against the exact Ising
conversion of the original objective. A digest alone is not proof of validity.
An edgeless Max-Cut problem has an exact zero bound, no witness, and the legacy
all-zero digest; its objective is identically zero.

Previously supported certificate formats remain readable. Malformed,
input-mismatched or resource-limit-exceeding certificates are rejected.
Within-solve reuse never replaces a fresh standalone verification.

## Termination and fallback

Ising statuses `TIME_LIMIT`, `INTERRUPTED` and `GAP_LIMIT` retain the best
available feasible spins and a checked bound. If the final native solve fails
and no exact optimum has already been established, `SOLVER_ERROR` retains the
prior feasible candidate and checked bound. Invalid inputs, model-setup errors
and preliminary relaxation failures can raise an exception. A failed final
certificate check raises rather than returning an unverified claim.

The QUBO/Max-Cut wrappers map unfinished Ising results to `HEURISTIC`; consult
the objective and bound to assess their quality. A valid positive gap is useful
evidence, but it is not a proof that the candidate is optimal.

`time_limit` is cooperative. Preparation, native calls, factorization and exact
verification can exceed the requested time. Failed or declined acceleration
does not restart an explicit `linprog` time budget. LP portfolio workers are
cancelled and joined before returning. No hard real-time guarantee is made.

Rejected proof proposals retain the strongest existing checked witness.
`IsingResult.certificate_fallbacks` records the stage and reason for caught
proof-construction or deadline failures; a fallback is not an invalid result.
`linprog` exposes `fallback_reason` after an accelerated exception when
`TurboSolver(fallback_on_error=True)` delegates to SciPy. Set that option to
`False` to propagate accelerated exceptions while diagnosing a failure.

## Input size and resources

The binary-product bridge recognizes complete exact envelopes with 32–8,191
binary base variables. Unsupported models/options retain SciPy execution.
Noncanonical sparse envelope matrices are declined before exact recognition;
canonicalize them deliberately if their rounded sum is the intended input.

Global Ising proofs support up to 8,192 vertices including any reference spin.
The dense path is limited to 2,048 vertices. Larger supported sparse factors
have explicit work and storage limits; unsuitable instances retain cut bounds.
These are proof-strategy limits, not a guarantee that every input of that size
fits a particular machine or solves quickly.

Sparse and dictionary QUBO inputs remain sparse during conversion. Stored
duplicates and opposite orientations are summed exactly without floating-point
canonicalization. Dictionary keys must be pairs of nonnegative integer indices;
the largest index plus one determines the number of variables. Memory still
scales with the variable count, nonzero coefficients and the solver's working
model. Dense input naturally requires quadratic storage.

LP method competition may use two model copies and two CPU cores. Optional
native graph operations require macOS and GMP; the Python/HiGHS fallback is
supported on macOS, Linux and Windows. Ordinary solver calls do not require
PyTorch or a trained model.
