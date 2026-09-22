# Application pilot kit

Use this kit to evaluate whether checking **answers you already have** is useful
for one concrete application. No customer data has been supplied. The included
three-case demo is synthetic and demonstrates the workflow only; it is not
customer evidence or a benchmark superiority claim.

The runner checks the mathematical objective encoded in each input and the
supplied assignment. Application feasibility, penalty modeling, data quality,
and business value require a separate review. Declare that review's status in
the manifest; the runner records the declaration without treating it as a proof.

## Run the synthetic demonstration

The scripts and fixtures ship in the [source distribution](https://github.com/steph4n-gh/libhighs_turbo/releases/download/v0.3.0/highs_turbo-0.3.0.tar.gz), not the wheel. Extract it or use a source checkout. From that directory, run these commands with the Python environment where you installed `highs_turbo`:

```bash
python pilots/run_pilot.py freeze pilots/example_manifest.json --output /tmp/highs-pilot-demo
python pilots/run_pilot.py run /tmp/highs-pilot-demo
```

`freeze` validates the manifest and input envelopes, copies the exact input
bytes, and records their SHA-256 hashes, the protocol hash, runtime versions and
runner hash **before any solve**. It also fingerprints the actual imported
package Python/JSON files and native libraries. A changed implementation blocks
a run, or fails an observation if it changes during a worker. Read `/tmp/highs-pilot-demo/protocol.json`
before running. The supplied files may subsequently change; the run uses their
frozen copies. Altering a frozen manifest, protocol, or input blocks execution.
Full coefficient and assignment validation happens in the certification API;
invalid models and answers are recorded as failed trials rather than silently
removed from the experiment. Missing or malformed input files are likewise
retained as input errors.

`run` performs three repetitions by default, in fresh sequential producer and
verification processes, rotating case order. Every observation is saved
immediately. It never overwrites an existing run. Create a new output directory
to repeat or revise an experiment.

The two optimal synthetic answers should meet the zero-gap target. The Max-Cut
example deliberately supplies an empty cut on a triangle and **misses** that
target: a valid certificate establishes a gap of 2. This demonstrates that a
verified bound and a successful quality target are different things. The demo
therefore normally exits with code **1**, while still producing its complete
report. Exit 0 means all trials passed their measured gates; exit 2 means the
frozen run was blocked before solving. Inspect `report.md` and `results.json`
instead of treating any certificate as evidence that the target was achieved.

Each successful producer saves a full portable bundle in `proofs/`; fresh
verification checks its captured byte hash and calls the package verifier.
`trials/` retains producer and verifier metadata, and `results.json` retains all
errors, timeouts and target misses. A producer artifact left after a timeout is
not labeled verified. Check any bundle later without solving:

```bash
python -m highs_turbo verify /tmp/highs-pilot-demo/proofs/qubo-optimal-r1.json
```

## Prepare a real pilot

Choose one application owner, one decision workflow, and approximately 12–16
representative instances. Include ordinary cases and the difficult cases the
application actually encounters. A smaller supplied set is allowed and clearly
flagged. Keep untouched cases separate from any data used to tune the model,
solver, or acceptance threshold.

For every case, obtain an existing feasible candidate from the application's
current process. Review that candidate in the **original application**, not
only its QUBO/Ising encoding. Define an absolute objective-gap target that would
change a decision, and a practical time/memory budget. Freeze these choices
before looking at solver results. The same target has different meaning across
objectives with different scales; do not compare raw gaps across applications.

Copy [the example manifest](example_manifest.json), set `evidence_type` to
`application_pilot`, identify the dataset and purpose in `name`/`description`,
and replace the synthetic input paths. Required declarations:

- `application_constraints.status`: `not_assessed`, `validated_separately`, or
  `not_applicable`.
- `application_constraints.notes`: who/what checked the original model and
  candidate, or why that check remains absent/not applicable. This is a manual
  declaration, not an automated validation.
- `cases`: unique simple `id` and `input` path relative to the manifest. Input
  paths may also be absolute. Keep private data and output directories outside
  version control unless sharing them is explicitly intended.

`defaults` applies to every case; each case can override the same settings:

| Setting | Meaning |
|---|---|
| `gap_target` | Nonnegative exact absolute gap, as an integer or fraction string such as `"1/10"`. |
| `solver_seconds` | Cooperative budget passed to `certify`; default 2 seconds. It is not a hard deadline. |
| `deadline_seconds` | End-to-end watchdog covering producer startup, read/solve/serialize, and fresh verification; default 30 seconds. A timed-out worker is terminated. OS scheduling and process cleanup may add a small overrun. |
| `max_observed_rss_mib` | Optional post-run threshold for the larger of the two workers' peak resident memories; `null` disables this gate. It is a measured acceptance budget, **not an OS-enforced memory cap**. |
| `seed` | Fixed nonnegative solver seed; default 0. Repetitions retain the same seed to expose timing/resource variation. |

`repetitions` defaults to 3. BLAS thread environment variables are fixed to one;
certification uses the package's controlled solver settings. Peak worker RSS is
available through the standard library on macOS/Linux. Elsewhere it is reported
as unavailable; a requested memory threshold cannot pass without a measurement.
The coordinator's small process memory is not included in that worker metric.

## Input format

Each JSON file has `model` and `answer`. Coefficients may be finite JSON numbers
or exact fraction strings. Numbers retain their binary64 meaning; fraction
strings express exact values such as `"1/3"`. Indices are zero-based.

```json
{
  "model": {
    "type": "qubo",
    "num_variables": 2,
    "coefficients": [[0, 0, -1], [1, 1, 2], [0, 1, 1]],
    "offset": "1/7"
  },
  "answer": [1, 0]
}
```

Supported model types use the public certification API:

- `ising`: `fields` list, `couplings` rows `[u, v, coefficient]`, optional
  `offset`; answer entries are integers -1 or +1.
- `qubo`: `num_variables`, `coefficients` rows `[i, j, coefficient]`, optional
  `offset`; answer entries are integers 0 or 1. The objective is `x.T @ Q @ x`.
- `maxcut`: `num_nodes`, `edges` rows `[u, v, weight]`; answer entries are integer
  partition bits. The objective is total crossing-edge weight.

The certificate's bound is a lower bound for minimization and an upper bound
for Max-Cut. The reported exact gap is nonnegative in both cases. Supplying an
answer does not currently warm-start the solver; certification runs the shared
solver to establish a bound and then assesses the supplied answer.

## Decide from the evidence

Record every trial, including failures. A trial passes only if the fresh
verifier accepts its bound, its exact gap meets the frozen target, and the
measured time/memory gates pass. A target miss does not establish that the
candidate is poor: its proof may simply be too loose. A verified result does
not establish that the original application model is appropriate.

`end_to_end_seconds` includes imports/process startup, reading the frozen input,
certification, proof serialization, and a new-process read and verification.
Separate process and API timings help explain the cost; excluding them from an
application decision would hide work. The harness checks its target only after
a complete certification attempt, so elapsed time is **time to obtain a
verified target-satisfying result**, not the first instant that the algorithm
crossed a target. A missed target has no successful time-to-target observation.

Review pass counts, failures, resource costs, and useful gap thresholds with the
application owner. Before claiming comparative advantage, evaluate an
appropriate independent baseline on the same frozen inputs/candidates and
budgets, including its verification costs where applicable. This kit currently
measures the certification workflow only; it does not claim a comparison it
has not run. If the target provides no practical decision value, narrow or stop
the effort before adding algorithms.
