# Check an answer from another optimizer

Keep your optimizer. Independently check how close its answer is to best possible.
`highs_turbo.certify` accepts an existing Ising, QUBO or Max-Cut answer and
produces a complete bound witness. `highs_turbo.verify` checks it without solving
again. A positive gap is a limit on possible improvement; zero proves optimality
for the encoded objective.

## Five-minute installed example

Install the portable wheel linked in [the README](README.md#installation). Save
this as `answer.json`:

```json
{
  "model": {
    "type": "qubo",
    "num_variables": 2,
    "coefficients": [[0, 0, -1], [1, 1, -1], [0, 1, 2]],
    "offset": 0
  },
  "answer": [0, 0]
}
```

Run two separate processes:

```bash
python -m highs_turbo certify --input answer.json --output proof.json --seconds 2
python -m highs_turbo verify proof.json
```

The installed `highs-turbo` command accepts the same arguments. The report uses
exact rational strings:

```json
{
  "model_type": "qubo",
  "objective_sense": "minimize",
  "candidate_objective": "0",
  "bound": "-1",
  "gap": "1",
  "bound_verified": true,
  "optimality_proven": false
}
```

The candidate is at most one objective unit worse than the optimum. Keep
`proof.json`, which includes the normalized model, answer, complete Ising
witness and informational producer metadata. The report or a hash alone is
insufficient to repeat verification. Invalid input or proof causes a nonzero
exit with a JSON error on stderr.

In Python:

```python
import json
from highs_turbo import certify, verify

source = json.load(open("answer.json"))
bundle = certify(source["model"], source["answer"], time_limit=2, seed=0)
report = verify(bundle)
assert report["bound_verified"]
```

## Model formats

Each input is exactly `{"model": {...}, "answer": [...]}`. Indices start at zero.
Every variable needs an integer assignment (booleans are rejected).

| Type | Model fields | Objective and answer |
| --- | --- | --- |
| `ising` | `fields`, `couplings`, optional `offset` | Minimize offset + Σ hᵢsᵢ + Σ Jᵢⱼsᵢsⱼ; answer entries -1 or +1. |
| `qubo` | `num_variables`, `coefficients`, optional `offset` | Minimize offset + Σ Qᵢⱼxᵢxⱼ; answer entries 0 or 1. |
| `maxcut` | `num_nodes`, `edges` | Maximize Σ wᵢⱼ[partitionᵢ ≠ partitionⱼ]; answer entries 0 or 1. |

`couplings`, `coefficients` and `edges` contain `[i, j, weight]` triples. Missing
Ising couplings and missing QUBO/Max-Cut term lists default to empty. Repeated
and reversed pairs add exactly. A QUBO matrix exported in full includes both
off-diagonal entries in the objective; do not double an already combined term.
Ising diagonal terms add to the constant, QUBO diagonal terms are linear, and
Max-Cut self-loops contribute zero.

Coefficients accept finite JSON numbers or exact strings such as `"1/3"` and
`"-7"`. JSON floating-point numbers denote their exact binary64 values. Use
`"1/10"` when you mean exactly one tenth. Decimal and exponent strings are not
accepted. Unknown model fields are rejected to avoid silently ignoring a
constraint. These models have binary-domain constraints only; check original
business constraints and any penalty encoding separately.

The JSON interface limits inputs to 100,000 variables, 1,000,000 terms and
4,096-bit coefficient numerators/denominators. Generation also requires the
transformed Ising absolute objective envelope (absolute offset plus absolute
field and coupling coefficients) to fit finite binary64 arithmetic; exact
verification retains the wider rational parsing limits. Individual proof strategies have
smaller limits; see [the API contract](API_CONTRACT.md). Accepted size does not
promise fast solution or low memory use.

## What verification means

Bundles use `version: 1`; the contained certificate has its own format version.
The verifier recomputes the candidate objective and checks the full witness
against the supplied model. The reported bound is a lower bound for minimization
and an upper bound for Max-Cut. `gap` is always nonnegative.

This proves mathematical validity, not authorship or authenticity. Another valid
answer can reuse the same bound; its gap is recomputed. Keep your own input hash
or compare the stored model to your intended input when receiving a bundle from
someone else. Producer timings and version strings are informational.

The supplied answer is checked but does not seed the bound solver's search.
`--seconds`/`time_limit` is a cooperative budget. Preparation and proof checking
can overrun it. The [pilot kit](pilots/README.md) supplies a separate process
watchdog and records the complete elapsed time, including proof export and
verification in a fresh process.
