"""Check a heuristic's Ising answer against an independently verified bound.

Run ``python examples/certify_external_answer.py --output /tmp/answer-proof.json``.
Then run ``python examples/certify_external_answer.py --verify /tmp/answer-proof.json``
without solving again. ``--input`` accepts JSON with ``fields``, ``couplings``
([u, v, weight] rows), ``spins`` (+1/-1 in field order), and optional ``offset``.
Coefficients may be JSON numbers or exact rational strings such as "1/3".
The default answer is a small deterministic example standing in for an external
heuristic. A verified positive gap bounds its possible suboptimality; it does
not assert that the external answer is optimal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def verify_bundle(bundle):
    """Keep the original example's format while reusing the installed checker."""
    from highs_turbo.certification import verify

    report = verify({"version": 1, "model": _model(bundle), "answer": bundle["spins"],
                     "certificate": bundle["certificate"]})
    return {"verified": report["bound_verified"], "external_energy": report["candidate_objective"],
            "exact_lower_bound": report["bound"], "exact_gap": report["gap"],
            "external_answer_is_proven_optimal": report["optimality_proven"]}


def _model(bundle):
    return {"type": "ising", "fields": bundle["fields"], "couplings": bundle["couplings"],
            "offset": bundle.get("offset", 0)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("external-answer-proof.json"))
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--seconds", type=float, default=2.0)
    args = parser.parse_args()
    if args.verify:
        started = time.perf_counter()
        report = verify_bundle(json.loads(args.verify.read_text()))
        report["verification_seconds"] = time.perf_counter() - started
        print(json.dumps(report, sort_keys=True))
        return
    from highs_turbo.certification import certify

    bundle = json.loads(args.input.read_text()) if args.input else {
        "fields": [0.5, -0.25, 0.0, 0.25],
        "couplings": [[0, 1, 1], [1, 2, 1], [0, 2, 1], [2, 3, -0.5]],
        "spins": [1, -1, 1, 1], "offset": 0,
    }
    certified = certify(_model(bundle), bundle["spins"], time_limit=args.seconds, seed=0)
    bundle["certificate"] = certified["certificate"]
    bundle["producer"] = certified["producer"]
    # Validate the external assignment before writing the portable artifact.
    verify_bundle(bundle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bundle, indent=2) + "\n")
    checked = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--verify", str(args.output.resolve())],
                             check=True, capture_output=True, text=True)
    report = json.loads(checked.stdout)
    report.update(artifact=str(args.output.resolve()), independent_process=True,
                  solve_seconds=bundle["producer"]["api_seconds"])
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
