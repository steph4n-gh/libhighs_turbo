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
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def verify_bundle(bundle):
    """Verify the original model, feasible answer, and full serialized proof."""
    from highs_turbo import verify_ising_certificate
    from highs_turbo.ising_cuts import IsingCertificate

    fields, spins = list(map(Fraction, bundle["fields"])), bundle["spins"]
    if len(spins) != len(fields) or any(type(s) is not int or s not in (-1, 1) for s in spins):
        raise ValueError("The external answer must contain one integer -1 or +1 per field")
    couplings = {}
    for u, v, weight in bundle["couplings"]:
        if type(u) is not int or type(v) is not int or not (0 <= u < len(spins) and 0 <= v < len(spins)):
            raise ValueError("Coupling endpoints must index the external answer")
        couplings[u, v] = couplings.get((u, v), Fraction()) + Fraction(weight)
    offset = Fraction(bundle.get("offset", 0))
    if not verify_ising_certificate(fields, couplings, bundle["certificate"], offset=offset):
        raise ValueError("Certificate does not verify against the supplied model")
    energy = Fraction(offset) + sum((Fraction(h) * s for h, s in zip(fields, spins)), Fraction())
    energy += sum((w * spins[u] * spins[v] for (u, v), w in couplings.items()), Fraction())
    bound = IsingCertificate.from_dict(bundle["certificate"]).lower_bound
    if bound > energy:
        raise ValueError("Certified lower bound exceeds the external feasible energy")
    return {"verified": True, "external_energy": str(energy), "exact_lower_bound": str(bound),
            "exact_gap": str(energy - bound), "external_answer_is_proven_optimal": energy == bound}


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
    from highs_turbo import solve_ising

    bundle = json.loads(args.input.read_text()) if args.input else {
        "fields": [0.5, -0.25, 0.0, 0.25],
        "couplings": [[0, 1, 1], [1, 2, 1], [0, 2, 1], [2, 3, -0.5]],
        "spins": [1, -1, 1, 1], "offset": 0,
    }
    couplings = {}
    for u, v, weight in bundle["couplings"]:
        couplings[u, v] = couplings.get((u, v), Fraction()) + Fraction(weight)
    began = time.perf_counter()
    result = solve_ising(list(map(Fraction, bundle["fields"])), couplings, offset=Fraction(bundle.get("offset", 0)),
                         time_limit=args.seconds, seed=0, threads=1)
    bundle["certificate"] = result.certificate.to_dict()
    bundle["producer"] = {"api_seconds": time.perf_counter() - began, "status": result.status}
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
