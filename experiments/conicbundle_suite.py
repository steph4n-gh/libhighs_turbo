"""Fixed-cut comparison using the unmodified ConicBundle 1.a.2 library.

Build conicbundle_fixed.cxx as described in CONICBUNDLE.md. Cut discovery
and generating the optional published Mixing starting point are free to this
baseline. Input conversion, the native call, and exact repairs are timed.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import scipy.sparse as sp

from experiments.check_published import mixing


def run(args):
    records = []
    for name in args.cases:
        data = json.loads((args.directory / f"{name}-relaxation.json").read_text())
        n, cuts = data["n"], data["cuts"]
        if n <= 2048:
            raise ValueError("This runner uses the large sparse certificate repair")
        for initialization in ("zero", "published_mixing_free"):
            prefix = args.directory / f"{name}-conicbundle-{initialization}-{args.seconds}"
            inp, out = prefix.with_suffix(".input"), prefix.with_suffix(".json")
            began = time.perf_counter()
            inp.write_text(
                f"{n} {len(data['edges'])} {len(cuts)}\n"
                + "".join(
                    f"{u} {v} {w:.17g}\n"
                    for (u, v), w in zip(data["edges"], data["weights"])
                )
                + "".join(
                    f"{len(c['indices'])} {c['rhs']}\n"
                    + "".join(
                        f"{i} {a}\n" for i, a in zip(c["indices"], c["coefficients"])
                    )
                    for c in cuts
                )
            )
            command = [str(args.binary), str(inp), str(out), str(args.seconds)]
            if initialization == "published_mixing_free":
                u, v = np.asarray(data["edges"]).T
                w = np.asarray(data["weights"]) / 2
                matrix = sp.csr_matrix(
                    (np.r_[w, w], (np.r_[u, v], np.r_[v, u])), shape=(n, n)
                )
                vectors = np.loadtxt(
                    args.directory / f"{name}-mixing2017-default.input.sol"
                )
                vectors /= np.maximum(np.linalg.norm(vectors, axis=1)[:, None], 1e-100)
                initial = np.r_[
                    -np.sum(vectors * (matrix @ vectors), axis=1), np.zeros(len(cuts))
                ]
                warm = prefix.with_suffix(".warm")
                np.savetxt(warm, initial)
                command.append(str(warm))
            solve_start = time.perf_counter()
            with prefix.with_suffix(".log").open("w") as stream:
                subprocess.run(
                    command, stdout=stream, stderr=subprocess.STDOUT, check=True,
                    timeout=args.seconds + 35, env=os.environ,
                )
            solve_seconds = time.perf_counter() - solve_start
            output = json.loads(out.read_text())
            dual = np.asarray(output["dual"])
            rhs = np.r_[np.ones(n), [2*c["rhs"]-sum(c["coefficients"]) for c in cuts]]
            # ConicBundle minimizes rhs*y + n*lambda_max(-Q - diag(y)
            # + sum(lambda*A)). Recover that spectral shift from its reported
            # objective, then offer the slack to the shared checker. This
            # numerical identity only proposes a factor; it is not trusted.
            spectral = (-output["numerical"] - rhs @ dual) / n
            adjusted = {**output, "dual": np.r_[-dual[:n]-spectral, dual[n:]].tolist()}
            checked = mixing(data, adjusted, str(out))
            record = dict(
                case=name, method="conicbundle", initialization=initialization,
                n=n, cuts=len(cuts), budget=args.seconds,
                seconds=time.perf_counter()-began, solve_seconds=solve_seconds,
                driver_seconds=output["seconds"], numerical=output["numerical"],
                status=output["status"], rank=output["rank"], spectral_shift=spectral,
                **checked,
            )
            records.append(record)
            (args.directory / "conicbundle-results.json").write_text(
                json.dumps(records, indent=2) + "\n"
            )
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=10)
    parser.add_argument("--cases", nargs="+", default=["pegasus-7111", "G56"])
    run(parser.parse_args())
