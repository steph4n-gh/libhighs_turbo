"""Compare the authors' unmodified 2017 Mixing executable with exact repair.

Build https://github.com/locuslab/mixing at the revision in the results report.
The executable's own solution export, input conversion, and both proof checks
are timed. Compilation and imports are excluded. No elapsed-time stopping is
available in that executable; report its measured time and actual settings.
"""

import argparse
from dataclasses import replace
from fractions import Fraction
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import scipy.sparse as sp

from highs_turbo.ising import _normalize
from highs_turbo.ising_cuts import _bound, check_certificate, make_certificate
from highs_turbo.ising_sparse import factor_witness


def run(args):
    records = []
    for case in args.cases:
        data = json.loads((args.directory / f"{case}.json").read_text())
        for label, options in [
            ("default", []),
            ("rank16-accurate", ["-k", "16", "-e", "1e-7"]),
        ]:
            began = time.perf_counter()
            labels, fields, weights, constant = _normalize(
                data["fields"], {(u, v): w for u, v, w in data["couplings"]}, 0
            )
            weights.update({(i, len(labels)): w for i, w in enumerate(fields) if w})
            edges = sorted(weights)
            n = max(max(e) for e in edges) + 1
            prefix = args.directory / f"{case}-mixing2017-{label}"
            inp = prefix.with_suffix(".input")
            inp.write_text(
                f"{n} {len(edges)}\n"
                + "".join(
                    f"{u + 1} {v + 1} {float(weights[u, v]):.17g}\n" for u, v in edges
                )
            )
            command = [str(args.binary), "-s", "maxcut", *options, str(inp)]
            solve_start = time.perf_counter()
            with prefix.with_suffix(".log").open("w") as stream:
                subprocess.run(
                    command,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=120,
                    env=os.environ,
                )
            solve_seconds = time.perf_counter() - solve_start
            check_start = time.perf_counter()
            vectors = np.loadtxt(str(inp) + ".sol")
            vectors /= np.maximum(np.linalg.norm(vectors, axis=1)[:, None], 1e-100)
            u, v = np.asarray(edges).T
            values = np.asarray([float(weights[e]) / 2 for e in edges])
            magnitude = float(np.max(abs(values)))
            matrix = sp.csr_matrix(
                (np.r_[values, values], (np.r_[u, v], np.r_[v, u])), shape=(n, n)
            )
            numerical = float(np.sum(vectors * (matrix @ vectors)))
            factor, denominator = factor_witness(matrix / magnitude, vectors, magnitude)
            source = make_certificate([], [], edges, weights, constant)
            bound = _bound(
                (),
                (),
                source.denominator,
                edges,
                weights,
                constant,
                (),
                denominator,
                factor,
            )
            proof = replace(
                source,
                lower_bound=bound,
                sparse_gram_factor=factor,
                gram_denominator=denominator,
            )
            assert check_certificate(proof, edges, weights, constant)
            check_seconds = time.perf_counter() - check_start
            record = dict(
                case=case,
                method="mixing2017",
                settings=label,
                rank=vectors.shape[1],
                seconds=time.perf_counter() - began,
                solve_seconds=solve_seconds,
                certificate_seconds=check_seconds,
                numerical=numerical,
                verified_bound=float(bound),
                exact=[bound.numerator, bound.denominator],
                factor_entries=len(factor[2]),
                command=command,
            )
            records.append(record)
            (args.directory / "mixing2017-results.json").write_text(
                json.dumps(records, indent=2) + "\n"
            )
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", default=["pegasus-6111", "G55"])
    run(parser.parse_args())
