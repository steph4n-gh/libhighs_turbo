"""Run AugmentedMixing against the active cuts exported by geometric_suite.py.

python experiments/published_suite.py --directory /tmp/geometric-final \
    --julia /path/to/julia --project /path/to/julia/environment

Install the revisions pinned in GEOMETRIC_ISING_RESULTS.md in that environment.
Imports and a tiny separate JIT warmup are excluded from solve time; relaxation
construction and exact bound repair are included. Selected cuts are given to
the baseline without charging our selection time, favoring the baseline.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(args):
    env = {
        **os.environ,
        "VECLIB_MAXIMUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    records = []
    settings = [
        ("default", 5, {}),
        ("tuned", 5, dict(mu_start=0.2, scaling=False, tau=1.3)),
        ("tuned", 30, dict(mu_start=0.2, scaling=False, tau=1.3)),
    ]
    for case in ["pegasus-4101", "G3"]:
        for label, seconds, options in settings:
            data = json.loads(
                (args.directory / (case + "-relaxation.json")).read_text()
            )
            data.pop("source_certificate", None)
            data.update(options)
            prefix = args.directory / f"{case}-mixing-{label}-{seconds}"
            inp, out = Path(str(prefix) + "-input.json"), prefix.with_suffix(".json")
            inp.write_text(json.dumps(data))
            command = [
                str(args.julia),
                "--project=" + str(args.project),
                "experiments/published_baselines.jl",
                "mixing",
                str(inp),
                str(out),
                str(seconds),
            ]
            with prefix.with_suffix(".log").open("w") as stream:
                subprocess.run(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=seconds + 120,
                )
            checked = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.check_published",
                    str(inp),
                    str(out),
                ],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )
            record = dict(
                case=case,
                settings=label,
                budget=seconds,
                **json.loads(checked.stdout.strip().splitlines()[-1]),
            )
            records.append(record)
            (args.directory / "published-results.json").write_text(
                json.dumps(records, indent=2) + "\n"
            )
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--julia", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    run(parser.parse_args())
