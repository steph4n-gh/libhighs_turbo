"""Generate the frozen four-case suite and run each solver in a fresh process.

python experiments/geometric_suite.py --directory /tmp/geometric-final --previous /path/to/4d27b7c
The output JSON records every run; a single run per case is a focused comparison,
not a statistical estimate. All solver processes run sequentially on one BLAS thread.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import urllib.request

import highspy
import networkx as nx
import numpy as np
import scipy
from dwave.graphs import pegasus_graph

ROOT = Path(__file__).resolve().parents[1]


def generate(directory, suite="geometric"):
    directory.mkdir(parents=True, exist_ok=True)
    cases = []
    seeds = (
        [(4101, False), (4102, True)]
        if suite == "geometric"
        else [(6111, False), (6112, True)]
    )
    size = 8 if suite == "geometric" else 16
    for seed, damaged in seeds:
        rng = np.random.default_rng(seed)
        graph = pegasus_graph(size)
        if damaged:
            graph.remove_nodes_from(
                rng.choice(sorted(graph), 70 if size == 8 else 280, replace=False)
            )
        labels = {node: i for i, node in enumerate(sorted(graph))}
        fields = (rng.integers(-3, 4, len(labels)) / 8).tolist()
        edges = sorted(tuple(sorted((labels[u], labels[v]))) for u, v in graph.edges)
        values = rng.integers(-8, 9, len(edges)) / 8
        cases.append(
            dict(
                name=f"pegasus-{seed}",
                size=size,
                seed=seed,
                damaged=damaged,
                fields=fields,
                couplings=[[u, v, float(w)] for (u, v), w in zip(edges, values) if w],
            )
        )
    for name in ["G3", "G13"] if suite == "geometric" else ["G55"]:
        url = f"https://web.stanford.edu/~yyye/yyye/Gset/{name}"
        raw = urllib.request.urlopen(url).read()
        lines = raw.decode().splitlines()
        n, m = map(int, lines[0].split())
        edges = [
            [int(u) - 1, int(v) - 1, float(w)]
            for u, v, w in (line.split() for line in lines[1:] if line.strip())
        ]
        assert len(edges) == m
        cases.append(
            dict(
                name=name,
                source=url,
                sha256=hashlib.sha256(raw).hexdigest(),
                fields=[0.0] * n,
                couplings=edges,
            )
        )
    paths = []
    for case in cases:
        path = directory / (case["name"] + ".json")
        path.write_text(json.dumps(case, sort_keys=True, separators=(",", ":")))
        paths.append(path)
    return paths


def run(args):
    paths = generate(args.directory, args.suite)
    env = {
        **os.environ,
        "VECLIB_MAXIMUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    results = dict(
        platform=platform.platform(),
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        highs=highspy.Highs().version(),
        networkx=nx.__version__,
        budget=args.seconds,
        records=[],
        cases={p.stem: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        suite=args.suite,
        production_hashes={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "highs_turbo").glob("ising*.py"))
        },
    )
    methods = [
        "previous_hybrid",
        "auto",
        "turbo_linprog",
        "native_highs",
        "scipy_linprog",
    ]
    if args.suite == "sparse":
        methods.remove("scipy_linprog")
    for index, path in enumerate(paths):
        for method in methods[index:] + methods[:index]:
            command = [
                sys.executable,
                str(ROOT / "examples/benchmark_geometric_ising.py"),
                "--case",
                str(path),
                "--method",
                method,
                "--seconds",
                str(args.seconds),
            ]
            if method == "previous_hybrid":
                command += ["--source", str(args.previous)]
            elif method == "auto":
                command += [
                    "--proof",
                    str(args.directory / (path.stem + "-relaxation.json")),
                ]
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=args.seconds + 30,
            )
            if completed.returncode:
                raise RuntimeError(completed.stderr)
            record = json.loads(completed.stdout.strip().splitlines()[-1])
            results["records"].append(record)
            (args.directory / "results.json").write_text(
                json.dumps(results, indent=2) + "\n"
            )
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--suite", choices=["geometric", "sparse"], default="geometric")
    run(parser.parse_args())
