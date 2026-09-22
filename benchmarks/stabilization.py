"""Frozen maintenance benchmark: fresh sequential processes and serialized proofs.

Prepare inputs without solving:
  python benchmarks/stabilization.py prepare
Run only when other tests/builds have finished:
  python benchmarks/stabilization.py run --baseline /tmp/highs-baseline \
      --artifacts /tmp/highs-stabilization-proofs
The baseline directory must contain an archive of the protocol's baseline
revision. Both revisions use this interpreter and identical dependencies.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import gzip
import hashlib
import itertools
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "benchmarks/stabilization_protocol.json"
CORPUS = ROOT / "benchmarks/stabilization_corpus.json.gz"
RESULTS = ROOT / "benchmarks/stabilization_results.json"


def packed_write(path, value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(gzip.compress(data, mtime=0) if path.suffix == ".gz" else data)


def read(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    import numpy as np
    import scipy.sparse as sp
    from dwave.graphs import pegasus_graph

    protocol = read(PROTOCOL)
    cases = []
    for spec in protocol["cases"]:
        case = dict(spec)
        rng = np.random.default_rng(spec.get("seed", 0))
        n = spec.get("variables")
        if spec["kind"] == "lp":
            m = spec["constraints"]
            if spec["generator"] == "sparse":
                matrix = sp.random(m, n, density=spec["density"], format="csr", random_state=rng,
                                   data_rvs=lambda size: rng.uniform(-1, 1, size))
            elif spec["generator"] == "redundant":
                base = rng.integers(-4, 5, size=(m // 10, n)).astype(float)
                matrix = sp.csr_matrix(np.tile(base, (10, 1)))
            else:
                matrix = sp.csr_matrix(rng.uniform(-1, 1, size=(m, n)))
            point = rng.uniform(0.2, 0.8, n)
            rhs = matrix @ point + rng.uniform(0.1, 1, m)
            if spec["generator"] == "redundant":
                rhs = np.tile(rhs[:m // 10], 10) + np.repeat(np.arange(10), m // 10)
            case.update(c=rng.normal(size=n).tolist(), rhs=rhs.tolist(),
                        matrix=[matrix.data.tolist(), matrix.indices.tolist(), matrix.indptr.tolist()])
        elif spec["kind"] == "qubo":
            entries = [[i, i, int(rng.integers(-4, 5))] for i in range(n)]
            entries += [[i, (i + 1) % n, int(rng.choice([-2, -1, 1, 2]))] for i in range(n)]
            case["entries"] = entries
        else:
            if spec["generator"] == "gset":
                try:
                    raw = urllib.request.urlopen(spec["source"], timeout=30).read()
                except Exception as error:
                    case["unavailable"] = f"{type(error).__name__}: {error}"
                    cases.append(case)
                    continue
                lines = raw.decode().splitlines()
                n, m = map(int, lines[0].split())
                edges = [[int(u) - 1, int(v) - 1, float(w)]
                         for u, v, w in (line.split() for line in lines[1:] if line.strip())]
                if len(edges) != m:
                    raise ValueError("Gset edge count does not match its header")
                fields = [0.0] * n
                case["source_sha256"] = hashlib.sha256(raw).hexdigest()
            else:
                if spec["generator"] == "pegasus":
                    graph = pegasus_graph(spec["size"])
                    graph.remove_nodes_from(rng.choice(sorted(graph), spec["damaged_nodes"], replace=False))
                    labels = {node: i for i, node in enumerate(sorted(graph))}
                    pairs = sorted(tuple(sorted((labels[u], labels[v]))) for u, v in graph.edges)
                    n = len(labels)
                elif spec["generator"] == "dense":
                    pairs = list(itertools.combinations(range(n), 2))
                else:
                    pairs = sorted({tuple(sorted((i, (i + step) % n))) for i in range(n) for step in (1, 3)})
                fields = (rng.integers(-3, 4, n) / 8).tolist()
                values = rng.integers(-8, 9, len(pairs)) / 8
                edges = [[u, v, float(w)] for (u, v), w in zip(pairs, values) if w]
            case.update(variables=n, fields=fields, couplings=edges)
            if spec.get("exact_oracle"):
                case["oracle_energy"] = str(min(
                    sum((Fraction(h) * s for h, s in zip(fields, spins)), Fraction())
                    + sum((Fraction(w) * spins[u] * spins[v] for u, v, w in edges), Fraction())
                    for spins in itertools.product((-1, 1), repeat=n)))
        cases.append(case)
    packed_write(CORPUS, {"protocol_sha256": digest(PROTOCOL), "cases": cases})
    print(json.dumps({"corpus": str(CORPUS), "sha256": digest(CORPUS),
                      "cases": [dict(name=c["name"], unavailable=c.get("unavailable")) for c in cases]}))


def cpu_seconds():
    return sum(r.ru_utime + r.ru_stime for r in
               (resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)))


def rss_bytes():
    # macOS reports bytes; Linux/BSD report KiB. This benchmark targets these hosts.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def model(case):
    if case["kind"] != "qubo":
        return case["fields"], {(u, v): w for u, v, w in case["couplings"]}, Fraction()
    h = [Fraction()] * case["variables"]
    J, offset = {}, Fraction()
    for u, v, weight in case["entries"]:
        w = Fraction(weight)
        if u == v:
            h[u] += w / 2
            offset += w / 2
        else:
            h[u] += w / 4
            h[v] += w / 4
            J[u, v] = J.get((u, v), Fraction()) + w / 4
            offset += w / 4
    return h, J, offset


def native_lp(c, matrix, rhs, method):
    """Same highspy model-building path as examples/benchmark_linprog.native_full."""
    import highspy
    import numpy as np
    import scipy.sparse as sp
    from scipy.optimize import OptimizeResult

    session = highspy.Highs()
    for option, value in (("output_flag", False), ("parallel", "off"), ("solver", method)):
        if session.setOptionValue(option, value) == highspy.HighsStatus.kError:
            raise RuntimeError(f"HiGHS rejected {option}")
    if session.addCols(len(c), c, np.zeros(len(c)), np.ones(len(c)), 0, [], [], []) == highspy.HighsStatus.kError:
        raise RuntimeError("HiGHS rejected columns")
    matrix = sp.csr_matrix(matrix)
    if session.addRows(len(rhs), np.full(len(rhs), -np.inf), rhs, matrix.nnz,
                       matrix.indptr, matrix.indices, matrix.data) == highspy.HighsStatus.kError:
        raise RuntimeError("HiGHS rejected rows")
    if session.run() == highspy.HighsStatus.kError:
        raise RuntimeError("HiGHS solve failed")
    status = session.getModelStatus()
    return OptimizeResult(x=np.asarray(session.getSolution().col_value), fun=session.getObjectiveValue(),
                          status=0 if status == highspy.HighsModelStatus.kOptimal else int(status),
                          success=status == highspy.HighsModelStatus.kOptimal,
                          message=session.modelStatusToString(status))


def worker(args):
    sys.path.insert(0, str(args.source))
    import numpy as np
    import scipy.sparse as sp
    from highs_turbo import linprog, solve_ising, solve_qubo

    case = next(c for c in read(CORPUS)["cases"] if c["name"] == args.case)
    record = dict(case=case["name"], kind=case["kind"], method=args.method, budget_seconds=case["seconds"])
    if case["kind"] == "lp":
        matrix = sp.csr_matrix(tuple(case["matrix"]), shape=(case["constraints"], case["variables"]))
        # Keep the dense caller path in the dense and row-recovery cases.
        supplied = matrix if case["generator"] == "sparse" else matrix.toarray()
        c, rhs = np.asarray(case["c"]), np.asarray(case["rhs"])
        began, cpu = time.perf_counter(), cpu_seconds()
        result = (native_lp(c, supplied, rhs, args.method.removeprefix("native-"))
                  if args.method.startswith("native-")
                  else linprog(c, A_ub=supplied, b_ub=rhs, bounds=(0, 1)))
        record.update(api_seconds=time.perf_counter() - began, cpu_seconds=cpu_seconds() - cpu,
                      peak_rss_bytes=rss_bytes(), status=result.status, message=result.message)
        if not result.success or result.x is None:
            raise ValueError(f"LP failed: {result.message}")
        residual = max(float(np.max(matrix @ result.x - rhs)), float(np.max(-result.x)),
                       float(np.max(result.x - 1)), 0.0)
        if residual > 1e-7 or not np.isclose(c @ result.x, result.fun, atol=1e-7, rtol=1e-9):
            raise ValueError("LP answer fails original-model feasibility/objective check")
        record.update(objective=float(result.fun), max_violation=residual,
                      strategy=result.get("turbo_strategy", args.method if args.method.startswith("native-") else "scipy"),
                      solver_iterations=result.get("turbo_solver_iterations"),
                      fallback_reasons=result.get("fallback_reason"), certificate_verified=None,
                      independent_verification_seconds=None)
    else:
        fields, couplings, offset = model(case)
        if case["kind"] == "qubo":
            rows, cols, values = zip(*case["entries"])
            Q = sp.csr_matrix((values, (rows, cols)), shape=(case["variables"], case["variables"]))
        began, cpu = time.perf_counter(), cpu_seconds()
        if case["kind"] == "qubo":
            result = solve_qubo(Q, time_limit=case["seconds"], seed=0, threads=1)
            spins = [int(2 * x - 1) for x in result.solution]
            proof, bound = result.bound_certificate, result.exact_rational_bound
        else:
            options = {"relaxation": "sdp"} if args.method == "sdp" else {}
            result = solve_ising(fields, couplings, time_limit=case["seconds"], seed=0, threads=1, **options)
            spins = [int(result.spins[i]) for i in range(len(fields))]
            proof, bound = result.certificate, result.exact_cut_lower_bound
        record.update(api_seconds=time.perf_counter() - began, cpu_seconds=cpu_seconds() - cpu,
                      peak_rss_bytes=rss_bytes(), status=result.status,
                      message=getattr(result, "message", None),
                      fallback_reasons=getattr(result, "certificate_fallbacks", None),
                      progress=getattr(result, "progress", None),
                      reported_lower_bound=result.lower_bound, reported_gap=getattr(result, "gap", None),
                      reported_bound_semantics="exact" if case["kind"] == "qubo" else "numerical")
        if any(s not in (-1, 1) for s in spins):
            raise ValueError("Returned assignment is not binary")
        energy = offset + sum((Fraction(h) * s for h, s in zip(fields, spins)), Fraction())
        energy += sum((Fraction(w) * spins[u] * spins[v] for (u, v), w in couplings.items()), Fraction())
        if float(energy) != result.energy or bound > energy:
            raise ValueError("Returned binary energy or bound does not match the original model")
        if "oracle_energy" in case and not bound <= Fraction(case["oracle_energy"]) <= energy:
            raise ValueError("Returned result excludes the exhaustive optimum")
        if "oracle_energy" in case and result.status == "OPTIMAL" and energy != Fraction(case["oracle_energy"]):
            raise ValueError("Reported optimal answer disagrees with exhaustive enumeration")
        packed_write(args.proof, {"case": case["name"], "spins": spins, "certificate": proof.to_dict()})
        record.update(exact_energy=str(energy), exact_bound=str(bound), exact_gap=str(energy - bound),
                      oracle_energy=case.get("oracle_energy"),
                      certificate_structure=dict(cuts=len(proof.cuts), dense_gram_rows=len(proof.gram_factor),
                                                 sparse_gram_entries=len(proof.sparse_gram_factor[2]) if proof.sparse_gram_factor else 0),
                      certificate_bytes=args.proof.stat().st_size,
                      certificate_sha256=digest(args.proof), proof_path=str(args.proof))
    record["overrun_seconds"] = (None if case["seconds"] is None
                                  else max(0.0, record["api_seconds"] - case["seconds"]))
    print(json.dumps(record, sort_keys=True))


def verify(args):
    sys.path.insert(0, str(args.source))
    from highs_turbo import verify_ising_certificate
    from highs_turbo.ising_cuts import IsingCertificate

    bundle = read(args.proof)
    case = next(c for c in read(CORPUS)["cases"] if c["name"] == bundle["case"])
    h, J, offset = model(case)
    proof = bundle["certificate"]
    began = time.perf_counter()
    valid = verify_ising_certificate(h, J, proof, offset=offset)
    elapsed = time.perf_counter() - began
    if not valid:
        raise ValueError("Saved certificate failed independent verification")
    bound = IsingCertificate.from_dict(proof).lower_bound
    spins = bundle["spins"]
    if len(spins) != len(h) or any(type(s) is not int or s not in (-1, 1) for s in spins):
        raise ValueError("Saved answer is not a feasible assignment")
    energy = offset + sum((Fraction(w) * s for w, s in zip(h, spins)), Fraction())
    energy += sum((Fraction(w) * spins[u] * spins[v] for (u, v), w in J.items()), Fraction())
    if bound > energy:
        raise ValueError("Saved lower bound exceeds feasible energy")
    print(json.dumps(dict(certificate_verified=True, independent_verification_seconds=elapsed,
                         verification_peak_rss_bytes=rss_bytes(), verified_exact_bound=str(bound))))


def run(args):
    import highspy
    import numpy
    import scipy

    protocol, corpus = read(PROTOCOL), read(CORPUS)
    if corpus["protocol_sha256"] != digest(PROTOCOL):
        raise ValueError("Protocol changed after preparation; freeze and prepare it before running")
    args.artifacts.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **protocol["thread_environment"], "PYTHONHASHSEED": "0"}
    sources = {"baseline": args.baseline.resolve(), "candidate": ROOT}
    dependency_versions = {}
    for package in ("highspy", "numpy", "scipy", "networkx", "dwave-graphs", "dwave-samplers"):
        try:
            dependency_versions[package] = version(package)
        except PackageNotFoundError:
            dependency_versions[package] = None
    evidence = dict(started_at=datetime.now(timezone.utc).isoformat(),
                    protocol_sha256=digest(PROTOCOL), corpus_sha256=digest(CORPUS),
                    platform=platform.platform(), python=platform.python_version(), numpy=numpy.__version__,
                    scipy=scipy.__version__, highspy=highspy.Highs().version(), interpreter=sys.executable, dependencies=dependency_versions,
                    baseline_revision=protocol["baseline_revision"], records=[], unavailable_cases=[],
                    native_extensions={name: {p.name: digest(p) for p in sorted((path / "highs_turbo").glob("*.so"))}
                                       for name, path in sources.items()},
                    sources={name: {str(p.relative_to(path)): digest(p) for p in sorted((path / "highs_turbo").rglob("*.py"))}
                             for name, path in sources.items()}, published_mixing=protocol["published_mixing"])
    if evidence["native_extensions"]["baseline"] != evidence["native_extensions"]["candidate"]:
        raise ValueError("Baseline and candidate native extensions differ")
    if args.case:
        selected = set(args.case.split(","))
    else:
        selected = {c["name"] for c in corpus["cases"]}
    for repetition in range(protocol["repetitions"]):
        for index, case in enumerate(corpus["cases"]):
            if case["name"] not in selected:
                continue
            if case.get("unavailable"):
                if repetition == 0:
                    evidence["unavailable_cases"].append(case)
                continue
            methods = protocol[case["kind"] + "_methods"]
            shift = (repetition + index) % len(methods)
            for method in methods[shift:] + methods[:shift]:
                proof = args.artifacts / f"{case['name']}-{method}-{repetition}.json.gz"
                source = sources["baseline" if method == "baseline" else "candidate"]
                command = [sys.executable, str(Path(__file__).resolve()), "worker", "--source", str(source),
                           "--case", case["name"], "--method", method, "--proof", str(proof)]
                began = time.perf_counter()
                completed = subprocess.run(command, cwd=args.artifacts, env=env, capture_output=True,
                                           text=True, timeout=case.get("watchdog_seconds", 120))
                if completed.returncode:
                    raise RuntimeError(f"{case['name']} {method}: {completed.stderr}")
                record = json.loads(completed.stdout.strip().splitlines()[-1])
                record.update(repetition=repetition, complete_worker_seconds=time.perf_counter() - began)
                if case["kind"] != "lp":
                    verified = subprocess.run([sys.executable, str(Path(__file__).resolve()), "verify",
                                               "--source", str(ROOT), "--proof", str(proof)],
                                              cwd=args.artifacts, env=env, capture_output=True, text=True, timeout=120)
                    if verified.returncode:
                        raise RuntimeError(verified.stderr)
                    record.update(json.loads(verified.stdout.strip().splitlines()[-1]))
                    if record["verified_exact_bound"] != record["exact_bound"]:
                        raise ValueError("Serialized bound changed during independent verification")
                evidence["records"].append(record)
                args.output.write_text(json.dumps(evidence, indent=2) + "\n")
                if case["kind"] == "lp":
                    objectives = [r["objective"] for r in evidence["records"] if r["case"] == case["name"]]
                    if max(objectives) - min(objectives) > 1e-7:
                        raise ValueError(f"{case['name']} LP objectives differ by more than 1e-7 across methods/repeats")
                print(json.dumps(record), flush=True)
    for name, path in sources.items():
        if evidence["sources"][name] != {str(p.relative_to(path)): digest(p) for p in sorted((path / "highs_turbo").rglob("*.py"))}:
            raise RuntimeError(f"{name} source changed during measurements; discard this run")
        if evidence["native_extensions"][name] != {p.name: digest(p) for p in sorted((path / "highs_turbo").glob("*.so"))}:
            raise RuntimeError(f"{name} native extension changed during measurements; discard this run")
    evidence["lp_objective_max_spreads"] = {
        case["name"]: max(r["objective"] for r in evidence["records"] if r["case"] == case["name"])
        - min(r["objective"] for r in evidence["records"] if r["case"] == case["name"])
        for case in corpus["cases"] if case["kind"] == "lp" and case["name"] in selected
    }
    evidence["completed_at"] = datetime.now(timezone.utc).isoformat()
    args.output.write_text(json.dumps(evidence, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "worker", "verify"))
    parser.add_argument("--source", type=Path, default=ROOT)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--artifacts", type=Path, default=Path("/tmp/highs-stabilization-proofs"))
    parser.add_argument("--output", type=Path, default=RESULTS)
    parser.add_argument("--case")
    parser.add_argument("--method")
    parser.add_argument("--proof", type=Path)
    args = parser.parse_args()
    if args.action == "run" and args.baseline is None:
        parser.error("run requires --baseline pointing to the archived baseline revision")
    {"prepare": prepare, "run": lambda: run(args), "worker": lambda: worker(args),
     "verify": lambda: verify(args)}[args.action]()


if __name__ == "__main__":
    main()
