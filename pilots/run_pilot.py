"""Freeze a small application pilot, then measure each certification workflow.

python pilots/run_pilot.py freeze pilots/example_manifest.json --output /tmp/highs-pilot-demo
python pilots/run_pilot.py run /tmp/highs-pilot-demo

This harness records evidence about the encoded objective and supplied answer.
It does not validate the application's modeling assumptions or constraints.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

# Permit running the checked-out kit with the installed dependency environment.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def runtime_versions():
    import highs_turbo
    versions = {"highs_turbo": highs_turbo.__version__, "python": platform.python_version()}
    for name in ("highspy", "numpy", "scipy", "networkx"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def package_identity():
    """Identify actual imported code/data and optional native implementation."""
    import highs_turbo
    root = Path(highs_turbo.__file__).resolve().parent
    suffixes = {".py", ".json", ".so", ".pyd", ".dll", ".dylib"}
    files = {str(p.relative_to(root)): sha256(p.read_bytes())
             for p in sorted(root.rglob("*")) if p.is_file() and p.suffix in suffixes}
    return {"directory": str(root), "files": files,
            "sha256": sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode())}


def peak_rss():
    if sys.platform not in ("darwin", "linux"):
        return None
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == "darwin" else value * 1024)
    except (ImportError, AttributeError, OSError):
        return None


def settings(defaults, case):
    values = {"gap_target": "0", "solver_seconds": 2.0, "deadline_seconds": 30.0,
              "max_observed_rss_mib": None, "seed": 0}
    if not isinstance(defaults, dict):
        raise ValueError("defaults must be an object")
    unknown = set(defaults) - set(values)
    if unknown:
        raise ValueError(f"Unknown default settings: {sorted(unknown)}")
    values.update(defaults)
    values.update({key: case[key] for key in values if key in case})
    target = values["gap_target"]
    if isinstance(target, bool) or not isinstance(target, (str, int)):
        raise ValueError("gap_target must be an exact fraction string or integer, such as '1/10'")
    target = Fraction(target)
    if target < 0:
        raise ValueError("gap_target must be nonnegative")
    values["gap_target"] = str(target)
    for name in ("solver_seconds", "deadline_seconds", "max_observed_rss_mib"):
        value = values[name]
        if name == "max_observed_rss_mib" and value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
        if value < 0 or (name != "solver_seconds" and value == 0):
            raise ValueError(f"{name} must be positive (solver_seconds may be zero)")
    if type(values["seed"]) is not int or not 0 <= values["seed"] < 2**31:
        raise ValueError("seed must be an integer in [0, 2**31)")
    return values


def freeze(manifest_path, directory):
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict) or manifest.get("version") != 1:
        raise ValueError("Manifest version must be 1")
    if manifest.get("evidence_type") not in ("synthetic_demo", "application_pilot"):
        raise ValueError("evidence_type must explicitly be synthetic_demo or application_pilot")
    repetitions = manifest.get("repetitions", 3)
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    attestation = manifest.get("application_constraints", {})
    if (not isinstance(attestation, dict)
            or attestation.get("status") not in ("not_assessed", "validated_separately", "not_applicable")
            or not isinstance(attestation.get("notes"), str) or not attestation["notes"].strip()):
        raise ValueError("Declare application_constraints.status and explanatory notes; this kit does not check them")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Manifest must contain at least one case")
    prepared, ids = [], set()
    for case in cases:
        case_id = case.get("id") if isinstance(case, dict) else None
        if not isinstance(case_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", case_id):
            raise ValueError("Every case needs a simple id (letters, digits, underscores, hyphens; at most 64 characters)")
        if case_id in ids:
            raise ValueError(f"Duplicate case id: {case_id}")
        ids.add(case_id)
        allowed = {"id", "input", "gap_target", "solver_seconds", "deadline_seconds", "max_observed_rss_mib", "seed"}
        if set(case) - allowed:
            raise ValueError(f"{case_id}: unknown case settings: {sorted(set(case) - allowed)}")
        if not isinstance(case.get("input"), str) or not case["input"]:
            raise ValueError(f"{case_id}: input must be a path")
        prepared.append({"id": case_id, "source_input": str((manifest_path.parent / case["input"]).resolve()),
                         "input": f"inputs/{case_id}.json", **settings(manifest.get("defaults", {}), case)})
    # No output is overwritten, and no optimizer is called during freezing.
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "inputs").mkdir()
    (directory / "manifest.json").write_bytes(manifest_bytes)
    for case in prepared:
        try:
            raw = Path(case["source_input"]).read_bytes()
            (directory / case["input"]).write_bytes(raw)
            case["input_sha256"] = sha256(raw)
            value = json.loads(raw)
            if (not isinstance(value, dict) or not isinstance(value.get("model"), dict)
                    or value["model"].get("type") not in ("ising", "qubo", "maxcut")
                    or not isinstance(value.get("answer"), list)):
                raise ValueError("Input must have a supported model object and answer list")
        except (OSError, ValueError) as error:
            case["input_error"] = f"{type(error).__name__}: {error}"
    warnings = []
    if not 12 <= len(cases) <= 16:
        warnings.append(f"Contains {len(cases)} cases; the suggested application pilot is 12–16 representative instances.")
    if manifest["evidence_type"] == "synthetic_demo":
        warnings.append("Synthetic demonstration only: these observations are not customer or application evidence.")
    if attestation["status"] == "not_assessed":
        warnings.append("Original application constraints have not been assessed; only the encoded objective is checked.")
    protocol = {"version": 1, "name": manifest.get("name", manifest_path.stem),
                "evidence_type": manifest["evidence_type"], "description": manifest.get("description", ""),
                "frozen_at": datetime.now(timezone.utc).isoformat(), "manifest_sha256": sha256(manifest_bytes),
                "repetitions": repetitions, "cases": prepared, "application_constraints": attestation,
                "runtime_versions_at_freeze": runtime_versions(), "package_identity_at_freeze": package_identity(),
                "python_executable": sys.executable,
                "platform": platform.platform(), "runner_sha256": sha256(Path(__file__).read_bytes()),
                "thread_environment": {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                                       "VECLIB_MAXIMUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONHASHSEED": "0"},
                "warnings": warnings}
    write_json(directory / "protocol.json", protocol)
    (directory / "protocol.sha256").write_text(sha256((directory / "protocol.json").read_bytes()) + "\n")
    return protocol


def check_frozen(directory, case_id=None):
    raw = (directory / "protocol.json").read_bytes()
    if sha256(raw) != (directory / "protocol.sha256").read_text().strip():
        raise ValueError("Frozen protocol hash mismatch; freeze a new run instead of editing this one")
    protocol = json.loads(raw)
    if sha256((directory / "manifest.json").read_bytes()) != protocol["manifest_sha256"]:
        raise ValueError("Frozen manifest hash mismatch")
    for case in protocol["cases"]:
        if case_id is not None and case["id"] != case_id:
            continue
        if "input_sha256" in case and sha256((directory / case["input"]).read_bytes()) != case["input_sha256"]:
            raise ValueError(f"Frozen input hash mismatch: {case['id']}")
    if package_identity()["sha256"] != protocol["package_identity_at_freeze"]["sha256"]:
        raise ValueError("Solver implementation changed since freeze; create a new frozen run")
    return protocol


def worker(mode, directory, case_id, repetition, proof_hash=None):
    protocol = check_frozen(directory, case_id)
    case = next(c for c in protocol["cases"] if c["id"] == case_id)
    stem = f"{case_id}-r{repetition}"
    proof_path = directory / "proofs" / f"{stem}.json"
    if mode == "produce":
        from highs_turbo.certification import certify
        value = json.loads((directory / case["input"]).read_bytes())
        began = time.perf_counter()
        bundle = certify(value["model"], value["answer"], time_limit=case["solver_seconds"], seed=case["seed"])
        api_seconds = time.perf_counter() - began
        write_json(proof_path, bundle)
        report = {"certify_api_seconds": api_seconds, "proof_sha256": sha256(proof_path.read_bytes())}
    else:
        from highs_turbo.certification import verify
        raw = proof_path.read_bytes()
        if sha256(raw) != proof_hash:
            raise ValueError("Produced proof hash mismatch before independent verification")
        began = time.perf_counter()
        report = verify(json.loads(raw))
        report["verify_api_seconds"] = time.perf_counter() - began
    identity = package_identity()
    if identity["sha256"] != protocol["package_identity_at_freeze"]["sha256"]:
        raise ValueError("Solver implementation changed during worker execution")
    report.update(peak_rss_bytes=peak_rss(), runtime_versions=runtime_versions(), package_identity=identity)
    write_json(directory / "trials" / f"{stem}-{mode}.json", report)


def child(mode, directory, case, repetition, timeout, proof_hash=None):
    command = [sys.executable, str(Path(__file__).resolve()), "worker", str(directory.resolve()),
               "--mode", mode, "--case", case["id"], "--repetition", str(repetition)]
    if proof_hash is not None:
        command.extend(["--proof-hash", proof_hash])
    if timeout <= 0:
        raise subprocess.TimeoutExpired(command, timeout)
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                   "VECLIB_MAXIMUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONHASHSEED": "0"}
    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout, env=environment)
    if completed.returncode:
        raise RuntimeError((completed.stderr or completed.stdout or f"Exit {completed.returncode}").strip()[-4000:])
    return json.loads((directory / "trials" / f"{case['id']}-r{repetition}-{mode}.json").read_text())


def trial(directory, case, repetition):
    began = time.perf_counter()
    record = {"case": case["id"], "repetition": repetition, "input_sha256": case.get("input_sha256"),
              "gap_target": case["gap_target"], "solver_seconds": case["solver_seconds"],
              "deadline_seconds": case["deadline_seconds"], "max_observed_rss_mib": case["max_observed_rss_mib"],
              "seed": case["seed"], "bound_verified": False, "quality_target_met": None,
              "producer_seconds": None, "verification_seconds": None,
              "failure_reasons": [], "proof": f"proofs/{case['id']}-r{repetition}.json"}
    phase, start = "produce", began
    try:
        if case.get("input_error"):
            record["failure_reasons"].append("input_error")
            record["error"] = case["input_error"]
        else:
            start = time.perf_counter()
            produced = child(phase, directory, case, repetition, case["deadline_seconds"] - (start - began))
            record.update(producer_seconds=time.perf_counter() - start, producer=produced,
                          peak_worker_rss_bytes=produced["peak_rss_bytes"])
            phase = "verify"
            start = time.perf_counter()
            verified = child(phase, directory, case, repetition,
                             case["deadline_seconds"] - (start - began), produced["proof_sha256"])
            record.update(verification_seconds=time.perf_counter() - start, verification=verified)
            if verified.get("bound_verified") is not True:
                raise ValueError("Verifier did not establish a checked bound")
            gap = Fraction(verified["gap"])
            if gap < 0:
                raise ValueError("Verifier returned a negative gap")
            record.update(bound_verified=True, quality_target_met=gap <= Fraction(case["gap_target"]),
                          candidate_objective=verified["candidate_objective"], bound=verified["bound"], gap=str(gap),
                          model_type=verified["model_type"], objective_sense=verified["objective_sense"],
                          optimality_proven=verified["optimality_proven"])
            if not record["quality_target_met"]:
                record["failure_reasons"].append("gap_target_missed")
            peaks = [part["peak_rss_bytes"] for part in (produced, verified) if part["peak_rss_bytes"] is not None]
            record["peak_worker_rss_bytes"] = max(peaks) if peaks else None
            if case["max_observed_rss_mib"] is not None:
                if len(peaks) < 2:
                    record["failure_reasons"].append("memory_measurement_unavailable")
                elif max(peaks) > case["max_observed_rss_mib"] * 2**20:
                    record["failure_reasons"].append("memory_budget_exceeded")
    except subprocess.TimeoutExpired:
        record["producer_seconds" if phase == "produce" else "verification_seconds"] = time.perf_counter() - start
        record["failure_reasons"].append("deadline_exceeded")
        record["error"] = f"End-to-end watchdog expired during {phase}; the worker was terminated"
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        record["producer_seconds" if phase == "produce" else "verification_seconds"] = time.perf_counter() - start
        record["failure_reasons"].append(f"{phase}_error")
        record["error"] = str(error)
    record["end_to_end_seconds"] = time.perf_counter() - began
    if record["end_to_end_seconds"] > case["deadline_seconds"] and "deadline_exceeded" not in record["failure_reasons"]:
        record["failure_reasons"].append("deadline_exceeded")
    record["outcome"] = "passed" if not record["failure_reasons"] else "failed"
    return record


def save_results(directory, evidence):
    write_json(directory / "results.json", evidence)
    lines = [f"# Pilot: {evidence.get('name', 'blocked run')}", "",
             f"Evidence type: **{evidence.get('evidence_type', 'untrusted protocol')}**.", "",
             "This checks the encoded objective and supplied answer. It does not establish business usefulness or validate application constraints.", ""]
    if evidence.get("fatal_error"):
        lines += ["Run blocked before starting solvers: " + evidence["fatal_error"], ""]
    for warning in evidence.get("warnings", []):
        lines += ["- " + warning]
    if evidence.get("warnings"):
        lines.append("")
    if evidence.get("application_constraints"):
        attestation = evidence["application_constraints"]
        lines += [f"Application-constraint declaration: `{attestation['status']}`. {attestation['notes']}", ""]
    lines += ["| Case | Repeat | Outcome | Checked gap / target | End-to-end seconds | Peak worker MiB | Failure reasons |",
              "|---|---:|---|---|---:|---:|---|"]
    for r in evidence["observations"]:
        peak = r.get("peak_worker_rss_bytes")
        memory = "unavailable" if peak is None else f"{peak / 2**20:.1f}"
        lines.append(f"| {r['case']} | {r['repetition']} | {r['outcome']} | {r.get('gap', 'unverified')} / {r['gap_target']} | "
                     f"{r['end_to_end_seconds']:.3f} | {memory} | {', '.join(r['failure_reasons']) or 'none'} |")
    passed = sum(r["outcome"] == "passed" for r in evidence["observations"])
    lines += ["", f"{passed}/{len(evidence['observations'])} observations met the checked-gap target and measured budgets.", "",
              "A target miss is retained as a failure; it does not prove the candidate is poor. A loose bound can fail to establish a good candidate's quality. "
              "Successful elapsed times are times to obtain a verified certificate satisfying the target with this fixed solver budget, not measured first-crossing times.", "",
              "End-to-end time includes fresh producer/verification process startup, input loading, proof serialization, and verification. "
              "Peak RSS is the maximum of the two sequential worker high-water marks; the optional memory budget is checked after execution, not enforced as an OS allocation limit. "
              "Raw results retain API and process timings, runtime versions, proof hashes, and all errors. Missing platform memory measurements remain unavailable.", ""]
    (directory / "report.md").write_text("\n".join(lines))


def run(directory):
    if (directory / "results.json").exists():
        raise ValueError("This run already has observations; freeze a new output directory to repeat it")
    evidence = {"started_at": datetime.now(timezone.utc).isoformat(), "observations": []}
    try:
        protocol = check_frozen(directory)
    except (OSError, ValueError, KeyError) as error:
        evidence["fatal_error"] = str(error)
        save_results(directory, evidence)
        return evidence
    evidence.update({key: protocol[key] for key in ("name", "evidence_type", "warnings", "application_constraints")})
    evidence.update(protocol_sha256=sha256((directory / "protocol.json").read_bytes()),
                    manifest_sha256=protocol["manifest_sha256"], runtime_versions_at_run=runtime_versions(),
                    runner_sha256_at_run=sha256(Path(__file__).read_bytes()), package_identity_at_run=package_identity())
    (directory / "proofs").mkdir()
    (directory / "trials").mkdir()
    save_results(directory, evidence)
    for repetition in range(1, protocol["repetitions"] + 1):
        cases = protocol["cases"]
        shift = (repetition - 1) % len(cases)
        for case in cases[shift:] + cases[:shift]:
            record = trial(directory, case, repetition)
            evidence["observations"].append(record)
            save_results(directory, evidence)
            print(json.dumps({key: record[key] for key in ("case", "repetition", "outcome", "failure_reasons")}), flush=True)
    evidence["completed_at"] = datetime.now(timezone.utc).isoformat()
    save_results(directory, evidence)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("freeze")
    p.add_argument("manifest", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("run")
    p.add_argument("directory", type=Path)
    p = sub.add_parser("worker", help=argparse.SUPPRESS)
    p.add_argument("directory", type=Path)
    p.add_argument("--mode", choices=("produce", "verify"), required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--repetition", type=int, required=True)
    p.add_argument("--proof-hash")
    args = parser.parse_args()
    if args.action == "freeze":
        protocol = freeze(args.manifest.resolve(), args.output.resolve())
        print(json.dumps({"directory": str(args.output.resolve()), "warnings": protocol["warnings"]}, indent=2))
    elif args.action == "run":
        result = run(args.directory.resolve())
        raise SystemExit(2 if result.get("fatal_error") else int(any(r["outcome"] != "passed" for r in result["observations"])))
    else:
        worker(args.mode, args.directory.resolve(), args.case, args.repetition, args.proof_hash)


if __name__ == "__main__":
    main()
