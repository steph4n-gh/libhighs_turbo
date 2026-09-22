"""Frozen inputs cannot drift, and failed pilot trials remain visible."""

import importlib.util
import json
from pathlib import Path

import pytest


KIT = Path(__file__).resolve().parents[1] / "pilots" / "run_pilot.py"
spec = importlib.util.spec_from_file_location("pilot_runner", KIT)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def manifest(tmp_path, cases):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({
        "version": 1, "name": "test-fixtures", "evidence_type": "synthetic_demo", "repetitions": 1,
        "defaults": {"solver_seconds": 1, "deadline_seconds": 20, "gap_target": "0"},
        "application_constraints": {"status": "not_applicable", "notes": "Synthetic mathematical test fixtures."},
        "cases": cases,
    }))
    return path


def write_input(tmp_path, name, answer):
    (tmp_path / name).write_text(json.dumps({"model": {"type": "ising", "fields": [1], "couplings": []},
                                           "answer": answer}))


@pytest.mark.parametrize("changed", ["protocol.json", "manifest.json", "inputs/case.json"])
def test_frozen_data_tampering_blocks_all_solves(tmp_path, monkeypatch, changed):
    write_input(tmp_path, "input.json", [-1])
    source = manifest(tmp_path, [{"id": "case", "input": "input.json"}])
    directory = tmp_path / "run"
    pilot.freeze(source, directory)
    (directory / changed).write_text((directory / changed).read_text() + " ")

    def forbidden(*args, **kwargs):
        raise AssertionError("No process may start after frozen data changes")

    monkeypatch.setattr(pilot, "child", forbidden)
    result = pilot.run(directory)
    assert "hash mismatch" in result["fatal_error"]
    assert result["observations"] == []
    assert "blocked before starting solvers" in (directory / "report.md").read_text()


def test_original_file_change_does_not_change_frozen_input(tmp_path):
    write_input(tmp_path, "input.json", [-1])
    source = manifest(tmp_path, [{"id": "case", "input": "input.json"}])
    directory = tmp_path / "run"
    protocol = pilot.freeze(source, directory)
    write_input(tmp_path, "input.json", [1])
    assert json.loads((directory / "inputs/case.json").read_text())["answer"] == [-1]
    assert pilot.check_frozen(directory)["cases"][0]["input_sha256"] == protocol["cases"][0]["input_sha256"]


def test_pilot_retains_target_miss_invalid_answer_and_deadline_then_continues(tmp_path):
    write_input(tmp_path, "good.json", [-1])
    write_input(tmp_path, "miss.json", [1])
    write_input(tmp_path, "invalid.json", [0])
    source = manifest(tmp_path, [
        {"id": "invalid", "input": "invalid.json"},
        {"id": "miss", "input": "miss.json"},
        {"id": "timeout", "input": "good.json", "deadline_seconds": 0.001},
        {"id": "memory", "input": "good.json", "max_observed_rss_mib": 0.0001},
        {"id": "missing", "input": "missing.json"},
        {"id": "good", "input": "good.json"},
    ])
    directory = tmp_path / "run"
    pilot.freeze(source, directory)
    result = pilot.run(directory)
    observations = {r["case"]: r for r in result["observations"]}
    assert len(observations) == 6
    assert observations["invalid"]["failure_reasons"] == ["produce_error"]
    assert observations["miss"]["failure_reasons"] == ["gap_target_missed"]
    assert observations["miss"]["bound_verified"] and observations["miss"]["gap"] == "2"
    assert "deadline_exceeded" in observations["timeout"]["failure_reasons"]
    assert not observations["timeout"]["bound_verified"]
    assert observations["memory"]["failure_reasons"] in (["memory_budget_exceeded"], ["memory_measurement_unavailable"])
    assert observations["memory"]["bound_verified"]
    assert observations["missing"]["failure_reasons"] == ["input_error"]
    good = observations["good"]
    assert good["outcome"] == "passed" and good["quality_target_met"]
    assert good["end_to_end_seconds"] >= good["producer_seconds"] + good["verification_seconds"]
    assert (directory / good["proof"]).is_file()
    assert good["producer"]["proof_sha256"] == pilot.sha256((directory / good["proof"]).read_bytes())
    assert "1/6 observations" in (directory / "report.md").read_text()
    with pytest.raises(ValueError, match="already has observations"):
        pilot.run(directory)


def test_proof_changed_between_processes_is_rejected(tmp_path):
    write_input(tmp_path, "input.json", [-1])
    source = manifest(tmp_path, [{"id": "case", "input": "input.json"}])
    directory = tmp_path / "run"
    protocol = pilot.freeze(source, directory)
    (directory / "proofs").mkdir()
    (directory / "trials").mkdir()
    produced = pilot.child("produce", directory, protocol["cases"][0], 1, 20)
    proof = directory / "proofs/case-r1.json"
    proof.write_text(proof.read_text() + " ")
    with pytest.raises(RuntimeError, match="proof hash mismatch"):
        pilot.child("verify", directory, protocol["cases"][0], 1, 20, produced["proof_sha256"])


def test_changed_solver_implementation_blocks_before_solving(tmp_path, monkeypatch):
    write_input(tmp_path, "input.json", [-1])
    source = manifest(tmp_path, [{"id": "case", "input": "input.json"}])
    directory = tmp_path / "run"
    pilot.freeze(source, directory)
    monkeypatch.setattr(pilot, "package_identity", lambda: {"sha256": "different-implementation"})
    monkeypatch.setattr(pilot, "child", lambda *args, **kwargs: pytest.fail("Must not start a changed solver"))
    result = pilot.run(directory)
    assert "implementation changed" in result["fatal_error"]
    assert not result["observations"]


def test_misspelled_target_setting_cannot_silently_use_default(tmp_path):
    write_input(tmp_path, "input.json", [-1])
    source = manifest(tmp_path, [{"id": "case", "input": "input.json", "gap_targte": "10"}])
    directory = tmp_path / "run"
    with pytest.raises(ValueError, match="unknown case settings"):
        pilot.freeze(source, directory)
    assert not directory.exists()
