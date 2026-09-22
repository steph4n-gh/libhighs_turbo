"""Original-objective oracles and portable external-answer certification."""

from copy import deepcopy
from fractions import Fraction
import itertools
import json
from pathlib import Path
import subprocess
import sys

import pytest

import highs_turbo.certification as certification
from highs_turbo.certification import certify, verify


MODELS = [
    {"type": "ising", "fields": ["1/3", -0.25, 0], "offset": "1/7",
     "couplings": [[0, 1, "2/3"], [1, 0, "1/3"], [1, 2, -2], [2, 2, "1/5"]]},
    {"type": "qubo", "num_variables": 3, "offset": "1/7",
     "coefficients": [[0, 0, "1/3"], [0, 1, -2], [1, 0, "-1/5"],
                      [0, 1, 0.1], [2, 2, -0.25]]},
    {"type": "maxcut", "num_nodes": 3,
     "edges": [[0, 1, "2/3"], [1, 0, "1/3"], [1, 2, -0.25], [0, 2, 2], [2, 2, 99]]},
]


def objective(model, answer):
    """Evaluate the original supplied terms directly, without Ising conversion."""
    value = Fraction(model.get("offset", 0))
    if model["type"] == "ising":
        value += sum((Fraction(h) * x for h, x in zip(model["fields"], answer)), Fraction())
        rows = model["couplings"]
    elif model["type"] == "qubo":
        rows = model["coefficients"]
    else:
        return sum((Fraction(w) for i, j, w in model["edges"] if answer[i] != answer[j]), Fraction())
    return value + sum((Fraction(w) * answer[i] * answer[j] for i, j, w in rows), Fraction())


@pytest.mark.parametrize("model", MODELS)
def test_exact_original_objective_and_bound_for_every_small_assignment(model, monkeypatch):
    original = deepcopy(model)
    domain = (-1, 1) if model["type"] == "ising" else (0, 1)
    answers = list(itertools.product(domain, repeat=3))
    bundle = certify(model, list(answers[0]), time_limit=1)
    bundle = json.loads(json.dumps(bundle, allow_nan=False))
    assert model == original
    monkeypatch.setattr(certification, "solve_ising", lambda *a, **k: pytest.fail("Verification must not solve"))
    values = [objective(model, answer) for answer in answers]
    optimum = max(values) if model["type"] == "maxcut" else min(values)
    for answer, expected in zip(answers, values):
        # A valid changed answer legitimately reuses the same mathematical bound.
        bundle["answer"] = list(answer)
        report = verify(bundle)
        assert Fraction(report["candidate_objective"]) == expected
        bound = Fraction(report["bound"])
        gap = Fraction(report["gap"])
        if model["type"] == "maxcut":
            assert bound >= optimum >= expected
            assert gap == bound - expected
        else:
            assert bound <= optimum <= expected
            assert gap == expected - bound
        assert report["bound_verified"] and report["optimality_proven"] == (gap == 0)


@pytest.mark.parametrize("model,answer,expected", [
    ({"type": "ising", "fields": ["1/3"], "offset": "1/7"}, [-1], "-4/21"),
    ({"type": "qubo", "num_variables": 1, "coefficients": [[0, 0, "-1/3"]], "offset": "1/7"}, [1], "-4/21"),
    ({"type": "maxcut", "num_nodes": 2, "edges": [[0, 1, "1/3"]]}, [0, 1], "1/3"),
    ({"type": "ising", "fields": []}, [], "0"),
    ({"type": "qubo", "num_variables": 0}, [], "0"),
    ({"type": "maxcut", "num_nodes": 3}, [0, 1, 0], "0"),
])
def test_exact_optimum_and_empty_models(model, answer, expected):
    report = verify(certify(model, answer, time_limit=1))
    assert report["candidate_objective"] == report["bound"] == expected
    assert report["optimality_proven"] and report["gap"] == "0"


@pytest.mark.parametrize("kind,rows", [("ising", "couplings"), ("qubo", "coefficients"), ("maxcut", "edges")])
def test_duplicate_float_terms_are_summed_exactly(kind, rows):
    model = {"type": kind, rows: [[0, 1, 1e16], [1, 0, 1], [0, 1, -1e16]]}
    model.update({"fields": [0, 0]} if kind == "ising" else {"num_variables" if kind == "qubo" else "num_nodes": 2})
    bundle = certify(model, [1, 1] if kind != "maxcut" else [0, 1], time_limit=0)
    assert bundle["model"][rows] == [[0, 1, "1"]]
    assert verify(bundle)["candidate_objective"] == "1"


def test_float_coefficients_preserve_binary64_and_inputs_are_copied():
    model = {"type": "ising", "fields": [0.1]}
    answer = [1]
    bundle = certify(model, answer, time_limit=0)
    model["fields"][0] = 999
    answer[0] = -1
    assert bundle["model"]["fields"] == [str(Fraction(0.1))]
    assert bundle["answer"] == [1]
    assert verify(bundle)["candidate_objective"] == str(Fraction(0.1))


def test_verified_bound_does_not_claim_a_suboptimal_answer_is_optimal():
    report = verify(certify({"type": "ising", "fields": [1]}, [1], time_limit=0))
    assert report == {"model_type": "ising", "objective_sense": "minimize",
                      "candidate_objective": "1", "bound": "-1", "gap": "2",
                      "bound_verified": True, "optimality_proven": False}


@pytest.mark.parametrize("model,rows", list(zip(MODELS, ["couplings", "coefficients", "edges"])))
def test_each_original_model_is_bound_to_its_certificate(model, rows):
    bundle = certify(model, [1, 1, 1], time_limit=0)
    bundle["model"][rows][0][2] = str(Fraction(bundle["model"][rows][0][2]) + 1)
    with pytest.raises(ValueError, match="Certificate does not verify"):
        verify(bundle)


@pytest.mark.parametrize("mutation", [
    lambda b: b.update(version=2),
    lambda b: b.update(version=True),
    lambda b: b["model"].update(offset="1"),
    lambda b: b["model"]["fields"].__setitem__(0, "7"),
    lambda b: b["answer"].__setitem__(0, 0),
    lambda b: b["answer"].__setitem__(0, True),
    lambda b: b["certificate"].update(problem_digest="0" * 64),
    lambda b: b["certificate"].update(lower_bound=[999, 1]),
    lambda b: b["certificate"].update(version=99),
    lambda b: b["certificate"].update(version=True),
    lambda b: b.update(certificate=None),
])
def test_rejects_modified_model_invalid_answer_and_invalid_witness(mutation):
    bundle = certify({"type": "ising", "fields": [1]}, [-1], time_limit=0)
    mutation(bundle)
    with pytest.raises(ValueError):
        verify(bundle)


@pytest.mark.parametrize("model,answer", [
    ({"type": "unknown"}, []),
    ({"type": "ising", "fields": [], "typo": 1}, []),
    ({"type": "ising", "fields": [True]}, [1]),
    ({"type": "ising", "fields": [float("nan")]}, [1]),
    ({"type": "ising", "fields": [float("inf")]}, [1]),
    ({"type": "ising", "fields": ["1/0"]}, [1]),
    ({"type": "ising", "fields": ["1e999999999999"]}, [1]),
    ({"type": "ising", "fields": [1 << 4096]}, [1]),
    ({"type": "ising", "fields": [0], "couplings": [[0, 1, 1]]}, [1]),
    ({"type": "qubo", "num_variables": 10**12}, []),
    ({"type": "qubo", "num_variables": True}, []),
    ({"type": "qubo", "num_variables": 1, "coefficients": [[False, 0, 1]]}, [1]),
    ({"type": "maxcut", "num_nodes": 1, "edges": [[-1, 0, 1]]}, [1]),
    ({"type": "maxcut", "num_nodes": 1, "edges": [[0, 0]]}, [1]),
    ({"type": "qubo", "num_variables": 1}, [2]),
    ({"type": "ising", "fields": [0]}, [1.0]),
])
def test_malformed_inputs_fail_before_solver_work(model, answer, monkeypatch):
    monkeypatch.setattr(certification, "solve_ising", lambda *a, **k: pytest.fail("Invalid input reached solver"))
    with pytest.raises(ValueError):
        certify(model, answer)


@pytest.mark.parametrize("options", [{"time_limit": -1}, {"time_limit": float("inf")},
                                    {"time_limit": True}, {"time_limit": 10**1000},
                                    {"seed": True}, {"seed": -1}, {"seed": 2**31}])
def test_invalid_solver_controls(options):
    with pytest.raises(ValueError):
        certify({"type": "ising", "fields": []}, [], **options)


@pytest.mark.parametrize("model,answer", [
    ({"type": "ising", "fields": [1 << 1024]}, [-1]),
    ({"type": "ising", "fields": [], "offset": 1 << 1024}, []),
    ({"type": "qubo", "num_variables": 1, "coefficients": [[0, 0, 1 << 1024]]}, [0]),
    ({"type": "maxcut", "num_nodes": 2, "edges": [[0, 1, 1 << 1024]]}, [0, 1]),
    ({"type": "ising", "fields": [1e308, -1e308]}, [-1, 1]),
])
def test_generation_rejects_binary64_objective_overflow_before_solving(model, answer, monkeypatch):
    monkeypatch.setattr(certification, "solve_ising", lambda *a, **k: pytest.fail("Oversized model reached solver"))
    with pytest.raises(ValueError, match="objective envelope.*finite binary64"):
        certify(model, answer, time_limit=0)


def test_verification_keeps_exact_range_beyond_binary64(monkeypatch):
    from highs_turbo.ising_cuts import make_certificate

    weight = Fraction(1 << 1024)
    witness = make_certificate([], [], [(0, 1)], {(0, 1): weight}, Fraction())
    bundle = {"version": 1, "model": {"type": "ising", "fields": [str(weight)]},
              "answer": [-1], "certificate": witness.to_dict()}
    monkeypatch.setattr(certification, "solve_ising", lambda *a, **k: pytest.fail("Verification must not solve"))
    report = verify(bundle)
    assert report["candidate_objective"] == report["bound"] == str(-weight)
    assert report["bound_verified"] and report["optimality_proven"]


@pytest.mark.parametrize("model", MODELS)
def test_cli_creates_and_replays_bundle_in_separate_processes(tmp_path, model):
    source, proof = tmp_path / "input.json", tmp_path / "proof.json"
    answer = [1, -1, 1] if model["type"] == "ising" else [0, 1, 0]
    source.write_text(json.dumps({"model": model, "answer": answer}))
    root = Path(__file__).resolve().parents[1]
    create = subprocess.run([sys.executable, "-m", "highs_turbo", "certify", "--input", str(source),
                             "--output", str(proof), "--seconds", "0"],
                            cwd=root, text=True, capture_output=True, timeout=30)
    assert create.returncode == 0, create.stderr
    replay = subprocess.run([sys.executable, "-m", "highs_turbo", "verify", str(proof)],
                            cwd=root, text=True, capture_output=True, timeout=30)
    assert replay.returncode == 0, replay.stderr
    assert json.loads(create.stdout) == json.loads(replay.stdout)
    assert json.loads(replay.stdout)["bound_verified"]
    bundle = json.loads(proof.read_text())
    bundle["certificate"]["problem_digest"] = "wrong"
    proof.write_text(json.dumps(bundle))
    invalid = subprocess.run([sys.executable, "-m", "highs_turbo", "verify", str(proof)],
                             cwd=root, text=True, capture_output=True, timeout=30)
    assert invalid.returncode != 0 and json.loads(invalid.stderr)["error"]


def test_cli_argument_errors_are_json():
    result = subprocess.run([sys.executable, "-m", "highs_turbo", "certify"],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode != 0
    assert "required" in json.loads(result.stderr)["error"]
