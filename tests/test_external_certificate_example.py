"""The documented external-answer workflow exports a portable, model-bound proof."""

import json
from pathlib import Path
import subprocess
import sys


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "certify_external_answer.py"


def test_external_answer_export_and_independent_verification(tmp_path):
    artifact = tmp_path / "answer.json"
    produced = subprocess.run([sys.executable, str(EXAMPLE), "--output", str(artifact)],
                              check=True, capture_output=True, text=True, timeout=30)
    report = json.loads(produced.stdout)
    assert report["verified"] and report["independent_process"]
    assert report["external_energy"] == "-1/2"
    bundle = json.loads(artifact.read_text())
    bundle["fields"][0] += 1
    artifact.write_text(json.dumps(bundle))
    replay = subprocess.run([sys.executable, str(EXAMPLE), "--verify", str(artifact)],
                            capture_output=True, text=True, timeout=30)
    assert replay.returncode != 0
    assert "Certificate does not verify" in replay.stderr
    bundle["spins"][0] = 0
    artifact.write_text(json.dumps(bundle))
    invalid = subprocess.run([sys.executable, str(EXAMPLE), "--verify", str(artifact)],
                             capture_output=True, text=True, timeout=30)
    assert invalid.returncode != 0
    assert "one integer -1 or +1" in invalid.stderr


def test_external_answer_rational_coefficients_preserve_exact_model(tmp_path):
    source = tmp_path / "answer-input.json"
    source.write_text(json.dumps({"fields": ["1/3"], "couplings": [], "spins": [-1], "offset": "1/7"}))
    produced = subprocess.run([sys.executable, str(EXAMPLE), "--input", str(source),
                               "--output", str(tmp_path / "answer-proof.json")],
                              check=True, capture_output=True, text=True, timeout=30)
    report = json.loads(produced.stdout)
    assert report["verified"] and report["independent_process"]
    assert report["external_energy"] == "-4/21"
    assert report["exact_gap"] == "0"
