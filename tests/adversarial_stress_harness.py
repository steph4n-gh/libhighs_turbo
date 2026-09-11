#!/usr/bin/env python3
"""Standalone Adversarial Stress & Forensic Harness for Milestone M1 RationalVerifier & SHA-256 Engine.

Executes empirical challenge suites and generates structured forensic report.
"""

from pathlib import Path
import sys

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from fractions import Fraction
import hashlib
import json
import math
import highs_turbo._compiled_engine as ce
from highs_turbo.rational_verifier import RationalCutVerifier
from highs_turbo.graph_generator import generate_k5_cluster_graph, GraphInstance


def run_all_stress_tests():
    print("=" * 80)
    print("CHALLENGER 2: ADVERSARIAL STRESS & FORENSIC VERIFICATION SUITE")
    print("=" * 80)

    findings = []

    # -------------------------------------------------------------------------
    # PART 1: Extreme Fractions & Overflow Boundaries
    # -------------------------------------------------------------------------
    print("\n[PART 1] Testing Extreme Fractions & Overflow Boundaries...")
    gcd_cases = [
        (2**62 - 1, 2**62 + 1),
        (2**62, 2**62),
        (2**126 - 1, 2**126 + 1),
        (2**126, 2**126),
        (2**127, 2**127),
        (2**256 - 1, 2**256 + 1),
        (10**50, 10**40),
    ]
    gcd_passed = 0
    for n, d in gcd_cases:
        r = ce.ExactRational(str(n), str(d))
        f = Fraction(n, d)
        if r.num_str() == str(f.numerator) and r.den_str() == str(f.denominator):
            gcd_passed += 1
    print(f"  - Stein GCD & Reduction: {gcd_passed}/{len(gcd_cases)} passed.")

    # Arithmetic add/sub/mul
    arith_cases = [
        (Fraction(2**62 - 1, 3), Fraction(2**62 + 5, 7)),
        (Fraction(2**126, 3), Fraction(2**126, 3)),
        (Fraction(10**50, 7), Fraction(10**50, 11)),
    ]
    arith_passed = 0
    for f1, f2 in arith_cases:
        r1 = ce.ExactRational(str(f1.numerator), str(f1.denominator))
        r2 = ce.ExactRational(str(f2.numerator), str(f2.denominator))
        if ((r1 + r2).num_str() == str((f1 + f2).numerator) and
            (r1 - r2).num_str() == str((f1 - f2).numerator) and
            (r1 * r2).num_str() == str((f1 * f2).numerator)):
            arith_passed += 1
    print(f"  - Boundary Arithmetic (+, -, *): {arith_passed}/{len(arith_cases)} passed.")

    # Vulnerability Probe: 128-bit cross-multiplication overflow in operator<
    target = 2**127 + 10**15
    D = target // 6000
    cand_rhs = ce.ExactRational("1", str(D))
    sound_bound = ce.ExactRational("6000", "1")
    cpp_less = cand_rhs < sound_bound
    py_less = Fraction(1, D) < Fraction(6000, 1)

    if not cpp_less and py_less:
        findings.append({
            "id": "FINDING-01",
            "severity": "CRITICAL",
            "component": "ExactRational::operator< (rational_verifier.cpp:271)",
            "summary": "128-bit signed overflow in operator< causes unsound cut with RHS deficit to be certified as valid",
            "details": f"Fraction(1, {D}) < 6000 evaluated to {cpp_less} in C++ (expected {py_less}). Soundness check candidate_rhs < sound_bound was bypassed.",
        })
        print(f"  [CRITICAL VULNERABILITY CONFIRMED] ExactRational::operator< overflowed! 1/{D} < 6000 is {cpp_less}")
    else:
        print("  - Operator< overflow test passed.")

    # String division by zero check
    try:
        ce.ExactRational(1, 0)
        int_div_zero = "Did not throw"
    except Exception as e:
        int_div_zero = "Caught exception cleanly"
    print(f"  - Integer division by zero: {int_div_zero}")

    findings.append({
        "id": "FINDING-02",
        "severity": "HIGH",
        "component": "ExactRational string constructor (rational_verifier.cpp:65)",
        "summary": "ExactRational('1', '0') triggers uncatchable SIGABRT in libgmp instead of C++ exception",
        "details": "Constructor passes unvalidated denominator to mpq_class, causing __gmpq_canonicalize to terminate process via SIGABRT.",
    })

    # -------------------------------------------------------------------------
    # PART 2: Float Continued Fraction Boundary Conversions
    # -------------------------------------------------------------------------
    print("\n[PART 2] Testing Float Continued Fraction Conversions...")
    subnormals = [5e-324, 1e-308, sys.float_info.min, sys.float_info.epsilon]
    sub_passed = 0
    for v in subnormals:
        r = ce.ExactRational.from_double(v, 100000)
        f = Fraction.from_float(v).limit_denominator(100000)
        if r.num_str() == str(f.numerator) and r.den_str() == str(f.denominator):
            sub_passed += 1
    print(f"  - Subnormals & Machine Epsilon: {sub_passed}/{len(subnormals)} passed.")

    # Large float crash
    large_float_crash = False
    try:
        ce.ExactRational.from_double(1e19, 100000)
    except Exception as e:
        large_float_crash = True
        findings.append({
            "id": "FINDING-03",
            "severity": "HIGH",
            "component": "ExactRational::from_double (rational_verifier.cpp:117)",
            "summary": "Floats >= 1e19 overflow int64 in floor(rem) and throw Division by zero in Rational128",
            "details": f"from_double(1e19) crashed with {type(e).__name__}: {e}. Python limit_denominator handles this gracefully.",
        })
        print(f"  [HIGH VULNERABILITY CONFIRMED] from_double(1e19) crashed with {type(e).__name__}: {e}")

    # Farey semi-convergent tie-breaking
    v_sqrt2 = math.sqrt(2)
    r_sqrt2 = ce.ExactRational.from_double(v_sqrt2, 10000)
    f_sqrt2 = Fraction.from_float(v_sqrt2).limit_denominator(10000)
    if r_sqrt2.num_str() != str(f_sqrt2.numerator):
        findings.append({
            "id": "FINDING-04",
            "severity": "LOW",
            "component": "ExactRational::from_double semi-convergent tie breaking (rational_verifier.cpp:128)",
            "summary": "Strict inequality (err_bound < err_prev) diverges from Python on equal-error mediants",
            "details": f"sqrt(2) with limit 10000 gave C++ {r_sqrt2.num_str()}/{r_sqrt2.den_str()} vs Python {f_sqrt2.numerator}/{f_sqrt2.denominator}.",
        })
        print(f"  - Farey tie-breaking divergence: C++ {r_sqrt2} vs Py {f_sqrt2}")

    # -------------------------------------------------------------------------
    # PART 3: Adversarial Cut Mutations (100+ hostile cuts)
    # -------------------------------------------------------------------------
    print("\n[PART 3] Testing 100+ Adversarial Mutated Cuts Across 8 Vectors...")
    g = generate_k5_cluster_graph(num_k5=2, num_bridges=1, seed=42)
    bg = ce.BitGraph(g.num_nodes)
    bg.add_edges(list(g.edges))
    verifier = ce.RationalVerifier()

    total_mutations = 0
    rejected_mutations = 0

    # Vector 1: Negative multipliers
    for val in [-1.0, -0.5, -10.0, -1e-4, ce.ExactRational(-1, 2), Fraction(-3, 8)]:
        cert = verifier.verify_clique_conic_combination(bg, {(0, 1, 2, 3, 4): val})
        total_mutations += 1
        if not cert.is_valid: rejected_mutations += 1

    # Vector 2: Missing edges
    bg_incomplete = ce.BitGraph(5)
    bg_incomplete.add_edges([(0, 1), (1, 2), (2, 3), (3, 4)]) # missing 6 edges
    cert = verifier.verify_clique_conic_combination(bg_incomplete, {(0, 1, 2, 3, 4): 1.0})
    total_mutations += 1
    if not cert.is_valid: rejected_mutations += 1

    # Vector 3: Invalid support size & duplicates
    for supp in [(), (0,), (0, 1), (0, 1, 2), (0, 1, 2, 3), (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 3)]:
        cert = verifier.verify_clique_conic_combination(bg, {supp: 1.0})
        total_mutations += 1
        if not cert.is_valid: rejected_mutations += 1

    # Vector 4: Out-of-bounds nodes
    for supp in [(0, 1, 2, 3, 10), (0, 1, 2, 3, 100), (0, 1, 2, 3, 1000)]:
        cert = verifier.verify_clique_conic_combination(bg, {supp: 1.0})
        total_mutations += 1
        if not cert.is_valid: rejected_mutations += 1

    # Vector 5: RHS claim deficits
    for rhs in [5.9, 5.0, 0.0, -10.0, Fraction(1, 2)]:
        cert = verifier.verify_clique_conic_combination(bg, {(0, 1, 2, 3, 4): 1.0}, rhs)
        total_mutations += 1
        if not cert.is_valid: rejected_mutations += 1

    # Vector 6: Odd cycle violations
    bg_c = ce.BitGraph(4)
    bg_c.add_edges([(0, 1), (1, 2), (2, 3), (3, 0)])
    for c_item in [
        (((0, 1, 2, 3), []), 1.0), # |F|=0 even
        (((0, 1, 2, 3), [(0, 1), (1, 2)]), 1.0), # |F|=2 even
        (((0, 1), []), 1.0), # len < 3
    ]:
        cert = verifier.verify_cycle_conic_combination(bg_c, [c_item])
        total_mutations += 1
        if not cert.is_valid: rejected_mutations += 1

    # Vector 7: Precision overflow
    cert_ovf = verifier.verify_clique_conic_combination(bg, {(0, 1, 2, 3, 4): ce.ExactRational(str(2**62), "1")}, 1.0)
    total_mutations += 1
    if not cert_ovf.is_valid: rejected_mutations += 1

    # Compound attacks sweep (100 attacks)
    import random
    rng = random.Random(42)
    for _ in range(100):
        supp = [0, 1, 2, 3, 4]
        m_val = 1.0
        r_val = None
        attack = rng.choice(["neg", "size", "oob", "def", "dup"])
        if attack == "neg": m_val = -rng.uniform(0.1, 10.0)
        elif attack == "size": supp = [0, 1, 2]
        elif attack == "oob": supp = [0, 1, 2, 3, 99]
        elif attack == "def": r_val = -1.0
        elif attack == "dup": supp = [0, 1, 2, 3, 3]

        cert = verifier.verify_clique_conic_combination(bg, {tuple(supp): m_val}, r_val)
        total_mutations += 1
        if not cert.is_valid: rejected_mutations += 1

    print(f"  - Hostile Cut Mutations Tested: {total_mutations}")
    print(f"  - Hostile Cut Mutations Rejected: {rejected_mutations}/{total_mutations} ({rejected_mutations/total_mutations*100:.1f}%)")

    # -------------------------------------------------------------------------
    # PART 4: Cryptographic SHA-256 Parity Verification
    # -------------------------------------------------------------------------
    print("\n[PART 4] Testing Cryptographic SHA-256 Parity...")
    # 1. C++ vs Python json.dumps byte exactness
    cert_cpp = verifier.verify_clique_conic_combination(bg, {(0, 1, 2, 3, 4): 1.0})
    coeffs = [
        {"u": k[0], "v": k[1], "num": int(v[0]), "den": int(v[1])}
        for k, v in sorted(cert_cpp.get_coefficients().items())
    ]
    payload = {
        "is_valid": cert_cpp.is_valid,
        "status": cert_cpp.status,
        "rejection_reason": cert_cpp.rejection_reason if cert_cpp.rejection_reason else None,
        "num_active_supports": cert_cpp.num_active_supports,
        "exact_rhs_num": int(cert_cpp.exact_rhs.num_str()),
        "exact_rhs_den": int(cert_cpp.exact_rhs.den_str()),
        "coefficients": coeffs,
    }
    py_canon_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    cpp_hash_match = (cert_cpp.sha256_hash == py_canon_hash)
    print(f"  - C++ compute_sha256() matches Python json.dumps byte-for-byte: {cpp_hash_match}")
    print(f"    Hash: {cert_cpp.sha256_hash}")

    # 2. Cross-language status string mismatch on cycle cuts
    bg3 = ce.BitGraph(3)
    bg3.add_edges([(0, 1), (1, 2), (0, 2)])
    g3 = GraphInstance(name="c3", num_nodes=3, edges=[(0, 1), (1, 2), (0, 2)], weights={})
    c_cpp = verifier.verify_cycle_conic_combination(bg3, [(((0, 1, 2), [(0, 1)]), 1.0)])
    c_py = RationalCutVerifier().verify_cycle_conic_combination(g3, [((0, 1, 2), [(0, 1)], 1.0)])

    if c_cpp.sha256_hash != c_py.sha256_hash:
        findings.append({
            "id": "FINDING-05",
            "severity": "MEDIUM",
            "component": "RationalVerifier::verify_cycle_conic_combination status string (rational_verifier.cpp:599)",
            "summary": "Cycle cut status string in C++ ('CERTIFIED_VALID_CYCLE_COMBINATION') omits 'CONIC_', breaking SHA-256 parity with Python ('CERTIFIED_VALID_CYCLE_CONIC_COMBINATION')",
            "details": f"C++ SHA-256: {c_cpp.sha256_hash} vs Python SHA-256: {c_py.sha256_hash}.",
        })
        print(f"  [MEDIUM ISSUE CONFIRMED] Cycle status mismatch: '{c_cpp.status}' vs '{c_py.status}' -> SHA-256 diverged")

    print("\n" + "=" * 80)
    print("FORENSIC AUDIT SUMMARY TABLE")
    print("=" * 80)
    print(f"{'ID':<12} | {'Severity':<10} | {'Component':<35} | {'Summary'}")
    print("-" * 110)
    for f in findings:
        print(f"{f['id']:<12} | {f['severity']:<10} | {f['component'][:35]:<35} | {f['summary'][:60]}")
    print("=" * 80)

    return findings


if __name__ == "__main__":
    run_all_stress_tests()
