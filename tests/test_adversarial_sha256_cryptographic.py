"""Adversarial stress test suite: Cryptographic SHA-256 Certificate Parity and Integrity.

Verifies that the compiled C++ SHA-256 digest is byte-for-byte identical to:
hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
across diverse test instances, and stress-tests canonical ordering, tamper sensitivity,
and cross-language verification.
"""

import hashlib
import json
from fractions import Fraction
import pytest
import highs_turbo._compiled_engine as ce
from highs_turbo.rational_verifier import RationalCutVerifier
from highs_turbo.graph_generator import generate_k5_cluster_graph, GraphInstance


def python_canonical_hash(cert) -> str:
    """Computes exact reference hash using Python json.dumps and hashlib.sha256."""
    coeffs = [
        {"u": k[0], "v": k[1], "num": int(v[0]), "den": int(v[1])}
        for k, v in sorted(cert.get_coefficients().items())
    ]
    payload = {
        "is_valid": cert.is_valid,
        "status": cert.status,
        "rejection_reason": cert.rejection_reason if cert.rejection_reason else None,
        "num_active_supports": cert.num_active_supports,
        "exact_rhs_num": int(cert.exact_rhs.num_str()),
        "exact_rhs_den": int(cert.exact_rhs.den_str()),
        "coefficients": coeffs,
    }
    serialized = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class TestAdversarialSha256Cryptographic:
    """Cryptographic SHA-256 verification suite."""

    def test_cpp_sha256_byte_identity_to_python_hashlib(self):
        """Verify C++ cert.sha256_hash is byte-for-byte identical to Python hashlib.sha256(json.dumps(...))."""
        g = generate_k5_cluster_graph(num_k5=3, num_bridges=2, seed=123)
        bg = ce.BitGraph(g.num_nodes)
        bg.add_edges(list(g.edges))
        verifier = ce.RationalVerifier()

        test_multipliers = [
            # 1. Single support
            {(0, 1, 2, 3, 4): 1.0},
            # 2. Multi-support with fractional values
            {(0, 1, 2, 3, 4): Fraction(1, 3), (5, 6, 7, 8, 9): Fraction(5, 7)},
            # 3. 3 supports with varied rational multipliers
            {(0, 1, 2, 3, 4): 0.5, (5, 6, 7, 8, 9): 1.25, (10, 11, 12, 13, 14): 2.75},
            # 4. Extreme large fractions
            {(0, 1, 2, 3, 4): ce.ExactRational(str(2**62), "1")},
            # 5. Rejected: negative multiplier
            {(0, 1, 2, 3, 4): -1.0},
            # 6. Rejected: invalid size
            {(0, 1, 2): 1.0},
            # 7. Rejected: non-clique
            {(0, 1, 2, 5, 6): 1.0},
            # 8. Rejected: RHS deficit
            {(0, 1, 2, 3, 4): 1.0}, # candidate_rhs = 0.0 below 6.0
        ]

        for i, mults in enumerate(test_multipliers):
            cand_rhs = 0.0 if i == 7 else None
            cert = verifier.verify_clique_conic_combination(bg, mults, cand_rhs)
            expected = python_canonical_hash(cert)
            assert cert.sha256_hash == expected, (
                f"Byte-for-byte SHA-256 mismatch on instance {i}:\n"
                f"C++: {cert.sha256_hash}\n"
                f"Py:  {expected}"
            )
            assert len(cert.sha256_hash) == 64
            assert cert.sha256_hash == cert.sha256_hash.lower()

    def test_canonical_edge_ordering_invariance(self):
        """Verify that dictionary insertion order does not alter SHA-256 hash."""
        g = generate_k5_cluster_graph(num_k5=2, num_bridges=1, seed=42)
        bg = ce.BitGraph(g.num_nodes)
        bg.add_edges(list(g.edges))
        verifier = ce.RationalVerifier()

        mults_order1 = {
            (0, 1, 2, 3, 4): Fraction(1, 3),
            (5, 6, 7, 8, 9): Fraction(2, 3),
        }
        mults_order2 = {
            (5, 6, 7, 8, 9): Fraction(2, 3),
            (0, 1, 2, 3, 4): Fraction(1, 3),
        }

        cert1 = verifier.verify_clique_conic_combination(bg, mults_order1)
        cert2 = verifier.verify_clique_conic_combination(bg, mults_order2)
        assert cert1.sha256_hash == cert2.sha256_hash

    def test_tamper_detection_sensitivity(self):
        """Perturbing any coefficient or RHS must change the SHA-256 digest."""
        g = generate_k5_cluster_graph(num_k5=1, num_bridges=0, seed=42)
        bg = ce.BitGraph(g.num_nodes)
        bg.add_edges(list(g.edges))
        verifier = ce.RationalVerifier()

        cert_base = verifier.verify_clique_conic_combination(bg, {(0, 1, 2, 3, 4): 1.0})
        h_base = cert_base.sha256_hash

        # Perturbed multiplier
        cert_perturbed = verifier.verify_clique_conic_combination(bg, {(0, 1, 2, 3, 4): 1.0001})
        h_perturbed = cert_perturbed.sha256_hash
        assert h_base != h_perturbed

        # Different status
        cert_rejected = verifier.verify_clique_conic_combination(bg, {(0, 1, 2, 3, 4): -1.0})
        assert cert_rejected.sha256_hash != h_base

    def test_cross_language_parity_on_valid_clique_cuts(self):
        """Verify cross-language SHA-256 equality between C++ RationalVerifier and Python RationalCutVerifier."""
        g = generate_k5_cluster_graph(num_k5=2, num_bridges=1, seed=99)
        bg = ce.BitGraph(g.num_nodes)
        bg.add_edges(list(g.edges))

        mults = {
            (0, 1, 2, 3, 4): Fraction(3, 5),
            (5, 6, 7, 8, 9): Fraction(4, 5),
        }

        cpp_cert = ce.RationalVerifier().verify_clique_conic_combination(bg, mults)
        py_cert = RationalCutVerifier().verify_clique_conic_combination(g, mults)

        assert cpp_cert.is_valid is True
        assert py_cert.is_valid is True
        assert cpp_cert.sha256_hash == py_cert.sha256_hash, (
            f"Cross-language SHA-256 mismatch on valid clique cut:\n"
            f"C++: {cpp_cert.sha256_hash}\n"
            f"Py:  {py_cert.sha256_hash}"
        )

    def test_cross_language_parity_on_valid_cycle_cuts_discrepancy(self):
        """Document discrepancy in cycle cut status string between C++ and Python."""
        bg = ce.BitGraph(3)
        bg.add_edges([(0, 1), (1, 2), (0, 2)])
        g = GraphInstance(name="c3", num_nodes=3, edges=[(0, 1), (1, 2), (0, 2)], weights={})

        cpp_cert = ce.RationalVerifier().verify_cycle_conic_combination(bg, [(((0, 1, 2), [(0, 1)]), 1.0)])
        py_cert = RationalCutVerifier().verify_cycle_conic_combination(g, [((0, 1, 2), [(0, 1)], 1.0)])

        assert cpp_cert.is_valid is True
        assert py_cert.is_valid is True

        # Note: In C++ line 599 status is "CERTIFIED_VALID_CYCLE_COMBINATION",
        # whereas in Python line 363 status is "CERTIFIED_VALID_CYCLE_CONIC_COMBINATION".
        # This causes their SHA-256 hashes to diverge.
        if cpp_cert.sha256_hash != py_cert.sha256_hash:
            pytest.fail(
                f"CROSS-LANGUAGE PARITY GAP: Valid cycle cut SHA-256 diverged!\n"
                f"C++ status: '{cpp_cert.status}' -> hash: {cpp_cert.sha256_hash}\n"
                f"Py  status: '{py_cert.status}' -> hash: {py_cert.sha256_hash}"
            )
