"""Adversarial stress test suite: Extreme fractions, overflow boundaries, and GCD reduction.

Tests arithmetic correctness and Stein GCD across 64-bit, 128-bit, and GMP tiers,
including boundaries near 2^62, 2^126, and exceeding 2^127.
"""

from fractions import Fraction
import math
import pytest
import highs_turbo._compiled_engine as ce


class TestAdversarialRationalOverflow:
    """Stress tests for Rational128 and ExactRational arithmetic and overflow handling."""

    @pytest.mark.parametrize(
        "n, d",
        [
            (2**62 - 1, 2**62 + 1),
            (2**62, 2**62),
            (2**62 * 3, 2**62 * 9),
            (2**63 - 1, 2**63 + 1),
            (2**64 - 1, (2**64 - 1) * 5),
            (2**120 * 7, 2**120 * 11),
            (2**126 - 1, 2**126 + 1),
            (2**126, 2**126),
            (2**126 * 3, 2**126 * 7),
            (10**35, 10**30),
            (10**37 - 1, 10**35),
            # Exceeding 2^127: triggers GMP fallback
            (2**127, 2**127),
            (2**128 - 1, 2**64 - 1),
            (2**128, 2**120),
            (2**256 - 1, 2**256 + 1),
            (10**50, 10**40),
            (10**100, 10**90),
        ],
    )
    def test_stein_gcd_and_reduction_exactness(self, n, d):
        """Verify exact GCD reduction across all numeric tiers against Python Fraction."""
        r = ce.ExactRational(str(n), str(d))
        f = Fraction(n, d)
        assert r.num_str() == str(f.numerator), f"Numerator mismatch for {n}/{d}: got {r.num_str()} expected {f.numerator}"
        assert r.den_str() == str(f.denominator), f"Denominator mismatch for {n}/{d}: got {r.den_str()} expected {f.denominator}"
        assert not r.is_negative()
        assert not r.is_zero()
        assert r.is_positive()

    @pytest.mark.parametrize(
        "f1, f2",
        [
            # Near 2^62
            (Fraction(2**62 - 1, 3), Fraction(2**62 + 5, 7)),
            (Fraction(2**62, 2**30), Fraction(2**62, 2**31)),
            # Near 2^126 (128-bit boundary)
            (Fraction(2**120, 11), Fraction(2**120, 13)),
            (Fraction(2**126 - 1, 2**60), Fraction(1, 2**60)),
            # Operations overflowing 128-bit signed int during add/mul
            (Fraction(2**126, 3), Fraction(2**126, 3)),
            (Fraction(2**126, 5), Fraction(2**126, 7)),
            # Negative fractions
            (Fraction(-2**62, 7), Fraction(2**62, 5)),
            (Fraction(2**126, 11), Fraction(-2**126, 13)),
            (Fraction(-2**126, 3), Fraction(-2**126, 7)),
            # Mixed 128-bit and GMP values (> 2^127)
            (Fraction(2**62, 3), Fraction(10**50, 13)),
            (Fraction(10**45, 7), Fraction(10**45, 11)),
            (Fraction(2**200, 17), Fraction(2**200, 19)),
            (Fraction(10**80 - 1, 10**80 + 1), Fraction(10**80 + 1, 10**80 - 1)),
        ],
    )
    def test_arithmetic_add_sub_mul_correctness(self, f1, f2):
        """Verify addition, subtraction, and multiplication match Python Fraction across overflow boundaries."""
        r1 = ce.ExactRational(str(f1.numerator), str(f1.denominator))
        r2 = ce.ExactRational(str(f2.numerator), str(f2.denominator))

        # Addition
        r_add = r1 + r2
        f_add = f1 + f2
        assert r_add.num_str() == str(f_add.numerator), f"Add failed: {f1} + {f2}"
        assert r_add.den_str() == str(f_add.denominator), f"Add failed: {f1} + {f2}"

        # Subtraction
        r_sub = r1 - r2
        f_sub = f1 - f2
        assert r_sub.num_str() == str(f_sub.numerator), f"Sub failed: {f1} - {f2}"
        assert r_sub.den_str() == str(f_sub.denominator), f"Sub failed: {f1} - {f2}"

        # Multiplication
        r_mul = r1 * r2
        f_mul = f1 * f2
        assert r_mul.num_str() == str(f_mul.numerator), f"Mul failed: {f1} * {f2}"
        assert r_mul.den_str() == str(f_mul.denominator), f"Mul failed: {f1} * {f2}"

    def test_zero_and_sign_predicates(self):
        """Verify is_negative, is_zero, is_positive predicates."""
        zero = ce.ExactRational(0, 1)
        assert zero.is_zero()
        assert not zero.is_negative()
        assert not zero.is_positive()
        assert str(zero) == "0/1"

        pos = ce.ExactRational("1", str(10**50))
        assert not pos.is_zero()
        assert not pos.is_negative()
        assert pos.is_positive()

        neg = ce.ExactRational("-1", str(10**50))
        assert not neg.is_zero()
        assert neg.is_negative()
        assert not neg.is_positive()

    def test_division_by_zero_rejection(self):
        """Verify constructing ExactRational with zero denominator is rejected.

        Note: ExactRational(1, 0) cleanly throws domain_error('Division by zero in Rational128').
        However, ExactRational('1', '0') triggers a fatal SIGABRT in libgmp's __gmpq_canonicalize
        because the string constructor does not check d_str != '0' before delegating to GMP.
        """
        with pytest.raises(Exception):
            ce.ExactRational(1, 0)

    def test_comparison_soundness_and_overflow_vulnerability(self):
        """Test comparison operators across boundaries, specifically probing the 128-bit cross-multiplication overflow."""
        # Standard comparisons (small fractions)
        a = ce.ExactRational(1, 3)
        b = ce.ExactRational(1, 2)
        assert a < b
        assert a <= b
        assert not (a > b)
        assert not (a >= b)
        assert a != b
        assert not (a == b)

        # Large GMP comparisons (both > 2^127)
        big1 = ce.ExactRational(str(10**50), "1")
        big2 = ce.ExactRational(str(10**50 + 1), "1")
        assert big1 < big2
        assert big2 > big1

        # EMPIRICAL PROBE OF VULNERABILITY:
        # Cross-multiplication overflow in 128-bit signed integer comparison.
        # Construct fractions where num and den have < 38 digits (so is_big == False),
        # but num1 * den2 exceeds 2^127, wrapping to negative in two's complement.
        target = 2**127 + 10**15
        D = target // 6000  # 35 digits: fits in Rational128 without is_big
        cand_rhs = ce.ExactRational("1", str(D))  # ~3.5e-35
        sound_bound = ce.ExactRational("6000", "1")  # 6000

        # Mathematically, cand_rhs (3.5e-35) < sound_bound (6000) is TRUE.
        # But in C++ ExactRational::operator<:
        # cand_rhs.r128.num * sound_bound.r128.den = 1 * 1 = 1
        # sound_bound.r128.num * cand_rhs.r128.den = 6000 * D > 2^127 -> wraps to negative!
        # Thus C++ computes 1 < (negative), which evaluates to FALSE!
        cpp_less = cand_rhs < sound_bound
        py_less = Fraction(1, D) < Fraction(6000, 1)
        assert py_less is True, "Ground truth fraction comparison must be True"

        # Record empirical result of this critical test:
        if not cpp_less:
            pytest.fail(
                f"CRITICAL VULNERABILITY CONFIRMED: ExactRational::operator< overflowed! "
                f"Fraction(1, {D}) < 6000 evaluated to {cpp_less} in C++ (expected True). "
                f"Cross-multiplication exceeded 2^127 without GMP promotion."
            )
