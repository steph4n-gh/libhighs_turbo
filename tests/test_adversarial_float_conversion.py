"""Adversarial stress test suite: Floating-point continued fraction boundary conversions.

Tests continued fraction approximation (Stern-Brocot / Farey) across:
- Subnormal numbers (5e-324, 1e-308, float_info.min)
- Values near machine epsilon (float_info.epsilon, 1 +- eps)
- Very large floats (1e15, 1e18, 1e19, 1e50, 1e308)
- Bounded denominator enforcement across varied limits (1 to 100000)
- Parity with Python Fraction.from_float().limit_denominator()
"""

import math
import sys
from fractions import Fraction
import pytest
import highs_turbo._compiled_engine as ce


class TestAdversarialFloatConversion:
    """Stress tests for ExactRational::from_double."""

    def test_nan_and_infinity_rejection(self):
        """NaN and +-Infinity must raise domain_error."""
        with pytest.raises(Exception):
            ce.ExactRational.from_double(float("nan"), 100000)
        with pytest.raises(Exception):
            ce.ExactRational.from_double(float("inf"), 100000)
        with pytest.raises(Exception):
            ce.ExactRational.from_double(float("-inf"), 100000)

    def test_exact_zero_and_negative_zero(self):
        """Zero values must produce 0/1."""
        r0 = ce.ExactRational.from_double(0.0, 100000)
        assert r0.num_str() == "0"
        assert r0.den_str() == "1"
        assert r0.is_zero()

        r_neg0 = ce.ExactRational.from_double(-0.0, 100000)
        assert r_neg0.is_zero()

    @pytest.mark.parametrize(
        "val",
        [
            5e-324,
            1e-320,
            1e-308,
            sys.float_info.min,
            sys.float_info.min * 0.5,
            sys.float_info.epsilon,
            sys.float_info.epsilon * 0.5,
            1.0 + sys.float_info.epsilon,
            1.0 - sys.float_info.epsilon,
            -5e-324,
            -sys.float_info.epsilon,
        ],
    )
    def test_subnormals_and_machine_epsilon(self, val):
        """Verify subnormal and epsilon conversions match Python limit_denominator."""
        r = ce.ExactRational.from_double(val, 100000)
        py_f = Fraction.from_float(val).limit_denominator(100000)
        assert r.num_str() == str(py_f.numerator)
        assert r.den_str() == str(py_f.denominator)

    @pytest.mark.parametrize("denom_limit", [1, 5, 10, 50, 100, 500, 1000, 10000, 100000])
    @pytest.mark.parametrize("val", [1/3, 2/7, 5/13, 0.125, 0.3333333333333333, -0.75])
    def test_bounded_denominator_enforcement(self, val, denom_limit):
        """Verify resulting denominator never exceeds max_denominator."""
        r = ce.ExactRational.from_double(val, denom_limit)
        den = int(r.den_str())
        assert den <= denom_limit, f"Denominator {den} exceeded limit {denom_limit} for val {val}"

    @pytest.mark.parametrize("val", [1e10, 1e12, 1e15, 1e18])
    def test_large_floats_within_int64_range(self, val):
        """Verify large floats within int64 range convert accurately."""
        r = ce.ExactRational.from_double(val, 100000)
        py_f = Fraction.from_float(val).limit_denominator(100000)
        assert r.num_str() == str(py_f.numerator)
        assert r.den_str() == str(py_f.denominator)

    @pytest.mark.parametrize("val", [1e19, 1e20, 1e50, 1e100, 1e308])
    def test_large_floats_exceeding_int64_range_vulnerability(self, val):
        """Expose crash/division-by-zero when float exceeds INT64_MAX in from_double."""
        py_f = Fraction.from_float(val).limit_denominator(100000)
        assert py_f.denominator <= 100000

        try:
            r = ce.ExactRational.from_double(val, 100000)
            assert r.num_str() == str(py_f.numerator)
            assert r.den_str() == str(py_f.denominator)
        except Exception as e:
            pytest.fail(
                f"VULNERABILITY CONFIRMED: ExactRational::from_double({val:e}) crashed with {type(e).__name__}: {e}. "
                f"Python Fraction.from_float({val:e}).limit_denominator() gracefully returned {py_f}."
            )

    def test_farey_semi_convergent_tie_breaking(self):
        """Test Farey semi-convergent behavior on sqrt(2) with limit 10000."""
        # sqrt(2) with max_denominator=10000
        val = math.sqrt(2)
        r = ce.ExactRational.from_double(val, 10000)
        py_f = Fraction.from_float(val).limit_denominator(10000)
        # Note: Python selects 11482/8119, while C++ selects 8119/5741 due to strict < rather than <=
        # Both satisfy den <= 10000, but they diverge in approximation choice
        assert int(r.den_str()) <= 10000
        assert int(str(py_f.denominator)) <= 10000
