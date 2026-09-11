#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <utility>
#include <memory>
#include <optional>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <iomanip>

#include <gmp.h>
#include <gmpxx.h>
#include <CommonCrypto/CommonDigest.h>

#include "bit_graph.hpp"

namespace highs_turbo {

// Fast 128-bit Binary GCD using count-trailing-zeros
static inline unsigned __int128 stein_gcd128(unsigned __int128 u, unsigned __int128 v) noexcept {
    if (u == 0) return v;
    if (v == 0) return u;

    auto ctz128 = [](unsigned __int128 x) -> int {
        uint64_t low = static_cast<uint64_t>(x);
        if (low != 0) return __builtin_ctzll(low);
        uint64_t high = static_cast<uint64_t>(x >> 64);
        return 64 + __builtin_ctzll(high);
    };

    int shift = ctz128(u | v);
    u >>= ctz128(u);
    do {
        v >>= ctz128(v);
        if (u > v) std::swap(u, v);
        v -= u;
    } while (v != 0);
    return u << shift;
}

// Fixed-width 128-bit rational
struct Rational128 {
    __int128_t num;
    __int128_t den; // Invariant: den > 0, gcd(|num|, den) == 1

    Rational128() : num(0), den(1) {}
    Rational128(__int128_t n, __int128_t d = 1) {
        if (d == 0) throw std::domain_error("Division by zero in Rational128");
        if (d < 0) { n = -n; d = -d; }
        unsigned __int128 un = (n < 0) ? -static_cast<unsigned __int128>(n) : static_cast<unsigned __int128>(n);
        unsigned __int128 g = stein_gcd128(un, static_cast<unsigned __int128>(d));
        num = n / static_cast<__int128_t>(g);
        den = d / static_cast<__int128_t>(g);
    }

    static bool add_checked(const Rational128& r1, const Rational128& r2, Rational128& out) noexcept {
        unsigned __int128 g = stein_gcd128(static_cast<unsigned __int128>(r1.den),
                                           static_cast<unsigned __int128>(r2.den));
        __int128_t d_g = r2.den / static_cast<__int128_t>(g);
        __int128_t b_g = r1.den / static_cast<__int128_t>(g);

        __int128_t term1, term2, new_num, new_den;
        if (__builtin_mul_overflow(r1.num, d_g, &term1)) return false;
        if (__builtin_mul_overflow(r2.num, b_g, &term2)) return false;
        if (__builtin_add_overflow(term1, term2, &new_num)) return false;
        if (__builtin_mul_overflow(b_g, r2.den, &new_den)) return false;

        unsigned __int128 un = (new_num < 0) ? -static_cast<unsigned __int128>(new_num) : static_cast<unsigned __int128>(new_num);
        unsigned __int128 g2 = stein_gcd128(un, static_cast<unsigned __int128>(new_den));
        out.num = new_num / static_cast<__int128_t>(g2);
        out.den = new_den / static_cast<__int128_t>(g2);
        return true;
    }

    static bool sub_checked(const Rational128& r1, const Rational128& r2, Rational128& out) noexcept {
        Rational128 neg_r2(-r2.num, r2.den);
        return add_checked(r1, neg_r2, out);
    }

    static bool mul_checked(const Rational128& r1, const Rational128& r2, Rational128& out) noexcept {
        unsigned __int128 g1 = stein_gcd128(
            (r1.num < 0) ? -static_cast<unsigned __int128>(r1.num) : static_cast<unsigned __int128>(r1.num),
            static_cast<unsigned __int128>(r2.den)
        );
        unsigned __int128 g2 = stein_gcd128(
            (r2.num < 0) ? -static_cast<unsigned __int128>(r2.num) : static_cast<unsigned __int128>(r2.num),
            static_cast<unsigned __int128>(r1.den)
        );

        __int128_t n1 = r1.num / static_cast<__int128_t>(g1);
        __int128_t d2 = r2.den / static_cast<__int128_t>(g1);
        __int128_t n2 = r2.num / static_cast<__int128_t>(g2);
        __int128_t d1 = r1.den / static_cast<__int128_t>(g2);

        __int128_t res_num, res_den;
        if (__builtin_mul_overflow(n1, n2, &res_num)) return false;
        if (__builtin_mul_overflow(d1, d2, &res_den)) return false;

        out.num = res_num;
        out.den = res_den;
        return true;
    }
};

// Tiered exact rational: zero-heap Rational128 with GMP fallback
class ExactRational {
public:
    bool is_big;
    Rational128 r128;
    mpq_class gmp_q;

    ExactRational();
    ExactRational(int64_t n, int64_t d = 1);
    ExactRational(const std::string& n_str, const std::string& d_str = "1");
    static ExactRational from_double(double val, int64_t max_denominator = 100000);

    ExactRational(const ExactRational& other);
    ExactRational& operator=(const ExactRational& other);
    ExactRational(ExactRational&& other) noexcept;
    ExactRational& operator=(ExactRational&& other) noexcept;

    static ExactRational from_rational128(const Rational128& r);

    ExactRational operator+(const ExactRational& other) const;
    ExactRational operator-(const ExactRational& other) const;
    ExactRational operator*(const ExactRational& other) const;
    ExactRational operator*(int64_t scalar) const;

    bool operator<(const ExactRational& other) const;
    bool operator<=(const ExactRational& other) const;
    bool operator>(const ExactRational& other) const;
    bool operator>=(const ExactRational& other) const;
    bool operator==(const ExactRational& other) const;
    bool operator!=(const ExactRational& other) const;

    bool is_negative() const;
    bool is_zero() const;
    bool is_positive() const;

    double to_double() const;
    std::string num_str() const;
    std::string den_str() const;
    std::string str() const;

    void promote_to_gmp();
};

// Verification Certificate
struct NativeVerificationCertificate {
    bool is_valid;
    std::string status;
    std::string rejection_reason; // empty if null
    uint64_t num_active_supports;
    std::map<std::pair<uint32_t, uint32_t>, ExactRational> exact_coefficients;
    ExactRational exact_rhs;
    std::string sha256_hash;

    NativeVerificationCertificate()
        : is_valid(false),
          status("INITIALIZED"),
          rejection_reason(""),
          num_active_supports(0),
          exact_rhs(0, 1),
          sha256_hash("") {}

    std::string compute_sha256();
    double exact_rhs_float() const { return exact_rhs.to_double(); }
};

// Rational Cut Verifier
class RationalVerifier {
private:
    int64_t denominator_limit;

public:
    explicit RationalVerifier(int64_t limit = 100000) : denominator_limit(limit) {}

    ExactRational to_rational(double val) const {
        return ExactRational::from_double(val, denominator_limit);
    }

    NativeVerificationCertificate verify_clique_conic_combination(
        const BitGraph& graph,
        const std::vector<std::pair<std::vector<uint32_t>, ExactRational>>& candidate_multipliers,
        const std::optional<ExactRational>& candidate_rhs = std::nullopt,
        double tol = 1e-6
    );

    NativeVerificationCertificate verify_cycle_conic_combination(
        const BitGraph& graph,
        const std::vector<std::pair<std::pair<std::vector<uint32_t>, std::vector<std::pair<uint32_t, uint32_t>>>, ExactRational>>& candidate_multipliers,
        const std::optional<ExactRational>& candidate_rhs = std::nullopt,
        double tol = 1e-6
    );
};

} // namespace highs_turbo
