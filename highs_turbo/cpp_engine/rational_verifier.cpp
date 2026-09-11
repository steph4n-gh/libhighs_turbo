#include "rational_verifier.hpp"

#include <cmath>
#include <iomanip>
#include <sstream>
#include <set>

namespace highs_turbo {

static std::string int128_to_string(__int128_t n) {
    if (n == 0) return "0";
    std::string s;
    bool neg = false;
    unsigned __int128 un;
    if (n < 0) {
        neg = true;
        un = -static_cast<unsigned __int128>(n);
    } else {
        un = static_cast<unsigned __int128>(n);
    }
    while (un > 0) {
        int rem = static_cast<int>(un % 10);
        s.push_back('0' + rem);
        un /= 10;
    }
    if (neg) s.push_back('-');
    std::reverse(s.begin(), s.end());
    return s;
}

static std::string escape_json(const std::string& str) {
    std::ostringstream ss;
    for (char c : str) {
        switch (c) {
            case '"': ss << "\\\""; break;
            case '\\': ss << "\\\\"; break;
            case '\b': ss << "\\b"; break;
            case '\f': ss << "\\f"; break;
            case '\n': ss << "\\n"; break;
            case '\r': ss << "\\r"; break;
            case '\t': ss << "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    ss << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c);
                } else {
                    ss << c;
                }
        }
    }
    return ss.str();
}

// Exact rational using continued fractions matching Python limit_denominator
ExactRational::ExactRational() : is_big(false), r128(0, 1), gmp_q(0, 1) {}

ExactRational::ExactRational(int64_t n, int64_t d)
    : is_big(false),
      r128(n, d),
      gmp_q(std::to_string(n) + "/" + std::to_string(d)) {}

ExactRational::ExactRational(const std::string& n_str, const std::string& d_str) {
    if (d_str.empty()) {
        throw std::invalid_argument("ExactRational denominator cannot be zero");
    }
    try {
        mpz_class d_test(d_str);
        if (d_test == 0) {
            throw std::invalid_argument("ExactRational denominator cannot be zero");
        }
    } catch (const std::invalid_argument&) {
        throw;
    } catch (...) {
        throw std::invalid_argument("Invalid denominator string: " + d_str);
    }

    // Try 128-bit parse first
    try {
        is_big = false;
        gmp_q = mpq_class(n_str + "/" + d_str);
        gmp_q.canonicalize();

        std::string num_s = gmp_q.get_num().get_str();
        std::string den_s = gmp_q.get_den().get_str();

        if (num_s.size() < 38 && den_s.size() < 38) {
            __int128_t n = 0;
            bool neg_n = (!num_s.empty() && num_s[0] == '-');
            size_t start_n = neg_n ? 1 : 0;
            for (size_t i = start_n; i < num_s.size(); ++i) {
                n = n * 10 + (num_s[i] - '0');
            }
            if (neg_n) n = -n;

            __int128_t d = 0;
            for (char c : den_s) {
                d = d * 10 + (c - '0');
            }
            r128 = Rational128(n, d);
        } else {
            is_big = true;
        }
    } catch (...) {
        is_big = true;
        gmp_q = mpq_class(n_str + "/" + d_str);
        gmp_q.canonicalize();
    }
}

ExactRational ExactRational::from_double(double val, int64_t max_denominator) {
    if (std::isnan(val) || std::isinf(val)) {
        throw std::domain_error("Cannot convert NaN or Inf to ExactRational");
    }
    if (val >= 1e19 || val <= -1e19) {
        ExactRational res;
        res.is_big = true;
        mpq_set_d(res.gmp_q.get_mpq_t(), val);
        res.gmp_q.canonicalize();
        return res;
    }
    ExactRational res;
    res.is_big = false;
    if (val == 0.0) {
        res.r128 = Rational128(0, 1);
        res.gmp_q = mpq_class(0, 1);
        return res;
    }

    // Continued fraction approximation (Stern-Brocot / Farey algorithm)
    // Identical to Python's limit_denominator()
    int sign = (val < 0) ? -1 : 1;
    double x = std::abs(val);

    int64_t p0 = 0, q0 = 1;
    int64_t p1 = 1, q1 = 0;
    double rem = x;

    while (true) {
        int64_t a = static_cast<int64_t>(std::floor(rem));
        int64_t p2 = a * p1 + p0;
        int64_t q2 = a * q1 + q0;

        if (q2 > max_denominator) {
            // Find intermediate semi-convergent
            int64_t k = (max_denominator - q0) / q1;
            int64_t p_bound = k * p1 + p0;
            int64_t q_bound = k * q1 + q0;
            double err_prev = std::abs(x - static_cast<double>(p1) / static_cast<double>(q1));
            double err_bound = std::abs(x - static_cast<double>(p_bound) / static_cast<double>(q_bound));
            if (err_bound < err_prev) {
                p1 = p_bound;
                q1 = q_bound;
            }
            break;
        }

        p0 = p1; q0 = q1;
        p1 = p2; q1 = q2;

        double diff = rem - static_cast<double>(a);
        if (diff < 1e-15) {
            break;
        }
        rem = 1.0 / diff;
    }

    res.r128 = Rational128(sign * p1, q1);
    res.gmp_q = mpq_class(std::to_string(sign * p1) + "/" + std::to_string(q1));
    return res;
}

ExactRational::ExactRational(const ExactRational& other)
    : is_big(other.is_big), r128(other.r128), gmp_q(other.gmp_q) {}

ExactRational& ExactRational::operator=(const ExactRational& other) {
    if (this != &other) {
        is_big = other.is_big;
        r128 = other.r128;
        gmp_q = other.gmp_q;
    }
    return *this;
}

ExactRational::ExactRational(ExactRational&& other) noexcept
    : is_big(other.is_big), r128(other.r128), gmp_q(std::move(other.gmp_q)) {}

ExactRational& ExactRational::operator=(ExactRational&& other) noexcept {
    if (this != &other) {
        is_big = other.is_big;
        r128 = other.r128;
        gmp_q = std::move(other.gmp_q);
    }
    return *this;
}

ExactRational ExactRational::from_rational128(const Rational128& r) {
    ExactRational res;
    res.is_big = false;
    res.r128 = r;
    return res;
}

void ExactRational::promote_to_gmp() {
    if (!is_big) {
        is_big = true;
        std::string n = int128_to_string(r128.num);
        std::string d = int128_to_string(r128.den);
        gmp_q = mpq_class(n + "/" + d);
        gmp_q.canonicalize();
    }
}

ExactRational ExactRational::operator+(const ExactRational& other) const {
    if (is_big || other.is_big) {
        ExactRational res;
        res.is_big = true;
        mpq_class a = is_big ? gmp_q : mpq_class(int128_to_string(r128.num) + "/" + int128_to_string(r128.den));
        mpq_class b = other.is_big ? other.gmp_q : mpq_class(int128_to_string(other.r128.num) + "/" + int128_to_string(other.r128.den));
        res.gmp_q = a + b;
        res.gmp_q.canonicalize();
        return res;
    }
    Rational128 out;
    if (Rational128::add_checked(r128, other.r128, out)) {
        return ExactRational::from_rational128(out);
    }
    // Overflow -> fallback to GMP
    ExactRational res;
    res.is_big = true;
    mpq_class a(int128_to_string(r128.num) + "/" + int128_to_string(r128.den));
    mpq_class b(int128_to_string(other.r128.num) + "/" + int128_to_string(other.r128.den));
    res.gmp_q = a + b;
    res.gmp_q.canonicalize();
    return res;
}

ExactRational ExactRational::operator-(const ExactRational& other) const {
    if (is_big || other.is_big) {
        ExactRational res;
        res.is_big = true;
        mpq_class a = is_big ? gmp_q : mpq_class(int128_to_string(r128.num) + "/" + int128_to_string(r128.den));
        mpq_class b = other.is_big ? other.gmp_q : mpq_class(int128_to_string(other.r128.num) + "/" + int128_to_string(other.r128.den));
        res.gmp_q = a - b;
        res.gmp_q.canonicalize();
        return res;
    }
    Rational128 out;
    if (Rational128::sub_checked(r128, other.r128, out)) {
        return ExactRational::from_rational128(out);
    }
    ExactRational res;
    res.is_big = true;
    mpq_class a(int128_to_string(r128.num) + "/" + int128_to_string(r128.den));
    mpq_class b(int128_to_string(other.r128.num) + "/" + int128_to_string(other.r128.den));
    res.gmp_q = a - b;
    res.gmp_q.canonicalize();
    return res;
}

ExactRational ExactRational::operator*(const ExactRational& other) const {
    if (is_big || other.is_big) {
        ExactRational res;
        res.is_big = true;
        mpq_class a = is_big ? gmp_q : mpq_class(int128_to_string(r128.num) + "/" + int128_to_string(r128.den));
        mpq_class b = other.is_big ? other.gmp_q : mpq_class(int128_to_string(other.r128.num) + "/" + int128_to_string(other.r128.den));
        res.gmp_q = a * b;
        res.gmp_q.canonicalize();
        return res;
    }
    Rational128 out;
    if (Rational128::mul_checked(r128, other.r128, out)) {
        return ExactRational::from_rational128(out);
    }
    ExactRational res;
    res.is_big = true;
    mpq_class a(int128_to_string(r128.num) + "/" + int128_to_string(r128.den));
    mpq_class b(int128_to_string(other.r128.num) + "/" + int128_to_string(other.r128.den));
    res.gmp_q = a * b;
    res.gmp_q.canonicalize();
    return res;
}

ExactRational ExactRational::operator*(int64_t scalar) const {
    return (*this) * ExactRational(scalar, 1);
}

bool ExactRational::operator<(const ExactRational& other) const {
    if (is_negative() != other.is_negative()) {
        return is_negative();
    }
    if (is_zero()) {
        return other.is_positive();
    }
    if (other.is_zero()) {
        return is_negative();
    }
    if (is_big || other.is_big) {
        mpq_class a = is_big ? gmp_q : mpq_class(int128_to_string(r128.num) + "/" + int128_to_string(r128.den));
        mpq_class b = other.is_big ? other.gmp_q : mpq_class(int128_to_string(other.r128.num) + "/" + int128_to_string(other.r128.den));
        return a < b;
    }
    __int128_t prod1, prod2;
    if (!__builtin_mul_overflow(r128.num, other.r128.den, &prod1) &&
        !__builtin_mul_overflow(other.r128.num, r128.den, &prod2)) {
        return prod1 < prod2;
    }
    Rational128 diff;
    if (Rational128::sub_checked(r128, other.r128, diff)) {
        return diff.num < 0;
    }
    mpq_class a(int128_to_string(r128.num) + "/" + int128_to_string(r128.den));
    mpq_class b(int128_to_string(other.r128.num) + "/" + int128_to_string(other.r128.den));
    return a < b;
}

bool ExactRational::operator<=(const ExactRational& other) const {
    return !(other < *this);
}

bool ExactRational::operator>(const ExactRational& other) const {
    return other < *this;
}

bool ExactRational::operator>=(const ExactRational& other) const {
    return !(*this < other);
}

bool ExactRational::operator==(const ExactRational& other) const {
    if (is_big || other.is_big) {
        mpq_class a = is_big ? gmp_q : mpq_class(int128_to_string(r128.num) + "/" + int128_to_string(r128.den));
        mpq_class b = other.is_big ? other.gmp_q : mpq_class(int128_to_string(other.r128.num) + "/" + int128_to_string(other.r128.den));
        return a == b;
    }
    return (r128.num == other.r128.num) && (r128.den == other.r128.den);
}

bool ExactRational::operator!=(const ExactRational& other) const {
    return !(*this == other);
}

bool ExactRational::is_negative() const {
    if (is_big) return gmp_q < 0;
    return r128.num < 0;
}

bool ExactRational::is_zero() const {
    if (is_big) return gmp_q == 0;
    return r128.num == 0;
}

bool ExactRational::is_positive() const {
    if (is_big) return gmp_q > 0;
    return r128.num > 0;
}

double ExactRational::to_double() const {
    if (is_big) return gmp_q.get_d();
    return static_cast<double>(r128.num) / static_cast<double>(r128.den);
}

std::string ExactRational::num_str() const {
    if (is_big) return gmp_q.get_num().get_str();
    return int128_to_string(r128.num);
}

std::string ExactRational::den_str() const {
    if (is_big) return gmp_q.get_den().get_str();
    return int128_to_string(r128.den);
}

std::string ExactRational::str() const {
    return num_str() + "/" + den_str();
}

std::string NativeVerificationCertificate::compute_sha256() {
    // Canonical JSON payload byte-for-byte matching Python VerificationCertificate.compute_sha256()
    std::ostringstream ss;
    ss << "{\"coefficients\": [";
    bool first = true;
    for (const auto& [edge, coeff] : exact_coefficients) {
        if (!first) ss << ", ";
        first = false;
        ss << "{\"den\": " << coeff.den_str()
           << ", \"num\": " << coeff.num_str()
           << ", \"u\": " << edge.first
           << ", \"v\": " << edge.second << "}";
    }
    ss << "], \"exact_rhs_den\": " << exact_rhs.den_str()
       << ", \"exact_rhs_num\": " << exact_rhs.num_str()
       << ", \"is_valid\": " << (is_valid ? "true" : "false")
       << ", \"num_active_supports\": " << num_active_supports
       << ", \"rejection_reason\": ";
    if (rejection_reason.empty()) {
        ss << "null";
    } else {
        ss << "\"" << escape_json(rejection_reason) << "\"";
    }
    ss << ", \"status\": \"" << status << "\"}";

    std::string payload = ss.str();

    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(payload.data(), static_cast<CC_LONG>(payload.size()), digest);

    std::ostringstream hex_ss;
    for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; ++i) {
        hex_ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(digest[i]);
    }
    sha256_hash = hex_ss.str();
    return sha256_hash;
}

NativeVerificationCertificate RationalVerifier::verify_clique_conic_combination(
    const BitGraph& graph,
    const std::vector<std::pair<std::vector<uint32_t>, ExactRational>>& candidate_multipliers,
    const std::optional<ExactRational>& candidate_rhs,
    double tol
) {
    NativeVerificationCertificate cert;
    std::map<std::vector<uint32_t>, ExactRational> exact_mults;

    // 1. Non-negativity and structural validity check
    for (const auto& [raw_clq, raw_val] : candidate_multipliers) {
        if (raw_val.is_negative()) {
            cert.is_valid = false;
            cert.status = "REJECTED_NEGATIVE_MULTIPLIER";
            std::ostringstream ss;
            ss << "Support [";
            for (size_t i = 0; i < raw_clq.size(); ++i) {
                if (i > 0) ss << ", ";
                ss << raw_clq[i];
            }
            ss << "] has negative multiplier: " << raw_val.str();
            cert.rejection_reason = ss.str();
            cert.compute_sha256();
            return cert;
        }

        if (raw_val.is_zero()) {
            continue;
        }

        if (raw_clq.size() != 5) {
            cert.is_valid = false;
            cert.status = "REJECTED_INVALID_SUPPORT_SIZE";
            std::ostringstream ss;
            ss << "Support size is " << raw_clq.size() << ", expected 5.";
            cert.rejection_reason = ss.str();
            cert.compute_sha256();
            return cert;
        }

        std::vector<uint32_t> sorted_clq = raw_clq;
        std::sort(sorted_clq.begin(), sorted_clq.end());

        // Check node bounds
        for (uint32_t v : sorted_clq) {
            if (v >= graph.num_nodes) {
                cert.is_valid = false;
                cert.status = "REJECTED_OUT_OF_BOUNDS_NODE";
                cert.rejection_reason = "Node " + std::to_string(v) + " outside bounds.";
                cert.compute_sha256();
                return cert;
            }
        }

        // Check duplicate vertices
        for (size_t i = 0; i + 1 < sorted_clq.size(); ++i) {
            if (sorted_clq[i] == sorted_clq[i + 1]) {
                cert.is_valid = false;
                cert.status = "REJECTED_INVALID_SUPPORT_SIZE";
                cert.rejection_reason = "Duplicate vertex in support.";
                cert.compute_sha256();
                return cert;
            }
        }

        // Check all 10 pairwise edges in graph
        for (size_t i = 0; i < 5; ++i) {
            for (size_t j = i + 1; j < 5; ++j) {
                uint32_t u = sorted_clq[i];
                uint32_t v = sorted_clq[j];
                if (!graph.has_edge(u, v)) {
                    cert.is_valid = false;
                    cert.status = "REJECTED_NON_CLIQUE_SUPPORT";
                    cert.rejection_reason = "Edge (" + std::to_string(u) + ", " + std::to_string(v) + ") missing in graph.";
                    cert.compute_sha256();
                    return cert;
                }
            }
        }

        exact_mults[sorted_clq] = raw_val;
    }

    // 2. Reconstruct exact rational coefficients and RHS
    std::map<std::pair<uint32_t, uint32_t>, ExactRational> exact_coeffs;
    ExactRational exact_rhs(0, 1);

    for (const auto& [clq, mult] : exact_mults) {
        exact_rhs = exact_rhs + (mult * 6);
        for (size_t i = 0; i < 5; ++i) {
            for (size_t j = i + 1; j < 5; ++j) {
                uint32_t u = clq[i];
                uint32_t v = clq[j];
                auto e = std::make_pair(std::min(u, v), std::max(u, v));
                exact_coeffs[e] = exact_coeffs[e] + mult;
            }
        }
    }

    // 3. RHS Soundness Check
    if (candidate_rhs.has_value()) {
        ExactRational tol_rational = ExactRational::from_double(tol, denominator_limit);
        ExactRational sound_bound = exact_rhs - tol_rational;
        if (*candidate_rhs < sound_bound) {
            cert.is_valid = false;
            cert.status = "REJECTED_UNSOUND_RHS_DEFICIT";
            cert.rejection_reason = "Claimed RHS " + candidate_rhs->str() + " is lower than sound bound " + exact_rhs.str();
            cert.compute_sha256();
            return cert;
        }
    }

    // Filter active nonzero coefficients
    std::map<std::pair<uint32_t, uint32_t>, ExactRational> active_coeffs;
    for (const auto& [e, c] : exact_coeffs) {
        if (c.is_positive()) {
            active_coeffs[e] = c;
        }
    }

    cert.is_valid = true;
    cert.status = "CERTIFIED_VALID_CONIC_COMBINATION";
    cert.rejection_reason = "";
    cert.num_active_supports = exact_mults.size();
    cert.exact_coefficients = active_coeffs;
    cert.exact_rhs = exact_rhs;
    cert.compute_sha256();

    return cert;
}

NativeVerificationCertificate RationalVerifier::verify_cycle_conic_combination(
    const BitGraph& graph,
    const std::vector<std::pair<std::pair<std::vector<uint32_t>, std::vector<std::pair<uint32_t, uint32_t>>>, ExactRational>>& candidate_multipliers,
    const std::optional<ExactRational>& candidate_rhs,
    double tol
) {
    NativeVerificationCertificate cert;
    ExactRational exact_rhs(0, 1);
    std::map<std::pair<uint32_t, uint32_t>, ExactRational> exact_coeffs;
    uint64_t active_count = 0;

    for (const auto& [support_data, mult] : candidate_multipliers) {
        const auto& cycle_nodes = support_data.first;
        const auto& odd_edges = support_data.second;

        if (mult.is_negative()) {
            cert.is_valid = false;
            cert.status = "REJECTED_NEGATIVE_MULTIPLIER";
            cert.rejection_reason = "Negative cycle multiplier";
            cert.compute_sha256();
            return cert;
        }

        if (mult.is_zero()) continue;

        if (cycle_nodes.size() < 3) {
            cert.is_valid = false;
            cert.status = "REJECTED_INVALID_CYCLE_LENGTH";
            cert.rejection_reason = "Cycle length must be >= 3";
            cert.compute_sha256();
            return cert;
        }

        if (odd_edges.size() % 2 == 0) {
            cert.is_valid = false;
            cert.status = "REJECTED_EVEN_F_CARDINALITY";
            cert.rejection_reason = "|F| must be odd, got " + std::to_string(odd_edges.size()) + ".";
            cert.compute_sha256();
            return cert;
        }

        // Check cycle edges exist in graph
        size_t L = cycle_nodes.size();
        std::set<std::pair<uint32_t, uint32_t>> cycle_edge_set;
        for (size_t i = 0; i < L; ++i) {
            uint32_t u = cycle_nodes[i];
            uint32_t v = cycle_nodes[(i + 1) % L];
            if (!graph.has_edge(u, v)) {
                cert.is_valid = false;
                cert.status = "REJECTED_NON_GRAPH_EDGE";
                cert.rejection_reason = "Edge (" + std::to_string(std::min(u, v)) + ", " + std::to_string(std::max(u, v)) + ") of cycle not present in graph.";
                cert.compute_sha256();
                return cert;
            }
            cycle_edge_set.insert(std::make_pair(std::min(u, v), std::max(u, v)));
        }

        // Check F is subset of C
        std::set<std::pair<uint32_t, uint32_t>> F_set;
        for (const auto& e : odd_edges) {
            auto can_e = std::make_pair(std::min(e.first, e.second), std::max(e.first, e.second));
            if (cycle_edge_set.find(can_e) == cycle_edge_set.end()) {
                cert.is_valid = false;
                cert.status = "REJECTED_F_EDGE_NOT_IN_CYCLE";
                cert.rejection_reason = "Edge (" + std::to_string(can_e.first) + ", " + std::to_string(can_e.second) + ") in F is not in cycle edges.";
                cert.compute_sha256();
                return cert;
            }
            F_set.insert(can_e);
        }

        int64_t rhs_val = static_cast<int64_t>(F_set.size()) - 1;
        exact_rhs = exact_rhs + (mult * rhs_val);
        active_count++;

        for (const auto& e : cycle_edge_set) {
            if (F_set.find(e) != F_set.end()) {
                exact_coeffs[e] = exact_coeffs[e] + mult;
            } else {
                exact_coeffs[e] = exact_coeffs[e] - mult;
            }
        }
    }

    if (candidate_rhs.has_value()) {
        ExactRational tol_rational = ExactRational::from_double(tol, denominator_limit);
        ExactRational sound_bound = exact_rhs - tol_rational;
        if (*candidate_rhs < sound_bound) {
            cert.is_valid = false;
            cert.status = "REJECTED_UNSOUND_RHS_DEFICIT";
            cert.rejection_reason = "Claimed RHS " + candidate_rhs->str() + " is lower than sound bound " + exact_rhs.str();
            cert.compute_sha256();
            return cert;
        }
    }

    // Filter active nonzero coefficients
    std::map<std::pair<uint32_t, uint32_t>, ExactRational> active_coeffs;
    for (const auto& [e, c] : exact_coeffs) {
        if (!c.is_zero()) {
            active_coeffs[e] = c;
        }
    }

    cert.is_valid = true;
    cert.status = "CERTIFIED_VALID_CYCLE_CONIC_COMBINATION";
    cert.rejection_reason = "";
    cert.num_active_supports = active_count;
    cert.exact_coefficients = active_coeffs;
    cert.exact_rhs = exact_rhs;
    cert.compute_sha256();

    return cert;
}

} // namespace highs_turbo
