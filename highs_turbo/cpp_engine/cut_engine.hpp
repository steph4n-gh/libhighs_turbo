#pragma once

#include <cstdint>
#include <vector>
#include <utility>
#include <string>
#include <map>
#include <memory>

#include "bit_graph.hpp"
#include "rational_verifier.hpp"

namespace highs_turbo {

struct ViolatedCut {
    std::string type; // "k5", "triangle", or "cycle"
    std::vector<uint32_t> nodes;
    std::vector<int> edge_indices;
    std::vector<double> coefficients;
    double rhs;
    double violation;
    uint64_t canonical_hash = 0;
};

struct SurrogateCut {
    std::vector<double> full_coefficients; // size m
    std::vector<int> nz_indices;
    std::vector<double> nz_values;
    double rhs;
    bool is_valid;
    std::string status;
    std::string certificate_hash;
    uint64_t num_active_supports;
};

class CutEngine {
private:
    RationalVerifier verifier;

public:
    explicit CutEngine(int64_t rational_denominator_limit = 100000)
        : verifier(rational_denominator_limit) {}

    // Separate violated K5 clique cuts given primal solution x* in [0, 1]^m
    std::vector<ViolatedCut> separate_violated_k5(
        const BitGraph& graph,
        const std::vector<std::pair<uint32_t, uint32_t>>& edge_list,
        const std::vector<double>& primal_sol,
        double threshold = 1e-4,
        size_t max_cuts = 0
    ) const;

    // Separate violated triangle cuts given primal solution x* in [0, 1]^m
    std::vector<ViolatedCut> separate_violated_triangles(
        const BitGraph& graph,
        const std::vector<std::pair<uint32_t, uint32_t>>& edge_list,
        const std::vector<double>& primal_sol,
        double threshold = 1e-4,
        size_t max_cuts = 0
    ) const;

    // Separate violated 4-cycle cuts given primal solution x* in [0, 1]^m
    std::vector<ViolatedCut> separate_violated_4cycles(
        const BitGraph& graph,
        const std::vector<std::pair<uint32_t, uint32_t>>& edge_list,
        const std::vector<double>& primal_sol,
        double threshold = 1e-4,
        size_t max_cuts = 0
    ) const;

    // Retain all cuts sharing the boundary violation tier, preserving isomorphism invariance
    static std::vector<ViolatedCut> select_canonical_cuts(
        std::vector<ViolatedCut> cuts,
        size_t max_cuts = 0
    );

    // Compile 1-row surrogate cut from K5 multipliers and certify via RationalVerifier
    SurrogateCut compile_surrogate_cut(
        const BitGraph& graph,
        const std::vector<std::pair<uint32_t, uint32_t>>& edge_list,
        const std::vector<std::pair<std::vector<uint32_t>, double>>& k5_multipliers,
        double tol = 1e-6
    );

    // Fast surrogate cut from discovered violated K5 cuts using violation-proportional weights
    SurrogateCut compile_violation_surrogate(
        const BitGraph& graph,
        const std::vector<std::pair<uint32_t, uint32_t>>& edge_list,
        const std::vector<ViolatedCut>& violated_cuts
    );
};

} // namespace highs_turbo
