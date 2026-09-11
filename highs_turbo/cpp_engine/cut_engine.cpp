#include "cut_engine.hpp"

#include <unordered_map>
#include <algorithm>
#include <stdexcept>

namespace highs_turbo {

// Custom hash for pair<uint32_t, uint32_t>
struct EdgeHash {
    std::size_t operator()(const std::pair<uint32_t, uint32_t>& e) const noexcept {
        return (static_cast<std::size_t>(e.first) << 32) ^ static_cast<std::size_t>(e.second);
    }
};

static std::unordered_map<std::pair<uint32_t, uint32_t>, int, EdgeHash> build_edge_map(
    const std::vector<std::pair<uint32_t, uint32_t>>& edge_list
) {
    std::unordered_map<std::pair<uint32_t, uint32_t>, int, EdgeHash> map;
    map.reserve(edge_list.size() * 2);
    for (size_t i = 0; i < edge_list.size(); ++i) {
        uint32_t u = edge_list[i].first;
        uint32_t v = edge_list[i].second;
        map[std::make_pair(std::min(u, v), std::max(u, v))] = static_cast<int>(i);
    }
    return map;
}

std::vector<ViolatedCut> CutEngine::select_canonical_cuts(
    std::vector<ViolatedCut> cuts,
    size_t max_cuts
) {
    if (cuts.empty()) return {};

    // Sort cuts canonically: violation descending, then invariant canonical_hash descending
    std::stable_sort(cuts.begin(), cuts.end(), [](const ViolatedCut& a, const ViolatedCut& b) {
        int64_t va = static_cast<int64_t>(std::round(a.violation * 1e8));
        int64_t vb = static_cast<int64_t>(std::round(b.violation * 1e8));
        if (va != vb) return va > vb;
        return a.canonical_hash > b.canonical_hash;
    });

    if (max_cuts == 0 || cuts.size() <= max_cuts) {
        return cuts;
    }

    // Retain all cuts sharing the boundary violation tier to ensure automorphism invariance
    auto get_tier_key = [](const ViolatedCut& c) {
        int64_t v = static_cast<int64_t>(std::round(c.violation * 1e8));
        return std::make_pair(v, c.canonical_hash);
    };

    size_t boundary_idx = max_cuts;
    auto boundary_key = get_tier_key(cuts[boundary_idx - 1]);

    size_t start_idx = boundary_idx - 1;
    while (start_idx > 0 && get_tier_key(cuts[start_idx - 1]) == boundary_key) {
        start_idx--;
    }

    size_t end_idx = boundary_idx;
    while (end_idx < cuts.size() && get_tier_key(cuts[end_idx]) == boundary_key) {
        end_idx++;
    }

    if (end_idx > max_cuts) {
        // Tier exceeds max_cuts, drop it entirely to avoid OOM and strictly enforce max_cuts while preserving invariance
        cuts.resize(start_idx);
    } else {
        cuts.resize(end_idx);
    }

    return cuts;
}

std::vector<ViolatedCut> CutEngine::separate_violated_k5(
    const BitGraph& graph,
    const std::vector<std::pair<uint32_t, uint32_t>>& edge_list,
    const std::vector<double>& primal_sol,
    double threshold,
    size_t max_cuts
) const {
    if (primal_sol.size() < edge_list.size()) {
        throw std::invalid_argument("primal_sol size must be >= edge count");
    }
    std::vector<ViolatedCut> violated;
    auto edge_map = build_edge_map(edge_list);
    auto cliques = graph.find_k5_cliques();
    auto colors = graph.compute_1wl_colors(2);

    for (const auto& clq : cliques) {
        double sum_x = 0.0;
        std::vector<int> cut_edges;
        cut_edges.reserve(10);
        bool complete = true;

        for (size_t i = 0; i < 5 && complete; ++i) {
            for (size_t j = i + 1; j < 5; ++j) {
                auto e = std::make_pair(std::min(clq[i], clq[j]), std::max(clq[i], clq[j]));
                auto it = edge_map.find(e);
                if (it == edge_map.end()) {
                    complete = false;
                    break;
                }
                int idx = it->second;
                cut_edges.push_back(idx);
                sum_x += primal_sol[idx];
            }
        }

        if (complete && sum_x > 6.0 + threshold) {
            ViolatedCut cut;
            cut.type = "k5";
            cut.nodes = clq;
            cut.edge_indices = cut_edges;
            cut.coefficients = std::vector<double>(10, 1.0);
            cut.rhs = 6.0;
            cut.violation = sum_x - 6.0;

            // Compute canonical automorphism invariant hash
            std::vector<uint64_t> node_colors;
            node_colors.reserve(5);
            for (uint32_t u : clq) {
                node_colors.push_back(u < colors.size() ? colors[u] : 0ULL);
            }
            std::sort(node_colors.begin(), node_colors.end());

            uint64_t h = 0xcbf29ce484222325ULL;
            for (uint64_t nc : node_colors) {
                h ^= nc + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            }
            int64_t qv = static_cast<int64_t>(std::round(cut.violation * 1e8));
            h ^= static_cast<uint64_t>(qv) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            cut.canonical_hash = h;

            violated.push_back(std::move(cut));
        }
    }

    return select_canonical_cuts(std::move(violated), max_cuts);
}

std::vector<ViolatedCut> CutEngine::separate_violated_triangles(
    const BitGraph& graph,
    const std::vector<std::pair<uint32_t, uint32_t>>& edge_list,
    const std::vector<double>& primal_sol,
    double threshold,
    size_t max_cuts
) const {
    if (primal_sol.size() < edge_list.size()) {
        throw std::invalid_argument("primal_sol size must be >= edge count");
    }
    std::vector<ViolatedCut> violated;
    auto edge_map = build_edge_map(edge_list);
    auto triangles = graph.find_triangles();
    auto colors = graph.compute_1wl_colors(2);

    for (const auto& tri : triangles) {
        uint32_t u = tri[0], v = tri[1], w = tri[2];
        auto e1 = std::make_pair(std::min(u, v), std::max(u, v));
        auto e2 = std::make_pair(std::min(v, w), std::max(v, w));
        auto e3 = std::make_pair(std::min(u, w), std::max(u, w));

        auto it1 = edge_map.find(e1);
        auto it2 = edge_map.find(e2);
        auto it3 = edge_map.find(e3);

        if (it1 == edge_map.end() || it2 == edge_map.end() || it3 == edge_map.end()) {
            continue;
        }

        int i1 = it1->second, i2 = it2->second, i3 = it3->second;
        double x1 = primal_sol[i1], x2 = primal_sol[i2], x3 = primal_sol[i3];

        auto make_cut = [&](const std::vector<int>& edge_idx, const std::vector<double>& coeffs, double rhs_val, double viol) {
            ViolatedCut cut;
            cut.type = "triangle";
            cut.nodes = {u, v, w};
            cut.edge_indices = edge_idx;
            cut.coefficients = coeffs;
            cut.rhs = rhs_val;
            cut.violation = viol;

            std::vector<uint64_t> node_colors = {
                u < colors.size() ? colors[u] : 0ULL,
                v < colors.size() ? colors[v] : 0ULL,
                w < colors.size() ? colors[w] : 0ULL,
            };
            std::sort(node_colors.begin(), node_colors.end());

            std::vector<uint32_t> edge_commons = {
                graph.count_common_neighbors(u, v),
                graph.count_common_neighbors(v, w),
                graph.count_common_neighbors(u, w),
            };
            std::sort(edge_commons.begin(), edge_commons.end());

            uint64_t h = 0xcbf29ce484222325ULL;
            for (uint64_t nc : node_colors) {
                h ^= nc + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            }
            for (uint32_t ec : edge_commons) {
                h ^= static_cast<uint64_t>(ec) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            }
            int64_t qv = static_cast<int64_t>(std::round(viol * 1e8));
            h ^= static_cast<uint64_t>(qv) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            cut.canonical_hash = h;
            return cut;
        };

        // Valid metric cycle inequalities (|F| in {1, 3}):
        // 1. x1 - x2 - x3 <= 0 (F = {e1})
        double v1 = x1 - x2 - x3;
        if (v1 > threshold) {
            violated.push_back(make_cut({i1, i2, i3}, {1.0, -1.0, -1.0}, 0.0, v1));
        }

        // 2. x2 - x1 - x3 <= 0 (F = {e2})
        double v2 = x2 - x1 - x3;
        if (v2 > threshold) {
            violated.push_back(make_cut({i1, i2, i3}, {-1.0, 1.0, -1.0}, 0.0, v2));
        }

        // 3. x3 - x1 - x2 <= 0 (F = {e3})
        double v3 = x3 - x1 - x2;
        if (v3 > threshold) {
            violated.push_back(make_cut({i1, i2, i3}, {-1.0, -1.0, 1.0}, 0.0, v3));
        }

        // 4. x1 + x2 + x3 <= 2 (F = {e1, e2, e3})
        double v4 = x1 + x2 + x3 - 2.0;
        if (v4 > threshold) {
            violated.push_back(make_cut({i1, i2, i3}, {1.0, 1.0, 1.0}, 2.0, v4));
        }
    }

    return select_canonical_cuts(std::move(violated), max_cuts);
}

std::vector<ViolatedCut> CutEngine::separate_violated_4cycles(
    const BitGraph& graph,
    const std::vector<std::pair<uint32_t, uint32_t>>& edge_list,
    const std::vector<double>& primal_sol,
    double threshold,
    size_t max_cuts
) const {
    if (primal_sol.size() < edge_list.size()) {
        throw std::invalid_argument("primal_sol size must be >= edge count");
    }
    std::vector<ViolatedCut> violated;
    auto edge_map = build_edge_map(edge_list);
    const size_t n = graph.num_nodes;
    const size_t w = graph.words_per_row;
    auto colors = graph.compute_1wl_colors(2);

    for (size_t u = 0; u < n; ++u) {
        for (size_t v = u + 1; v < n; ++v) {
            std::vector<uint32_t> commons;
            const uint64_t* r_u = graph.row(u);
            const uint64_t* r_v = graph.row(v);

            for (size_t kw = 0; kw < w; ++kw) {
                uint64_t common = r_u[kw] & r_v[kw];
                while (common != 0) {
                    int bit = __builtin_ctzll(common);
                    uint32_t node_w = static_cast<uint32_t>((kw << 6) + bit);
                    common &= (common - 1);
                    if (node_w < n) {
                        commons.push_back(node_w);
                    }
                }
            }

            if (commons.size() < 2) continue;

            for (size_t i = 0; i < commons.size(); ++i) {
                uint32_t w1 = commons[i];
                for (size_t j = i + 1; j < commons.size(); ++j) {
                    uint32_t w2 = commons[j];

                    if (u > w1) continue;

                    auto e1 = std::make_pair(std::min(static_cast<uint32_t>(u), w1), std::max(static_cast<uint32_t>(u), w1));
                    auto e2 = std::make_pair(std::min(w1, static_cast<uint32_t>(v)), std::max(w1, static_cast<uint32_t>(v)));
                    auto e3 = std::make_pair(std::min(static_cast<uint32_t>(v), w2), std::max(static_cast<uint32_t>(v), w2));
                    auto e4 = std::make_pair(std::min(w2, static_cast<uint32_t>(u)), std::max(w2, static_cast<uint32_t>(u)));

                    auto it1 = edge_map.find(e1);
                    auto it2 = edge_map.find(e2);
                    auto it3 = edge_map.find(e3);
                    auto it4 = edge_map.find(e4);

                    if (it1 == edge_map.end() || it2 == edge_map.end() ||
                        it3 == edge_map.end() || it4 == edge_map.end()) {
                        continue;
                    }

                    int i1 = it1->second, i2 = it2->second, i3 = it3->second, i4 = it4->second;
                    double x1 = primal_sol[i1], x2 = primal_sol[i2], x3 = primal_sol[i3], x4 = primal_sol[i4];

                    std::vector<std::tuple<std::vector<int>, std::vector<double>, double, uint32_t, uint32_t>> combos = {
                        {{i1, i2, i3, i4}, {1.0, 1.0, 1.0, -1.0}, x1 + x2 + x3 - x4 - 2.0, w2, static_cast<uint32_t>(u)},
                        {{i1, i2, i4, i3}, {1.0, 1.0, 1.0, -1.0}, x1 + x2 + x4 - x3 - 2.0, static_cast<uint32_t>(v), w2},
                        {{i1, i3, i4, i2}, {1.0, 1.0, 1.0, -1.0}, x1 + x3 + x4 - x2 - 2.0, w1, static_cast<uint32_t>(v)},
                        {{i2, i3, i4, i1}, {1.0, 1.0, 1.0, -1.0}, x2 + x3 + x4 - x1 - 2.0, static_cast<uint32_t>(u), w1},
                    };

                    std::vector<uint32_t> cycle_commons = {
                        graph.count_common_neighbors(static_cast<uint32_t>(u), w1),
                        graph.count_common_neighbors(w1, static_cast<uint32_t>(v)),
                        graph.count_common_neighbors(static_cast<uint32_t>(v), w2),
                        graph.count_common_neighbors(w2, static_cast<uint32_t>(u)),
                    };
                    std::sort(cycle_commons.begin(), cycle_commons.end());

                    for (const auto& [idx_list, coeff_list, viol, odd_u, odd_v] : combos) {
                        if (viol > threshold) {
                            ViolatedCut cut;
                            cut.type = "cycle";
                            cut.nodes = {static_cast<uint32_t>(u), w1, static_cast<uint32_t>(v), w2};
                            cut.edge_indices = idx_list;
                            cut.coefficients = coeff_list;
                            cut.rhs = 2.0;
                            cut.violation = viol;

                            std::vector<uint64_t> node_colors = {
                                colors[u], colors[w1], colors[v], colors[w2]
                            };
                            std::sort(node_colors.begin(), node_colors.end());

                            uint64_t odd_c1 = std::min(colors[odd_u], colors[odd_v]);
                            uint64_t odd_c2 = std::max(colors[odd_u], colors[odd_v]);

                            uint64_t h = 0xcbf29ce484222325ULL;
                            for (uint64_t nc : node_colors) {
                                h ^= nc + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
                            }
                            for (uint32_t cc : cycle_commons) {
                                h ^= static_cast<uint64_t>(cc) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
                            }
                            h ^= odd_c1 + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
                            h ^= odd_c2 + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
                            int64_t qv = static_cast<int64_t>(std::round(viol * 1e8));
                            h ^= static_cast<uint64_t>(qv) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
                            cut.canonical_hash = h;

                            violated.push_back(std::move(cut));
                        }
                    }
                }
            }
        }
    }

    return select_canonical_cuts(std::move(violated), max_cuts);
}

SurrogateCut CutEngine::compile_surrogate_cut(
    const BitGraph& graph,
    const std::vector<std::pair<uint32_t, uint32_t>>& edge_list,
    const std::vector<std::pair<std::vector<uint32_t>, double>>& k5_multipliers,
    double tol
) {
    SurrogateCut surr;
    surr.full_coefficients.assign(edge_list.size(), 0.0);
    surr.rhs = 0.0;
    surr.is_valid = false;
    surr.num_active_supports = 0;

    std::vector<std::pair<std::vector<uint32_t>, ExactRational>> exact_mults;
    exact_mults.reserve(k5_multipliers.size());

    for (const auto& [clq, mult] : k5_multipliers) {
        exact_mults.emplace_back(clq, verifier.to_rational(mult));
    }

    NativeVerificationCertificate cert = verifier.verify_clique_conic_combination(
        graph, exact_mults, std::nullopt, tol
    );

    surr.is_valid = cert.is_valid;
    surr.status = cert.status;
    surr.certificate_hash = cert.sha256_hash;
    surr.num_active_supports = cert.num_active_supports;
    surr.rhs = cert.exact_rhs_float();

    if (!cert.is_valid) {
        return surr;
    }

    auto edge_map = build_edge_map(edge_list);

    for (const auto& [e, coeff] : cert.exact_coefficients) {
        auto it = edge_map.find(e);
        if (it != edge_map.end()) {
            int idx = it->second;
            double val = coeff.to_double();
            surr.full_coefficients[idx] = val;
            if (std::abs(val) > 1e-12) {
                surr.nz_indices.push_back(idx);
                surr.nz_values.push_back(val);
            }
        }
    }

    return surr;
}

SurrogateCut CutEngine::compile_violation_surrogate(
    const BitGraph& graph,
    const std::vector<std::pair<uint32_t, uint32_t>>& edge_list,
    const std::vector<ViolatedCut>& violated_cuts
) {
    if (violated_cuts.empty()) {
        SurrogateCut empty_cut;
        empty_cut.full_coefficients.assign(edge_list.size(), 0.0);
        empty_cut.rhs = 0.0;
        empty_cut.is_valid = true;
        empty_cut.status = "CERTIFIED_EMPTY_CONIC_COMBINATION";
        empty_cut.certificate_hash = std::string(64, '0');
        empty_cut.num_active_supports = 0;
        return empty_cut;
    }

    bool all_k5 = true;
    for (const auto& cut : violated_cuts) {
        if (cut.type != "k5") {
            all_k5 = false;
            break;
        }
    }

    if (all_k5) {
        std::vector<std::pair<std::vector<uint32_t>, double>> mults;
        mults.reserve(violated_cuts.size());
        for (const auto& cut : violated_cuts) {
            if (cut.violation > 0) {
                mults.emplace_back(cut.nodes, cut.violation);
            }
        }
        return compile_surrogate_cut(graph, edge_list, mults);
    }

    // Compile cycle cuts (triangles / 4-cycles)
    std::vector<std::pair<std::pair<std::vector<uint32_t>, std::vector<std::pair<uint32_t, uint32_t>>>, ExactRational>> cycle_mults;
    cycle_mults.reserve(violated_cuts.size());

    for (const auto& cut : violated_cuts) {
        if (cut.violation <= 0) continue;
        std::vector<std::pair<uint32_t, uint32_t>> odd_edges;
        for (size_t k = 0; k < cut.edge_indices.size(); ++k) {
            if (cut.coefficients[k] > 0) {
                int e_idx = cut.edge_indices[k];
                if (e_idx >= 0 && static_cast<size_t>(e_idx) < edge_list.size()) {
                    odd_edges.push_back(edge_list[e_idx]);
                }
            }
        }
        if (odd_edges.size() % 2 == 1) {
            cycle_mults.push_back({{cut.nodes, odd_edges}, verifier.to_rational(cut.violation)});
        }
    }

    NativeVerificationCertificate cert = verifier.verify_cycle_conic_combination(graph, cycle_mults);
    SurrogateCut surr;
    surr.full_coefficients.assign(edge_list.size(), 0.0);
    surr.is_valid = cert.is_valid;
    surr.status = cert.status;
    surr.certificate_hash = cert.sha256_hash;
    surr.num_active_supports = cert.num_active_supports;
    surr.rhs = cert.exact_rhs_float();

    if (!cert.is_valid) return surr;

    auto edge_map = build_edge_map(edge_list);
    for (const auto& [e, coeff] : cert.exact_coefficients) {
        auto it = edge_map.find(e);
        if (it != edge_map.end()) {
            int idx = it->second;
            double val = coeff.to_double();
            surr.full_coefficients[idx] = val;
            if (std::abs(val) > 1e-12) {
                surr.nz_indices.push_back(idx);
                surr.nz_values.push_back(val);
            }
        }
    }
    return surr;
}

} // namespace highs_turbo
