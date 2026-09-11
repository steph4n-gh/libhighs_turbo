#include "bit_graph.hpp"
#include <cmath>
#include <unordered_map>
#include <algorithm>

namespace highs_turbo {

BitGraph::BitGraph(size_t n)
    : num_nodes(n),
      words_per_row((((n + 63) / 64 + 7) / 8) * 8),
      num_edges_count(0) {
    total_words = num_nodes * words_per_row;
    if (total_words == 0) {
        bit_adj = nullptr;
    } else {
        // Allocate 64-byte aligned memory matching CPU cache line and SIMD vectors
        bit_adj = static_cast<uint64_t*>(
            ::operator new[](total_words * sizeof(uint64_t), std::align_val_t(64))
        );
        std::memset(bit_adj, 0, total_words * sizeof(uint64_t));
    }
}

BitGraph::~BitGraph() {
    if (bit_adj) {
        ::operator delete[](bit_adj, std::align_val_t(64));
        bit_adj = nullptr;
    }
}

BitGraph::BitGraph(BitGraph&& other) noexcept
    : num_nodes(other.num_nodes),
      words_per_row(other.words_per_row),
      total_words(other.total_words),
      bit_adj(other.bit_adj),
      num_edges_count(other.num_edges_count) {
    other.bit_adj = nullptr;
    other.num_nodes = 0;
    other.words_per_row = 0;
    other.total_words = 0;
    other.num_edges_count = 0;
}

BitGraph& BitGraph::operator=(BitGraph&& other) noexcept {
    if (this != &other) {
        if (bit_adj) {
            ::operator delete[](bit_adj, std::align_val_t(64));
        }
        num_nodes = other.num_nodes;
        words_per_row = other.words_per_row;
        total_words = other.total_words;
        bit_adj = other.bit_adj;
        num_edges_count = other.num_edges_count;

        other.bit_adj = nullptr;
        other.num_nodes = 0;
        other.words_per_row = 0;
        other.total_words = 0;
        other.num_edges_count = 0;
    }
    return *this;
}

void BitGraph::add_edge(uint32_t u, uint32_t v) {
    if (u == v || u >= num_nodes || v >= num_nodes) return;
    size_t u_idx = static_cast<size_t>(u) * words_per_row + (v >> 6);
    uint64_t v_mask = (1ULL << (v & 63));
    if ((bit_adj[u_idx] & v_mask) == 0) {
        bit_adj[u_idx] |= v_mask;
        bit_adj[static_cast<size_t>(v) * words_per_row + (u >> 6)] |= (1ULL << (u & 63));
        num_edges_count++;
    }
}

void BitGraph::add_edges(const std::vector<std::pair<uint32_t, uint32_t>>& edges) {
    for (const auto& [u, v] : edges) {
        add_edge(u, v);
    }
}

bool BitGraph::has_edge(uint32_t u, uint32_t v) const noexcept {
    if (u >= num_nodes || v >= num_nodes || u == v) return false;
    return (bit_adj[static_cast<size_t>(u) * words_per_row + (v >> 6)] & (1ULL << (v & 63))) != 0;
}

uint32_t BitGraph::degree(uint32_t u) const noexcept {
    if (u >= num_nodes) return 0;
    const uint64_t* r_u = row(u);
    uint32_t d = 0;
    for (size_t k = 0; k < words_per_row; ++k) {
        d += __builtin_popcountll(r_u[k]);
    }
    return d;
}

uint32_t BitGraph::count_common_neighbors(uint32_t u, uint32_t v) const noexcept {
    if (u >= num_nodes || v >= num_nodes || u == v) return 0;
    const uint64_t* r_u = row(u);
    const uint64_t* r_v = row(v);
    uint32_t count = 0;
    for (size_t k = 0; k < words_per_row; ++k) {
        count += __builtin_popcountll(r_u[k] & r_v[k]);
    }
    return count;
}

uint32_t BitGraph::count_triangles_on_edge(uint32_t u, uint32_t v) const noexcept {
    return count_common_neighbors(u, v);
}

uint64_t BitGraph::count_triangles() const noexcept {
    uint64_t total = 0;
    const size_t w = words_per_row;

    for (size_t u = 0; u < num_nodes; ++u) {
        const uint64_t* r_u = row(u);
        size_t w_u = u >> 6;
        uint64_t mask_u = ((u & 63) == 63) ? 0ULL : ~((1ULL << ((u & 63) + 1)) - 1);

        for (size_t kw = w_u; kw < w; ++kw) {
            uint64_t u_word = r_u[kw] & (kw == w_u ? mask_u : ~0ULL);
            while (u_word != 0) {
                int bit_v = __builtin_ctzll(u_word);
                size_t v = (kw << 6) + bit_v;
                u_word &= (u_word - 1);

                // Common neighbors w > v
                const uint64_t* r_v = row(v);
                size_t w_v = v >> 6;
                uint64_t mask_v = ((v & 63) == 63) ? 0ULL : ~((1ULL << ((v & 63) + 1)) - 1);

                for (size_t kw_v = w_v; kw_v < w; ++kw_v) {
                    uint64_t m = (kw_v == w_v ? mask_v : ~0ULL);
                    total += __builtin_popcountll(r_u[kw_v] & r_v[kw_v] & m);
                }
            }
        }
    }
    return total;
}

std::vector<std::vector<uint32_t>> BitGraph::find_triangles() const {
    std::vector<std::vector<uint32_t>> triangles;
    const size_t w = words_per_row;

    for (size_t u = 0; u < num_nodes; ++u) {
        const uint64_t* r_u = row(u);
        size_t w_u = u >> 6;
        uint64_t mask_u = ((u & 63) == 63) ? 0ULL : ~((1ULL << ((u & 63) + 1)) - 1);

        for (size_t kw = w_u; kw < w; ++kw) {
            uint64_t u_word = r_u[kw] & (kw == w_u ? mask_u : ~0ULL);
            while (u_word != 0) {
                int bit_v = __builtin_ctzll(u_word);
                size_t v = (kw << 6) + bit_v;
                u_word &= (u_word - 1);

                // Common neighbors w > v
                const uint64_t* r_v = row(v);
                size_t w_v = v >> 6;
                uint64_t mask_v = ((v & 63) == 63) ? 0ULL : ~((1ULL << ((v & 63) + 1)) - 1);

                for (size_t kw_v = w_v; kw_v < w; ++kw_v) {
                    uint64_t m = (kw_v == w_v ? mask_v : ~0ULL);
                    uint64_t common = r_u[kw_v] & r_v[kw_v] & m;
                    while (common != 0) {
                        int bit_w = __builtin_ctzll(common);
                        size_t node_w = (kw_v << 6) + bit_w;
                        common &= (common - 1);

                        triangles.push_back({
                            static_cast<uint32_t>(u),
                            static_cast<uint32_t>(v),
                            static_cast<uint32_t>(node_w)
                        });
                    }
                }
            }
        }
    }
    return triangles;
}

std::vector<std::vector<uint32_t>> BitGraph::find_k5_cliques() const {
    std::vector<std::vector<uint32_t>> results;
    const size_t n = num_nodes;
    const size_t w = words_per_row;
    if (n < 5 || w == 0) return results;

    std::vector<uint64_t> C2(w, 0);
    std::vector<uint64_t> C3(w, 0);
    std::vector<uint64_t> C4(w, 0);

    for (size_t v1 = 0; v1 < n; ++v1) {
        const uint64_t* r1 = row(v1);
        size_t w1 = v1 >> 6;
        uint64_t mask1 = ((v1 & 63) == 63) ? 0ULL : ~((1ULL << ((v1 & 63) + 1)) - 1);

        for (size_t kw1 = w1; kw1 < w; ++kw1) {
            uint64_t c1_word = r1[kw1] & (kw1 == w1 ? mask1 : ~0ULL);
            while (c1_word != 0) {
                int b1 = __builtin_ctzll(c1_word);
                size_t v2 = (kw1 << 6) + b1;
                c1_word &= (c1_word - 1);

                // Level 2: C2 = N(v1) & N(v2) & (v > v2)
                const uint64_t* r2 = row(v2);
                size_t w2 = v2 >> 6;
                uint64_t mask2 = ((v2 & 63) == 63) ? 0ULL : ~((1ULL << ((v2 & 63) + 1)) - 1);
                uint32_t c2_pop = 0;

                for (size_t kw2 = w2; kw2 < w; ++kw2) {
                    uint64_t m = (kw2 == w2 ? mask2 : ~0ULL);
                    C2[kw2] = r1[kw2] & r2[kw2] & m;
                    c2_pop += __builtin_popcountll(C2[kw2]);
                }
                if (c2_pop < 3) continue; // Prune: need at least 3 more nodes

                // Level 3: iterate v3 in C2
                for (size_t kw2 = w2; kw2 < w; ++kw2) {
                    uint64_t c2_word = C2[kw2];
                    while (c2_word != 0) {
                        int b2 = __builtin_ctzll(c2_word);
                        size_t v3 = (kw2 << 6) + b2;
                        c2_word &= (c2_word - 1);

                        // C3 = C2 & N(v3) & (v > v3)
                        const uint64_t* r3 = row(v3);
                        size_t w3 = v3 >> 6;
                        uint64_t mask3 = ((v3 & 63) == 63) ? 0ULL : ~((1ULL << ((v3 & 63) + 1)) - 1);
                        uint32_t c3_pop = 0;

                        for (size_t kw3 = w3; kw3 < w; ++kw3) {
                            uint64_t m = (kw3 == w3 ? mask3 : ~0ULL);
                            C3[kw3] = C2[kw3] & r3[kw3] & m;
                            c3_pop += __builtin_popcountll(C3[kw3]);
                        }
                        if (c3_pop < 2) continue; // Prune: need at least 2 more nodes

                        // Level 4: iterate v4 in C3
                        for (size_t kw3 = w3; kw3 < w; ++kw3) {
                            uint64_t c3_word = C3[kw3];
                            while (c3_word != 0) {
                                int b3 = __builtin_ctzll(c3_word);
                                size_t v4 = (kw3 << 6) + b3;
                                c3_word &= (c3_word - 1);

                                // C4 = C3 & N(v4) & (v > v4)
                                const uint64_t* r4 = row(v4);
                                size_t w4 = v4 >> 6;
                                uint64_t mask4 = ((v4 & 63) == 63) ? 0ULL : ~((1ULL << ((v4 & 63) + 1)) - 1);
                                uint32_t c4_pop = 0;

                                for (size_t kw4 = w4; kw4 < w; ++kw4) {
                                    uint64_t m = (kw4 == w4 ? mask4 : ~0ULL);
                                    C4[kw4] = C3[kw4] & r4[kw4] & m;
                                    c4_pop += __builtin_popcountll(C4[kw4]);
                                }
                                if (c4_pop < 1) continue; // Prune: need at least 1 more node

                                // Level 5: collect all v5 in C4
                                for (size_t kw4 = w4; kw4 < w; ++kw4) {
                                    uint64_t c4_word = C4[kw4];
                                    while (c4_word != 0) {
                                        int b4 = __builtin_ctzll(c4_word);
                                        size_t v5 = (kw4 << 6) + b4;
                                        c4_word &= (c4_word - 1);

                                        results.push_back({
                                            static_cast<uint32_t>(v1),
                                            static_cast<uint32_t>(v2),
                                            static_cast<uint32_t>(v3),
                                            static_cast<uint32_t>(v4),
                                            static_cast<uint32_t>(v5)
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return results;
}

std::vector<std::pair<uint32_t, uint32_t>> BitGraph::get_edges() const {
    std::vector<std::pair<uint32_t, uint32_t>> edge_list;
    edge_list.reserve(num_edges_count);
    const size_t w = words_per_row;

    for (size_t u = 0; u < num_nodes; ++u) {
        const uint64_t* r_u = row(u);
        size_t w_u = u >> 6;
        uint64_t mask_u = ((u & 63) == 63) ? 0ULL : ~((1ULL << ((u & 63) + 1)) - 1);

        for (size_t kw = w_u; kw < w; ++kw) {
            uint64_t u_word = r_u[kw] & (kw == w_u ? mask_u : ~0ULL);
            while (u_word != 0) {
                int bit_v = __builtin_ctzll(u_word);
                size_t v = (kw << 6) + bit_v;
                u_word &= (u_word - 1);

                edge_list.emplace_back(static_cast<uint32_t>(u), static_cast<uint32_t>(v));
            }
        }
    }
    return edge_list;
}

std::vector<uint64_t> BitGraph::compute_1wl_colors(size_t max_iters) const {
    const size_t n = num_nodes;
    if (n == 0) return {};
    std::vector<uint64_t> colors(n);
    for (size_t u = 0; u < n; ++u) {
        colors[u] = degree(static_cast<uint32_t>(u));
    }

    auto hash_combine = [](uint64_t seed, uint64_t val) -> uint64_t {
        return seed ^ (val + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
    };

    const size_t w = words_per_row;
    for (size_t it = 0; it < max_iters; ++it) {
        std::vector<uint64_t> next_colors(n);
        for (size_t u = 0; u < n; ++u) {
            const uint64_t* r_u = row(u);
            std::vector<uint64_t> nbr_colors;
            nbr_colors.reserve(degree(static_cast<uint32_t>(u)));
            for (size_t kw = 0; kw < w; ++kw) {
                uint64_t word = r_u[kw];
                while (word != 0) {
                    int bit = __builtin_ctzll(word);
                    size_t v = (kw << 6) + bit;
                    word &= (word - 1);
                    if (v < n) {
                        nbr_colors.push_back(colors[v]);
                    }
                }
            }
            std::sort(nbr_colors.begin(), nbr_colors.end());
            uint64_t h = colors[u] * 0x517cc1b727220a95ULL;
            for (uint64_t nc : nbr_colors) {
                h = hash_combine(h, nc);
            }
            next_colors[u] = h;
        }
        colors = std::move(next_colors);
    }
    return colors;
}

struct LocalEdgeHash {
    std::size_t operator()(const std::pair<uint32_t, uint32_t>& e) const noexcept {
        return (static_cast<std::size_t>(e.first) << 32) ^ static_cast<std::size_t>(e.second);
    }
};

BitGraph::NativeTopologicalFeatures BitGraph::extract_topological_features(
    const std::vector<std::pair<uint32_t, uint32_t>>& edge_list,
    const std::vector<float>& edge_weights
) const {
    const size_t n = num_nodes;
    const size_t m = edge_list.size();
    NativeTopologicalFeatures feats;
    feats.num_nodes = n;
    feats.num_edges = m;
    feats.node_features.assign(n * 5, 0.0f);
    feats.edge_features.assign(m * 8, 0.0f);

    if (n == 0) return feats;

    // 1. Precompute degrees using SIMD popcount
    std::vector<uint32_t> deg(n, 0);
    for (size_t u = 0; u < n; ++u) {
        deg[u] = degree(static_cast<uint32_t>(u));
    }

    // 2. K5 cliques
    auto k5_cliques = find_k5_cliques();
    std::vector<uint32_t> k5_per_node(n, 0);
    std::unordered_map<std::pair<uint32_t, uint32_t>, uint32_t, LocalEdgeHash> k5_per_edge;
    if (!k5_cliques.empty()) {
        k5_per_edge.reserve(k5_cliques.size() * 10);
        for (const auto& clq : k5_cliques) {
            for (uint32_t u : clq) {
                if (u < n) k5_per_node[u]++;
            }
            for (size_t i = 0; i < 5; ++i) {
                for (size_t j = i + 1; j < 5; ++j) {
                    uint32_t u = clq[i], v = clq[j];
                    k5_per_edge[std::make_pair(std::min(u, v), std::max(u, v))]++;
                }
            }
        }
    }

    // 3. Triangles per node and edge common neighbors via SIMD bitwise AND & popcounts
    std::vector<uint32_t> tri_per_node(n, 0);
    std::vector<uint32_t> edge_common(m, 0);
    std::vector<float> edge_aa(m, 0.0f);

    const size_t w = words_per_row;
    for (size_t idx = 0; idx < m; ++idx) {
        uint32_t u = edge_list[idx].first;
        uint32_t v = edge_list[idx].second;
        if (u >= n || v >= n || u == v) continue;

        const uint64_t* r_u = row(u);
        const uint64_t* r_v = row(v);
        uint32_t num_common = 0;
        float aa = 0.0f;

        for (size_t kw = 0; kw < w; ++kw) {
            uint64_t common = r_u[kw] & r_v[kw];
            if (common != 0) {
                num_common += __builtin_popcountll(common);
                uint64_t temp = common;
                while (temp != 0) {
                    int bit = __builtin_ctzll(temp);
                    size_t node_w = (kw << 6) + bit;
                    temp &= (temp - 1);
                    if (node_w < n) {
                        aa += 1.0f / std::log(float(deg[node_w]) + 1.1f);
                    }
                }
            }
        }
        edge_common[idx] = num_common;
        edge_aa[idx] = aa;
        tri_per_node[u] += num_common;
        tri_per_node[v] += num_common;
    }

    // Fill node features (5 dims): [norm_deg, norm_tri, norm_k5, cc, log_deg]
    float max_possible_triangles = (n >= 3) ? float((n - 1) * (n - 2) / 2) : 1.0f;
    if (max_possible_triangles < 1.0f) max_possible_triangles = 1.0f;
    float denom_nodes = float(std::max(size_t(1), n - 1));
    float total_k5 = float(std::max(size_t(1), k5_cliques.size()));

    for (size_t u = 0; u < n; ++u) {
        uint32_t d = deg[u];
        uint32_t actual_tri = tri_per_node[u] / 2;
        float norm_deg = float(d) / denom_nodes;
        float norm_tri = float(actual_tri) / max_possible_triangles;
        float norm_k5 = k5_cliques.empty() ? 0.0f : (float(k5_per_node[u]) / total_k5);
        float cc = (d >= 2) ? (float(2 * actual_tri) / float(d * (d - 1))) : 0.0f;
        float log_deg = std::log1p(float(d));

        size_t base = u * 5;
        feats.node_features[base + 0] = norm_deg;
        feats.node_features[base + 1] = norm_tri;
        feats.node_features[base + 2] = norm_k5;
        feats.node_features[base + 3] = cc;
        feats.node_features[base + 4] = log_deg;
    }

    // Fill edge features (8 dims):
    // [harm_deg, harm_rarity, num_common, jaccard, aa, k5_part, weight, deg_diff]
    for (size_t idx = 0; idx < m; ++idx) {
        uint32_t u = edge_list[idx].first;
        uint32_t v = edge_list[idx].second;
        uint32_t du = (u < n) ? deg[u] : 0;
        uint32_t dv = (v < n) ? deg[v] : 0;
        uint32_t num_common = edge_common[idx];

        float harm_deg = (du + dv > 0) ? (2.0f * float(du) * float(dv) / float(du + dv)) : 0.0f;
        float harm_rarity = (1.0f / (1.0f + float(num_common))) * harm_deg;
        uint32_t union_size = du + dv - num_common;
        float jaccard = float(num_common) / float(std::max(uint32_t(1), union_size));
        float aa = edge_aa[idx];

        auto edge_pair = std::make_pair(std::min(u, v), std::max(u, v));
        auto it_k5 = k5_per_edge.find(edge_pair);
        float k5_part = (it_k5 != k5_per_edge.end()) ? float(it_k5->second) : 0.0f;

        float w_val = (idx < edge_weights.size()) ? edge_weights[idx] : 1.0f;
        float deg_diff = float(std::abs(int(du) - int(dv)));

        size_t base = idx * 8;
        feats.edge_features[base + 0] = harm_deg;
        feats.edge_features[base + 1] = harm_rarity;
        feats.edge_features[base + 2] = float(num_common);
        feats.edge_features[base + 3] = jaccard;
        feats.edge_features[base + 4] = aa;
        feats.edge_features[base + 5] = k5_part;
        feats.edge_features[base + 6] = w_val;
        feats.edge_features[base + 7] = deg_diff;
    }

    return feats;
}

} // namespace highs_turbo
