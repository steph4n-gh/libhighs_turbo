#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <utility>
#include <stdexcept>
#include <cstring>
#include <new>
#include <algorithm>

namespace highs_turbo {

class BitGraph {
public:
    size_t num_nodes;
    size_t words_per_row;
    size_t total_words;
    uint64_t* bit_adj; // 64-byte aligned contiguous buffer
    size_t num_edges_count;

    explicit BitGraph(size_t n);
    ~BitGraph();

    // Move semantics
    BitGraph(BitGraph&& other) noexcept;
    BitGraph& operator=(BitGraph&& other) noexcept;

    // Non-copyable to prevent accidental large memory copies
    BitGraph(const BitGraph&) = delete;
    BitGraph& operator=(const BitGraph&) = delete;

    inline const uint64_t* row(size_t u) const noexcept {
        return bit_adj + (u * words_per_row);
    }

    inline uint64_t* row(size_t u) noexcept {
        return bit_adj + (u * words_per_row);
    }

    void add_edge(uint32_t u, uint32_t v);
    void add_edges(const std::vector<std::pair<uint32_t, uint32_t>>& edges);
    bool has_edge(uint32_t u, uint32_t v) const noexcept;

    uint32_t degree(uint32_t u) const noexcept;
    uint32_t count_common_neighbors(uint32_t u, uint32_t v) const noexcept;
    uint32_t count_triangles_on_edge(uint32_t u, uint32_t v) const noexcept;

    uint64_t count_triangles() const noexcept;
    std::vector<std::vector<uint32_t>> find_triangles() const;
    std::vector<std::vector<uint32_t>> find_k5_cliques() const;

    std::vector<uint64_t> compute_1wl_colors(size_t max_iters = 3) const;

    struct NativeTopologicalFeatures {
        size_t num_nodes;
        size_t num_edges;
        std::vector<float> node_features; // num_nodes * 5
        std::vector<float> edge_features; // num_edges * 8
    };

    NativeTopologicalFeatures extract_topological_features(
        const std::vector<std::pair<uint32_t, uint32_t>>& edge_list,
        const std::vector<float>& edge_weights = {}
    ) const;

    std::vector<std::pair<uint32_t, uint32_t>> get_edges() const;
    size_t get_num_nodes() const noexcept { return num_nodes; }
    size_t get_num_edges() const noexcept { return num_edges_count; }
};

} // namespace highs_turbo
