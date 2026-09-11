#include <iostream>
#include <vector>
#include <cassert>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <set>

#include "bit_graph.hpp"
#include "cut_engine.hpp"

using namespace highs_turbo;

static uint64_t n_choose_k(uint64_t n, uint64_t k) {
    if (k > n) return 0;
    if (k == 0 || k == n) return 1;
    if (k > n / 2) k = n - k;
    uint64_t res = 1;
    for (uint64_t i = 1; i <= k; ++i) {
        res = res * (n - i + 1) / i;
    }
    return res;
}

void test_word_boundary_graphs() {
    std::cout << ">>> Running test_word_boundary_graphs..." << std::endl;
    std::vector<size_t> boundary_sizes = {
        0, 1, 62, 63, 64, 65, 66, 126, 127, 128, 129, 130, 191, 192, 193, 255, 256, 257
    };

    for (size_t n : boundary_sizes) {
        // 1. Empty Graph
        {
            BitGraph bg(n);
            assert(bg.get_num_nodes() == n);
            assert(bg.get_num_edges() == 0);
            assert(bg.count_triangles() == 0);
            assert(bg.find_triangles().empty());
            assert(bg.find_k5_cliques().empty());
            assert(bg.get_edges().empty());
            if (n > 0) {
                assert(bg.degree(0) == 0);
                assert(bg.degree(n - 1) == 0);
            }
        }

        // 2. Star Graph centered at 0
        if (n >= 2) {
            BitGraph bg(n);
            for (uint32_t i = 1; i < n; ++i) {
                bg.add_edge(0, i);
            }
            assert(bg.get_num_edges() == n - 1);
            assert(bg.degree(0) == n - 1);
            assert(bg.count_triangles() == 0);
            assert(bg.find_triangles().empty());
            assert(bg.find_k5_cliques().empty());
            auto edges = bg.get_edges();
            assert(edges.size() == n - 1);
            for (const auto& [u, v] : edges) {
                assert(u < v);
                assert(u == 0);
            }
        }

        // 3. Star Graph centered at node with (u & 63) == 63
        if (n >= 64) {
            uint32_t center = 63;
            BitGraph bg(n);
            for (uint32_t i = 0; i < n; ++i) {
                if (i != center) bg.add_edge(center, i);
            }
            assert(bg.get_num_edges() == n - 1);
            assert(bg.degree(center) == n - 1);
            assert(bg.count_triangles() == 0);
            assert(bg.find_triangles().empty());
            assert(bg.find_k5_cliques().empty());
            auto edges = bg.get_edges();
            assert(edges.size() == n - 1);
            for (const auto& [u, v] : edges) {
                assert(u < v);
                assert(u == center || v == center);
            }
        }

        // 4. Star Graph centered at node with (u & 63) == 63 on word 1 (node 127)
        if (n >= 128) {
            uint32_t center = 127;
            BitGraph bg(n);
            for (uint32_t i = 0; i < n; ++i) {
                if (i != center) bg.add_edge(center, i);
            }
            assert(bg.get_num_edges() == n - 1);
            assert(bg.degree(center) == n - 1);
            assert(bg.count_triangles() == 0);
            auto edges = bg.get_edges();
            assert(edges.size() == n - 1);
            for (const auto& [u, v] : edges) {
                assert(u < v);
                assert(u == center || v == center);
            }
        }
    }
    std::cout << "  Passed word boundary star and empty graphs." << std::endl;
}

void test_complete_graphs_boundary() {
    std::cout << ">>> Running test_complete_graphs_boundary..." << std::endl;
    // Complete graphs Kn at boundaries
    std::vector<size_t> boundary_ns = {63, 64, 65, 127, 128, 129};

    for (size_t n : boundary_ns) {
        BitGraph bg(n);
        for (uint32_t i = 0; i < n; ++i) {
            for (uint32_t j = i + 1; j < n; ++j) {
                bg.add_edge(i, j);
            }
        }

        uint64_t expected_edges = n * (n - 1) / 2;
        assert(bg.get_num_edges() == expected_edges);

        auto edges = bg.get_edges();
        assert(edges.size() == expected_edges);
        for (const auto& [u, v] : edges) {
            assert(u < v);
            assert(bg.has_edge(u, v));
        }

        uint64_t expected_triangles = n * (n - 1) * (n - 2) / 6;
        uint64_t actual_triangles = bg.count_triangles();
        assert(actual_triangles == expected_triangles);

        // Boundary common neighbor queries
        if (n >= 64) {
            assert(bg.count_common_neighbors(62, 63) == n - 2);
        }
        if (n >= 65) {
            assert(bg.count_common_neighbors(63, 64) == n - 2);
        }
        if (n >= 128) {
            assert(bg.count_common_neighbors(126, 127) == n - 2);
        }
        if (n >= 129) {
            assert(bg.count_common_neighbors(127, 128) == n - 2);
        }

        // For n = 63, 64, 65, verify find_triangles() count and canonical sorting
        if (n <= 65) {
            auto tris = bg.find_triangles();
            assert(tris.size() == expected_triangles);
            for (const auto& tri : tris) {
                assert(tri.size() == 3);
                assert(tri[0] < tri[1] && tri[1] < tri[2]);
            }
        }

        // For n = 63, 64, verify find_k5_cliques() count and canonical sorting
        if (n <= 64) {
            uint64_t expected_k5 = n_choose_k(n, 5);
            auto k5s = bg.find_k5_cliques();
            assert(k5s.size() == expected_k5);
            for (const auto& clq : k5s) {
                assert(clq.size() == 5);
                assert(clq[0] < clq[1] && clq[1] < clq[2] && clq[2] < clq[3] && clq[3] < clq[4]);
            }
        }
    }
    std::cout << "  Passed complete graphs boundary." << std::endl;
}

void test_boundary_crossing_cliques() {
    std::cout << ">>> Running test_boundary_crossing_cliques..." << std::endl;
    // Test planted cliques spanning exactly across word boundary (60..65) and (124..130)
    size_t n = 200;
    BitGraph bg(n);

    // Planted K5 at {61, 62, 63, 64, 65}
    std::vector<uint32_t> k5_nodes_1 = {61, 62, 63, 64, 65};
    for (size_t i = 0; i < k5_nodes_1.size(); ++i) {
        for (size_t j = i + 1; j < k5_nodes_1.size(); ++j) {
            bg.add_edge(k5_nodes_1[i], k5_nodes_1[j]);
        }
    }

    // Planted K5 at {125, 126, 127, 128, 129}
    std::vector<uint32_t> k5_nodes_2 = {125, 126, 127, 128, 129};
    for (size_t i = 0; i < k5_nodes_2.size(); ++i) {
        for (size_t j = i + 1; j < k5_nodes_2.size(); ++j) {
            bg.add_edge(k5_nodes_2[i], k5_nodes_2[j]);
        }
    }

    // Expected triangles: 10 from each K5 = 20
    assert(bg.count_triangles() == 20);
    auto tris = bg.find_triangles();
    assert(tris.size() == 20);

    // Expected K5 cliques: exactly 2
    auto k5s = bg.find_k5_cliques();
    assert(k5s.size() == 2);
    assert(k5s[0] == k5_nodes_1);
    assert(k5s[1] == k5_nodes_2);

    // Edge list verification
    auto edges = bg.get_edges();
    assert(edges.size() == 20);
    for (const auto& [u, v] : edges) {
        assert(u < v);
    }
    std::cout << "  Passed boundary crossing cliques." << std::endl;
}

void test_cut_engine_bounds_check() {
    std::cout << ">>> Running test_cut_engine_bounds_check..." << std::endl;
    CutEngine engine;
    BitGraph bg(10);
    // Add a K5 on {0, 1, 2, 3, 4} (10 edges)
    std::vector<std::pair<uint32_t, uint32_t>> edge_list;
    for (uint32_t i = 0; i < 5; ++i) {
        for (uint32_t j = i + 1; j < 5; ++j) {
            bg.add_edge(i, j);
            edge_list.emplace_back(i, j);
        }
    }
    assert(edge_list.size() == 10);

    // Test 1: separate_violated_k5 with short primal_sol
    std::vector<std::vector<double>> bad_primals = {
        {}, // 0 elements
        {1.0}, // 1 element
        {1.0, 1.0, 1.0, 1.0, 1.0}, // 5 elements
        {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0} // 9 elements (edge_list has 10)
    };

    for (const auto& bad_primal : bad_primals) {
        bool caught_k5 = false;
        try {
            engine.separate_violated_k5(bg, edge_list, bad_primal);
        } catch (const std::invalid_argument& e) {
            caught_k5 = true;
            std::string msg = e.what();
            assert(msg.find("primal_sol size must be >= edge count") != std::string::npos);
        }
        assert(caught_k5);

        bool caught_tri = false;
        try {
            engine.separate_violated_triangles(bg, edge_list, bad_primal);
        } catch (const std::invalid_argument& e) {
            caught_tri = true;
            std::string msg = e.what();
            assert(msg.find("primal_sol size must be >= edge count") != std::string::npos);
        }
        assert(caught_tri);
    }

    // Test 2: Valid primal_sol with exact size (10)
    std::vector<double> valid_primal(10, 0.9);
    auto k5_cuts = engine.separate_violated_k5(bg, edge_list, valid_primal);
    // sum_x = 9.0 > 6.0, violation = 3.0
    assert(k5_cuts.size() == 1);
    assert(k5_cuts[0].violation > 2.9);

    auto tri_cuts = engine.separate_violated_triangles(bg, edge_list, valid_primal);
    assert(!tri_cuts.empty());

    // Test 3: primal_sol larger than edge_list (oversized is valid)
    std::vector<double> oversized_primal(20, 0.9);
    auto k5_oversized = engine.separate_violated_k5(bg, edge_list, oversized_primal);
    assert(k5_oversized.size() == 1);

    std::cout << "  Passed CutEngine bounds check." << std::endl;
}

void test_scalability_and_submillisecond_latency() {
    std::cout << ">>> Running test_scalability_and_submillisecond_latency..." << std::endl;
    std::vector<size_t> sizes = {1000, 2000, 3000, 4000, 5000};

    for (size_t n : sizes) {
        auto t_start = std::chrono::high_resolution_clock::now();
        BitGraph bg(n);
        auto t_alloc = std::chrono::high_resolution_clock::now();
        double alloc_ms = std::chrono::duration<double, std::milli>(t_alloc - t_start).count();

        // Verify 64-byte row alignment
        assert((bg.words_per_row * sizeof(uint64_t)) % 64 == 0);

        // Add 10,000 edges
        size_t num_test_edges = 10000;
        for (size_t e = 0; e < num_test_edges; ++e) {
            uint32_t u = (e * 37) % n;
            uint32_t v = (e * 73 + 1) % n;
            bg.add_edge(u, v);
        }

        // Measure has_edge query latency
        size_t num_queries = 20000;
        auto t0 = std::chrono::high_resolution_clock::now();
        uint64_t dummy = 0;
        for (size_t q = 0; q < num_queries; ++q) {
            uint32_t u = (q * 17) % n;
            uint32_t v = (q * 31 + 5) % n;
            if (bg.has_edge(u, v)) dummy++;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double avg_has_edge_ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / num_queries;

        // Measure count_common_neighbors query latency
        size_t cn_queries = 5000;
        auto t2 = std::chrono::high_resolution_clock::now();
        uint64_t dummy_cn = 0;
        for (size_t q = 0; q < cn_queries; ++q) {
            uint32_t u = (q * 19) % n;
            uint32_t v = (q * 23 + 7) % n;
            dummy_cn += bg.count_common_neighbors(u, v);
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        double avg_cn_ns = std::chrono::duration<double, std::nano>(t3 - t2).count() / cn_queries;

        std::cout << "  N=" << n 
                  << " | Alloc: " << alloc_ms << " ms"
                  << " | has_edge: " << avg_has_edge_ns << " ns"
                  << " | common_neighbors: " << avg_cn_ns << " ns"
                  << " (dummy=" << dummy + dummy_cn << ")"
                  << std::endl;

        // Verify latency is sub-millisecond (sub-microsecond in fact)
        assert(avg_has_edge_ns < 1000.0); // < 1 microsecond << 1 millisecond
        assert(avg_cn_ns < 10000.0);      // < 10 microseconds << 1 millisecond
    }
    std::cout << "  Passed scalability and submillisecond latency." << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Running Adversarial UBSan & ASan Suite" << std::endl;
    std::cout << "========================================" << std::endl;

    test_word_boundary_graphs();
    test_complete_graphs_boundary();
    test_boundary_crossing_cliques();
    test_cut_engine_bounds_check();
    test_scalability_and_submillisecond_latency();

    std::cout << "========================================" << std::endl;
    std::cout << "ALL ADVERSARIAL SANITIZER CHECKS PASSED!" << std::endl;
    std::cout << "========================================" << std::endl;
    return 0;
}
