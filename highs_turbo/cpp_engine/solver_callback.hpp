#pragma once

#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <limits>

#include "cut_engine.hpp"

namespace highs_turbo {

constexpr double HIGHS_INFINITY = 1e30;

struct AddedRow {
    double lower;
    double upper;
    std::vector<int> indices;
    std::vector<double> values;
    std::string label;
};

struct WarmStartStats {
    int simplex_iterations;
    double initial_objective;
    double final_objective;
    double bound_improvement;
    double wall_clock_seconds;
};

class SolverCallbackBridge {
private:
    size_t num_edges;
    std::vector<AddedRow> row_pool;
    std::vector<SurrogateCut> certified_surrogate_cuts;
    double latest_primal_obj;
    double latest_dual_obj;

public:
    explicit SolverCallbackBridge(size_t m)
        : num_edges(m), latest_primal_obj(0.0), latest_dual_obj(0.0) {}

    void add_cut_row(
        double upper,
        const std::vector<int>& indices,
        const std::vector<double>& values,
        const std::string& label = ""
    );

    void add_surrogate_cut(const SurrogateCut& cut, const std::string& label = "");

    size_t get_num_rows() const noexcept { return row_pool.size(); }
    const std::vector<AddedRow>& get_rows() const noexcept { return row_pool; }
    const AddedRow& get_latest_row() const;

    size_t get_num_certified_cuts() const noexcept { return certified_surrogate_cuts.size(); }
    const std::vector<SurrogateCut>& get_certified_cuts() const noexcept { return certified_surrogate_cuts; }

    void clear() {
        row_pool.clear();
        certified_surrogate_cuts.clear();
    }

    // Formats row data for HiGHS addRow or addRows call
    void get_flat_row_data(
        std::vector<double>& out_lower,
        std::vector<double>& out_upper,
        std::vector<int>& out_starts,
        std::vector<int>& out_indices,
        std::vector<double>& out_values
    ) const;

};

} // namespace highs_turbo
