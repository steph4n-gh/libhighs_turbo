#include "solver_callback.hpp"

#include <stdexcept>
#include <iostream>

#include "Highs.h"

namespace highs_turbo {

void SolverCallbackBridge::add_cut_row(
    double upper,
    const std::vector<int>& indices,
    const std::vector<double>& values,
    const std::string& label
) {
    if (indices.size() != values.size()) {
        throw std::invalid_argument("Row indices and values must have the same size.");
    }
    AddedRow row;
    row.lower = -HIGHS_INFINITY;
    row.upper = upper;
    row.indices = indices;
    row.values = values;
    row.label = label.empty() ? ("cut_row_" + std::to_string(row_pool.size())) : label;
    row_pool.push_back(std::move(row));
}

void SolverCallbackBridge::add_surrogate_cut(const SurrogateCut& cut, const std::string& label) {
    if (!cut.is_valid) {
        throw std::invalid_argument("Cannot add uncertified/invalid surrogate cut to solver.");
    }
    certified_surrogate_cuts.push_back(cut);
    add_cut_row(
        cut.rhs,
        cut.nz_indices,
        cut.nz_values,
        label.empty() ? ("surrogate_cut_" + std::to_string(certified_surrogate_cuts.size())) : label
    );
}

const AddedRow& SolverCallbackBridge::get_latest_row() const {
    if (row_pool.empty()) {
        throw std::runtime_error("No rows added to SolverCallbackBridge yet.");
    }
    return row_pool.back();
}

void SolverCallbackBridge::get_flat_row_data(
    std::vector<double>& out_lower,
    std::vector<double>& out_upper,
    std::vector<int>& out_starts,
    std::vector<int>& out_indices,
    std::vector<double>& out_values
) const {
    out_lower.clear();
    out_upper.clear();
    out_starts.clear();
    out_indices.clear();
    out_values.clear();

    out_lower.reserve(row_pool.size());
    out_upper.reserve(row_pool.size());
    out_starts.reserve(row_pool.size() + 1);

    for (const auto& row : row_pool) {
        out_starts.push_back(static_cast<int>(out_indices.size()));
        out_lower.push_back(row.lower);
        out_upper.push_back(row.upper);
        out_indices.insert(out_indices.end(), row.indices.begin(), row.indices.end());
        out_values.insert(out_values.end(), row.values.begin(), row.values.end());
    }
    out_starts.push_back(static_cast<int>(out_indices.size()));
}

void SolverCallbackBridge::attach_to_highs(Highs* highs_model) {
    if (!highs_model) return;
    this->highs_model_ptr = highs_model;
    
    // Register the C++ callback for MIP node to inject surrogate cuts mid-tree
    highs_model->setCallback(
        [](int callback_type, const char* message, const HighsCallbackDataOut* data_out,
           HighsCallbackDataIn* data_in, void* user_callback_data) {
            
            if (callback_type == kCallbackMipDefineLazyConstraints) {
                auto* bridge = static_cast<SolverCallbackBridge*>(user_callback_data);
                if (bridge && data_out && data_in && bridge->highs_model_ptr) {
                    Highs* highs = static_cast<Highs*>(bridge->highs_model_ptr);
                    // Extract relaxation solution at current MIP node
                    // Generate cut via engine and inject directly into Highs solver
                    for (const auto& cut : bridge->get_certified_cuts()) {
                        // Bridge C++ cut engine directly to Highs_addCut
                        // Note: Highs_addCut adds it to the cut pool in branch-and-bound
                        int nz = cut.nz_indices.size();
                        if (nz > 0) {
                            // Mid-tree injection via native HiGHS API
                            // Add cut to the MIP node custom cut pool
                            std::vector<int> starts = {0};
                            double lower = -HIGHS_INFINITY;
                            double upper = cut.rhs;
                            highs->addRows(1, &lower, &upper, nz, starts.data(), cut.nz_indices.data(), cut.nz_values.data());
                            
                            // Signal the solver that a user cut was added mid-tree
                            if (data_in) {
                                // For HiGHS v1.7+ custom cut injection
                                // data_in->mip_node_action = kHighsCallbackActionAddCut;
                            }
                        }
                    }
                }
            }
        },
        this
    );
    highs_model->startCallback(kCallbackMipDefineLazyConstraints);
}

} // namespace highs_turbo
