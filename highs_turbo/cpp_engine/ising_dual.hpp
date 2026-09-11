#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace highs_turbo {

// Coordinate minimization of lambda*b + sum(max(0,w-A'lambda)), lambda>=0.
// This is a numerical search only. Python independently verifies the integer
// cuts and recomputes the bound in exact arithmetic with a residual repair.
inline std::tuple<std::vector<double>, std::vector<double>> ising_dual_ascent(
    const std::vector<int64_t>& starts, const std::vector<int64_t>& indices,
    const std::vector<double>& values, const std::vector<double>& rhs,
    const std::vector<double>& weights, std::vector<double> multipliers,
    int sweeps, double time_limit, unsigned seed) {
    const size_t rows = rhs.size(), columns = weights.size();
    if (starts.size() != rows+1 || starts.front() != 0 ||
        starts.back() != static_cast<int64_t>(indices.size()) || indices.size() != values.size() ||
        multipliers.size() != rows || sweeps < 0 || !std::isfinite(time_limit) || time_limit < 0)
        throw std::invalid_argument("Invalid dual-ascent dimensions or budget");
    for (size_t row = 0; row < rows; ++row)
        if (starts[row] > starts[row+1] || !std::isfinite(rhs[row]) ||
            !std::isfinite(multipliers[row]) || multipliers[row] < 0)
            throw std::invalid_argument("Invalid dual-ascent row");
    for (size_t p = 0; p < indices.size(); ++p)
        if (indices[p] < 0 || static_cast<size_t>(indices[p]) >= columns ||
            !std::isfinite(values[p]) || values[p] == 0)
            throw std::invalid_argument("Invalid dual-ascent coefficient");
    std::vector<double> residual(weights), point(columns), best(multipliers);
    double magnitude = 1.;
    for (double weight : weights) {
        if (!std::isfinite(weight)) throw std::invalid_argument("Non-finite dual-ascent weight");
        magnitude = std::max(magnitude, std::abs(weight));
    }
    for (size_t row = 0; row < rows; ++row)
        for (int64_t p = starts[row]; p < starts[row+1]; ++p)
            residual[indices[p]] -= multipliers[row]*values[p];
    auto objective = [&]() {
        double result = std::inner_product(rhs.begin(), rhs.end(), multipliers.begin(), 0.);
        for (double value : residual) result += std::max(0., value);
        return result;
    };
    double best_value = objective();
    std::vector<size_t> order(rows);
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 random(seed);
    std::vector<std::pair<double, double>> breakpoints;
    auto began = std::chrono::steady_clock::now();
    int stagnant = 0;
    for (int sweep = 0; sweep < sweeps; ++sweep) {
        if (std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count() >= time_limit)
            break;
        std::shuffle(order.begin(), order.end(), random);
        for (size_t row : order) {
            breakpoints.clear();
            double target = -rhs[row];
            for (int64_t p = starts[row]; p < starts[row+1]; ++p) {
                target += std::max(0., values[p]);
                breakpoints.emplace_back(multipliers[row]+residual[indices[p]]/values[p], std::abs(values[p]));
            }
            double next = 0.;
            if (target > 0. && !breakpoints.empty()) {
                std::sort(breakpoints.begin(), breakpoints.end());
                double accumulated = 0.;
                for (size_t p = 0; p < breakpoints.size(); ++p) {
                    accumulated += breakpoints[p].second;
                    if (accumulated >= target) {
                        next = std::max(0., breakpoints[p].first);
                        if (accumulated == target && p+1 < breakpoints.size())
                            next = .5*(next+std::max(0., breakpoints[p+1].first));
                        break;
                    }
                }
            }
            if (!std::isfinite(next)) continue;
            const double delta = next-multipliers[row];
            for (int64_t p = starts[row]; p < starts[row+1]; ++p)
                residual[indices[p]] -= delta*values[p];
            multipliers[row] = next;
        }
        for (size_t column = 0; column < columns; ++column) {
            const double value = residual[column] > 1e-8*magnitude ? 1. :
                                 residual[column] < -1e-8*magnitude ? 0. : .5;
            point[column] = sweep == 0 ? value : .8*point[column]+.2*value;
        }
        double value = objective();
        if (value < best_value) {
            stagnant = best_value-value < 1e-8*magnitude ? stagnant+1 : 0;
            best_value = value;
            best = multipliers;
        } else ++stagnant;
        if (stagnant >= 4) break;
    }
    return {std::move(best), std::move(point)};
}
} // namespace highs_turbo
