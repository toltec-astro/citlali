#pragma once

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <limits>
#include <vector>

#include <Eigen/Core>

namespace citlali::pipeline {

template <class Calib>
std::vector<int> diagnostic_array_ids(const Calib &calib, int fill_value) {
    std::vector<int> ids(static_cast<std::size_t>(calib.n_arrays),
                         fill_value);
    for (Eigen::Index i=0; i<calib.n_arrays; ++i) {
        ids[static_cast<std::size_t>(i)] = static_cast<int>(calib.arrays(i));
    }
    return ids;
}

inline double rtcdiag_percentile_sorted(
    const std::vector<double> &sorted_values, double pct) {
    if (sorted_values.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (sorted_values.size() == 1) {
        return sorted_values.front();
    }
    pct = std::min(100.0, std::max(0.0, pct));
    const double pos =
        (pct / 100.0) * static_cast<double>(sorted_values.size() - 1);
    const auto lo = static_cast<std::size_t>(std::floor(pos));
    const auto hi = static_cast<std::size_t>(std::ceil(pos));
    const double frac = pos - static_cast<double>(lo);
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac;
}

}  // namespace citlali::pipeline
