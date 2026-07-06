#pragma once

// Beammap prior and final-QC statistics helpers.

namespace beammap_prior_qc_stats {

inline double median_or_nan(const std::vector<double> &values) {
    if (values.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    Eigen::Map<const Eigen::VectorXd> vec(values.data(), static_cast<Eigen::Index>(values.size()));
    return tula::alg::median(vec);
}

inline double quantile(std::vector<double> values, double q) {
    if (values.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    q = std::clamp(q, 0.0, 1.0);
    std::sort(values.begin(), values.end());
    const double pos = q * static_cast<double>(values.size() - 1);
    const auto lo = static_cast<std::size_t>(std::floor(pos));
    const auto hi = static_cast<std::size_t>(std::ceil(pos));
    if (lo == hi) {
        return values[lo];
    }
    const double frac = pos - static_cast<double>(lo);
    return values[lo] * (1.0 - frac) + values[hi] * frac;
}

} // namespace beammap_prior_qc_stats
