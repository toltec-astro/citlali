#pragma once

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <limits>
#include <vector>

#include <Eigen/Core>
#include <tula/algorithm/ei_stats.h>

namespace citlali::pipeline {

struct MapdiagTailStats {
    double frac_abs3 = std::numeric_limits<double>::quiet_NaN();
    double frac_pos3 = std::numeric_limits<double>::quiet_NaN();
    double frac_neg3 = std::numeric_limits<double>::quiet_NaN();
    double excess_abs3 = std::numeric_limits<double>::quiet_NaN();
    double excess_pos3 = std::numeric_limits<double>::quiet_NaN();
    double excess_neg3 = std::numeric_limits<double>::quiet_NaN();
    double skew = std::numeric_limits<double>::quiet_NaN();
};

inline double mapdiag_vector_median(const std::vector<double> &values,
                                    double fill_value) {
    if (values.empty()) {
        return fill_value;
    }
    Eigen::Map<const Eigen::VectorXd> mapped(
        values.data(), static_cast<Eigen::Index>(values.size()));
    return tula::alg::median(mapped);
}

inline double mapdiag_vector_quantile(std::vector<double> values, double q,
                                      double fill_value) {
    if (values.empty()) {
        return fill_value;
    }
    q = std::clamp(q, 0.0, 1.0);
    std::sort(values.begin(), values.end());
    const double pos = q * static_cast<double>(values.size() - 1);
    const std::size_t i0 = static_cast<std::size_t>(std::floor(pos));
    const std::size_t i1 = static_cast<std::size_t>(std::ceil(pos));
    const double frac = pos - static_cast<double>(i0);
    return values[i0] * (1.0 - frac) + values[i1] * frac;
}

inline std::vector<double> mapdiag_collect_masked_values(
    const Eigen::MatrixXd &matrix, const Eigen::ArrayXXd &mask) {
    std::vector<double> values;
    values.reserve(static_cast<std::size_t>(mask.sum()));
    for (Eigen::Index r=0; r<matrix.rows(); ++r) {
        for (Eigen::Index c=0; c<matrix.cols(); ++c) {
            const double value = matrix(r, c);
            if (mask(r, c) > 0.0 && std::isfinite(value)) {
                values.push_back(value);
            }
        }
    }
    return values;
}

inline double mapdiag_masked_median(const Eigen::MatrixXd &matrix,
                                    const Eigen::ArrayXXd &mask,
                                    double fill_value) {
    return mapdiag_vector_median(
        mapdiag_collect_masked_values(matrix, mask), fill_value);
}

inline MapdiagTailStats mapdiag_tail_stats(const std::vector<double> &values,
                                           double fill_value) {
    MapdiagTailStats stats;
    if (values.size() < 8) {
        return stats;
    }
    const double center = mapdiag_vector_median(values, fill_value);
    if (!std::isfinite(center)) {
        return stats;
    }
    std::vector<double> abs_dev;
    abs_dev.reserve(values.size());
    for (const auto &value : values) {
        abs_dev.push_back(std::abs(value - center));
    }
    const double mad = mapdiag_vector_median(abs_dev, fill_value);
    const double robust_sigma = 1.4826 * mad;
    if (!std::isfinite(robust_sigma) ||
        robust_sigma <= std::numeric_limits<double>::epsilon()) {
        return stats;
    }

    std::size_t n_abs = 0;
    std::size_t n_pos = 0;
    std::size_t n_neg = 0;
    double skew_sum = 0.0;
    for (const auto &value : values) {
        const double z = (value - center) / robust_sigma;
        if (!std::isfinite(z)) {
            continue;
        }
        if (std::abs(z) >= 3.0) {
            ++n_abs;
        }
        if (z >= 3.0) {
            ++n_pos;
        }
        if (z <= -3.0) {
            ++n_neg;
        }
        skew_sum += z * z * z;
    }

    const double n = static_cast<double>(values.size());
    stats.frac_abs3 = static_cast<double>(n_abs) / n;
    stats.frac_pos3 = static_cast<double>(n_pos) / n;
    stats.frac_neg3 = static_cast<double>(n_neg) / n;
    constexpr double gauss_pos3 = 1.3498980316300959e-3;
    constexpr double gauss_abs3 = 2.6997960632601918e-3;
    stats.excess_abs3 = stats.frac_abs3 / gauss_abs3;
    stats.excess_pos3 = stats.frac_pos3 / gauss_pos3;
    stats.excess_neg3 = stats.frac_neg3 / gauss_pos3;
    stats.skew = skew_sum / n;
    return stats;
}

}  // namespace citlali::pipeline
