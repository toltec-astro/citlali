#pragma once

// Included by mapdiag_stats_core.h inside namespace citlali::pipeline.

inline double mapdiag_vector_median(const std::vector<double> &values,
                                    double fill_value) {
    if (values.empty()) {
        return fill_value;
    }
    Eigen::Map<const Eigen::VectorXd> mapped(
        values.data(), static_cast<Eigen::Index>(values.size()));
    return tula::alg::median(mapped);
}

inline double MapdiagStatsContext::median(
    const std::vector<double> &values) const {
    return mapdiag_vector_median(values, fill_value);
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

inline double MapdiagStatsContext::quantile(
    std::vector<double> values, double q) const {
    return mapdiag_vector_quantile(std::move(values), q, fill_value);
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

inline std::vector<double> MapdiagStatsContext::collect_masked_values(
    const Eigen::MatrixXd &matrix, const Eigen::ArrayXXd &mask) const {
    return mapdiag_collect_masked_values(matrix, mask);
}

inline double mapdiag_masked_median(const Eigen::MatrixXd &matrix,
                                    const Eigen::ArrayXXd &mask,
                                    double fill_value) {
    return mapdiag_vector_median(
        mapdiag_collect_masked_values(matrix, mask), fill_value);
}

inline double mapdiag_positive_sqrt_or_fill(double value, double fill_value) {
    if (std::isfinite(value) && value > std::numeric_limits<double>::epsilon()) {
        return std::sqrt(value);
    }
    return fill_value;
}

inline double mapdiag_positive_denominator_ratio_or_fill(double numerator,
                                                         double denominator,
                                                         double fill_value) {
    if (std::isfinite(numerator) && std::isfinite(denominator) &&
        denominator > std::numeric_limits<double>::epsilon()) {
        return numerator / denominator;
    }
    return fill_value;
}

