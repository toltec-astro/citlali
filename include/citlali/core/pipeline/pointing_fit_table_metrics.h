#pragma once

#include <cmath>
#include <limits>
#include <optional>

#include <Eigen/Core>

namespace citlali::pipeline {

struct PointingFitTableMetrics {
    float legacy_sig2noise;
    float peak_over_full_map_rms;
    float fit_sig2noise;
};

inline float pointing_finite_ratio(double numerator, double denominator) {
    if (!std::isfinite(numerator) || !std::isfinite(denominator) ||
        denominator <= 0.0) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    return static_cast<float>(numerator / denominator);
}

inline PointingFitTableMetrics pointing_fit_table_metrics(
    double amplitude, double amplitude_error, double full_map_rms) {
    const float peak_over_full_map_rms =
        pointing_finite_ratio(amplitude, full_map_rms);
    return {
        peak_over_full_map_rms,
        peak_over_full_map_rms,
        pointing_finite_ratio(amplitude, amplitude_error),
    };
}

inline std::optional<float> pointing_fits_header_value(float value) {
    if (!std::isfinite(value)) {
        return std::nullopt;
    }
    return value;
}

inline Eigen::Index pointing_fit_table_legacy_sig2noise_column(
    Eigen::Index n_params) {
    return 2 * n_params + 1;
}

inline Eigen::Index pointing_fit_table_peak_over_full_map_rms_column(
    Eigen::Index n_params) {
    return 2 * n_params + 2;
}

inline Eigen::Index pointing_fit_table_fit_sig2noise_column(
    Eigen::Index n_params) {
    return 2 * n_params + 3;
}

inline Eigen::Index pointing_fit_table_column_count(Eigen::Index n_params) {
    return 2 * n_params + 4;
}

}  // namespace citlali::pipeline
