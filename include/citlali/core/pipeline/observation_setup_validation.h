#pragma once

#include <citlali/core/error/error.h>

#include <Eigen/Core>
#include <fmt/format.h>

#include <cmath>
#include <string_view>

namespace citlali::pipeline {

inline void require_matching_detector_count(Eigen::Index input_count,
                                            Eigen::Index calibration_count) {
    if (input_count != calibration_count) {
        throw citlali::error::io(fmt::format(
            "detector count mismatch between data files ({}) and calibration table ({})",
            input_count, calibration_count));
    }
}

inline double reconcile_sample_rate_hz(double reference_hz,
                                       double observed_hz,
                                       int network_index) {
    if (reference_hz != -1.0 && observed_hz != reference_hz) {
        throw citlali::error::io(fmt::format(
            "sample rate mismatch for Toltec network {}: observed={} Hz reference={} Hz",
            network_index, observed_hz, reference_hz));
    }
    return observed_hz;
}

inline void require_positive_sample_rate_hz(double sample_rate_hz,
                                            std::string_view context) {
    if (!std::isfinite(sample_rate_hz) || sample_rate_hz <= 0.0) {
        throw citlali::error::io(fmt::format(
            "invalid or missing sample rate in {}: {} Hz", context,
            sample_rate_hz));
    }
}

inline void require_nonnegative_extinction_tau(double tau,
                                               std::string_view array_name) {
    if (tau < 0.0) {
        throw citlali::error::runtime(fmt::format(
            "calculated mean {} tau {} is negative", array_name, tau));
    }
}

inline void require_polarization_frequency_groups(bool all_unmatched) {
    if (all_unmatched) {
        throw citlali::error::io(
            "polarized reduction requires matched frequency groups in the calibration table");
    }
}

inline void require_bounded_nonpolarimetric_profile(
    bool polarization_requested) {
    if (polarization_requested) {
        throw citlali::error::invalid_config(
            "polarization processing is unavailable in the bounded nonpolarimetric SCI-ALIGN-001 profile");
    }
}

inline void require_iir_below_nyquist_hz(bool below_nyquist,
                                         double frequency_hz,
                                         double nyquist_hz) {
    if (!below_nyquist) {
        throw citlali::error::invalid_config(fmt::format(
            "timestream raw IIR filter frequency {} Hz must be below Nyquist {} Hz",
            frequency_hz, nyquist_hz));
    }
}

}  // namespace citlali::pipeline
