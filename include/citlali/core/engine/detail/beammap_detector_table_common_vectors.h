#pragma once

// Beammap detector table common vector helpers.

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <limits>

#include <citlali/core/config/timestream_config.h>

namespace beammap_detector_table_vectors {

inline Eigen::VectorXd double_or_nan(const Eigen::VectorXd &values,
                                     Eigen::Index n_dets,
                                     double scale = 1.0) {
    Eigen::VectorXd out =
        Eigen::VectorXd::Constant(n_dets, std::numeric_limits<double>::quiet_NaN());
    if (values.size() == n_dets) {
        out = (scale * values.array()).matrix();
    }
    return out;
}

inline Eigen::VectorXd int_or_nan(const Eigen::VectorXi &values,
                                  Eigen::Index n_dets) {
    Eigen::VectorXd out =
        Eigen::VectorXd::Constant(n_dets, std::numeric_limits<double>::quiet_NaN());
    if (values.size() == n_dets) {
        out = values.cast<double>();
    }
    return out;
}

inline Eigen::VectorXd positive_scaled_threshold(const Eigen::VectorXd &values,
                                                 Eigen::Index n_dets,
                                                 double scale) {
    Eigen::VectorXd out =
        Eigen::VectorXd::Constant(n_dets, std::numeric_limits<double>::quiet_NaN());
    if (scale <= 0.0 || values.size() != n_dets) {
        return out;
    }
    for (Eigen::Index i = 0; i < n_dets; ++i) {
        if (std::isfinite(values(i)) && values(i) > 0.0) {
            out(i) = scale * values(i);
        }
    }
    return out;
}

struct FruitLoopsSupportVectors {
    Eigen::VectorXd npix;
    Eigen::VectorXd signal_sum;
    Eigen::VectorXd x_span_arcsec;
    Eigen::VectorXd y_span_arcsec;
};

struct FruitLoopsQCVectors {
    Eigen::VectorXd source_x_t;
    Eigen::VectorXd source_y_t;
    Eigen::VectorXd local_sigma;
    Eigen::VectorXd local_sigma_npix;
    Eigen::VectorXd amp_ref;
    Eigen::VectorXd peak_threshold;
    Eigen::VectorXd snr_threshold;
    Eigen::VectorXd adaptive_threshold;
    Eigen::VectorXd support_radius_arcsec;
    FruitLoopsSupportVectors support;
};

struct FitQCSignalVectors {
    Eigen::VectorXd map_rms;
    Eigen::VectorXd fit_sig2noise;
    Eigen::VectorXd map_sig2noise;
    Eigen::VectorXd n_weight_pos;
};

} // namespace beammap_detector_table_vectors
