#pragma once

// Beammap detector table vector helpers.

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

} // namespace beammap_detector_table_vectors
