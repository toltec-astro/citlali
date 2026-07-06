#pragma once

// Beammap empirical-template calibration utility helpers.

namespace beammap_empirical_template_utils {

inline double median_finite(std::vector<double> values) {
    values.erase(std::remove_if(values.begin(), values.end(),
                                [](double v) { return !std::isfinite(v); }),
                 values.end());
    if (values.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    Eigen::Map<Eigen::VectorXd> vec(values.data(), static_cast<Eigen::Index>(values.size()));
    return tula::alg::median(vec);
}

inline double bilinear_sample(const Eigen::MatrixXd &map, double row, double col) {
    if (!std::isfinite(row) || !std::isfinite(col)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const Eigen::Index r0 = static_cast<Eigen::Index>(std::floor(row));
    const Eigen::Index c0 = static_cast<Eigen::Index>(std::floor(col));
    const Eigen::Index r1 = r0 + 1;
    const Eigen::Index c1 = c0 + 1;
    if (r0 < 0 || c0 < 0 || r1 >= map.rows() || c1 >= map.cols()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double v00 = map(r0, c0);
    const double v01 = map(r0, c1);
    const double v10 = map(r1, c0);
    const double v11 = map(r1, c1);
    if (!std::isfinite(v00) || !std::isfinite(v01) ||
        !std::isfinite(v10) || !std::isfinite(v11)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double fr = row - static_cast<double>(r0);
    const double fc = col - static_cast<double>(c0);
    const double v0 = (1.0 - fc) * v00 + fc * v01;
    const double v1 = (1.0 - fc) * v10 + fc * v11;
    return (1.0 - fr) * v0 + fr * v1;
}

} // namespace beammap_empirical_template_utils
