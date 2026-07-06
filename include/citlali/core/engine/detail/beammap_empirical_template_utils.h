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

inline double edge_baseline(
    const Eigen::MatrixXd &map, double row0, double col0,
    Eigen::Index template_radius_pix) {
    const Eigen::Index side = 2 * template_radius_pix + 1;
    std::vector<double> edge;
    edge.reserve(static_cast<std::size_t>(4 * side));
    for (Eigen::Index k = -template_radius_pix; k <= template_radius_pix; ++k) {
        edge.push_back(bilinear_sample(
            map, row0 - template_radius_pix, col0 + k));
        edge.push_back(bilinear_sample(
            map, row0 + template_radius_pix, col0 + k));
        edge.push_back(bilinear_sample(
            map, row0 + k, col0 - template_radius_pix));
        edge.push_back(bilinear_sample(
            map, row0 + k, col0 + template_radius_pix));
    }
    return median_finite(std::move(edge));
}

inline double local_peak(
    const Eigen::MatrixXd &map, double row0, double col0, double baseline,
    Eigen::Index peak_radius_pix) {
    double peak = -std::numeric_limits<double>::infinity();
    for (Eigen::Index dr = -peak_radius_pix; dr <= peak_radius_pix; ++dr) {
        for (Eigen::Index dc = -peak_radius_pix; dc <= peak_radius_pix; ++dc) {
            if (dr * dr + dc * dc > peak_radius_pix * peak_radius_pix) {
                continue;
            }
            const double value = bilinear_sample(map, row0 + dr, col0 + dc);
            if (std::isfinite(value)) {
                peak = std::max(peak, value - baseline);
            }
        }
    }
    return std::isfinite(peak) ? peak : std::numeric_limits<double>::quiet_NaN();
}

inline bool extract_normalized_cut(
    const Eigen::MatrixXd &map, double row0, double col0,
    Eigen::Index template_radius_pix, Eigen::Index peak_radius_pix,
    Eigen::MatrixXd &cut, double &peak_amp) {
    const double baseline = edge_baseline(map, row0, col0, template_radius_pix);
    if (!std::isfinite(baseline)) {
        return false;
    }
    peak_amp = local_peak(map, row0, col0, baseline, peak_radius_pix);
    if (!std::isfinite(peak_amp) || peak_amp <= 0.0) {
        return false;
    }

    const Eigen::Index side = 2 * template_radius_pix + 1;
    const Eigen::Index center = template_radius_pix;
    cut.resize(side, side);
    cut.setConstant(std::numeric_limits<double>::quiet_NaN());
    for (Eigen::Index rr = 0; rr < side; ++rr) {
        const Eigen::Index dr = rr - center;
        for (Eigen::Index cc = 0; cc < side; ++cc) {
            const Eigen::Index dc = cc - center;
            const double value = bilinear_sample(map, row0 + dr, col0 + dc);
            if (std::isfinite(value)) {
                cut(rr, cc) = (value - baseline) / peak_amp;
            }
        }
    }
    return true;
}

} // namespace beammap_empirical_template_utils
