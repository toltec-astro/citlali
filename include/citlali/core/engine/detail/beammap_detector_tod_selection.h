#pragma once

// Beammap detector-specific TOD scan selection helpers.

namespace beammap_detector_tod_selection {

inline std::vector<Eigen::Index> uniform_scan_indices(int n_uniform,
                                                      Eigen::Index n_scans) {
    std::vector<Eigen::Index> scans;
    scans.reserve(static_cast<std::size_t>(n_uniform));
    for (int i = 0; i < n_uniform; ++i) {
        Eigen::Index scan_index = (n_scans - 1) / 2;
        if (n_uniform > 1) {
            const double frac = static_cast<double>(i) /
                                static_cast<double>(n_uniform - 1);
            scan_index = static_cast<Eigen::Index>(std::llround(frac * (n_scans - 1)));
        }
        scans.push_back(std::clamp<Eigen::Index>(scan_index, 0, n_scans - 1));
    }
    return scans;
}

inline std::vector<Eigen::Index> dense_scan_window(Eigen::Index center_scan,
                                                   int n_dense,
                                                   Eigen::Index n_scans) {
    std::vector<Eigen::Index> scans;
    scans.reserve(static_cast<std::size_t>(n_dense));
    if (n_dense <= 0) {
        return scans;
    }
    if (n_dense > n_scans) {
        for (int i = 0; i < n_dense; ++i) {
            Eigen::Index scan_index = 0;
            if (n_dense > 1) {
                const double frac = static_cast<double>(i) /
                                    static_cast<double>(n_dense - 1);
                scan_index = static_cast<Eigen::Index>(std::llround(frac * (n_scans - 1)));
            }
            scans.push_back(std::clamp<Eigen::Index>(scan_index, 0, n_scans - 1));
        }
        return scans;
    }

    Eigen::Index first_dense =
        center_scan - static_cast<Eigen::Index>((n_dense - 1) / 2);
    first_dense = std::clamp<Eigen::Index>(
        first_dense, 0, std::max<Eigen::Index>(0, n_scans - static_cast<Eigen::Index>(n_dense)));
    for (int i = 0; i < n_dense; ++i) {
        scans.push_back(first_dense + static_cast<Eigen::Index>(i));
    }
    return scans;
}

inline std::size_t flat_detector_slot(Eigen::Index det,
                                      Eigen::Index slot,
                                      Eigen::Index n_slots) {
    return static_cast<std::size_t>(det) * static_cast<std::size_t>(n_slots) +
           static_cast<std::size_t>(slot);
}

template <class GoodFits>
inline std::pair<double, double> detector_source_position(
    Eigen::Index det,
    const GoodFits &good_fits,
    const Eigen::MatrixXd &params,
    const Eigen::VectorXd &apt_x_t,
    const Eigen::VectorXd &apt_y_t,
    double pixel_size_rad,
    Eigen::Index n_cols,
    Eigen::Index n_rows,
    bool &used_fit) {
    used_fit = false;
    double x_arcsec = std::numeric_limits<double>::quiet_NaN();
    double y_arcsec = std::numeric_limits<double>::quiet_NaN();
    const bool fit_ok = det < good_fits.size() && good_fits(det) &&
                        det < params.rows() && params.cols() > 2 &&
                        std::isfinite(params(det, 1)) &&
                        std::isfinite(params(det, 2));
    if (fit_ok) {
        x_arcsec = RAD_TO_ASEC * pixel_size_rad *
                   (params(det, 1) - (n_cols - 1) / 2.0);
        y_arcsec = RAD_TO_ASEC * pixel_size_rad *
                   (params(det, 2) - (n_rows - 1) / 2.0);
        used_fit = true;
    }
    if ((!std::isfinite(x_arcsec) || !std::isfinite(y_arcsec)) &&
        det < apt_x_t.size() && det < apt_y_t.size()) {
        x_arcsec = apt_x_t(det);
        y_arcsec = apt_y_t(det);
    }
    return {x_arcsec, y_arcsec};
}

} // namespace beammap_detector_tod_selection
