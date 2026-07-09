#pragma once

// Beammap detector-specific TOD source selection helpers.

#include <Eigen/Core>

#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace beammap_detector_tod_selection {

template <class SampledTelData, class PointingOffsets, class PixelAxes,
          class MapGrouping>
inline Eigen::Index scan_distances_for_detector_source(
    Eigen::Index det,
    double source_x_arcsec,
    double source_y_arcsec,
    Eigen::Index n_scans,
    Eigen::Index n_sampled,
    const std::vector<Eigen::Index> &sampled_scan,
    SampledTelData &sampled_tel_data,
    const Eigen::VectorXd &apt_x_t,
    const Eigen::VectorXd &apt_y_t,
    const PixelAxes &pixel_axes,
    PointingOffsets &pointing_offsets,
    MapGrouping map_grouping,
    std::vector<double> &distances_arcsec) {
    Eigen::Index best_scan = (n_scans - 1) / 2;
    distances_arcsec.assign(static_cast<std::size_t>(n_scans),
                            std::numeric_limits<double>::quiet_NaN());
    if (!std::isfinite(source_x_arcsec) || !std::isfinite(source_y_arcsec) ||
        det < 0 || det >= apt_x_t.size() || det >= apt_y_t.size() ||
        !std::isfinite(apt_x_t(det)) || !std::isfinite(apt_y_t(det))) {
        return best_scan;
    }

    double best_d2 = std::numeric_limits<double>::infinity();
    const double source_x_rad = source_x_arcsec * ASEC_TO_RAD;
    const double source_y_rad = source_y_arcsec * ASEC_TO_RAD;
    // Use the detector pointing that built the map, then find where that
    // pointing passes closest to the fitted source location.
    auto [lat, lon] = engine_utils::calc_det_pointing(
        sampled_tel_data, apt_x_t(det), apt_y_t(det), pixel_axes,
        pointing_offsets, map_grouping, true);

    std::vector<double> best_d2_by_scan(static_cast<std::size_t>(n_scans),
                                        std::numeric_limits<double>::infinity());
    for (Eigen::Index sample_i = 0; sample_i < n_sampled; ++sample_i) {
        if (sample_i >= lat.size() || sample_i >= lon.size()) {
            continue;
        }
        const double y = lat(sample_i) - source_y_rad;
        const double x = lon(sample_i) - source_x_rad;
        if (!std::isfinite(x) || !std::isfinite(y)) {
            continue;
        }
        const auto scan_index = sampled_scan[static_cast<std::size_t>(sample_i)];
        if (scan_index < 0 || scan_index >= n_scans) {
            continue;
        }
        const double d2 = x * x + y * y;
        auto &scan_best = best_d2_by_scan[static_cast<std::size_t>(scan_index)];
        if (d2 < scan_best) {
            scan_best = d2;
        }
        if (d2 < best_d2) {
            best_d2 = d2;
            best_scan = scan_index;
        }
    }
    for (Eigen::Index scan_index = 0; scan_index < n_scans; ++scan_index) {
        const double d2 = best_d2_by_scan[static_cast<std::size_t>(scan_index)];
        if (std::isfinite(d2)) {
            distances_arcsec[static_cast<std::size_t>(scan_index)] =
                std::sqrt(d2) * RAD_TO_ASEC;
        }
    }
    return best_scan;
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
