#pragma once

// Beammap detector table fruit-loops vector helpers.

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <limits>

#include <citlali/core/engine/detail/beammap_detector_table_common_vectors.h>

namespace beammap_detector_table_vectors {

template <class PtcProc, class MapBuffer>
FruitLoopsSupportVectors fruitloops_support_vectors(
    const PtcProc &ptcproc,
    const MapBuffer &omb,
    Eigen::Index n_dets,
    const Eigen::VectorXd &adaptive_threshold,
    double pix_to_arcsec,
    double fill_value) {
    FruitLoopsSupportVectors out{
        Eigen::VectorXd::Constant(n_dets, fill_value),
        Eigen::VectorXd::Constant(n_dets, fill_value),
        Eigen::VectorXd::Constant(n_dets, fill_value),
        Eigen::VectorXd::Constant(n_dets, fill_value)};

    for (Eigen::Index i = 0; i < n_dets; ++i) {
        if (i >= static_cast<Eigen::Index>(omb.signal.size()) ||
            i >= static_cast<Eigen::Index>(omb.weight.size()) ||
            omb.signal[i].rows() != omb.n_rows ||
            omb.signal[i].cols() != omb.n_cols ||
            omb.weight[i].rows() != omb.n_rows ||
            omb.weight[i].cols() != omb.n_cols) {
            continue;
        }
        const double threshold = adaptive_threshold(i);
        if (!std::isfinite(threshold) || threshold <= 0.0) {
            continue;
        }
        if (ptcproc.fruit_loops_source_valid.size() != n_dets ||
            ptcproc.fruit_loops_source_valid(i) == 0 ||
            !std::isfinite(ptcproc.fruit_loops_source_lat(i)) ||
            !std::isfinite(ptcproc.fruit_loops_source_lon(i))) {
            continue;
        }

        const double center_row =
            ptcproc.fruit_loops_source_lat(i) / omb.pixel_size_rad +
            (omb.n_rows - 1) / 2.0;
        const double center_col =
            ptcproc.fruit_loops_source_lon(i) / omb.pixel_size_rad +
            (omb.n_cols - 1) / 2.0;
        if (!std::isfinite(center_row) || !std::isfinite(center_col)) {
            continue;
        }

        double support_radius_pix = std::numeric_limits<double>::infinity();
        const double support_radius_rad =
            (ptcproc.fruit_loops_adaptive_support_radius_rad.size() == n_dets)
                ? ptcproc.fruit_loops_adaptive_support_radius_rad(i)
                : fill_value;
        if (std::isfinite(support_radius_rad) && support_radius_rad > 0.0) {
            support_radius_pix = support_radius_rad / omb.pixel_size_rad;
        }

        Eigen::Index npix = 0;
        double signal_sum = 0.0;
        double min_x = std::numeric_limits<double>::infinity();
        double max_x = -std::numeric_limits<double>::infinity();
        double min_y = std::numeric_limits<double>::infinity();
        double max_y = -std::numeric_limits<double>::infinity();
        for (Eigen::Index row = 0; row < omb.n_rows; ++row) {
            const double drow_pix = static_cast<double>(row) - center_row;
            for (Eigen::Index col = 0; col < omb.n_cols; ++col) {
                const double weight = omb.weight[i](row, col);
                const double signal = omb.signal[i](row, col);
                if (!std::isfinite(weight) || weight <= 0.0 ||
                    !std::isfinite(signal)) {
                    continue;
                }
                const double dcol_pix = static_cast<double>(col) - center_col;
                if (std::sqrt(drow_pix * drow_pix + dcol_pix * dcol_pix) >
                    support_radius_pix) {
                    continue;
                }
                bool include_pixel = false;
                if (citlali::config::is_upper_fruit_loops_mode(
                        ptcproc.fruit_mode)) {
                    include_pixel = signal >= threshold;
                }
                else if (citlali::config::is_lower_fruit_loops_mode(
                             ptcproc.fruit_mode)) {
                    include_pixel = signal <= -std::abs(threshold);
                }
                else {
                    include_pixel = std::abs(signal) >= threshold;
                }
                if (!include_pixel) {
                    continue;
                }
                const double x_arcsec = dcol_pix * pix_to_arcsec;
                const double y_arcsec = drow_pix * pix_to_arcsec;
                min_x = std::min(min_x, x_arcsec);
                max_x = std::max(max_x, x_arcsec);
                min_y = std::min(min_y, y_arcsec);
                max_y = std::max(max_y, y_arcsec);
                signal_sum += signal;
                ++npix;
            }
        }
        out.npix(i) = static_cast<double>(npix);
        out.signal_sum(i) = signal_sum;
        if (npix > 0) {
            out.x_span_arcsec(i) = max_x - min_x;
            out.y_span_arcsec(i) = max_y - min_y;
        }
    }

    return out;
}

template <class PtcProc, class MapBuffer>
FruitLoopsQCVectors fruitloops_qc_vectors(
    const PtcProc &ptcproc,
    const MapBuffer &omb,
    Eigen::Index n_dets,
    double pix_to_arcsec,
    double fill_value) {
    FruitLoopsQCVectors out{
        double_or_nan(ptcproc.fruit_loops_source_lon, n_dets, RAD_TO_ASEC),
        double_or_nan(ptcproc.fruit_loops_source_lat, n_dets, RAD_TO_ASEC),
        double_or_nan(ptcproc.fruit_loops_local_sigma_map, n_dets),
        int_or_nan(ptcproc.fruit_loops_local_sigma_npix, n_dets),
        double_or_nan(ptcproc.fruit_loops_amp_ref, n_dets),
        Eigen::VectorXd::Constant(n_dets, fill_value),
        Eigen::VectorXd::Constant(n_dets, fill_value),
        double_or_nan(ptcproc.fruit_loops_adaptive_threshold, n_dets),
        double_or_nan(ptcproc.fruit_loops_adaptive_support_radius_rad,
                      n_dets, RAD_TO_ASEC),
        {Eigen::VectorXd(), Eigen::VectorXd(), Eigen::VectorXd(),
         Eigen::VectorXd()}};

    out.peak_threshold = positive_scaled_threshold(
        out.amp_ref, n_dets, ptcproc.fruit_loops_peak_fraction_limit);
    out.snr_threshold = positive_scaled_threshold(
        out.local_sigma, n_dets, ptcproc.fruit_loops_local_snr_floor);
    out.support = fruitloops_support_vectors(
        ptcproc, omb, n_dets, out.adaptive_threshold, pix_to_arcsec,
        fill_value);
    return out;
}

} // namespace beammap_detector_table_vectors
