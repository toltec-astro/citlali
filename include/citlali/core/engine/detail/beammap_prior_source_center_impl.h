#pragma once

// Beammap prior source-center implementation detail.
// Include this only after Beammap has been declared.

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <citlali/core/pipeline/reduction_config_accessors.h>

bool Beammap::find_map_weighted_peak(Eigen::Index map_index, Eigen::Index &best_row,
                                     Eigen::Index &best_col, double &best_snr) const {
    best_row = -1;
    best_col = -1;
    best_snr = -std::numeric_limits<double>::infinity();

    if (map_index < 0 || map_index >= map_indices.n_maps) {
        return false;
    }

    const auto &sig = omb.signal[map_index];
    const auto &wt = omb.weight[map_index];
    if (sig.rows() <= 0 || sig.cols() <= 0 || wt.rows() != sig.rows() || wt.cols() != sig.cols()) {
        return false;
    }

    const double center_row = static_cast<double>(sig.rows() - 1) / 2.0;
    const double center_col = static_cast<double>(sig.cols() - 1) / 2.0;
    const double radius_pix = map_fitter.fitting_region_pix;
    const double radius2 = radius_pix * radius_pix;

    auto scan = [&](bool apply_radius) {
        bool found = false;
        for (Eigen::Index row = 0; row < sig.rows(); ++row) {
            for (Eigen::Index col = 0; col < sig.cols(); ++col) {
                const double s = sig(row, col);
                const double w = wt(row, col);
                if (!std::isfinite(s) || !std::isfinite(w) || w <= 0.0) {
                    continue;
                }
                if (apply_radius) {
                    const double dr = static_cast<double>(row) - center_row;
                    const double dc = static_cast<double>(col) - center_col;
                    if (dr * dr + dc * dc >= radius2) {
                        continue;
                    }
                }
                const double snr = s * std::sqrt(w);
                if (!std::isfinite(snr)) {
                    continue;
                }
                if (!found || snr > best_snr) {
                    best_row = row;
                    best_col = col;
                    best_snr = snr;
                    found = true;
                }
            }
        }
        return found;
    };

    if (radius_pix > 0.0 && scan(true)) {
        return true;
    }
    return scan(false);
}

void Beammap::clear_beammap_detector_source_centers() {
    ptcproc.fruit_loops_source_lat.resize(0);
    ptcproc.fruit_loops_source_lon.resize(0);
    ptcproc.fruit_loops_source_valid.resize(0);
    rtcproc.kernel.clear_source_centers();
}

bool Beammap::has_complete_beammap_detector_source_center_state() const {
    return p0.rows() == map_indices.n_maps &&
           p0.cols() >= 3 &&
           good_fits.size() == map_indices.n_maps;
}

bool Beammap::has_valid_previous_beammap_source_center(Eigen::Index map_index) const {
    return good_fits(map_index) &&
           std::isfinite(p0(map_index, 0)) && p0(map_index, 0) > 0.0 &&
           std::isfinite(p0(map_index, 1)) &&
           std::isfinite(p0(map_index, 2));
}

void Beammap::record_beammap_detector_source_center(
    Eigen::Index map_index,
    Eigen::VectorXd &kernel_source_a_fwhm_rad,
    Eigen::VectorXd &kernel_source_b_fwhm_rad,
    std::vector<double> &fwhm_arcsec_values,
    Beammap::BeammapDetectorSourceCenterStats &source_center_stats) {
    ptcproc.fruit_loops_source_lat(map_index) =
        (p0(map_index, 2) - (omb.n_rows - 1) / 2.0) * omb.pixel_size_rad;
    ptcproc.fruit_loops_source_lon(map_index) =
        (p0(map_index, 1) - (omb.n_cols - 1) / 2.0) * omb.pixel_size_rad;
    ptcproc.fruit_loops_source_valid(map_index) = 1;
    source_center_stats.n_valid++;

    if (p0.cols() > 4 &&
        std::isfinite(p0(map_index, 3)) && p0(map_index, 3) > 0.0 &&
        std::isfinite(p0(map_index, 4)) && p0(map_index, 4) > 0.0) {
        kernel_source_a_fwhm_rad(map_index) =
            STD_TO_FWHM * omb.pixel_size_rad * p0(map_index, 3);
        kernel_source_b_fwhm_rad(map_index) =
            STD_TO_FWHM * omb.pixel_size_rad * p0(map_index, 4);
        const double mean_fwhm_arcsec =
            RAD_TO_ASEC * (kernel_source_a_fwhm_rad(map_index) +
                           kernel_source_b_fwhm_rad(map_index)) / 2.0;
        if (std::isfinite(mean_fwhm_arcsec) && mean_fwhm_arcsec > 0.0) {
            fwhm_arcsec_values.push_back(mean_fwhm_arcsec);
            source_center_stats.n_valid_fwhm++;
        }
    }
}

double Beammap::median_beammap_source_fwhm_arcsec(
    std::vector<double> fwhm_arcsec_values) const {
    if (fwhm_arcsec_values.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    std::sort(fwhm_arcsec_values.begin(), fwhm_arcsec_values.end());
    return fwhm_arcsec_values[fwhm_arcsec_values.size() / 2];
}

void Beammap::publish_beammap_detector_kernel_source_centers(
    const Eigen::VectorXd &kernel_source_a_fwhm_rad,
    const Eigen::VectorXd &kernel_source_b_fwhm_rad,
    const std::vector<double> &fwhm_arcsec_values,
    const Beammap::BeammapDetectorSourceCenterStats &source_center_stats) {
    if (!rtcproc.run_kernel) {
        return;
    }

    const double median_fwhm_arcsec =
        median_beammap_source_fwhm_arcsec(fwhm_arcsec_values);
    rtcproc.kernel.set_source_centers(ptcproc.fruit_loops_source_lat,
                                      ptcproc.fruit_loops_source_lon,
                                      ptcproc.fruit_loops_source_valid,
                                      kernel_source_a_fwhm_rad,
                                      kernel_source_b_fwhm_rad);
    logger->info(
        "beammap detector kernel placement using previous-fit centers for {}/{} detector maps on iter {}; fitted kernel FWHM available for {}/{} maps (median={:.3f} arcsec)",
        source_center_stats.n_valid, map_indices.n_maps, current_iter,
        source_center_stats.n_valid_fwhm, map_indices.n_maps, median_fwhm_arcsec);
}

void Beammap::configure_detector_source_centers_from_previous_fit() {
    if (citlali::pipeline::mapmaking_config(*this).grouping !=
        citlali::config::MapGrouping::detector) {
        clear_beammap_detector_source_centers();
        return;
    }

    if (!is_beammap_measurement_iter(current_iter)) {
        clear_beammap_detector_source_centers();
        logger->info(
            "beammap detector source centers unavailable on iter {} phase={}: locator pass has no previous fits "
            "(ptc_mask_radius={:.3f} arcsec)",
            current_iter, beammap_iter_phase_name(current_iter), ptcproc.mask_radius_arcsec);
        return;
    }

    if (!has_complete_beammap_detector_source_center_state()) {
        clear_beammap_detector_source_centers();
        logger->warn(
            "beammap detector source centers unavailable on iter {}: previous-fit state is incomplete "
            "(p0={}x{}, good_fits={})",
            current_iter, p0.rows(), p0.cols(), good_fits.size());
        return;
    }

    ptcproc.fruit_loops_source_lat = Eigen::VectorXd::Zero(map_indices.n_maps);
    ptcproc.fruit_loops_source_lon = Eigen::VectorXd::Zero(map_indices.n_maps);
    ptcproc.fruit_loops_source_valid = Eigen::VectorXi::Zero(map_indices.n_maps);
    Eigen::VectorXd kernel_source_a_fwhm_rad = Eigen::VectorXd::Zero(map_indices.n_maps);
    Eigen::VectorXd kernel_source_b_fwhm_rad = Eigen::VectorXd::Zero(map_indices.n_maps);

    BeammapDetectorSourceCenterStats source_center_stats;
    std::vector<double> fwhm_arcsec_values;
    for (Eigen::Index i = 0; i < map_indices.n_maps; ++i) {
        if (!has_valid_previous_beammap_source_center(i)) {
            continue;
        }
        record_beammap_detector_source_center(
            i, kernel_source_a_fwhm_rad, kernel_source_b_fwhm_rad,
            fwhm_arcsec_values, source_center_stats);
    }

    logger->info(
        "beammap detector source centers using previous-fit centers for {}/{} detector maps "
        "on iter {} (ptc_mask_radius={:.3f} arcsec)",
        source_center_stats.n_valid, map_indices.n_maps, current_iter,
        ptcproc.mask_radius_arcsec);

    publish_beammap_detector_kernel_source_centers(
        kernel_source_a_fwhm_rad, kernel_source_b_fwhm_rad,
        fwhm_arcsec_values, source_center_stats);
}
