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

void Beammap::configure_detector_source_centers_from_previous_fit() {
    if (citlali::pipeline::mapmaking_config(*this).grouping !=
        citlali::config::MapGrouping::detector) {
        ptcproc.fruit_loops_source_lat.resize(0);
        ptcproc.fruit_loops_source_lon.resize(0);
        ptcproc.fruit_loops_source_valid.resize(0);
        rtcproc.kernel.clear_source_centers();
        return;
    }

    if (!is_beammap_measurement_iter(current_iter)) {
        ptcproc.fruit_loops_source_lat.resize(0);
        ptcproc.fruit_loops_source_lon.resize(0);
        ptcproc.fruit_loops_source_valid.resize(0);
        rtcproc.kernel.clear_source_centers();
        logger->info(
            "beammap detector source centers unavailable on iter {} phase={}: locator pass has no previous fits "
            "(ptc_mask_radius={:.3f} arcsec)",
            current_iter, beammap_iter_phase_name(current_iter), ptcproc.mask_radius_arcsec);
        return;
    }

    if (p0.rows() != map_indices.n_maps || p0.cols() < 3 || good_fits.size() != map_indices.n_maps) {
        ptcproc.fruit_loops_source_lat.resize(0);
        ptcproc.fruit_loops_source_lon.resize(0);
        ptcproc.fruit_loops_source_valid.resize(0);
        rtcproc.kernel.clear_source_centers();
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

    Eigen::Index n_valid = 0;
    Eigen::Index n_valid_fwhm = 0;
    std::vector<double> fwhm_arcsec_values;
    for (Eigen::Index i = 0; i < map_indices.n_maps; ++i) {
        if (!good_fits(i) ||
            !std::isfinite(p0(i, 0)) || p0(i, 0) <= 0.0 ||
            !std::isfinite(p0(i, 1)) || !std::isfinite(p0(i, 2))) {
            continue;
        }
        ptcproc.fruit_loops_source_lat(i) =
            (p0(i, 2) - (omb.n_rows - 1) / 2.0) * omb.pixel_size_rad;
        ptcproc.fruit_loops_source_lon(i) =
            (p0(i, 1) - (omb.n_cols - 1) / 2.0) * omb.pixel_size_rad;
        ptcproc.fruit_loops_source_valid(i) = 1;
        n_valid++;

        if (p0.cols() > 4 &&
            std::isfinite(p0(i, 3)) && p0(i, 3) > 0.0 &&
            std::isfinite(p0(i, 4)) && p0(i, 4) > 0.0) {
            kernel_source_a_fwhm_rad(i) = STD_TO_FWHM * omb.pixel_size_rad * p0(i, 3);
            kernel_source_b_fwhm_rad(i) = STD_TO_FWHM * omb.pixel_size_rad * p0(i, 4);
            const double mean_fwhm_arcsec =
                RAD_TO_ASEC * (kernel_source_a_fwhm_rad(i) + kernel_source_b_fwhm_rad(i)) / 2.0;
            if (std::isfinite(mean_fwhm_arcsec) && mean_fwhm_arcsec > 0.0) {
                fwhm_arcsec_values.push_back(mean_fwhm_arcsec);
                n_valid_fwhm++;
            }
        }
    }

    logger->info(
        "beammap detector source centers using previous-fit centers for {}/{} detector maps "
        "on iter {} (ptc_mask_radius={:.3f} arcsec)",
        n_valid, map_indices.n_maps, current_iter, ptcproc.mask_radius_arcsec);

    if (rtcproc.run_kernel) {
        double median_fwhm_arcsec = std::numeric_limits<double>::quiet_NaN();
        if (!fwhm_arcsec_values.empty()) {
            std::sort(fwhm_arcsec_values.begin(), fwhm_arcsec_values.end());
            median_fwhm_arcsec = fwhm_arcsec_values[fwhm_arcsec_values.size() / 2];
        }
        rtcproc.kernel.set_source_centers(ptcproc.fruit_loops_source_lat,
                                          ptcproc.fruit_loops_source_lon,
                                          ptcproc.fruit_loops_source_valid,
                                          kernel_source_a_fwhm_rad,
                                          kernel_source_b_fwhm_rad);
        logger->info(
            "beammap detector kernel placement using previous-fit centers for {}/{} detector maps on iter {}; fitted kernel FWHM available for {}/{} maps (median={:.3f} arcsec)",
            n_valid, map_indices.n_maps, current_iter, n_valid_fwhm, map_indices.n_maps, median_fwhm_arcsec);
    }
}
