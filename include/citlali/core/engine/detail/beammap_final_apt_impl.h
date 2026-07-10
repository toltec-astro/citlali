#pragma once

// Beammap final APT/calibration implementation detail.
// Include this only after Beammap has been declared.

#include <citlali/core/pipeline/reduction_config_accessors.h>

void Beammap::calculate_beammap_detector_sensitivities(
    const std::string &map_parallel_policy) {
    const auto &beammap_config = citlali::pipeline::beammap_config(*this);
    logger->info("calculating sensitivity");
    const auto &sens_psd_limits_hz =
        beammap_config.flagging.sens_psd_limits_hz;
    // parallelize on detectors
    grppi::map(tula::grppi_utils::dyn_ex(map_parallel_policy), det_in_vec, det_out_vec, [&](auto i) {
        Eigen::MatrixXd det_sens, noise_flux;
        // calc sensitivity within psd freq range
        calc_sensitivity(ptcs, det_sens, noise_flux, telescope.d_fsmp, i, {sens_psd_limits_hz[0], sens_psd_limits_hz[1]});
        // copy into apt table
        calib.apt["sens"](i) = tula::alg::median(det_sens);

        return 0;
    });
}

void Beammap::populate_beammap_detector_fit_apt_columns() {
    // rescale fit params from pixel to on-sky units
    calib.apt["amp"] = params.col(0);
    calib.apt["x_t"] = RAD_TO_ASEC*omb.pixel_size_rad*(params.col(1).array() - (omb.n_cols - 1)/2.0);
    calib.apt["y_t"] = RAD_TO_ASEC*omb.pixel_size_rad*(params.col(2).array() - (omb.n_rows - 1)/2.0);
    calib.apt["a_fwhm"] = RAD_TO_ASEC*STD_TO_FWHM*omb.pixel_size_rad*(params.col(3));
    calib.apt["b_fwhm"] = RAD_TO_ASEC*STD_TO_FWHM*omb.pixel_size_rad*(params.col(4));
    calib.apt["angle"] = params.col(5);

    // rescale fit errors from pixel to on-sky units
    calib.apt["amp_err"] = perrors.col(0);
    calib.apt["x_t_err"] = RAD_TO_ASEC*omb.pixel_size_rad*(perrors.col(1));
    calib.apt["y_t_err"] = RAD_TO_ASEC*omb.pixel_size_rad*(perrors.col(2));
    calib.apt["a_fwhm_err"] = RAD_TO_ASEC*STD_TO_FWHM*omb.pixel_size_rad*(perrors.col(3));
    calib.apt["b_fwhm_err"] = RAD_TO_ASEC*STD_TO_FWHM*omb.pixel_size_rad*(perrors.col(4));
    calib.apt["angle_err"] = perrors.col(5);

    // add convergence iteration to apt table
    calib.apt["converge_iter"] = converge_iter.cast<double> ();
}

void Beammap::populate_beammap_mask_diagnostic_apt_columns() {
    if (rfi_mask_samples_flagged.size() == calib.n_dets) {
        calib.apt["rfi_masked_samples"] = rfi_mask_samples_flagged.cast<double>();
    }
    if (rfi_mask_scans_flagged.size() == calib.n_dets) {
        calib.apt["rfi_masked_scans"] = rfi_mask_scans_flagged.cast<double>();
    }
    if (scan_band_mask_samples_flagged.size() == calib.n_dets) {
        calib.apt["scan_band_masked_samples"] = scan_band_mask_samples_flagged.cast<double>();
    }
    if (scan_band_mask_rows_flagged.size() == calib.n_dets) {
        calib.apt["scan_band_masked_rows"] = scan_band_mask_rows_flagged.cast<double>();
    }
    if (scan_band_mask_edge_code.size() == calib.n_dets) {
        calib.apt["scan_band_masked_edge"] = scan_band_mask_edge_code.cast<double>();
    }
    if (scan_band_mask_rejected.size() == calib.n_dets) {
        calib.apt["scan_band_mask_rejected"] = scan_band_mask_rejected.cast<double>();
    }

    const auto &beammap_config = citlali::pipeline::beammap_config(*this);
    if (beammap_config.rfi_mask.enabled &&
        rfi_mask_samples_flagged.size() == calib.n_dets &&
        rfi_mask_scans_flagged.size() == calib.n_dets) {
        const Eigen::Index n_det_masked = (rfi_mask_scans_flagged.array() > 0).count();
        logger->info("beammap rfi mask summary: {} detectors affected, {} total samples masked",
                     n_det_masked, static_cast<long long>(rfi_mask_samples_flagged.cast<double>().sum()));
    }
}

void Beammap::log_beammap_final_bound_summary() {
    if (fit_diag_bound_nhit.size() == map_indices.n_maps &&
        fit_diag_hit_lower.rows() == map_indices.n_maps && fit_diag_hit_upper.rows() == map_indices.n_maps &&
        fit_diag_hit_lower.cols() >= 6 && fit_diag_hit_upper.cols() >= 6) {
        const Eigen::Index n_bound_any = (fit_diag_bound_nhit.array() > 0).count();
        Eigen::VectorXi low_hits = fit_diag_hit_lower.colwise().sum().transpose();
        Eigen::VectorXi high_hits = fit_diag_hit_upper.colwise().sum().transpose();
        logger->info(
            "beammap final bound-hit summary: any_hit={}/{} amp(lo/hi)={}/{} x(lo/hi)={}/{} y(lo/hi)={}/{} a(lo/hi)={}/{} b(lo/hi)={}/{} angle(lo/hi)={}/{}",
            n_bound_any, map_indices.n_maps,
            low_hits(0), high_hits(0),
            low_hits(1), high_hits(1),
            low_hits(2), high_hits(2),
            low_hits(3), high_hits(3),
            low_hits(4), high_hits(4),
            low_hits(5), high_hits(5));
    }
}

void Beammap::write_beammap_final_prior_diagnostics_to_apt() {
    if (final_prior_slot_index_diag.size() == calib.n_dets) {
        calib.apt["final_prior_slot_index"] = final_prior_slot_index_diag.cast<double>();
    }
    if (final_prior_d2_diag.size() == calib.n_dets) {
        calib.apt["final_prior_d2"] = final_prior_d2_diag;
    }
}

void Beammap::refresh_beammap_final_calibration_products() {
    calib.setup();
    update_beammap_array_source_flux_density();
    log_final_network_qc_summary();
}
