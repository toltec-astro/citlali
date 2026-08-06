#pragma once

// Beammap implementation detail.
// Include this only after Beammap has been declared.

void Beammap::assign_beammap_kids_tone_indices() {
    calib.apt["kids_tone"].resize(calib.n_dets);

    Eigen::Index j = 0;
    calib.apt["kids_tone"](0) = 0;
    for (Eigen::Index i=1; i<calib.n_dets; ++i) {
        if (calib.apt["nw"](i) > calib.apt["nw"](i-1)) {
            j = 0;
        }
        else {
            j++;
        }

        calib.apt["kids_tone"](i) = j;
    }
}

void Beammap::register_beammap_kids_tone_column() {
    calib.apt_header_keys.push_back("kids_tone");
    calib.apt_header_units["kids_tone"] = "N/A";
}

void Beammap::setup_beammap_kids_tone_column() {
    assign_beammap_kids_tone_indices();
    register_beammap_kids_tone_column();
}

void Beammap::resize_beammap_scan_buffers() {
    ptcs0.resize(telescope.scan_indices.cols());
    calib_scans0.resize(telescope.scan_indices.cols());
}

void Beammap::reset_beammap_fit_buffers() {
    p0.setZero(map_indices.n_maps, map_fitter.n_params);
    perror0.setZero(map_indices.n_maps, map_fitter.n_params);
    params.setZero(map_indices.n_maps, map_fitter.n_params);
    perrors.setZero(map_indices.n_maps, map_fitter.n_params);
    fit_diag_init_params.setZero(map_indices.n_maps, map_fitter.n_params);
    fit_diag_lower_limits.setZero(map_indices.n_maps, map_fitter.n_params);
    fit_diag_upper_limits.setZero(map_indices.n_maps, map_fitter.n_params);
    fit_diag_hit_lower.setZero(map_indices.n_maps, map_fitter.n_params);
    fit_diag_hit_upper.setZero(map_indices.n_maps, map_fitter.n_params);
    fit_diag_bound_code.setZero(map_indices.n_maps);
    fit_diag_bound_nhit.setZero(map_indices.n_maps);
    prior_diag_values.resize(map_indices.n_maps, n_prior_diag_cols);
    prior_diag_values.setConstant(std::numeric_limits<double>::quiet_NaN());

    good_fits.setZero(map_indices.n_maps);
}

void Beammap::reset_beammap_mask_diagnostics() {
    rfi_mask_samples_flagged = Eigen::VectorXi::Zero(calib.n_dets);
    rfi_mask_scans_flagged = Eigen::VectorXi::Zero(calib.n_dets);
    scan_band_mask_samples_flagged = Eigen::VectorXi::Zero(calib.n_dets);
    scan_band_mask_rows_flagged = Eigen::VectorXi::Zero(calib.n_dets);
    scan_band_mask_edge_code = Eigen::VectorXi::Zero(calib.n_dets);
    scan_band_mask_rejected = Eigen::VectorXi::Zero(calib.n_dets);
    final_prior_d2_diag = Eigen::VectorXd::Constant(calib.n_dets, std::numeric_limits<double>::quiet_NaN());
    final_prior_slot_index_diag = Eigen::VectorXi::Constant(calib.n_dets, -1);
}

void Beammap::reset_beammap_convergence_state() {
    converged.setZero(map_indices.n_maps);
    converge_iter.resize(map_indices.n_maps);
    converge_iter.setConstant(1);
    current_iter = 0;
}

void Beammap::resize_beammap_state_buffers() {
    beammap_direction_selection = {};
    beammap_direction_selection_initialized = false;
    beammap_direction_products.reset();
    resize_beammap_scan_buffers();
    reset_beammap_fit_buffers();
    reset_beammap_mask_diagnostics();
    reset_beammap_convergence_state();
}
