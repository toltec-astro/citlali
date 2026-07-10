#pragma once

// Beammap fit initialization implementation detail.
// Include this only after Beammap has been declared.

void Beammap::require_beammap_fit_map_geometry(Eigen::Index map_index) const {
    if (omb.signal[map_index].rows() != omb.n_rows ||
        omb.signal[map_index].cols() != omb.n_cols ||
        omb.weight[map_index].rows() != omb.n_rows ||
        omb.weight[map_index].cols() != omb.n_cols) {
        logger->error(
            "beammap fit map={} geometry mismatch: signal={}x{} weight={}x{} expected={}x{}",
            map_index, omb.signal[map_index].rows(),
            omb.signal[map_index].cols(), omb.weight[map_index].rows(),
            omb.weight[map_index].cols(), omb.n_rows, omb.n_cols);
        std::exit(EXIT_FAILURE);
    }
}

void Beammap::log_beammap_fit_map_stats(Eigen::Index map_index) const {
    const auto &sig = omb.signal[map_index];
    const auto &wt = omb.weight[map_index];
    const Eigen::Index n_pix = sig.size();
    const Eigen::Index sig_finite = sig.array().isFinite().count();
    const Eigen::Index wt_finite = wt.array().isFinite().count();
    const Eigen::Index wt_pos = (wt.array() > 0.0).count();
    logger->debug(
        "beammap fit map={} stats: sig_finite={}/{} wt_finite={}/{} wt_pos={}/{} sig[min,max]=({:.6g}, {:.6g}) wt[min,max]=({:.6g}, {:.6g})",
        map_index, sig_finite, n_pix, wt_finite, n_pix, wt_pos, n_pix,
        sig.minCoeff(), sig.maxCoeff(), wt.minCoeff(), wt.maxCoeff());
}

bool Beammap::prepare_beammap_fit_map(Eigen::Index map_index) {
    if (has_beammap_prior_diagnostics()) {
        reset_beammap_prior_diagnostics(map_index);
    }

    const Eigen::Index n_weight_pos =
        (omb.weight[map_index].array() > 0.0).count();
    if (n_weight_pos < map_fitter.n_params) {
        logger->warn(
            "beammap fit map={} skipped: insufficient weighted pixels ({})",
            map_index, n_weight_pos);
        clear_beammap_fit_result(map_index);
        return false;
    }
    return true;
}

double Beammap::beammap_init_fwhm_pix(Eigen::Index map_index) {
    const auto array = map_indices.maps_to_arrays(map_index);
    return toltec_io.array_fwhm_arcsec[array] * ASEC_TO_RAD /
           omb.pixel_size_rad;
}

void Beammap::reset_beammap_fit_diagnostics(Eigen::Index map_index) {
    fit_diag_init_params.row(map_index).setZero();
    fit_diag_lower_limits.row(map_index).setZero();
    fit_diag_upper_limits.row(map_index).setZero();
    fit_diag_hit_lower.row(map_index).setZero();
    fit_diag_hit_upper.row(map_index).setZero();
    fit_diag_bound_code(map_index) = 0;
    fit_diag_bound_nhit(map_index) = 0;
}

void Beammap::clear_beammap_fit_result(Eigen::Index map_index) {
    params.row(map_index).setZero();
    perrors.row(map_index).setZero();
    reset_beammap_fit_diagnostics(map_index);
    good_fits(map_index) = false;
}

void Beammap::restore_converged_beammap_fit_result(Eigen::Index map_index) {
    params.row(map_index) = p0.row(map_index);
    perrors.row(map_index) = perror0.row(map_index);
}

bool Beammap::has_beammap_prior_diagnostics() const {
    return prior_diag_values.rows() == map_indices.n_maps &&
           prior_diag_values.cols() == n_prior_diag_cols;
}

void Beammap::reset_beammap_prior_diagnostics(Eigen::Index map_index) {
    prior_diag_values.row(map_index).setConstant(
        std::numeric_limits<double>::quiet_NaN());
    prior_diag_values(map_index, prior_init_mode_col) = -1.0;
    prior_diag_values(map_index, prior_used_col) = 0.0;
    prior_diag_values(map_index, prior_fallback_blind_col) = 0.0;
    prior_diag_values(map_index, prior_no_candidate_reason_col) = 0.0;
    prior_diag_values(map_index, prior_slot_index_col) = -1.0;
}
