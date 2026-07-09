#pragma once

// Beammap fit initialization implementation detail.
// Include this only after Beammap has been declared.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>

#include <citlali/core/pipeline/reduction_config_accessors.h>

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

bool Beammap::beammap_prior_position_compatible(
    Eigen::Index map_index, double row, double col,
    double derot_elev_rad, double prior_max_d2,
    double &d2_out) {
    const int array_int = static_cast<int>(map_indices.maps_to_arrays(map_index));
    const int nw_int = static_cast<int>(std::lround(calib.apt["nw"](map_index)));
    const double pix_to_arcsec = RAD_TO_ASEC * omb.pixel_size_rad;
    const double col0 = static_cast<double>(omb.n_cols - 1) / 2.0;
    const double row0 = static_cast<double>(omb.n_rows - 1) / 2.0;
    const double x_raw = pix_to_arcsec * (col - col0);
    const double y_raw = pix_to_arcsec * (row - row0);
    double x_prior = std::numeric_limits<double>::quiet_NaN();
    double y_prior = std::numeric_limits<double>::quiet_NaN();
    d2_out = std::numeric_limits<double>::infinity();
    int slot_index = -1;
    if (!observed_to_prior_frame(array_int, x_raw, y_raw, derot_elev_rad,
                                 x_prior, y_prior, nullptr, nullptr, true)) {
        return false;
    }
    if (!match_prior_slot(array_int, nw_int, x_prior, y_prior,
                          d2_out, slot_index)) {
        return false;
    }
    static_cast<void>(slot_index);
    return prior_max_d2 <= 0.0 || d2_out <= prior_max_d2;
}

bool Beammap::beammap_prior_allows_peak_switch(Eigen::Index map_index,
                                               double prev_row, double prev_col,
                                               Eigen::Index peak_row,
                                               Eigen::Index peak_col) {
    const double derot_elev_rad = get_prior_derot_elev_rad();
    const double prior_max_d2 = effective_prior_max_d2();

    double prev_prior_d2 = std::numeric_limits<double>::infinity();
    double peak_prior_d2 = std::numeric_limits<double>::infinity();
    const bool prev_prior_ok =
        beammap_prior_position_compatible(
            map_index, prev_row, prev_col, derot_elev_rad, prior_max_d2,
            prev_prior_d2);
    const bool peak_prior_ok = beammap_prior_position_compatible(
        map_index,
        static_cast<double>(peak_row), static_cast<double>(peak_col),
        derot_elev_rad, prior_max_d2, peak_prior_d2);
    const bool prior_allows_switch = peak_prior_ok || !prev_prior_ok;
    if (!prior_allows_switch) {
        logger->debug(
            "beammap fit map={} kept previous init over stronger weighted peak because prior d2 prev={} peak={} max_d2={}",
            map_index, prev_prior_d2, peak_prior_d2, prior_max_d2);
    }
    return prior_allows_switch;
}

bool Beammap::has_previous_beammap_fit_init_candidate(
    Eigen::Index map_index, bool measurement_iter) const {
    return measurement_iter &&
           good_fits(map_index) &&
           p0.cols() > 2 &&
           std::isfinite(p0(map_index, 0)) && p0(map_index, 0) > 0.0 &&
           std::isfinite(p0(map_index, 1)) &&
           std::isfinite(p0(map_index, 2));
}

bool Beammap::read_previous_beammap_fit_seed(
    Eigen::Index map_index, double prev_row, double prev_col,
    double &seed_signal, double &seed_weight) const {
    seed_signal = std::numeric_limits<double>::quiet_NaN();
    seed_weight = std::numeric_limits<double>::quiet_NaN();

    Eigen::Index prev_row_i =
        static_cast<Eigen::Index>(std::llround(prev_row));
    Eigen::Index prev_col_i =
        static_cast<Eigen::Index>(std::llround(prev_col));
    if (prev_row_i < 0 || prev_row_i >= omb.signal[map_index].rows() ||
        prev_col_i < 0 || prev_col_i >= omb.signal[map_index].cols()) {
        return false;
    }

    seed_weight = omb.weight[map_index](prev_row_i, prev_col_i);
    seed_signal = omb.signal[map_index](prev_row_i, prev_col_i);
    return std::isfinite(seed_weight) && seed_weight > 0.0 &&
           std::isfinite(seed_signal) && seed_signal > 0.0;
}

bool Beammap::should_reject_previous_beammap_fit_for_peak(
    Eigen::Index map_index, double prev_row, double prev_col,
    double seed_signal, double seed_weight, bool can_try_prior,
    double init_fwhm) {
    Eigen::Index peak_row = -1;
    Eigen::Index peak_col = -1;
    double peak_snr = -std::numeric_limits<double>::infinity();
    if (!find_map_weighted_peak(map_index, peak_row, peak_col, peak_snr) ||
        peak_row < 0 || peak_col < 0 || !std::isfinite(peak_snr)) {
        return false;
    }

    const double prev_snr = seed_signal * std::sqrt(seed_weight);
    const double dr = static_cast<double>(peak_row) - prev_row;
    const double dc = static_cast<double>(peak_col) - prev_col;
    const double dist_pix = std::sqrt(dr * dr + dc * dc);
    const double min_switch_dist_pix = std::max(1.0, init_fwhm);
    constexpr double min_switch_snr_ratio = 1.25;
    const bool prior_allows_switch =
        !can_try_prior ||
        beammap_prior_allows_peak_switch(
            map_index, prev_row, prev_col, peak_row, peak_col);
    if (std::isfinite(prev_snr) &&
        peak_snr > min_switch_snr_ratio * prev_snr &&
        dist_pix > min_switch_dist_pix &&
        prior_allows_switch) {
        logger->debug(
            "beammap fit map={} rejected previous init: current weighted peak row={} col={} snr={} is {} pix from previous row={} col={} snr={}",
            map_index, peak_row, peak_col, peak_snr, dist_pix,
            prev_row, prev_col, prev_snr);
        return true;
    }
    return false;
}

Beammap::BeammapPreviousFitInit Beammap::choose_previous_beammap_fit_init(
    Eigen::Index map_index, bool measurement_iter, bool can_try_prior,
    double init_fwhm) {
    BeammapPreviousFitInit result;

    if (!has_previous_beammap_fit_init_candidate(map_index, measurement_iter)) {
        return result;
    }

    const double prev_col = p0(map_index, 1);
    const double prev_row = p0(map_index, 2);
    double seed_signal = std::numeric_limits<double>::quiet_NaN();
    double seed_weight = std::numeric_limits<double>::quiet_NaN();
    bool prev_seed_valid =
        read_previous_beammap_fit_seed(
            map_index, prev_row, prev_col, seed_signal, seed_weight);
    if (prev_seed_valid &&
        should_reject_previous_beammap_fit_for_peak(
            map_index, prev_row, prev_col, seed_signal, seed_weight,
            can_try_prior, init_fwhm)) {
        prev_seed_valid = false;
        result.rejected_by_peak = true;
    }

    if (prev_seed_valid) {
        result.valid = true;
        result.col = prev_col;
        result.row = prev_row;
    }
    else {
        logger->debug(
            "beammap fit map={} rejected previous init at row={} col={} due to invalid/no-weight/non-positive seed pixel",
            map_index, prev_row, prev_col);
    }
    return result;
}

void Beammap::record_beammap_prior_init_mode(
    Eigen::Index map_index, const BeammapFitInitSelection &init_selection) {
    if (!has_beammap_prior_diagnostics()) {
        return;
    }
    if (init_selection.from_previous) {
        prior_diag_values(map_index, prior_init_mode_col) = 1.0;
    }
    else if (init_selection.from_prior) {
        prior_diag_values(map_index, prior_init_mode_col) = 2.0;
    }
    else {
        prior_diag_values(map_index, prior_init_mode_col) = 0.0;
    }
}

Beammap::BeammapFitInitSelection Beammap::choose_beammap_fit_init(
    Eigen::Index map_index, bool measurement_iter, bool can_try_prior,
    double init_fwhm, BeammapFitIterationStats &fit_stats) {
    BeammapFitInitSelection selection;

    const auto prev_init = choose_previous_beammap_fit_init(
        map_index, measurement_iter, can_try_prior, init_fwhm);
    if (prev_init.rejected_by_peak) {
        fit_stats.prev_rejected_by_peak++;
    }
    if (prev_init.valid) {
        selection.col = prev_init.col;
        selection.row = prev_init.row;
        selection.from_previous = true;
        selection.mode = BeammapFitInitMode::Previous;
        fit_stats.init_prev++;
        record_beammap_prior_init_mode(map_index, selection);
        return selection;
    }

    if (can_try_prior) {
        if (choose_prior_guided_init(map_index, selection.row, selection.col)) {
            selection.from_prior = true;
            selection.mode = BeammapFitInitMode::Prior;
            fit_stats.init_prior++;
        }
        else if (!citlali::pipeline::beammap_config(*this)
                      .priors.fallback_blind) {
            if (has_beammap_prior_diagnostics()) {
                prior_diag_values(map_index, prior_init_mode_col) = -1.0;
            }
            logger->warn(
                "beammap fit map={} skipped: no prior-guided init candidate and fallback_blind=false",
                map_index);
            fit_stats.init_skip++;
            selection.skip_fit = true;
            return selection;
        }
        else if (has_beammap_prior_diagnostics()) {
            prior_diag_values(map_index, prior_fallback_blind_col) = 1.0;
        }
    }

    if (!selection.from_prior) {
        fit_stats.init_blind++;
    }
    record_beammap_prior_init_mode(map_index, selection);
    return selection;
}

const char *Beammap::beammap_fit_init_mode_name(
    BeammapFitInitMode init_mode) const {
    switch (init_mode) {
        case BeammapFitInitMode::Previous:
            return "previous";
        case BeammapFitInitMode::Prior:
            return "prior";
        case BeammapFitInitMode::Blind:
            return "blind";
    }
    return "unknown";
}
