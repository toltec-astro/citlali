#pragma once

// Beammap fit diagnostics implementation detail.
// Include this only after Beammap has been declared.

#include <cmath>

Beammap::BeammapFitAttemptFlags Beammap::beammap_fit_attempt_flags(
    const engine_utils::mapFitter::FitDiagnostics &fit_diag) const {
    BeammapFitAttemptFlags flags;
    if (fit_diag.valid &&
        fit_diag.init_params.size() > 0 &&
        fit_diag.lower_limits.size() > 0 &&
        fit_diag.upper_limits.size() > 0) {
        const double init_amp = fit_diag.init_params(0);
        const double amp_low = fit_diag.lower_limits(0);
        const double amp_high = fit_diag.upper_limits(0);
        flags.init_amp_zero =
            std::isfinite(init_amp) && std::abs(init_amp) <= 1e-12;
        flags.amp_bounds_zero =
            std::isfinite(amp_low) && std::isfinite(amp_high) &&
            std::abs(amp_high - amp_low) <= 1e-12;
    }
    return flags;
}

void Beammap::record_beammap_fit_attempt_stats(
    BeammapFitIterationStats &fit_stats, BeammapFitInitMode init_mode,
    bool good_fit, bool init_amp_zero, bool amp_bounds_zero) {
    switch (init_mode) {
        case BeammapFitInitMode::Previous:
            fit_stats.attempt_prev++;
            if (!good_fit) {
                fit_stats.fail_prev++;
            }
            if (init_amp_zero) {
                fit_stats.init_amp_zero_prev++;
            }
            if (amp_bounds_zero) {
                fit_stats.amp_bounds_zero_prev++;
            }
            break;
        case BeammapFitInitMode::Prior:
            fit_stats.attempt_prior++;
            if (!good_fit) {
                fit_stats.fail_prior++;
            }
            if (init_amp_zero) {
                fit_stats.init_amp_zero_prior++;
            }
            if (amp_bounds_zero) {
                fit_stats.amp_bounds_zero_prior++;
            }
            break;
        case BeammapFitInitMode::Blind:
            fit_stats.attempt_blind++;
            if (!good_fit) {
                fit_stats.fail_blind++;
            }
            if (init_amp_zero) {
                fit_stats.init_amp_zero_blind++;
            }
            if (amp_bounds_zero) {
                fit_stats.amp_bounds_zero_blind++;
            }
            break;
    }
}

bool Beammap::has_complete_beammap_fit_diagnostics(
    const engine_utils::mapFitter::FitDiagnostics &fit_diag) const {
    return fit_diag.valid &&
           fit_diag.init_params.size() == map_fitter.n_params &&
           fit_diag.lower_limits.size() == map_fitter.n_params &&
           fit_diag.upper_limits.size() == map_fitter.n_params &&
           fit_diag.hit_lower.size() == map_fitter.n_params &&
           fit_diag.hit_upper.size() == map_fitter.n_params;
}

void Beammap::record_beammap_fit_diagnostics(
    Eigen::Index map_index,
    const engine_utils::mapFitter::FitDiagnostics &fit_diag,
    BeammapFitIterationStats &fit_stats) {
    if (!has_complete_beammap_fit_diagnostics(fit_diag)) {
        reset_beammap_fit_diagnostics(map_index);
        return;
    }

    fit_diag_init_params.row(map_index) = fit_diag.init_params.transpose();
    fit_diag_lower_limits.row(map_index) = fit_diag.lower_limits.transpose();
    fit_diag_upper_limits.row(map_index) = fit_diag.upper_limits.transpose();
    fit_diag_hit_lower.row(map_index) = fit_diag.hit_lower.transpose();
    fit_diag_hit_upper.row(map_index) = fit_diag.hit_upper.transpose();

    int bound_code = 0;
    int bound_nhit = 0;
    for (int p = 0; p < map_fitter.n_params; ++p) {
        const bool hit_low = fit_diag.hit_lower(p) != 0;
        const bool hit_high = fit_diag.hit_upper(p) != 0;
        if (hit_low) {
            bound_code |= (1 << (2 * p));
            fit_stats.bound_low(p)++;
            bound_nhit++;
        }
        if (hit_high) {
            bound_code |= (1 << (2 * p + 1));
            fit_stats.bound_high(p)++;
            bound_nhit++;
        }
    }
    fit_diag_bound_code(map_index) = bound_code;
    fit_diag_bound_nhit(map_index) = bound_nhit;
    if (bound_nhit > 0) {
        fit_stats.bound_any++;
    }
}

void Beammap::log_beammap_fit_iteration_stats(
    const BeammapFitIterationStats &fit_stats) {
    logger->info("beammap init summary (iter {}): previous={} prior={} blind={} skipped={} prev_rejected_by_peak={}",
                 current_iter, fit_stats.init_prev, fit_stats.init_prior, fit_stats.init_blind, fit_stats.init_skip,
                 fit_stats.prev_rejected_by_peak);
    logger->info(
        "beammap fit diagnostics (iter {}): prev fail={}/{} init_amp_zero={}/{} amp_bounds_zero={}/{} | "
        "prior fail={}/{} init_amp_zero={}/{} amp_bounds_zero={}/{} | "
        "blind fail={}/{} init_amp_zero={}/{} amp_bounds_zero={}/{}",
        current_iter,
        fit_stats.fail_prev, fit_stats.attempt_prev, fit_stats.init_amp_zero_prev, fit_stats.attempt_prev,
        fit_stats.amp_bounds_zero_prev, fit_stats.attempt_prev,
        fit_stats.fail_prior, fit_stats.attempt_prior, fit_stats.init_amp_zero_prior, fit_stats.attempt_prior,
        fit_stats.amp_bounds_zero_prior, fit_stats.attempt_prior,
        fit_stats.fail_blind, fit_stats.attempt_blind, fit_stats.init_amp_zero_blind, fit_stats.attempt_blind,
        fit_stats.amp_bounds_zero_blind, fit_stats.attempt_blind);

    if (map_fitter.n_params >= 6) {
        logger->info(
            "beammap fit bound summary (iter {}): any_hit={}/{} amp(lo/hi)={}/{} x(lo/hi)={}/{} y(lo/hi)={}/{} a(lo/hi)={}/{} b(lo/hi)={}/{} angle(lo/hi)={}/{}",
            current_iter, fit_stats.bound_any, map_indices.n_maps,
            fit_stats.bound_low(0), fit_stats.bound_high(0),
            fit_stats.bound_low(1), fit_stats.bound_high(1),
            fit_stats.bound_low(2), fit_stats.bound_high(2),
            fit_stats.bound_low(3), fit_stats.bound_high(3),
            fit_stats.bound_low(4), fit_stats.bound_high(4),
            fit_stats.bound_low(5), fit_stats.bound_high(5));
    }
    else {
        logger->info("beammap fit bound summary (iter {}): any_hit={}/{}",
                     current_iter, fit_stats.bound_any, map_indices.n_maps);
    }
    logger->info("number of good fits {}/{}", static_cast<long long>(good_fits.cast<int>().sum()), map_indices.n_maps);
}
