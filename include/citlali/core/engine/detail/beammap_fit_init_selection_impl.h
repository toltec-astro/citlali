#pragma once

// Beammap fit initialization implementation detail.
// Include this only after Beammap has been declared.

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

bool Beammap::skip_beammap_fit_without_prior_fallback(
    Eigen::Index map_index,
    BeammapFitInitSelection &selection,
    BeammapFitIterationStats &fit_stats) {
    if (has_beammap_prior_diagnostics()) {
        prior_diag_values(map_index, prior_init_mode_col) = -1.0;
    }
    logger->warn(
        "beammap fit map={} skipped: no prior-guided init candidate and fallback_blind=false",
        map_index);
    fit_stats.init_skip++;
    selection.skip_fit = true;
    return true;
}

void Beammap::record_beammap_fit_prior_fallback_blind(Eigen::Index map_index) {
    if (has_beammap_prior_diagnostics()) {
        prior_diag_values(map_index, prior_fallback_blind_col) = 1.0;
    }
}

bool Beammap::try_beammap_prior_fit_init(
    Eigen::Index map_index,
    BeammapFitInitSelection &selection,
    BeammapFitIterationStats &fit_stats) {
    if (choose_prior_guided_init(map_index, selection.row, selection.col)) {
        selection.from_prior = true;
        selection.mode = BeammapFitInitMode::Prior;
        fit_stats.init_prior++;
        return true;
    }

    if (!citlali::pipeline::beammap_config(*this).priors.fallback_blind) {
        return skip_beammap_fit_without_prior_fallback(
            map_index, selection, fit_stats);
    }

    record_beammap_fit_prior_fallback_blind(map_index);
    return false;
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
        if (try_beammap_prior_fit_init(map_index, selection, fit_stats) &&
            selection.skip_fit) {
            return selection;
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
