#pragma once

// Beammap fit-stage implementation detail.
// Include this only after Beammap has been declared.

#include <citlali/core/engine/detail/beammap_fit_diagnostics_impl.h>
#include <citlali/core/engine/detail/beammap_fit_init_impl.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>
#include <citlali/core/pipeline/post_processing_provenance_lifecycle.h>
#include <citlali/core/pipeline/stage_profile.h>

bool Beammap::can_use_beammap_fit_priors(bool detector_grouping) const {
    return
        citlali::pipeline::beammap_config(*this).priors.enabled &&
        beammap_soft_priors_loaded && detector_grouping;
}

void Beammap::maybe_update_beammap_prior_frame_for_fit(bool can_use_priors) {
    if (can_use_priors) {
        update_prior_frame_estimates();
    }
}

void Beammap::fit_single_beammap_map(
    Eigen::Index map_index,
    bool measurement_iter,
    bool can_use_priors,
    BeammapFitIterationStats &fit_stats) {
    logger->debug("beammap fit checkpoint: map={} begin converged={}",
                  map_index, converged(map_index));

    require_beammap_fit_map_geometry(map_index);
    log_beammap_fit_map_stats(map_index);

    // only fit if not converged
    if (!converged(map_index)) {
        if (!prepare_beammap_fit_map(map_index)) {
            return;
        }

        const double init_fwhm = beammap_init_fwhm_pix(map_index);
        const auto init_selection = choose_beammap_fit_init(
            map_index, measurement_iter, can_use_priors, init_fwhm, fit_stats);
        if (init_selection.skip_fit) {
            clear_beammap_fit_result(map_index);
            return;
        }
        logger->debug("beammap fit map={} init mode={} row={:.3f} col={:.3f}",
                      map_index, beammap_fit_init_mode_name(init_selection.mode),
                      init_selection.row, init_selection.col);
        // fit the maps
        logger->debug("beammap fit checkpoint: map={} call fit_to_gaussian",
                      map_index);
        engine_utils::mapFitter::FitDiagnostics fit_diag;
        auto [det_params, det_perror, good_fit] =
            map_fitter.fit_to_gaussian<engine_utils::mapFitter::beammap>(
                omb.signal[map_index], omb.weight[map_index], init_fwhm,
                init_selection.row, init_selection.col, &fit_diag);
        logger->debug(
            "beammap fit checkpoint: map={} fit_to_gaussian returned good_fit={}",
            map_index, good_fit);

        if (!(det_params.array().isFinite().all() &&
              det_perror.array().isFinite().all())) {
            det_params.setZero();
            det_perror.setZero();
            good_fit = false;
        }

        params.row(map_index) = det_params;
        perrors.row(map_index) = det_perror;
        good_fits(map_index) = good_fit;

        const auto fit_flags = beammap_fit_attempt_flags(fit_diag);
        record_beammap_fit_attempt_stats(
            fit_stats, init_selection.mode, good_fit,
            fit_flags.init_amp_zero, fit_flags.amp_bounds_zero);
        record_beammap_fit_diagnostics(map_index, fit_diag, fit_stats);
    }
    // otherwise keep value from previous iteration
    else {
        restore_converged_beammap_fit_result(map_index);
    }

    logger->debug("beammap fit checkpoint: map={} end good_fit={}",
                  map_index, good_fits(map_index));
}

void Beammap::fit_beammap_maps(bool detector_grouping, bool measurement_iter) {
    BeammapFitIterationStats fit_stats(map_fitter.n_params);

    logger->info("fitting maps");
    logger->info("beammap fit diagnostics enabled");
    const bool can_use_priors = can_use_beammap_fit_priors(detector_grouping);
    maybe_update_beammap_prior_frame_for_fit(can_use_priors);

    // Run beammap fits sequentially. This avoids allocator/covariance instability
    // observed with parallel Ceres fits on some systems.
    {
        const auto fit_profile_scope =
            citlali::pipeline::profile_stage(
                "beammap.fit_maps", logger,
                "iter=" + std::to_string(current_iter) +
                    " phase=" + beammap_iter_phase_name(current_iter));
        for (Eigen::Index i = 0; i < map_indices.n_maps; ++i) {
            fit_single_beammap_map(i, measurement_iter, can_use_priors, fit_stats);
        }
    }

    log_beammap_fit_iteration_stats(fit_stats);

    const auto attempt_count = fit_stats.attempt_prev +
        fit_stats.attempt_prior + fit_stats.attempt_blind;
    const auto failure_count = fit_stats.fail_prev +
        fit_stats.fail_prior + fit_stats.fail_blind;
    citlali::pipeline::record_post_processing_beammap_fits_completed(
        citlali::pipeline::post_processing_plan(*this),
        static_cast<std::size_t>(attempt_count),
        static_cast<std::size_t>(attempt_count - failure_count));
}
