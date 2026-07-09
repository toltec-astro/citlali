#pragma once

// Beammap fit-stage implementation detail.
// Include this only after Beammap has been declared.

#include <citlali/core/engine/detail/beammap_fit_diagnostics_impl.h>
#include <citlali/core/engine/detail/beammap_fit_init_impl.h>
#include <citlali/core/pipeline/stage_profile.h>

void Beammap::fit_beammap_maps(bool detector_grouping, bool measurement_iter) {
    BeammapFitIterationStats fit_stats(map_fitter.n_params);

    logger->info("fitting maps");
    logger->info("beammap fit diagnostics enabled");
    if (typed_config.beammap.priors.enabled && beammap_soft_priors_loaded &&
        detector_grouping) {
        update_prior_frame_estimates();
    }
    // Run beammap fits sequentially. This avoids allocator/covariance instability
    // observed with parallel Ceres fits on some systems.
    {
        const auto fit_profile_scope =
            citlali::pipeline::profile_stage(
                "beammap.fit_maps", logger,
                "iter=" + std::to_string(current_iter) +
                    " phase=" + beammap_iter_phase_name(current_iter));
        for (Eigen::Index i = 0; i < map_indices.n_maps; ++i) {
            logger->debug("beammap fit checkpoint: map={} begin converged={}", i, converged(i));

            require_beammap_fit_map_geometry(i);
            log_beammap_fit_map_stats(i);

            // only fit if not converged
            if (!converged(i)) {
                if (!prepare_beammap_fit_map(i)) {
                    continue;
                }

                const double init_fwhm = beammap_init_fwhm_pix(i);
                const bool can_try_prior =
                    typed_config.beammap.priors.enabled && beammap_soft_priors_loaded &&
                    detector_grouping;
                const auto init_selection = choose_beammap_fit_init(
                    i, measurement_iter, can_try_prior, init_fwhm, fit_stats);
                if (init_selection.skip_fit) {
                    clear_beammap_fit_result(i);
                    continue;
                }
                logger->debug("beammap fit map={} init mode={} row={:.3f} col={:.3f}",
                              i, beammap_fit_init_mode_name(init_selection.mode),
                              init_selection.row, init_selection.col);
                // fit the maps
                logger->debug("beammap fit checkpoint: map={} call fit_to_gaussian", i);
                engine_utils::mapFitter::FitDiagnostics fit_diag;
                auto [det_params, det_perror, good_fit] =
                    map_fitter.fit_to_gaussian<engine_utils::mapFitter::beammap>(
                        omb.signal[i], omb.weight[i], init_fwhm,
                        init_selection.row, init_selection.col, &fit_diag);
                logger->debug(
                    "beammap fit checkpoint: map={} fit_to_gaussian returned good_fit={}",
                    i, good_fit);

                if (!(det_params.array().isFinite().all() &&
                      det_perror.array().isFinite().all())) {
                    det_params.setZero();
                    det_perror.setZero();
                    good_fit = false;
                }

                params.row(i) = det_params;
                perrors.row(i) = det_perror;
                good_fits(i) = good_fit;

                const auto fit_flags = beammap_fit_attempt_flags(fit_diag);
                record_beammap_fit_attempt_stats(
                    fit_stats, init_selection.mode, good_fit,
                    fit_flags.init_amp_zero, fit_flags.amp_bounds_zero);
                record_beammap_fit_diagnostics(i, fit_diag, fit_stats);
            }
            // otherwise keep value from previous iteration
            else {
                restore_converged_beammap_fit_result(i);
            }

            logger->debug("beammap fit checkpoint: map={} end good_fit={}", i, good_fits(i));
        }
    }

    log_beammap_fit_iteration_stats(fit_stats);
}
