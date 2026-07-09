#pragma once

// Beammap implementation detail.
// Include this only after Beammap has been declared.

#include <citlali/core/engine/detail/beammap_mapmaking_stage_impl.h>
#include <citlali/core/engine/detail/beammap_ptc_cleaning_impl.h>
#include <citlali/core/engine/detail/beammap_fit_stage_impl.h>
#include <citlali/core/pipeline/output_policy.h>

bool Beammap::update_beammap_convergence_state() {
    if (!has_completed_beammap_measurement_iter(current_iter)) {
        return false;
    }

    // only do convergence test if tolerance is above zero, otherwise run all iterations
    const auto &iteration_config = typed_config.beammap.iteration;
    if (citlali::pipeline::mapmaking_enabled(*this) &&
        iteration_config.tolerance > 0) {
        // loop through maps and check if it is converged
        logger->info("checking convergence in fitted-source aperture radius={:.3f} arcsec",
                     iteration_config.convergence_radius_arcsec);
        const auto convergence_profile_scope =
            citlali::pipeline::profile_stage(
                "beammap.convergence", logger,
                "iter=" + std::to_string(current_iter) +
                    " radius_arcsec=" +
                    std::to_string(
                        iteration_config.convergence_radius_arcsec));
        Eigen::VectorXd convergence_delta =
            Eigen::VectorXd::Constant(n_maps, std::numeric_limits<double>::quiet_NaN());
        grppi::map(tula::grppi_utils::dyn_ex(omb.parallel_policy), det_in_vec, det_out_vec, [&](auto i) {
            if (!converged(i)) {
                const double delta = calc_beammap_convergence_delta(i);
                convergence_delta(i) = delta;
                if (std::isfinite(delta) &&
                    delta <= iteration_config.tolerance) {
                    // set as converged
                    converged(i) = true;
                    // set convergence iteration
                    converge_iter(i) = current_iter;
                }
            }
            return 0;
        });

        Eigen::Index n_delta_finite = 0;
        Eigen::Index n_delta_invalid = 0;
        double max_delta = 0.0;
        for (Eigen::Index i = 0; i < convergence_delta.size(); ++i) {
            if (std::isfinite(convergence_delta(i))) {
                n_delta_finite++;
                max_delta = std::max(max_delta, convergence_delta(i));
            }
            else if (!converged(i)) {
                n_delta_invalid++;
            }
        }

        logger->info(
            "{} maps converged on iter {} (finite_metrics={} invalid_metrics={} max_delta={})",
            (converged.array() == true).count(), current_iter,
            n_delta_finite, n_delta_invalid, max_delta);

        // stop if all maps converged
        if ((converged.array() == true).all()) {
            logger->info("all maps converged");
            return true;
        }
    }
    else {
        logger->info("bypassing convergence check");
    }
    return false;
}

bool Beammap::advance_beammap_iteration_state() {
    bool keep_going = true;

    // increment loop iteration
    current_iter++;

    if (current_iter <
        static_cast<Eigen::Index>(
            typed_config.beammap.iteration.max_iterations)) {
        // check if all detectors are converged
        if ((converged.array() == true).all()) {
            logger->info("all maps converged");
            keep_going = false;
        }
        else if (update_beammap_convergence_state()) {
            keep_going = false;
        }

        // set previous iteration fits to current iteration fits
        p0 = params;
        perror0 = perrors;
    }
    else {
        logger->info("max iteration reached");
        keep_going = false;
    }

    return keep_going;
}

void Beammap::write_or_clear_beammap_ptc_products_for_iter(int completed_iter,
                                                           bool keep_going) {
    const bool beammap_iter_is_final = !keep_going;
    // The default is the actual last attempted iteration, including early
    // convergence, so the saved PTC reflects the final cleaning state.
    const int beammap_tod_output_iter =
        citlali::pipeline::default_beammap_tod_output_iter();
    const bool write_beammap_ptc_this_iter =
        (beammap_tod_output_iter < 0 && beammap_iter_is_final) ||
        (beammap_tod_output_iter >= 0 && completed_iter == beammap_tod_output_iter);
    if (write_beammap_ptc_this_iter) {
        write_beammap_ptc_products(completed_iter);
    }
    else {
        clear_beammap_ptc_diagnostics();
    }
}

template <class KidsProc, class RawObs>
void Beammap::run_loop(KidsProc &kidsproc, RawObs &rawobs) {
    // variable to control iteration
    bool keep_going = true;

    // declare random number generator
    boost::random::mt19937 eng;

    // boost random number generator (0,1)
    boost::random::uniform_int_distribution<> rands{0,1};
    const bool detector_grouping =
        typed_config.mapmaking.grouping ==
        citlali::config::MapGrouping::detector;

    log_beammap_masking_config();

    // iterative loop
    while (keep_going) {
        const bool locator_iter = is_beammap_locator_iter(current_iter);
        const bool measurement_iter = is_beammap_measurement_iter(current_iter);
        const bool first_measurement_iter = is_beammap_first_measurement_iter(current_iter);
        logger->info(
            "starting iter {} phase={} locator_iter={} measurement_start_iter={}",
            current_iter, beammap_iter_phase_name(current_iter),
            typed_config.beammap.phase_strategy.locator_iter,
            typed_config.beammap.phase_strategy.measurement_start_iter);

        const bool rerun_source_aware_rtc =
            maybe_run_beammap_source_aware_rtc(
                kidsproc, rawobs, first_measurement_iter, detector_grouping);

        prepare_beammap_iteration_state(
            rerun_source_aware_rtc, measurement_iter, first_measurement_iter,
            detector_grouping);

        // cleaning (separate from mapmaking loop due to jinc mapmaking parallelization)
        run_beammap_ptc_cleaning_pass(
            locator_iter, measurement_iter, detector_grouping);

        run_beammap_mapmaking_stage(
            locator_iter, measurement_iter, detector_grouping, rands, eng);

        const int completed_iter = current_iter;
        keep_going = advance_beammap_iteration_state();
        write_or_clear_beammap_ptc_products_for_iter(completed_iter, keep_going);
    }
}
