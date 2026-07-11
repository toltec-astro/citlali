#pragma once

// Beammap mapmaking stage implementation detail.
// Include this only after Beammap has been declared.

void Beammap::prepare_beammap_iteration_state(bool rerun_source_aware_rtc,
                                              bool measurement_iter,
                                              bool first_measurement_iter,
                                              bool detector_grouping) {
    ptcs = ptcs0;
    calib_scans = calib_scans0;

    const auto &rfi_config =
        citlali::pipeline::beammap_config(*this).rfi_mask;
    if (rfi_config.enabled && detector_grouping &&
        rfi_mask_samples_flagged.size() == calib.n_dets &&
        rfi_mask_scans_flagged.size() == calib.n_dets) {
        rfi_mask_samples_flagged.setZero();
        rfi_mask_scans_flagged.setZero();
    }

    const bool skip_centered_kernel_map_feedback = rerun_source_aware_rtc;
    ptcproc.fruit_loops_kernel_feedback_enabled = !skip_centered_kernel_map_feedback;
    if (skip_centered_kernel_map_feedback) {
        logger->info(
            "beammap detector kernel map feedback disabled on iter {} while building the first source-aware kernel map",
            current_iter);
    }

    // copy previous-iteration maps for source-aperture convergence tests
    const auto &iteration_config =
        citlali::pipeline::beammap_config(*this).iteration;
    if (citlali::pipeline::mapmaking_enabled(*this) &&
        iteration_config.tolerance > 0.0 &&
        measurement_iter) {
        omb_copy.signal = omb.signal;
        omb_copy.weight = omb.weight;
    }

    if (citlali::pipeline::fruit_loops_config(*this).enabled) {
        if (first_measurement_iter && !omb.noise.empty()) {
            omb.calc_median_rms();
        }
        if (measurement_iter) {
            ptcproc.configure_fruit_loops_adaptive_gate(
                omb, calib, citlali::pipeline::active_map_grouping_name(*this),
                false);
        }
    }
}
