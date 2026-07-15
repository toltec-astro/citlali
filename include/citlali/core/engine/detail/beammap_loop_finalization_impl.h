#pragma once

// Beammap loop-pipeline implementation detail.
// Include this only after Beammap has been declared.

#include <citlali/core/engine/detail/beammap_final_apt_impl.h>
#include <citlali/core/engine/detail/beammap_final_tod_pointing_impl.h>

void Beammap::finalize_beammap_detector_grouping_outputs(
    const std::string &map_parallel_policy,
    citlali::config::MapGrouping mapmaking_grouping) {
    calculate_beammap_detector_sensitivities(map_parallel_policy);
    populate_beammap_detector_fit_apt_columns();
    populate_beammap_mask_diagnostic_apt_columns();
    log_beammap_final_bound_summary();

    // flag detectors in apt based on config limits
    set_apt_flags();

    // subtract reference detector position and derotate
    process_apt();
    apply_final_network_position_flags();
    update_final_prior_match_diagnostics();
    write_beammap_final_prior_diagnostics_to_apt();
    refresh_beammap_final_calibration_products();
    update_beammap_final_tod_pointing(map_parallel_policy, mapmaking_grouping);
}

void Beammap::finalize_beammap_non_detector_grouping_outputs(
    citlali::pipeline::StageProfileCollector &stage_profile) {
    citlali::pipeline::calculate_map_psd_with_log(
        omb, stage_profile, logger, "calculating map psd");
    citlali::pipeline::calculate_map_histogram_with_log(
        omb, stage_profile, logger, "calculating map histogram");
}
