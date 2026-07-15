#pragma once

// Beammap implementation detail.
// Include this only after Beammap has been declared.

#include <citlali/core/engine/detail/beammap_apt_keys.h>
#include <citlali/core/pipeline/map_diagnostics.h>
#include <citlali/core/pipeline/output_policy.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>

#include <citlali/core/engine/detail/beammap_loop_finalization_impl.h>

template <class KidsProc, class RawObs>
void Beammap::loop_pipeline(
    KidsProc &kidsproc, RawObs &rawobs,
    citlali::pipeline::StageProfileCollector &stage_profile) {
    const auto &mapmaking_config = citlali::pipeline::mapmaking_config(*this);
    const bool detector_grouping =
        mapmaking_config.grouping ==
        citlali::config::MapGrouping::detector;

    // run iterative stage
    run_loop(kidsproc, rawobs, stage_profile);
    ptcproc.fruit_loops_kernel_feedback_enabled = true;

    // write map summary
    if (citlali::pipeline::verbose_runtime_enabled(*this)) {
        write_map_summary(omb);
    }

    // empty initial ptcdata vector to save memory
    ptcs0.clear();

    const auto map_parallel_policy = omb.parallel_policy;

    if (detector_grouping) {
        finalize_beammap_detector_grouping_outputs(
            map_parallel_policy, mapmaking_config.grouping);
    }
    else {
        finalize_beammap_non_detector_grouping_outputs(stage_profile);
    }
}
