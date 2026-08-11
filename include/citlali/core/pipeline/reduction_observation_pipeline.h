#pragma once

#include <citlali/core/pipeline/coherent_iq_mode_sidecar.h>
#include <citlali/core/pipeline/observation_fruit_loop_map_loading.h>
#include <citlali/core/pipeline/observation_output_execution.h>
#include <citlali/core/pipeline/observation_pipeline.h>
#include <citlali/core/pipeline/raw_timestream_provenance_lifecycle.h>
#include <citlali/core/pipeline/stage_profile.h>

namespace citlali::pipeline {

template <bool IsBeammap, auto RawObsMap, auto FilteredObsMap, bool FitMaps,
          class TodProc, class KidsProc, class RawObs, class Logger>
void run_reduction_observation_pipeline(TodProc &todproc, KidsProc &kidsproc,
                                        const RawObs &rawobs,
                                        StageProfileCollector &stage_profile,
                                        const Logger &logger) {
    auto &engine = todproc.engine();

    load_observation_fruit_loop_maps_if_needed<IsBeammap>(engine, logger);
    setup_and_run_observation_pipeline(
        engine, kidsproc, rawobs, stage_profile, logger);
    finalize_complete_calibration_product_identity_if_available(engine);
    write_coherent_iq_mode_sidecar_if_requested(
        engine, rawobs, logger);
    write_observation_outputs_and_accumulate<RawObsMap, FilteredObsMap,
                                             FitMaps>(
        todproc, stage_profile, logger);
    const auto raw_provenance_path =
        publish_completed_raw_timestream_provenance<IsBeammap>(engine);
    if (raw_provenance_path) {
        logger->info("raw timestream provenance sidecar: {}",
                     raw_provenance_path->string());
    }
}

}  // namespace citlali::pipeline
