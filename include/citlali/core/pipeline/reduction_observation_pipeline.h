#pragma once

#include <citlali/core/pipeline/coherent_iq_mode_sidecar.h>
#include <citlali/core/pipeline/observation_fruit_loop_map_loading.h>
#include <citlali/core/pipeline/observation_output_execution.h>
#include <citlali/core/pipeline/observation_pipeline.h>
#include <citlali/core/pipeline/jinc_processing_provenance.h>
#include <citlali/core/pipeline/product_index_file.h>
#include <citlali/core/pipeline/raw_timestream_provenance_lifecycle.h>
#include <citlali/core/pipeline/stage_profile.h>

#include <type_traits>

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
    complete_raw_timestream_observation_if_available<IsBeammap>(engine);
    bind_jinc_processing_realization_if_available(engine);
    write_coherent_iq_mode_sidecar_if_requested(
        engine, rawobs, logger);
    write_observation_outputs_and_accumulate<RawObsMap, FilteredObsMap,
                                             FitMaps>(
        todproc, stage_profile, logger);
    const auto raw_provenance_path =
        publish_completed_raw_timestream_provenance<IsBeammap>(engine);
    if (raw_provenance_path) {
        if constexpr (has_raw_timestream_plan_v<
                          std::remove_reference_t<decltype(engine)>>) {
            const auto &plan = raw_timestream_plan(engine);
            if (plan.observation &&
                plan.observation->native_consumer_route ==
                    NativeConsumerRoute::native_required) {
                // The observation index becomes visible only after the
                // complete compact-v2 lineage sidecar has committed. The
                // reduction-root index is published at the final session
                // boundary after every required reduction sidecar commits.
                write_final_product_index_file(
                    engine.output_paths.obsnum_dir_name,
                    {*raw_provenance_path});
            }
        }
        logger->info("raw timestream provenance sidecar: {}",
                     raw_provenance_path->string());
    }
}

}  // namespace citlali::pipeline
