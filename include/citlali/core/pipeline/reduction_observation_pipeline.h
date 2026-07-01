#pragma once

#include <citlali/core/pipeline/fruit_loop_map_loading.h>
#include <citlali/core/pipeline/observation_output_execution.h>
#include <citlali/core/pipeline/observation_pipeline.h>

namespace citlali::pipeline {

template <bool IsBeammap, auto RawObsMap, auto FilteredObsMap, bool FitMaps,
          class TodProc, class KidsProc, class RawObs, class Logger>
void run_reduction_observation_pipeline(TodProc &todproc, KidsProc &kidsproc,
                                        const RawObs &rawobs,
                                        const Logger &logger) {
    auto &engine = todproc.engine();

    load_observation_fruit_loop_maps_if_needed<IsBeammap>(engine, logger);
    setup_and_run_observation_pipeline(engine, kidsproc, rawobs, logger);
    write_observation_outputs_and_accumulate<RawObsMap, FilteredObsMap,
                                             FitMaps>(todproc, logger);
}

}  // namespace citlali::pipeline
