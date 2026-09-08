#pragma once

#include <citlali/core/pipeline/fruit_loop_iteration_state.h>
#include <citlali/core/pipeline/processed_timestream_execution_plan.h>
#include <citlali/core/pipeline/reduction_iteration.h>
#include <citlali/core/pipeline/reduction_iteration_setup.h>
#include <citlali/core/pipeline/stage_profile.h>

namespace citlali::pipeline {

template <bool IsBeammap, auto RawObsMap, auto FilteredObsMap,
          auto RawCoaddMap, auto FilteredCoaddMap, bool FitMaps,
          class KidsDataProc, class TodProc, class IOCoordinator,
          class CitlaliConfig, class ConfigFilepaths, class MapExtents,
          class MapCoords, class DateObsFactory, class Logger>
bool run_reduction_iterations(
    TodProc &todproc, const IOCoordinator &co, CitlaliConfig &citlali_config,
    const ConfigFilepaths &config_filepaths, MapExtents &map_extents,
    MapCoords &map_coords, DateObsFactory &&date_obs_factory,
    StageProfileCollector &stage_profile,
    const Logger &logger) {
    ReductionIterationState iteration_state;
    auto &engine = todproc.engine();
    initialize_reduction_iterations(engine, iteration_state, logger);

    while (fruit_loop_iteration_pending(engine, iteration_state)) {
        if (!run_reduction_iteration<
                IsBeammap, RawObsMap, FilteredObsMap, RawCoaddMap,
                FilteredCoaddMap, FitMaps, KidsDataProc>(
                todproc, co, citlali_config, config_filepaths, map_extents,
                map_coords, date_obs_factory, stage_profile, logger)) {
            return false;
        }
    }
    if constexpr (has_processed_timestream_plan_v<decltype(engine)>) {
        auto &plan = processed_timestream_plan(engine);
        if (plan.initialized) {
            record_processed_timestream_iteration_result(
                plan, engine.iteration.fruit_iter,
                iteration_state.fruit_loops_converged);
        }
    }
    return true;
}

}  // namespace citlali::pipeline
