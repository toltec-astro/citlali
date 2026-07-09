#pragma once

#include <citlali/core/pipeline/reduction_iteration_outputs.h>
#include <citlali/core/pipeline/reduction_iteration_setup.h>
#include <citlali/core/pipeline/reduction_observation_loop.h>
#include <citlali/core/pipeline/stage_profile.h>

#include <string>

namespace citlali::pipeline {

template <bool IsBeammap, auto RawObsMap, auto FilteredObsMap,
          auto RawCoaddMap, auto FilteredCoaddMap, bool FitMaps,
          class KidsDataProc, class TodProc, class IOCoordinator,
          class CitlaliConfig, class ConfigFilepaths, class MapExtents,
          class MapCoords, class DateObsFactory, class Logger>
bool run_reduction_iteration(
    TodProc &todproc, const IOCoordinator &co, CitlaliConfig &citlali_config,
    const ConfigFilepaths &config_filepaths, MapExtents &map_extents,
    MapCoords &map_coords, DateObsFactory &&date_obs_factory,
    const Logger &logger) {
    const auto profile_scope = profile_stage(
        "reduction.iteration", logger,
        "fruit_iter=" + std::to_string(todproc.engine().iteration.fruit_iter));

    begin_reduction_iteration(todproc, config_filepaths, logger);

    if (!run_reduction_iteration_observations<
            IsBeammap, RawObsMap, FilteredObsMap, FitMaps, KidsDataProc>(
            todproc, co, citlali_config, map_extents, map_coords,
            date_obs_factory, logger)) {
        return false;
    }

    finish_reduction_iteration<RawCoaddMap, FilteredCoaddMap>(todproc,
                                                              logger);
    return true;
}

}  // namespace citlali::pipeline
