#pragma once

#include <citlali/core/pipeline/initial_reduction_geometry_execution.h>
#include <citlali/core/pipeline/reduction_iteration_loop.h>
#include <citlali/core/pipeline/stage_profile.h>

namespace citlali::pipeline {

template <bool IsBeammap, auto RawObsMap, auto FilteredObsMap,
          auto RawCoaddMap, auto FilteredCoaddMap, bool FitMaps,
          class KidsDataProc, class TodProc, class IOCoordinator,
          class CitlaliConfig, class ConfigFilepaths, class MapExtents,
          class MapCoords, class DateObsFactory, class Logger>
bool run_reduction_pipeline(
    TodProc &todproc, const IOCoordinator &co, CitlaliConfig &citlali_config,
    const ConfigFilepaths &config_filepaths, MapExtents &map_extents,
    MapCoords &map_coords, DateObsFactory &&date_obs_factory,
    const Logger &logger) {
    reset_stage_profile();

    {
        const auto profile_scope =
            profile_stage("reduction.prepare_initial_geometry", logger);
        if (!prepare_initial_reduction_geometry<IsBeammap, KidsDataProc>(
                todproc, co, citlali_config, map_extents, map_coords,
                logger)) {
            return false;
        }
    }

    const auto profile_scope = profile_stage("reduction.iterations", logger);
    return run_reduction_iterations<
        IsBeammap, RawObsMap, FilteredObsMap, RawCoaddMap, FilteredCoaddMap,
        FitMaps, KidsDataProc>(
        todproc, co, citlali_config, config_filepaths, map_extents,
        map_coords, date_obs_factory, logger);
}

}  // namespace citlali::pipeline
