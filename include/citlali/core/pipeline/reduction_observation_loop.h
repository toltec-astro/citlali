#pragma once

#include <citlali/core/pipeline/kids_metadata.h>
#include <citlali/core/pipeline/reduction_observation.h>

#include <cstddef>

namespace citlali::pipeline {

template <bool IsBeammap, auto RawObsMap, auto FilteredObsMap, bool FitMaps,
          class KidsDataProc, class TodProc, class IOCoordinator,
          class CitlaliConfig, class MapExtents, class MapCoords,
          class DateObsFactory, class Logger>
bool run_reduction_observation_at_index(
    TodProc &todproc, const IOCoordinator &co, CitlaliConfig &citlali_config,
    MapExtents &map_extents, MapCoords &map_coords,
    std::size_t observation_index, DateObsFactory &&date_obs_factory,
    const Logger &logger) {
    logger->info("starting reduction of observation {}/{}",
                 observation_index + 1, co.n_inputs());
    auto kidsproc = make_kids_data_proc<KidsDataProc>(citlali_config);
    const auto &rawobs = co.inputs()[observation_index];
    auto rawobs_kids_meta = load_rawobs_kids_meta(kidsproc, rawobs, logger);

    return run_reduction_observation<IsBeammap, RawObsMap, FilteredObsMap,
                                     FitMaps>(
        todproc, kidsproc, rawobs, rawobs_kids_meta, co.n_inputs() > 1,
        map_extents, map_coords, observation_index,
        date_obs_factory(todproc.engine()), logger);
}

template <bool IsBeammap, auto RawObsMap, auto FilteredObsMap, bool FitMaps,
          class KidsDataProc, class TodProc, class IOCoordinator,
          class CitlaliConfig, class MapExtents, class MapCoords,
          class DateObsFactory, class Logger>
bool run_reduction_iteration_observations(
    TodProc &todproc, const IOCoordinator &co, CitlaliConfig &citlali_config,
    MapExtents &map_extents, MapCoords &map_coords,
    DateObsFactory &&date_obs_factory, const Logger &logger) {
    for (std::size_t observation_index = 0; observation_index < co.n_inputs();
         ++observation_index) {
        if (!run_reduction_observation_at_index<
                IsBeammap, RawObsMap, FilteredObsMap, FitMaps,
                KidsDataProc>(
                todproc, co, citlali_config, map_extents, map_coords,
                observation_index, date_obs_factory, logger)) {
            return false;
        }
    }
    return true;
}

}  // namespace citlali::pipeline
