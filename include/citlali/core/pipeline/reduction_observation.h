#pragma once

#include <citlali/core/pipeline/reduction_observation_inputs.h>
#include <citlali/core/pipeline/reduction_observation_pipeline.h>

#include <cstddef>
#include <utility>

namespace citlali::pipeline {

template <bool IsBeammap, auto RawObsMap, auto FilteredObsMap, bool FitMaps,
          class TodProc, class KidsProc, class RawObs, class RawObsKidsMeta,
          class MapExtents, class MapCoords, class DateObs, class Logger>
bool run_reduction_observation(
    TodProc &todproc, KidsProc &kidsproc, const RawObs &rawobs,
    const RawObsKidsMeta &rawobs_kids_meta, bool has_multiple_inputs,
    MapExtents &map_extents, MapCoords &map_coords,
    std::size_t observation_index, DateObs &&date_obs, const Logger &logger) {
    if (!prepare_reduction_observation_inputs<IsBeammap>(
            todproc, rawobs, rawobs_kids_meta, has_multiple_inputs,
            map_extents, map_coords, observation_index,
            std::forward<DateObs>(date_obs), logger)) {
        return false;
    }

    run_reduction_observation_pipeline<IsBeammap, RawObsMap, FilteredObsMap,
                                       FitMaps>(
        todproc, kidsproc, rawobs, logger);
    return true;
}

}  // namespace citlali::pipeline
