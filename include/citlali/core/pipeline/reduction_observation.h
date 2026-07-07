#pragma once

#include <citlali/core/pipeline/reduction_observation_inputs.h>
#include <citlali/core/pipeline/reduction_observation_pipeline.h>
#include <citlali/core/pipeline/stage_profile.h>

#include <cstddef>
#include <string>
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
    const auto profile_context =
        "observation_index=" + std::to_string(observation_index);

    {
        const auto profile_scope = profile_stage(
            "observation.prepare_inputs", logger, profile_context);
        if (!prepare_reduction_observation_inputs<IsBeammap>(
                todproc, rawobs, rawobs_kids_meta, has_multiple_inputs,
                map_extents, map_coords, observation_index,
                std::forward<DateObs>(date_obs), logger)) {
            return false;
        }
    }

    {
        const auto profile_scope = profile_stage(
            "observation.pipeline", logger, profile_context);
        run_reduction_observation_pipeline<IsBeammap, RawObsMap,
                                           FilteredObsMap, FitMaps>(
            todproc, kidsproc, rawobs, logger);
    }
    return true;
}

template <bool IsBeammap, auto RawObsMap, auto FilteredObsMap, bool FitMaps,
          class TodProc, class KidsProc, class ObservationContext,
          class MapExtents, class MapCoords, class Logger>
bool run_reduction_observation_context(
    TodProc &todproc, KidsProc &kidsproc, ObservationContext &context,
    MapExtents &map_extents, MapCoords &map_coords,
    const Logger &logger) {
    return run_reduction_observation<IsBeammap, RawObsMap, FilteredObsMap,
                                     FitMaps>(
        todproc, kidsproc, context.rawobs, context.rawobs_kids_meta,
        context.has_multiple_inputs, map_extents, map_coords,
        context.observation_index, std::move(context.date_obs), logger);
}

}  // namespace citlali::pipeline
