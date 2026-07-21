#pragma once

#include <citlali/core/pipeline/reduction_observation_inputs.h>
#include <citlali/core/pipeline/reduction_observation_pipeline.h>
#include <citlali/core/pipeline/learning_housekeeping_qa.h>
#include <citlali/core/pipeline/stage_profile.h>

#include <cstddef>
#include <string>
#include <utility>

namespace citlali::pipeline {

template <bool IsBeammap, auto RawObsMap, auto FilteredObsMap, bool FitMaps,
          class TodProc, class KidsProc, class RawObs, class RawObsKidsMeta,
          class MapExtents, class MapCoords, class DateObsFactory, class Logger>
bool run_reduction_observation(
    TodProc &todproc, KidsProc &kidsproc, const RawObs &rawobs,
    const RawObsKidsMeta &rawobs_kids_meta, bool has_multiple_inputs,
    MapExtents &map_extents, MapCoords &map_coords,
    std::size_t observation_index, DateObsFactory &&date_obs_factory,
    StageProfileCollector &stage_profile,
    const Logger &logger) {
    const auto profile_context =
        "observation_index=" + std::to_string(observation_index);

    {
        const auto profile_scope = profile_stage(stage_profile,
            "observation.prepare_inputs", logger, profile_context);
        if (!prepare_reduction_observation_inputs<IsBeammap>(
                todproc, rawobs, rawobs_kids_meta, has_multiple_inputs,
                map_extents, map_coords, observation_index,
                std::forward<DateObsFactory>(date_obs_factory), logger)) {
            return false;
        }
    }

    {
        const auto profile_scope = profile_stage(stage_profile,
            "observation.pipeline", logger, profile_context);
        run_reduction_observation_pipeline<IsBeammap, RawObsMap,
                                           FilteredObsMap, FitMaps>(
            todproc, kidsproc, rawobs, stage_profile, logger);
    }
    write_learning_housekeeping_qa_if_available(
        todproc.engine(), rawobs, observation_index == 0, logger);
    return true;
}

template <bool IsBeammap, auto RawObsMap, auto FilteredObsMap, bool FitMaps,
          class TodProc, class KidsProc, class ObservationContext,
          class MapExtents, class MapCoords, class DateObsFactory,
          class Logger>
bool run_reduction_observation_context(
    TodProc &todproc, KidsProc &kidsproc, ObservationContext &context,
    MapExtents &map_extents, MapCoords &map_coords,
    DateObsFactory &&date_obs_factory, StageProfileCollector &stage_profile,
    const Logger &logger) {
    return run_reduction_observation<IsBeammap, RawObsMap, FilteredObsMap,
                                     FitMaps>(
        todproc, kidsproc, context.rawobs, context.rawobs_kids_meta,
        context.has_multiple_inputs, map_extents, map_coords,
        context.observation_index,
        std::forward<DateObsFactory>(date_obs_factory), stage_profile,
        logger);
}

}  // namespace citlali::pipeline
