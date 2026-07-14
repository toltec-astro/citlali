#pragma once

#include <citlali/core/pipeline/observation_buffer_allocation.h>
#include <citlali/core/pipeline/observation_map_access.h>
#include <citlali/core/pipeline/mapmaking_provenance_lifecycle.h>
#include <citlali/core/pipeline/pointing_provenance_lifecycle.h>

#include <cstddef>

namespace citlali::pipeline {

template <class TodProc, class MapExtents, class MapCoords, class Logger>
void allocate_observation_map_buffers_if_needed(
    TodProc &todproc, MapExtents &map_extents, MapCoords &map_coords,
    std::size_t observation_index, const Logger &logger) {
    auto &engine = todproc.engine();

    if (!should_allocate_observation_map_buffers(engine)) {
        return;
    }

    allocate_observation_map_buffers(
        todproc, observation_map_extent_at(map_extents, observation_index),
        observation_map_coord_at(map_coords, observation_index),
        logger);
    begin_mapmaking_observation_if_available(engine, observation_index);
    begin_pointing_observation_if_available(engine);
}

template <class TodProc, class MapExtents, class MapCoords, class Logger>
void allocate_reduction_observation_map_buffers_if_needed(
    TodProc &todproc, MapExtents &map_extents, MapCoords &map_coords,
    std::size_t observation_index, const Logger &logger) {
    allocate_observation_map_buffers_if_needed(
        todproc, map_extents, map_coords, observation_index, logger);
}

}  // namespace citlali::pipeline
