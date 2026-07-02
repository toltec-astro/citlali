#pragma once

#include <citlali/core/pipeline/observation_buffer_policy.h>
#include <citlali/core/pipeline/observation_map_access.h>

#include <cstddef>

namespace citlali::pipeline {

template <class TodProc, class Logger>
void calculate_observation_map_count(TodProc &todproc,
                                     const Logger &logger) {
    logger->info("calculating number of maps");
    todproc.calc_map_num();
}

template <class TodProc, class MapExtent, class MapCoord, class Logger>
void allocate_observation_signal_map_buffer(TodProc &todproc,
                                            MapExtent &map_extent,
                                            MapCoord &map_coord,
                                            const Logger &logger) {
    logger->info("allocating obs map buffer");
    todproc.allocate_omb(map_extent, map_coord);
}

template <class TodProc, class Logger>
void allocate_observation_noise_map_buffer(TodProc &todproc,
                                           const Logger &logger) {
    auto &engine = todproc.engine();

    logger->info("allocating obs noise maps");
    todproc.allocate_nmb(engine.omb);
}

template <class TodProc, class MapExtent, class MapCoord, class Logger>
void allocate_observation_map_buffers(TodProc &todproc,
                                      MapExtent &map_extent,
                                      MapCoord &map_coord,
                                      const Logger &logger) {
    auto &engine = todproc.engine();

    calculate_observation_map_count(todproc, logger);
    allocate_observation_signal_map_buffer(
        todproc, map_extent, map_coord, logger);
    configure_observation_pixel_contribution_targets(engine);

    if (should_allocate_observation_noise_maps(engine)) {
        allocate_observation_noise_map_buffer(todproc, logger);
    }
}

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
}

template <class TodProc, class MapExtents, class MapCoords, class Logger>
void allocate_reduction_observation_map_buffers_if_needed(
    TodProc &todproc, MapExtents &map_extents, MapCoords &map_coords,
    std::size_t observation_index, const Logger &logger) {
    allocate_observation_map_buffers_if_needed(
        todproc, map_extents, map_coords, observation_index, logger);
}

}  // namespace citlali::pipeline
