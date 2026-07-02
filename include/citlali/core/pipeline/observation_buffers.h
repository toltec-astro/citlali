#pragma once

#include <cstddef>

namespace citlali::pipeline {

template <class Engine>
bool should_allocate_observation_map_buffers(const Engine &engine) {
    return engine.run_mapmaking;
}

template <class MapExtents>
decltype(auto) observation_map_extent_at(MapExtents &map_extents,
                                         std::size_t observation_index) {
    return map_extents[observation_index];
}

template <class TodProc, class MapExtent, class MapCoord, class Logger>
void allocate_observation_map_buffers(TodProc &todproc,
                                      MapExtent &map_extent,
                                      MapCoord &map_coord,
                                      const Logger &logger) {
    auto &engine = todproc.engine();

    logger->info("calculating number of maps");
    todproc.calc_map_num();
    logger->info("allocating obs map buffer");
    todproc.allocate_omb(map_extent, map_coord);
    engine.configure_map_pixel_contribution_targets(engine.omb, "raw_obs");

    if (engine.run_noise &&
        (!engine.run_coadd || engine.map_method == "jinc")) {
        logger->info("allocating obs noise maps");
        todproc.allocate_nmb(engine.omb);
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
        map_coords[observation_index],
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
