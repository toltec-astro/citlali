#pragma once

#include <cstddef>

namespace citlali::pipeline {

template <class MapExtents>
decltype(auto) observation_map_extent_at(MapExtents &map_extents,
                                         std::size_t observation_index) {
    return map_extents[observation_index];
}

template <class MapCoords>
decltype(auto) observation_map_coord_at(MapCoords &map_coords,
                                        std::size_t observation_index) {
    return map_coords[observation_index];
}

}  // namespace citlali::pipeline
