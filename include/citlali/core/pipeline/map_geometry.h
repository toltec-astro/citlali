#pragma once

#include <vector>

namespace citlali::pipeline {

template <class TodProc>
struct ReductionMapGeometry {
    std::vector<typename TodProc::map_extent_t> extents;
    std::vector<typename TodProc::map_coord_t> coords;
};

}  // namespace citlali::pipeline
