#pragma once

#include <citlali/core/config/mapmaking_config.h>
#include <citlali/core/config/runtime_config.h>

namespace citlali::pipeline {

inline citlali::config::MapGrouping automatic_map_grouping_for_reduction(
    citlali::config::ReductionType reduction_type) {
    return citlali::config::is_beammap_reduction_type(reduction_type)
               ? citlali::config::MapGrouping::detector
               : citlali::config::MapGrouping::array;
}

inline bool detector_map_grouping_disallowed(
    citlali::config::ReductionType reduction_type,
    citlali::config::MapGrouping map_grouping) {
    return citlali::config::is_detector_map_grouping(map_grouping) &&
           !citlali::config::is_beammap_reduction_type(reduction_type);
}

inline citlali::config::MapGrouping effective_map_grouping_for_reduction(
    citlali::config::ReductionType reduction_type,
    citlali::config::MapGrouping requested_grouping) {
    auto grouping = requested_grouping;
    if (citlali::config::is_automatic_map_grouping(grouping)) {
        grouping = automatic_map_grouping_for_reduction(reduction_type);
    }
    if (detector_map_grouping_disallowed(reduction_type, grouping)) {
        grouping = citlali::config::MapGrouping::array;
    }
    return grouping;
}

}  // namespace citlali::pipeline
