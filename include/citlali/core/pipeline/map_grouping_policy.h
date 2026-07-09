#pragma once

#include <string>

#include <citlali/core/config/mapmaking_config.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>

namespace citlali::pipeline {

inline std::string map_grouping_name(
    citlali::config::MapGrouping grouping) {
    return std::string(citlali::config::to_string(grouping));
}

template <class Engine>
std::string active_map_grouping_name(const Engine &engine) {
    return map_grouping_name(mapmaking_config(engine).grouping);
}

}  // namespace citlali::pipeline
