#pragma once

#include <citlali/core/config/mapmaking_config.h>

#include <string>

#include <Eigen/Core>

namespace citlali::pipeline {

template <class Calib>
std::string map_layer_name(Eigen::Index i, citlali::config::MapGrouping grouping,
                           const Calib &calib) {
    std::string map_name;

    if (citlali::config::is_array_map_grouping(grouping)) {
        return map_name;
    }

    if (citlali::config::is_network_map_grouping(grouping)) {
        map_name += "nw_" + std::to_string(calib.nws(i)) + "_";
    }
    else if (citlali::config::is_frequency_group_map_grouping(grouping)) {
        const Eigen::Index n_fg = calib.fg.size();
        if (n_fg > 0) {
            map_name += "fg_" + std::to_string(calib.fg(i % n_fg)) + "_";
        }
    }
    else if (citlali::config::is_detector_map_grouping(grouping)) {
        map_name += "det_" + std::to_string(i) + "_";
    }

    return map_name;
}

}  // namespace citlali::pipeline
