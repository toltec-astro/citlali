#pragma once

#include <citlali/core/config/mapmaking_config.h>

#include <string>

#include <Eigen/Core>

namespace citlali::pipeline {

template <class Calib>
std::string map_layer_name(Eigen::Index i, citlali::config::MapGrouping grouping,
                           const Calib &calib) {
    std::string map_name;

    if (grouping == citlali::config::MapGrouping::array) {
        return map_name;
    }

    if (grouping == citlali::config::MapGrouping::network) {
        map_name += "nw_" + std::to_string(calib.nws(i)) + "_";
    }
    else if (grouping == citlali::config::MapGrouping::frequency_group) {
        const Eigen::Index n_fg = calib.fg.size();
        if (n_fg > 0) {
            map_name += "fg_" + std::to_string(calib.fg(i % n_fg)) + "_";
        }
    }
    else if (grouping == citlali::config::MapGrouping::detector) {
        map_name += "det_" + std::to_string(i) + "_";
    }

    return map_name;
}

}  // namespace citlali::pipeline
