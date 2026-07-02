#pragma once

#include <string>

#include <Eigen/Core>

namespace citlali::pipeline {

template <class Calib>
std::string map_layer_name(Eigen::Index i, const std::string &map_grouping,
                           const Calib &calib) {
    std::string map_name;

    if (map_grouping == "array") {
        return map_name;
    }

    if (map_grouping == "nw") {
        map_name += "nw_" + std::to_string(calib.nws(i)) + "_";
    }
    else if (map_grouping == "fg") {
        const Eigen::Index n_fg = calib.fg.size();
        if (n_fg > 0) {
            map_name += "fg_" + std::to_string(calib.fg(i % n_fg)) + "_";
        }
    }
    else if (map_grouping == "detector") {
        map_name += "det_" + std::to_string(i) + "_";
    }

    return map_name;
}

}  // namespace citlali::pipeline
