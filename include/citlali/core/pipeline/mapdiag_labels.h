#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace citlali::pipeline {

struct MapdiagMapLabels {
    std::string array_name;
    std::string stokes_name;
    std::string map_name;
};

inline MapdiagMapLabels make_mapdiag_map_labels(
    const std::string &array_name, const std::string &stokes_name,
    const std::string &map_name) {
    return {array_name, stokes_name, map_name};
}

inline std::vector<std::string> mapdiag_obsnum_labels(
    const std::vector<std::string> &obsnums,
    const std::string &fallback_obsnum) {
    std::vector<std::string> labels = obsnums;
    if (labels.empty()) {
        labels.push_back(fallback_obsnum);
    }
    return labels;
}

inline std::vector<std::string> mapdiag_dateobs_labels(
    std::vector<std::string> labels, std::size_t n_obsnums) {
    if (labels.empty()) {
        labels.push_back("");
    }
    if (labels.size() > n_obsnums) {
        labels.resize(n_obsnums);
    }
    if (labels.size() < n_obsnums) {
        labels.resize(n_obsnums, "");
    }
    return labels;
}

}  // namespace citlali::pipeline
