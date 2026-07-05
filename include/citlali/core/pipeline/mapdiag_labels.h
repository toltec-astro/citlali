#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace citlali::pipeline {

struct MapdiagMapLabels {
    std::string array_name;
    std::string stokes_name;
    std::string map_name;
};

struct MapdiagMapLabelRefs {
    std::vector<std::string> &array_names;
    std::vector<std::string> &stokes_names;
    std::vector<std::string> &map_names;
};

inline MapdiagMapLabels make_mapdiag_map_labels(
    const std::string &array_name, const std::string &stokes_name,
    const std::string &map_name) {
    return {array_name, stokes_name, map_name};
}

inline void assign_mapdiag_map_labels(std::size_t idx,
                                      const MapdiagMapLabels &labels,
                                      MapdiagMapLabelRefs refs) {
    refs.array_names[idx] = labels.array_name;
    refs.stokes_names[idx] = labels.stokes_name;
    refs.map_names[idx] = labels.map_name;
}

template <class ArrayNameMap, class Arrays, class StokesParams,
          class GetMapName, class WriteIndices>
void assign_mapdiag_map_labels_from_indices(
    std::size_t idx, Eigen::Index map_i, const WriteIndices &indices,
    ArrayNameMap &array_name_map, const Arrays &arrays,
    StokesParams &stokes_params, const GetMapName &get_map_name,
    MapdiagMapLabelRefs refs) {
    const auto labels = make_mapdiag_map_labels(
        array_name_map[arrays[indices.map_index]],
        stokes_params[indices.stokes_index], get_map_name(map_i));
    assign_mapdiag_map_labels(idx, labels, refs);
}

inline std::string mapdiag_weight_hdu_name(const std::string &map_name,
                                           const std::string &stokes_name) {
    return "weight_" + map_name + stokes_name;
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
