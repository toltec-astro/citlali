#pragma once

#include <citlali/core/config/mapmaking_config.h>
#include <citlali/core/pipeline/map_index_state.h>

#include <tula/eigen.h>

#include <map>
#include <stdexcept>
#include <unordered_map>

namespace citlali::pipeline {

template <class Calib>
Eigen::VectorXI detector_map_indices_for_grouping(
    citlali::config::MapGrouping grouping, const Calib &calib) {
    Eigen::VectorXI group_ids;
    if (citlali::config::is_network_map_grouping(grouping)) {
        group_ids = calib.apt.at("nw").template cast<Eigen::Index>();
    }
    else if (citlali::config::is_array_map_grouping(grouping)) {
        group_ids = calib.apt.at("array").template cast<Eigen::Index>();
    }
    else if (citlali::config::is_detector_map_grouping(grouping)) {
        group_ids = Eigen::VectorXI::LinSpaced(
            calib.n_dets, 0, calib.n_dets - 1);
    }
    else if (citlali::config::is_frequency_group_map_grouping(grouping)) {
        group_ids = calib.apt.at("fg").template cast<Eigen::Index>();
    }
    else {
        throw std::invalid_argument(
            "detector map indices require a resolved map grouping");
    }

    if (group_ids.size() != calib.n_dets) {
        throw std::invalid_argument(
            "detector map-group column cardinality does not match APT");
    }

    Eigen::VectorXI map_indices(calib.n_dets);
    if (!citlali::config::is_frequency_group_map_grouping(grouping)) {
        std::unordered_map<Eigen::Index, Eigen::Index> group_to_index;
        Eigen::Index next_index = 0;
        for (Eigen::Index i = 0; i < group_ids.size(); ++i) {
            const auto [entry, inserted] =
                group_to_index.emplace(group_ids(i), next_index);
            map_indices(i) = entry->second;
            if (inserted) {
                ++next_index;
            }
        }
        return map_indices;
    }

    std::map<Eigen::Index, Eigen::Index> fg_to_index;
    std::map<Eigen::Index, Eigen::Index> array_to_index;
    for (Eigen::Index i = 0; i < calib.fg.size(); ++i) {
        fg_to_index.emplace(calib.fg(i), i);
    }
    for (Eigen::Index i = 0; i < calib.arrays.size(); ++i) {
        array_to_index.emplace(calib.arrays(i), i);
    }
    for (Eigen::Index i = 0; i < group_ids.size(); ++i) {
        const auto fg = fg_to_index.find(group_ids(i));
        const auto array = array_to_index.find(calib.apt.at("array")(i));
        if (fg == fg_to_index.end() || array == array_to_index.end()) {
            throw std::invalid_argument(
                "APT frequency-group identity is absent from calibration order");
        }
        map_indices(i) =
            fg->second + calib.fg.size() * array->second;
    }
    return map_indices;
}

template <class Calib, class NetworkToArrayMap>
Eigen::VectorXI map_array_indices_for_grouping(
    citlali::config::MapGrouping grouping, Calib &calib,
    NetworkToArrayMap &nw_to_array_map) {
    Eigen::VectorXI array_indices;

    if (citlali::config::is_detector_map_grouping(grouping)) {
        array_indices = calib.apt["array"].template cast<Eigen::Index>();
    }
    else if (citlali::config::is_array_map_grouping(grouping)) {
        array_indices = calib.arrays;
    }
    else if (citlali::config::is_network_map_grouping(grouping)) {
        array_indices.resize(calib.nws.size());
        for (Eigen::Index i = 0; i < calib.nws.size(); ++i) {
            array_indices(i) = nw_to_array_map[calib.nws(i)];
        }
    }
    else if (citlali::config::is_frequency_group_map_grouping(grouping)) {
        array_indices.resize(calib.fg.size() * calib.n_arrays);
        Eigen::Index j = 0;
        for (Eigen::Index i = 0; i < calib.n_arrays; ++i) {
            array_indices.segment(j, calib.fg.size())
                .setConstant(calib.arrays(i));
            j += calib.fg.size();
        }
    }

    return array_indices;
}

template <class Polarization, class MapsToArrays, class MapsToStokes,
          class ArraysToMaps>
void populate_map_index_mappings(const Eigen::VectorXI &array_indices,
                                 Eigen::Index n_maps,
                                 const Polarization &polarization,
                                 MapsToArrays &maps_to_arrays,
                                 MapsToStokes &maps_to_stokes,
                                 ArraysToMaps &arrays_to_maps) {
    maps_to_arrays.resize(n_maps);
    maps_to_stokes.resize(n_maps);
    arrays_to_maps.resize(n_maps);

    Eigen::Index j = 0;
    for (const auto &stokes_entry : polarization.stokes_params) {
        const auto stokes_index = stokes_entry.first;
        maps_to_arrays.segment(j, array_indices.size()) = array_indices;
        maps_to_stokes.segment(j, array_indices.size()).setConstant(
            stokes_index);
        j += array_indices.size();
    }

    Eigen::Index index = 0;
    arrays_to_maps(0) = index;
    for (Eigen::Index i = 1; i < n_maps; ++i) {
        if (maps_to_arrays(i) > maps_to_arrays(i - 1)) {
            index++;
        }
        else if (maps_to_arrays(i) < maps_to_arrays(i - 1)) {
            index = 0;
        }
        arrays_to_maps(i) = index;
    }
}

template <class Polarization>
MapIndexState make_map_index_state(const Eigen::VectorXI &array_indices,
                                   Eigen::Index n_maps,
                                   const Polarization &polarization) {
    MapIndexState state;
    state.n_maps = n_maps;
    populate_map_index_mappings(
        array_indices, state.n_maps, polarization, state.maps_to_arrays,
        state.maps_to_stokes, state.arrays_to_maps);
    return state;
}

}  // namespace citlali::pipeline
