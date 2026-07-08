#pragma once

#include <citlali/core/config/mapmaking_config.h>

#include <tula/eigen.h>

namespace citlali::pipeline {

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

}  // namespace citlali::pipeline
