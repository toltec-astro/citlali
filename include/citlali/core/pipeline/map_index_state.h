#pragma once

#include <Eigen/Core>

namespace citlali::pipeline {

struct MapIndexState {
    Eigen::Index n_maps = 0;
    Eigen::VectorXI maps_to_arrays;
    Eigen::VectorXI arrays_to_maps;
    Eigen::VectorXI maps_to_stokes;
};

inline bool has_maps(const MapIndexState &state) {
    return state.n_maps > 0;
}

}  // namespace citlali::pipeline
