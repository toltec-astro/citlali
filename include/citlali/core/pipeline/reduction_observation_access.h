#pragma once

#include <cstddef>

namespace citlali::pipeline {

template <class IOCoordinator>
std::size_t reduction_observation_count(const IOCoordinator &co) {
    return co.n_inputs();
}

template <class IOCoordinator>
bool has_multiple_reduction_observations(const IOCoordinator &co) {
    return reduction_observation_count(co) > 1;
}

}  // namespace citlali::pipeline
