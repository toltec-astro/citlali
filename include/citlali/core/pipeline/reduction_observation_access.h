#pragma once

#include <cstddef>

namespace citlali::pipeline {

template <class IOCoordinator>
std::size_t reduction_observation_count(const IOCoordinator &co) {
    return co.n_inputs();
}

}  // namespace citlali::pipeline
