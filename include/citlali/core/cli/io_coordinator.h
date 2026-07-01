#pragma once

namespace citlali::cli {

template <class IOCoordinator, class Config>
IOCoordinator make_io_coordinator_from_config(const Config &config) {
    return IOCoordinator::from_config(config);
}

}  // namespace citlali::cli
