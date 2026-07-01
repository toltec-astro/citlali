#pragma once

#include <citlali/core/cli/config_loading.h>

#include <utility>

namespace citlali::cli {

template <class Config, class IOCoordinator>
struct StandardReductionInputs {
    LoadedConfigFiles<Config> loaded_config;
    IOCoordinator coordinator;
};

template <class Config, class IOCoordinator>
StandardReductionInputs<Config, IOCoordinator> make_standard_reduction_inputs(
    LoadedConfigFiles<Config> loaded_config, IOCoordinator coordinator) {
    return {std::move(loaded_config), std::move(coordinator)};
}

}  // namespace citlali::cli
