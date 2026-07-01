#pragma once

#include <citlali/core/cli/config_loading.h>
#include <citlali/core/cli/io_coordinator.h>

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

template <class IOCoordinator, class RuntimeConfig, class Logger>
auto load_standard_reduction_inputs(const RuntimeConfig &runtime_config,
                                    const Logger &logger) {
    auto loaded_config = load_merged_yaml_config_files(runtime_config, logger);
    auto coordinator =
        make_io_coordinator_from_config<IOCoordinator>(loaded_config.config);
    return make_standard_reduction_inputs(
        std::move(loaded_config), std::move(coordinator));
}

}  // namespace citlali::cli
