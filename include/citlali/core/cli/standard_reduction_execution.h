#pragma once

#include <citlali/core/cli/reduction_execution.h>
#include <citlali/core/cli/standard_reduction_types.h>

#include <ostream>

namespace citlali::cli {

template <class KidsDataProc, class IOCoordinator, class Config,
          class ConfigFilepaths, class Logger>
int run_standard_citlali_reduction_variant(
    StandardTodProcessorVariant &todproc, const IOCoordinator &co,
    Config &config, const ConfigFilepaths &config_filepaths,
    const Logger &logger, std::ostream &os) {
    return run_standard_cli_reduction_variant<
        StandardBeammapTodProcessor, StandardPointingTodProcessor,
        KidsDataProc>(todproc, co, config, config_filepaths, logger, os);
}

}  // namespace citlali::cli
