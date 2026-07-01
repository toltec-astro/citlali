#pragma once

#include <citlali/core/pipeline/observation_preflight.h>

namespace citlali::cli {

template <class TodProc, class Config, class Logger,
          class EnableDebugLogging, class ConfigureThreads>
bool prepare_reduction_runtime(TodProc &todproc, Config &config,
                               const Logger &logger,
                               EnableDebugLogging &&enable_debug_logging,
                               ConfigureThreads &&configure_threads) {
    auto &engine = todproc.engine();
    if (!citlali::pipeline::load_and_validate_engine_config(
            engine, config, logger)) {
        return false;
    }

    citlali::pipeline::configure_verbose_logging_if_requested(
        engine, logger, enable_debug_logging);
    configure_threads(engine);
    return true;
}

}  // namespace citlali::cli
