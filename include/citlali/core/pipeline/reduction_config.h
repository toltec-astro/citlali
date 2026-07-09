#pragma once

#include <citlali/core/pipeline/runtime_policy.h>

namespace citlali::pipeline {

template <class Engine>
bool engine_config_has_errors(const Engine &engine) {
    return !engine.config_diagnostics.missing_keys.empty() || !engine.config_diagnostics.invalid_keys.empty();
}

template <class Engine, class Config, class Logger>
bool load_and_validate_engine_config(Engine &engine, Config &config,
                                     const Logger &logger) {
    logger->info("getting citlali config");
    engine.get_citlali_config(config);

    if (!engine_config_has_errors(engine)) {
        return true;
    }

    logger->error("missing or invalid keys were found!");
    logger->error(
        "see for default config: "
        "https://github.com/toltec-astro/citlali/blob/v4.x/data/config.yaml");
    return false;
}

template <class Engine, class Logger, class EnableDebugLogging>
void configure_verbose_logging_if_requested(
    const Engine &engine, const Logger &logger,
    EnableDebugLogging &&enable_debug_logging) {
    if (!verbose_runtime_enabled(engine)) {
        return;
    }

    enable_debug_logging();
    logger->debug("running in verbose mode. setting log level=debug.");
}

}  // namespace citlali::pipeline
