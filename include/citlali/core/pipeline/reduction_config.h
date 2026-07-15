#pragma once

#include <citlali/core/pipeline/config_schema_validation.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>
#include <citlali/core/pipeline/runtime_policy.h>

namespace citlali::pipeline {

template <class Engine>
bool engine_config_has_errors(const Engine &engine) {
    return config_diagnostics(engine).has_errors();
}

template <class Engine, class Config, class Logger>
bool load_and_validate_engine_config(Engine &engine, Config &config,
                                     const Logger &logger) {
    logger->info("getting citlali config");
    auto &diagnostics = config_diagnostics(engine);
    diagnostics = {};
    validate_low_level_config_schema(config, diagnostics);
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
