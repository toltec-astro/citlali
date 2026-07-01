#pragma once

#include <citlali/core/pipeline/reduction_config.h>
#include <fmt/core.h>
#include <tula/formatter/container.h>

#include <ostream>
#include <utility>

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

template <class Engine>
void report_engine_config_errors(const Engine &engine, std::ostream &os) {
    os << fmt::format("missing keys={}", engine.missing_keys) << "\n";
    os << fmt::format("invalid keys={}", engine.invalid_keys) << "\n";
}

template <class TodProc, class Config, class Logger,
          class EnableDebugLogging, class ConfigureThreads>
bool prepare_reduction_runtime_or_report_errors(
    TodProc &todproc, Config &config, const Logger &logger,
    EnableDebugLogging &&enable_debug_logging,
    ConfigureThreads &&configure_threads, std::ostream &os) {
    if (prepare_reduction_runtime(
            todproc, config, logger,
            std::forward<EnableDebugLogging>(enable_debug_logging),
            std::forward<ConfigureThreads>(configure_threads))) {
        return true;
    }

    report_engine_config_errors(todproc.engine(), os);
    return false;
}

}  // namespace citlali::cli
