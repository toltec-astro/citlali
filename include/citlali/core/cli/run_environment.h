#pragma once

#include <citlali/core/cli/abort_backtrace.h>
#include <citlali/core/cli/hdf5_diagnostics.h>
#include <citlali/core/cli/kids_data_spec.h>
#include <citlali/core/cli/run_logging.h>
#include <spdlog/spdlog.h>

#include <memory>

namespace citlali::cli {

struct CliRunEnvironment {
    spdlog::level::level_enum previous_log_level;
    RunLoggers run_loggers;
    std::shared_ptr<spdlog::logger> logger;
};

inline CliRunEnvironment configure_citlali_cli_run_environment() {
    suppress_optional_hdf5_diagnostics();

    auto previous_log_level = spdlog::get_level();
    auto run_loggers = configure_run_loggers(previous_log_level);
    auto logger = run_loggers.logger;

    install_abort_backtrace_handler();
    log_kids_data_spec(logger);

    return {previous_log_level, run_loggers, logger};
}

inline void restore_citlali_cli_run_environment(
    const CliRunEnvironment &environment) {
    restore_default_sink_level(
        environment.run_loggers, environment.previous_log_level);
}

}  // namespace citlali::cli
