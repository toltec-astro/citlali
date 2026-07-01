#pragma once

#include <citlali/core/cli/reduction_runtime.h>
#include <citlali/core/cli/runtime_setup.h>
#include <spdlog/spdlog.h>

#include <ostream>

namespace citlali::cli {

template <class TodProc, class Config, class Logger>
bool prepare_cli_reduction_runtime_or_report_errors(
    TodProc &todproc, Config &config, const Logger &logger,
    std::ostream &os) {
    return prepare_reduction_runtime_or_report_errors(
        todproc, config, logger,
        []() { spdlog::set_level(spdlog::level::debug); },
        [&](const auto &engine) {
            configure_citlali_runtime_threads(engine, logger);
        },
        os);
}

}  // namespace citlali::cli
