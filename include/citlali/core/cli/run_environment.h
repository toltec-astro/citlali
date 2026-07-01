#pragma once

#include <citlali/core/cli/run_logging.h>
#include <spdlog/spdlog.h>

#include <memory>

namespace citlali::cli {

struct CliRunEnvironment {
    spdlog::level::level_enum previous_log_level;
    RunLoggers run_loggers;
    std::shared_ptr<spdlog::logger> logger;
};

}  // namespace citlali::cli
