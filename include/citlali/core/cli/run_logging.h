#pragma once

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <memory>
#include <vector>

namespace citlali::cli {

struct RunLoggers {
    spdlog::sink_ptr default_sink;
    std::shared_ptr<spdlog::logger> logger;
};

inline RunLoggers configure_run_loggers(spdlog::level::level_enum log_level) {
    std::vector<spdlog::sink_ptr> sinks_default;
    auto default_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    default_sink->set_level(spdlog::level::critical);
    sinks_default.push_back(default_sink);

    auto default_logger = std::make_shared<spdlog::logger>(
        "console", sinks_default.begin(), sinks_default.end());
    spdlog::register_logger(default_logger);
    spdlog::set_default_logger(default_logger);
    default_logger->flush_on(spdlog::level::info);

    std::vector<spdlog::sink_ptr> sinks;
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    sinks.push_back(console_sink);

    auto logger = std::make_shared<spdlog::logger>(
        "citlali_logger", sinks.begin(), sinks.end());
    spdlog::register_logger(logger);
    logger->flush_on(spdlog::level::info);

    spdlog::set_level(log_level);

    return {default_sink, logger};
}

inline void restore_default_sink_level(const RunLoggers &run_loggers,
                                       spdlog::level::level_enum log_level) {
    run_loggers.default_sink->set_level(log_level);
}

}  // namespace citlali::cli
