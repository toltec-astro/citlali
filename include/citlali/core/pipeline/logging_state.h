#pragma once

#include <memory>

#include <spdlog/spdlog.h>

namespace citlali::pipeline {

struct LoggingState {
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");
};

}  // namespace citlali::pipeline
