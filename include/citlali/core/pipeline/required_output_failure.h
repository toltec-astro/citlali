#pragma once

#include <citlali/core/error/error.h>

#include <string>
#include <utility>

namespace citlali::pipeline {

template <class Logger>
[[noreturn]] void fail_required_output(const Logger &logger,
                                       std::string message) {
    logger->error("{}", message);
    throw citlali::error::output(std::move(message));
}

}  // namespace citlali::pipeline
