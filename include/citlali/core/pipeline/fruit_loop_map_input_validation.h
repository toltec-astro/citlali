#pragma once

#include <citlali/core/error/error.h>

#include <string>
#include <utility>

namespace citlali::pipeline {

[[noreturn]] inline void fail_fruit_loop_map_request(std::string message) {
    throw citlali::error::invalid_config(
        "invalid fruit-loop map request: " + std::move(message));
}

[[noreturn]] inline void fail_fruit_loop_map_input(std::string message) {
    throw citlali::error::io(
        "invalid fruit-loop map input: " + std::move(message));
}

inline void require_fruit_loop_map_request(
    bool condition, std::string message) {
    if (!condition) {
        fail_fruit_loop_map_request(std::move(message));
    }
}

inline void require_fruit_loop_map_input(
    bool condition, std::string message) {
    if (!condition) {
        fail_fruit_loop_map_input(std::move(message));
    }
}

}  // namespace citlali::pipeline
