#pragma once

#include <citlali/core/pipeline/config_parse_tracking.h>

#include <utility>
#include <vector>

namespace citlali::pipeline {

template <class Config, class Key, class Value, class Diagnostics>
void read_optional_raw_request_value(
    Config &config, const Key &key, Value &target, Diagnostics &diagnostics,
    std::vector<Value> accepted = {}, std::vector<Value> minimum = {},
    std::vector<Value> maximum = {}) {
    Value value = target;
    read_optional_mirrored_config_value(
        config, key, value, target, diagnostics, std::move(accepted),
        std::move(minimum), std::move(maximum));
}

}  // namespace citlali::pipeline
