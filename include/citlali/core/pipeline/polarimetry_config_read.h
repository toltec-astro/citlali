#pragma once

#include <citlali/core/config/timestream_config.h>
#include <citlali/core/pipeline/config_parse_tracking.h>

#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace citlali::pipeline {

template <class Config, class Key, class Enum, class Parser,
          class Diagnostics>
void read_polarimetry_enum(
    Config &config, const Key &key, Enum &target, Parser parser,
    Diagnostics &diagnostics, std::vector<std::string> accepted_values) {
    std::string value{citlali::config::to_string(target)};
    read_config_value_if_clean(
        config, key, value,
        [&target, &parser, &key, &diagnostics](const auto &parsed_value) {
            if (const auto parsed = parser(parsed_value)) {
                target = *parsed;
            } else {
                add_invalid_config_key(
                    key, diagnostics.invalid_key_paths());
            }
        },
        diagnostics, std::move(accepted_values));
}

template <class Config, class Diagnostics>
void read_polarimetry_request_config(
    Config &config, citlali::config::TimestreamPolarimetryConfig &request,
    Diagnostics &diagnostics) {
    request = {};
    read_config_value(
        config, request.enabled, diagnostics,
        std::tuple{"timestream", "polarimetry", "enabled"});
    read_polarimetry_enum(
        config, std::tuple{"timestream", "polarimetry", "grouping"},
        request.grouping, citlali::config::parse_polarimetry_grouping,
        diagnostics, {"fg", "loc"});
    read_polarimetry_enum(
        config, std::tuple{"timestream", "polarimetry", "ignore_hwpr"},
        request.hwpr_policy,
        citlali::config::parse_polarimetry_hwpr_policy, diagnostics,
        {"auto", "true", "false"});
}

}  // namespace citlali::pipeline
