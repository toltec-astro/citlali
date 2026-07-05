#pragma once

#include <citlali/core/config/timestream_config.h>
#include <citlali/core/engine/detail/config_parse_tracking.h>

#include <string>
#include <tuple>

namespace citlali::engine_detail {

template <class Config, class MissingKeys, class InvalidKeys,
          class TimestreamConfig>
void read_timestream_enabled_config(Config &config, bool &enabled,
                                    TimestreamConfig &typed_config,
                                    MissingKeys &missing_keys,
                                    InvalidKeys &invalid_keys) {
    const auto missing_before = missing_keys.size();
    const auto invalid_before = invalid_keys.size();
    ::get_config_value(config, enabled, missing_keys, invalid_keys,
                       std::tuple{"timestream", "enabled"});
    if (config_parse_clean(
            missing_keys, invalid_keys, missing_before, invalid_before)) {
        typed_config.enabled = enabled;
    }
}

template <class Config, class MissingKeys, class InvalidKeys,
          class TimestreamConfig>
void read_timestream_type_config(Config &config, std::string &type,
                                 TimestreamConfig &typed_config,
                                 MissingKeys &missing_keys,
                                 InvalidKeys &invalid_keys) {
    const auto missing_before = missing_keys.size();
    const auto invalid_before = invalid_keys.size();
    ::get_config_value(config, type, missing_keys, invalid_keys,
                       std::tuple{"timestream", "type"});
    if (!config_parse_clean(
            missing_keys, invalid_keys, missing_before, invalid_before)) {
        return;
    }
    if (auto parsed = citlali::config::parse_tod_type(type)) {
        typed_config.type = *parsed;
    }
}

template <class Config, class MissingKeys, class InvalidKeys,
          class TimestreamConfig>
void read_raw_tod_output_enabled_config(Config &config, bool &enabled,
                                        TimestreamConfig &typed_config,
                                        MissingKeys &missing_keys,
                                        InvalidKeys &invalid_keys) {
    const auto missing_before = missing_keys.size();
    const auto invalid_before = invalid_keys.size();
    ::get_config_value(
        config, enabled, missing_keys, invalid_keys,
        std::tuple{"timestream", "raw_time_chunk", "output", "enabled"});
    if (config_parse_clean(
            missing_keys, invalid_keys, missing_before, invalid_before)) {
        typed_config.output.raw_time_chunk_enabled = enabled;
        typed_config.output.raw_time_chunk.enabled = enabled;
    }
}

template <class Config, class MissingKeys, class InvalidKeys,
          class TimestreamConfig>
void read_processed_tod_output_enabled_config(Config &config, bool &enabled,
                                              TimestreamConfig &typed_config,
                                              MissingKeys &missing_keys,
                                              InvalidKeys &invalid_keys) {
    const auto missing_before = missing_keys.size();
    const auto invalid_before = invalid_keys.size();
    ::get_config_value(config, enabled, missing_keys, invalid_keys,
                       std::tuple{"timestream", "processed_time_chunk",
                                  "output", "enabled"});
    if (config_parse_clean(
            missing_keys, invalid_keys, missing_before, invalid_before)) {
        typed_config.output.processed_time_chunk_enabled = enabled;
        typed_config.output.processed_time_chunk.enabled = enabled;
    }
}

}  // namespace citlali::engine_detail
