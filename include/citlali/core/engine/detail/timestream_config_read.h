#pragma once

#include <citlali/core/config/timestream_config.h>
#include <citlali/core/engine/detail/config_parse_tracking.h>
#include <citlali/core/pipeline/tod_output_selection.h>

#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

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

template <class Config, class Key, class MissingKeys, class InvalidKeys,
          class StreamOutputConfig>
void read_tod_stream_output_mode_config(
    Config &config, const Key &key, bool output_enabled,
    const std::vector<std::string> &allowed_modes, std::string &mode,
    bool &mini, bool &outer, StreamOutputConfig &typed_stream,
    MissingKeys &missing_keys, InvalidKeys &invalid_keys) {
    if (!output_enabled || !config.has(key)) {
        return;
    }

    const auto missing_before = missing_keys.size();
    const auto invalid_before = invalid_keys.size();
    ::get_config_value(config, mode, missing_keys, invalid_keys, key,
                       allowed_modes);
    if (config_parse_clean(
            missing_keys, invalid_keys, missing_before, invalid_before)) {
        if (auto parsed =
                citlali::config::parse_tod_stream_output_mode(mode)) {
            typed_stream.mode = *parsed;
        }
    }
    citlali::pipeline::apply_tod_output_mode_flags(mode, mini, outer);
}

template <class Config, class Key, class MissingKeys, class InvalidKeys,
          class ContextSamples, class StreamOutputConfig>
void read_tod_stream_outer_context_config(
    Config &config, const Key &key, bool output_enabled,
    ContextSamples &outer_context_samples, StreamOutputConfig &typed_stream,
    MissingKeys &missing_keys, InvalidKeys &invalid_keys) {
    if (!output_enabled || !config.has(key)) {
        return;
    }

    using value_type = std::decay_t<ContextSamples>;
    const auto missing_before = missing_keys.size();
    const auto invalid_before = invalid_keys.size();
    ::get_config_value(
        config, outer_context_samples, missing_keys, invalid_keys, key,
        std::vector<value_type>{}, std::vector<value_type>{0});
    if (config_parse_clean(
            missing_keys, invalid_keys, missing_before, invalid_before)) {
        typed_stream.outer_context_samples =
            static_cast<int>(outer_context_samples);
    }
}

template <class TimestreamConfig>
void sync_tod_output_type_config(bool raw_time_chunk_enabled,
                                 bool processed_time_chunk_enabled,
                                 bool &output_enabled,
                                 std::string &output_type,
                                 TimestreamConfig &typed_config) {
    output_enabled = false;
    if (auto requested_output_type =
            citlali::pipeline::requested_tod_output_type_name(
                raw_time_chunk_enabled, processed_time_chunk_enabled)) {
        output_enabled = true;
        output_type = *requested_output_type;
    }
    if (!output_enabled) {
        return;
    }
    if (auto parsed = citlali::config::parse_tod_output_type(output_type)) {
        typed_config.output.type = *parsed;
    }
}

template <class Config, class Key, class Param, class Target,
          class MissingKeys, class InvalidKeys>
void read_mirrored_config_value(
    Config &config, const Key &key, Param &param, Target &target,
    MissingKeys &missing_keys, InvalidKeys &invalid_keys,
    std::vector<std::decay_t<Param>> accepted_values = {},
    std::vector<std::decay_t<Param>> min_values = {},
    std::vector<std::decay_t<Param>> max_values = {}) {
    const auto missing_before = missing_keys.size();
    const auto invalid_before = invalid_keys.size();
    ::get_config_value(config, param, missing_keys, invalid_keys, key,
                       std::move(accepted_values), std::move(min_values),
                       std::move(max_values));
    mirror_if_config_parsed(target, param, missing_keys, invalid_keys,
                            missing_before, invalid_before);
}

}  // namespace citlali::engine_detail
