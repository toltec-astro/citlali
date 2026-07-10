#pragma once

#include <citlali/core/config/timestream_config.h>
#include <citlali/core/pipeline/config_parse_tracking.h>

#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace citlali::pipeline {

template <class Config, class MissingKeys, class InvalidKeys,
          class TimestreamConfig>
void read_timestream_enabled_config(Config &config, bool &enabled,
                                    TimestreamConfig &typed_config,
                                    MissingKeys &missing_keys,
                                    InvalidKeys &invalid_keys) {
    citlali::pipeline::read_mirrored_config_value(
        config, std::tuple{"timestream", "enabled"}, enabled,
        typed_config.enabled, missing_keys, invalid_keys);
}

template <class Config, class Diagnostics, class TimestreamConfig>
void read_timestream_enabled_config(Config &config, bool &enabled,
                                    TimestreamConfig &typed_config,
                                    Diagnostics &diagnostics) {
    read_timestream_enabled_config(
        config, enabled, typed_config, diagnostics.missing_key_paths(),
        diagnostics.invalid_key_paths());
}

template <class Config, class MissingKeys, class InvalidKeys,
          class TimestreamConfig>
void read_timestream_type_config(Config &config, std::string &type,
                                 TimestreamConfig &typed_config,
                                 MissingKeys &missing_keys,
                                 InvalidKeys &invalid_keys) {
    citlali::pipeline::read_parsed_mirrored_config_value(
        config, std::tuple{"timestream", "type"}, type, typed_config.type,
        citlali::config::parse_tod_type, missing_keys, invalid_keys);
}

template <class Config, class Diagnostics, class TimestreamConfig>
void read_timestream_type_config(Config &config, std::string &type,
                                 TimestreamConfig &typed_config,
                                 Diagnostics &diagnostics) {
    read_timestream_type_config(
        config, type, typed_config, diagnostics.missing_key_paths(),
        diagnostics.invalid_key_paths());
}

template <class Config, class MissingKeys, class InvalidKeys,
          class TimestreamConfig>
void read_auxiliary_quadrature_channel_config(Config &config,
                                              TimestreamConfig &typed_config,
                                              MissingKeys &missing_keys,
                                              InvalidKeys &invalid_keys) {
    auto &channel = typed_config.auxiliary_channels.quadrature_r;
    citlali::pipeline::read_optional_mirrored_config_value(
        config,
        std::tuple{"timestream", "auxiliary_channels", "quadrature_r",
                   "enabled"},
        channel.enabled, channel.enabled, missing_keys, invalid_keys);
    citlali::pipeline::read_optional_mirrored_config_value(
        config,
        std::tuple{"timestream", "auxiliary_channels", "quadrature_r", "name"},
        channel.name, channel.name, missing_keys, invalid_keys);

    std::string source_type{
        std::string(citlali::config::to_string(channel.source_type))};
    citlali::pipeline::read_optional_parsed_mirrored_config_value(
        config,
        std::tuple{"timestream", "auxiliary_channels", "quadrature_r",
                   "source_type"},
        source_type, channel.source_type, citlali::config::parse_tod_type,
        missing_keys, invalid_keys, {"xs", "rs", "is", "qs"});

    citlali::pipeline::read_optional_mirrored_config_value(
        config,
        std::tuple{"timestream", "auxiliary_channels", "quadrature_r",
                   "native_unit"},
        channel.native_unit, channel.native_unit, missing_keys, invalid_keys);

    std::string calibration_policy{std::string(
        citlali::config::to_string(channel.calibration_policy))};
    citlali::pipeline::read_optional_parsed_mirrored_config_value(
        config,
        std::tuple{"timestream", "auxiliary_channels", "quadrature_r",
                   "calibration_policy"},
        calibration_policy, channel.calibration_policy,
        citlali::config::parse_auxiliary_measured_channel_calibration_policy,
        missing_keys, invalid_keys,
        {"native", "primary_equivalent", "sky_equivalent"});

    citlali::pipeline::read_optional_mirrored_config_value(
        config,
        std::tuple{"timestream", "auxiliary_channels", "quadrature_r",
                   "apply_primary_transfer"},
        channel.apply_primary_linear_transfer,
        channel.apply_primary_linear_transfer, missing_keys, invalid_keys);
    citlali::pipeline::read_optional_mirrored_config_value(
        config,
        std::tuple{"timestream", "auxiliary_channels", "quadrature_r",
                   "use_for_science_map"},
        channel.use_for_science_map, channel.use_for_science_map,
        missing_keys, invalid_keys);
    citlali::pipeline::read_optional_mirrored_config_value(
        config,
        std::tuple{"timestream", "auxiliary_channels", "quadrature_r",
                   "diagnostics_enabled"},
        channel.diagnostics_enabled, channel.diagnostics_enabled,
        missing_keys, invalid_keys);
}

template <class Config, class Diagnostics, class TimestreamConfig>
void read_auxiliary_quadrature_channel_config(Config &config,
                                              TimestreamConfig &typed_config,
                                              Diagnostics &diagnostics) {
    read_auxiliary_quadrature_channel_config(
        config, typed_config, diagnostics.missing_key_paths(),
        diagnostics.invalid_key_paths());
}

template <class Config, class TimestreamConfig, class Diagnostics>
bool read_timestream_core_config(Config &config,
                                 TimestreamConfig &typed_config,
                                 Diagnostics &diagnostics) {
    bool enabled = typed_config.enabled;
    read_timestream_enabled_config(
        config, enabled, typed_config, diagnostics);
    if (!enabled) {
        return false;
    }

    std::string type{
        std::string(citlali::config::to_string(typed_config.type))};
    read_timestream_type_config(
        config, type, typed_config, diagnostics);
    read_auxiliary_quadrature_channel_config(
        config, typed_config, diagnostics);
    return true;
}

template <class Config, class MissingKeys, class InvalidKeys,
          class TimestreamConfig>
void read_raw_tod_output_enabled_config(Config &config, bool &enabled,
                                        TimestreamConfig &typed_config,
                                        MissingKeys &missing_keys,
                                        InvalidKeys &invalid_keys) {
    citlali::pipeline::read_config_value_if_clean(
        config,
        std::tuple{"timestream", "raw_time_chunk", "output", "enabled"},
        enabled,
        [&typed_config](bool output_enabled) {
            typed_config.output.raw_time_chunk_enabled = output_enabled;
            typed_config.output.raw_time_chunk.enabled = output_enabled;
        },
        missing_keys, invalid_keys);
}

template <class Config, class Diagnostics, class TimestreamConfig>
void read_raw_tod_output_enabled_config(Config &config, bool &enabled,
                                        TimestreamConfig &typed_config,
                                        Diagnostics &diagnostics) {
    read_raw_tod_output_enabled_config(
        config, enabled, typed_config, diagnostics.missing_key_paths(),
        diagnostics.invalid_key_paths());
}

template <class Config, class MissingKeys, class InvalidKeys,
          class TimestreamConfig>
void read_processed_tod_output_enabled_config(Config &config, bool &enabled,
                                              TimestreamConfig &typed_config,
                                              MissingKeys &missing_keys,
                                              InvalidKeys &invalid_keys) {
    citlali::pipeline::read_config_value_if_clean(
        config,
        std::tuple{"timestream", "processed_time_chunk", "output",
                   "enabled"},
        enabled,
        [&typed_config](bool output_enabled) {
            typed_config.output.processed_time_chunk_enabled = output_enabled;
            typed_config.output.processed_time_chunk.enabled = output_enabled;
        },
        missing_keys, invalid_keys);
}

template <class Config, class Diagnostics, class TimestreamConfig>
void read_processed_tod_output_enabled_config(Config &config, bool &enabled,
                                              TimestreamConfig &typed_config,
                                              Diagnostics &diagnostics) {
    read_processed_tod_output_enabled_config(
        config, enabled, typed_config, diagnostics.missing_key_paths(),
        diagnostics.invalid_key_paths());
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
    citlali::pipeline::read_config_value_if_clean(
        config, key, mode,
        [&typed_stream](const std::string &mode_name) {
            if (auto parsed =
                    citlali::config::parse_tod_stream_output_mode(mode_name)) {
                typed_stream.mode = *parsed;
            }
        },
        missing_keys, invalid_keys, allowed_modes);
    citlali::config::TodStreamOutputMode parsed_mode =
        citlali::config::TodStreamOutputMode::full;
    if (auto parsed = citlali::config::parse_tod_stream_output_mode(mode)) {
        parsed_mode = *parsed;
    }
    citlali::config::TodStreamOutputMode stream_mode =
        citlali::pipeline::config_parse_clean(
            missing_keys, invalid_keys, missing_before, invalid_before)
            ? typed_stream.mode
            : parsed_mode;
    mini = citlali::config::is_mini_tod_stream_output_mode(stream_mode);
    outer = citlali::config::is_outer_tod_stream_output_mode(stream_mode);
}

template <class Config, class Key, class Diagnostics,
          class StreamOutputConfig>
void read_tod_stream_output_mode_config(
    Config &config, const Key &key, bool output_enabled,
    const std::vector<std::string> &allowed_modes, std::string &mode,
    bool &mini, bool &outer, StreamOutputConfig &typed_stream,
    Diagnostics &diagnostics) {
    read_tod_stream_output_mode_config(
        config, key, output_enabled, allowed_modes, mode, mini, outer,
        typed_stream, diagnostics.missing_key_paths(),
        diagnostics.invalid_key_paths());
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
    citlali::pipeline::read_config_value_if_clean(
        config, key, outer_context_samples,
        [&typed_stream](value_type count) {
            typed_stream.outer_context_samples = static_cast<int>(count);
        },
        missing_keys, invalid_keys, {}, {0});
}

template <class Config, class Key, class Diagnostics, class ContextSamples,
          class StreamOutputConfig>
void read_tod_stream_outer_context_config(
    Config &config, const Key &key, bool output_enabled,
    ContextSamples &outer_context_samples, StreamOutputConfig &typed_stream,
    Diagnostics &diagnostics) {
    read_tod_stream_outer_context_config(
        config, key, output_enabled, outer_context_samples, typed_stream,
        diagnostics.missing_key_paths(), diagnostics.invalid_key_paths());
}

template <class TimestreamConfig>
void sync_tod_output_type_config(bool raw_time_chunk_enabled,
                                 bool processed_time_chunk_enabled,
                                 bool &output_enabled,
                                 TimestreamConfig &typed_config) {
    output_enabled = false;
    const auto output_type = citlali::config::enabled_tod_output_type(
        raw_time_chunk_enabled, processed_time_chunk_enabled);
    if (citlali::config::is_tod_output_enabled(output_type)) {
        typed_config.output.type = output_type;
        output_enabled = true;
        return;
    }
    typed_config.output.type = citlali::config::TodOutputType::none;
}

}  // namespace citlali::pipeline
