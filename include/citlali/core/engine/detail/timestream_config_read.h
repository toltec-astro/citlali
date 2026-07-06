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
    read_mirrored_config_value(
        config, std::tuple{"timestream", "enabled"}, enabled,
        typed_config.enabled, missing_keys, invalid_keys);
}

template <class Config, class MissingKeys, class InvalidKeys,
          class TimestreamConfig>
void read_timestream_type_config(Config &config, std::string &type,
                                 TimestreamConfig &typed_config,
                                 MissingKeys &missing_keys,
                                 InvalidKeys &invalid_keys) {
    read_parsed_mirrored_config_value(
        config, std::tuple{"timestream", "type"}, type, typed_config.type,
        citlali::config::parse_tod_type, missing_keys, invalid_keys);
}

template <class Config, class MissingKeys, class InvalidKeys,
          class TimestreamConfig>
void read_raw_tod_output_enabled_config(Config &config, bool &enabled,
                                        TimestreamConfig &typed_config,
                                        MissingKeys &missing_keys,
                                        InvalidKeys &invalid_keys) {
    read_config_value_if_clean(
        config,
        std::tuple{"timestream", "raw_time_chunk", "output", "enabled"},
        enabled,
        [&typed_config](bool output_enabled) {
            typed_config.output.raw_time_chunk_enabled = output_enabled;
            typed_config.output.raw_time_chunk.enabled = output_enabled;
        },
        missing_keys, invalid_keys);
}

template <class Config, class MissingKeys, class InvalidKeys,
          class TimestreamConfig>
void read_processed_tod_output_enabled_config(Config &config, bool &enabled,
                                              TimestreamConfig &typed_config,
                                              MissingKeys &missing_keys,
                                              InvalidKeys &invalid_keys) {
    read_config_value_if_clean(
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

    read_config_value_if_clean(
        config, key, mode,
        [&typed_stream](const std::string &mode_name) {
            if (auto parsed =
                    citlali::config::parse_tod_stream_output_mode(mode_name)) {
                typed_stream.mode = *parsed;
            }
        },
        missing_keys, invalid_keys, allowed_modes);
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
    read_config_value_if_clean(
        config, key, outer_context_samples,
        [&typed_stream](value_type count) {
            typed_stream.outer_context_samples = static_cast<int>(count);
        },
        missing_keys, invalid_keys, {}, {0});
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

template <class Chunks>
void sync_legacy_tod_output_selection_state(
    bool raw_time_chunk_enabled, bool processed_time_chunk_enabled,
    bool raw_chunk_select_enabled, bool processed_chunk_select_enabled,
    Chunks &raw_output_chunks, Chunks &processed_output_chunks,
    bool &stored_raw_chunk_select_enabled,
    bool &stored_processed_chunk_select_enabled,
    Chunks &stored_raw_output_chunks, Chunks &stored_processed_output_chunks,
    bool &legacy_chunk_select_enabled, Chunks &legacy_output_chunks) {
    stored_raw_chunk_select_enabled = raw_chunk_select_enabled;
    stored_processed_chunk_select_enabled = processed_chunk_select_enabled;
    stored_raw_output_chunks = std::move(raw_output_chunks);
    stored_processed_output_chunks = std::move(processed_output_chunks);

    citlali::pipeline::align_legacy_tod_output_selection(
        raw_time_chunk_enabled, processed_time_chunk_enabled,
        stored_raw_chunk_select_enabled, stored_processed_chunk_select_enabled,
        stored_raw_output_chunks, stored_processed_output_chunks,
        legacy_chunk_select_enabled, legacy_output_chunks);
}

}  // namespace citlali::engine_detail
